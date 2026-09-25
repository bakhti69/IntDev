"""Live demo: upload a clip from the camera, get the events, an annotated
playback, the event timeline and the accident-risk curve.

Runs on CPU (e.g. a free Hugging Face Space):  python demo/app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent   # repo checkout or bundled Space
sys.path.insert(0, str(ROOT / "src"))

from trafficwatch.api import process  # noqa: E402
from trafficwatch.config import Settings  # noqa: E402
from trafficwatch.render import render  # noqa: E402
from trafficwatch.video import video_info  # noqa: E402
from trafficwatch.viz import EVENT_COLORS  # noqa: E402

MAX_SECONDS = 120
MAX_MB = 200
# CPU-friendly settings: fewer analysed frames, smaller input size
CPU_SETTINGS = Settings(imgsz=640, stride=4, risk_stride=8, risk_imgsz=640, batch=4)


def _hex(bgr) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def timeline_svg(events: list[list], duration: float, width: int = 900) -> str:
    labels = sorted({e[2] for e in events})
    row_h, left = 22, 150
    height = max(1, len(labels)) * row_h + 30
    scale = (width - left - 10) / max(duration, 1e-6)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
             f'style="font:12px system-ui, sans-serif">']
    for i, lab in enumerate(labels):
        y = 5 + i * row_h
        parts.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" fill="currentColor">{lab}</text>')
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{width - left - 10}" height="{row_h - 6}" '
                     f'fill="currentColor" opacity="0.06"/>')
        for s, e, lb in events:
            if lb == lab:
                parts.append(f'<rect x="{left + s * scale:.1f}" y="{y + 3}" width="{max(2, (e - s) * scale):.1f}" '
                             f'height="{row_h - 6}" rx="3" fill="{_hex(EVENT_COLORS.get(lab, (150, 150, 150)))}">'
                             f'<title>{lab}: {s:.1f}-{e:.1f} s</title></rect>')
    if not labels:
        parts.append(f'<text x="{width / 2}" y="18" text-anchor="middle" fill="currentColor">no events</text>')
    y_axis = height - 8
    for k in range(0, int(duration) + 1, max(5, int(duration / 10) // 5 * 5 or 5)):
        parts.append(f'<text x="{left + k * scale:.1f}" y="{y_axis}" text-anchor="middle" '
                     f'fill="currentColor" opacity="0.7">{k}s</text>')
    parts.append("</svg>")
    return "".join(parts)


def risk_svg(risk: list[list[float]], duration: float, width: int = 900, height: int = 140) -> str:
    left, pad = 40, 10
    sx = (width - left - pad) / max(duration, 1e-6)
    sy = height - 30
    pts = " ".join(f"{left + t * sx:.1f},{pad + (1 - r) * sy:.1f}" for t, r in risk[::2])
    y_th = pad + 0.5 * sy
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
            f'style="font:12px system-ui, sans-serif">'
            f'<line x1="{left}" x2="{width - pad}" y1="{y_th}" y2="{y_th}" stroke="#d33" stroke-dasharray="4 4"/>'
            f'<text x="{left - 6}" y="{y_th + 4}" text-anchor="end" fill="#d33">0.5</text>'
            f'<text x="{left - 6}" y="{pad + 4}" text-anchor="end" fill="currentColor">1</text>'
            f'<text x="{left - 6}" y="{pad + sy + 4}" text-anchor="end" fill="currentColor">0</text>'
            f'<polyline fill="none" stroke="#e8590c" stroke-width="1.5" points="{pts}"/></svg>')


def run(video_path: str | None, with_risk: bool, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload an .mp4 first.")
    if Path(video_path).stat().st_size > MAX_MB * 1e6:
        raise gr.Error(f"File is larger than {MAX_MB} MB.")
    info = video_info(video_path)
    if info.duration > MAX_SECONDS + 1:
        raise gr.Error(f"Clip is {info.duration:.0f} s long; the demo accepts up to {MAX_SECONDS} s.")

    def report(frac: float, stage: str) -> None:
        span = {"detect": (0.0, 0.55), "risk": (0.55, 0.85)}[stage]
        progress(span[0] + frac * (span[1] - span[0]), desc=f"{stage}…")

    rep = process(video_path, CPU_SETTINGS, risk=None if with_risk else [], progress=report)
    progress(0.87, desc="rendering…")
    out_dir = Path(tempfile.mkdtemp(prefix="trafficwatch_"))
    video_out = render(rep.analysis, rep.events, rep.risk or None, out_dir / "annotated.mp4")
    json_out = out_dir / "events.json"
    json_out.write_text(json.dumps({"video": Path(video_path).name, "events": rep.events,
                                    "risk": rep.risk}, indent=1))
    rows = [[f"{s:.2f}", f"{e:.2f}", lab] for s, e, lab in rep.events]
    scene = rep.analysis.scene
    reg = f"ok ({scene.inliers} matches)" if scene.registered else "failed – layout rescaled only"
    note = (f"{len(rep.events)} event(s) · {info.width}×{info.height} @ {info.fps:.1f} fps, "
            f"{info.duration:.1f} s · scene registration: {reg}")
    risk_html = risk_svg(rep.risk, info.duration) if rep.risk else "<p>Risk curve skipped.</p>"
    return str(video_out), timeline_svg(rep.events, info.duration), risk_html, rows, str(json_out), note


with gr.Blocks(title="TrafficWatch – live demo") as demo:
    gr.Markdown(
        "## TrafficWatch — traffic events & accident anticipation\n"
        f"Upload an **.mp4 up to {MAX_SECONDS} s / {MAX_MB} MB** from the competition camera. "
        "The scene layout (lanes, stop line, crosswalks) is calibrated for that camera; other views "
        "still run but zone-based classes will be unreliable. CPU inference: expect roughly "
        "2× the clip length."
    )
    with gr.Row():
        inp = gr.Video(label="Input clip", sources=["upload"])
        out_video = gr.Video(label="Annotated playback")
    with gr.Row():
        with_risk = gr.Checkbox(value=True, label="Compute the accident-risk curve (Part B)")
        btn = gr.Button("Detect events", variant="primary")
    status = gr.Markdown()
    gr.Markdown("### Event timeline")
    timeline = gr.HTML()
    gr.Markdown("### Accident risk (P(accident within 5 s); alarm at 0.5)")
    risk = gr.HTML()
    table = gr.Dataframe(headers=["start_sec", "end_sec", "label"], label="Events", interactive=False)
    download = gr.File(label="events.json")
    btn.click(run, [inp, with_risk], [out_video, timeline, risk, table, download, status],
              api_name="detect", concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0")
