"""A video whose analysis dies must not take the harness down with it."""
from __future__ import annotations

from trafficwatch.cli import run_isolated


def test_failing_video_gives_empty_events(tmp_path, capsys):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert run_isolated(str(broken)) == []
    assert "retrying with single-threaded decoding" in capsys.readouterr().err
