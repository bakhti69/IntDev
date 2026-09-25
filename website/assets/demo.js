// Live demo: sends the upload to the Hugging Face Space (demo/app.py, endpoint /detect)
// and renders the answer with the same timeline / risk components as the results section.
// Loaded lazily so that a blocked CDN shows an error instead of breaking the page.
const CLIENT_URL = window.GRADIO_CLIENT_URL || "https://cdn.jsdelivr.net/npm/@gradio/client@2.7.0/dist/index.min.js";

const S = window.SITE || {};
const $ = (s) => document.querySelector(s);
const MAX_MB = 200, MAX_SEC = 120;
let file = null;

const status = (msg) => { $("#demo-status").textContent = msg; };
const bar = $("#demo-progress");
const spaceUrl = `https://huggingface.co/spaces/${S.space}`;
$("#demo-fallback").innerHTML = `Demo backend: <a href="${spaceUrl}" target="_blank" rel="noopener">${S.space}</a> on Hugging Face Spaces (also usable directly there).`;

function pick(f) {
  if (!f) return;
  if (!/\.mp4$/i.test(f.name)) { status("Please choose an .mp4 file."); return; }
  if (f.size > MAX_MB * 1e6) { status(`File is ${(f.size / 1e6).toFixed(0)} MB; the limit is ${MAX_MB} MB.`); return; }
  const probe = document.createElement("video");
  probe.preload = "metadata";
  probe.onloadedmetadata = () => {
    URL.revokeObjectURL(probe.src);
    if (probe.duration > MAX_SEC + 1) { status(`Clip is ${probe.duration.toFixed(0)} s; the demo accepts up to ${MAX_SEC} s.`); return; }
    file = f;
    $("#file-name").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB · ${probe.duration.toFixed(1)} s`;
    $("#run").disabled = false;
    status("Ready.");
  };
  probe.onerror = () => { file = f; $("#run").disabled = false; $("#file-name").textContent = f.name; };
  probe.src = URL.createObjectURL(f);
}

const drop = $("#drop");
$("#file").addEventListener("change", (e) => pick(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

$("#run").addEventListener("click", async () => {
  if (!file) return;
  const TW = window.TW;
  $("#run").disabled = true;
  $("#demo-out").innerHTML = "";
  bar.hidden = false; bar.removeAttribute("value");
  try {
    status("Connecting to the model server (a sleeping Space can take ~1 min to wake)…");
    const { Client, handle_file } = await import(CLIENT_URL);
    const app = await Client.connect(S.space);
    status("Uploading…");
    const job = app.submit("/detect", { video_path: handle_file(file), with_risk: $("#with-risk").checked });
    let data = null;
    for await (const msg of job) {
      if (msg.type === "status") {
        const p = msg.progress_data && msg.progress_data[0];
        if (p && p.progress != null) { bar.value = p.progress; status(`${p.desc || "processing"} ${(100 * p.progress).toFixed(0)} %`); }
        else if (msg.stage === "pending") status(msg.queue_size ? `Queued (position ${msg.position ?? "?"})…` : "Processing…");
        else if (msg.stage === "error") throw new Error(msg.message || "server error");
      } else if (msg.type === "data") { data = msg.data; break; }
    }
    if (!data) throw new Error("no result");
    const [videoOut, , , , jsonOut, note] = data;
    const result = await fetch(jsonOut.url).then((r) => r.json());
    const duration = Math.max(1, ...result.risk.map((r) => r[0]), ...result.events.map((e) => e[1]));
    const out = $("#demo-out");
    const video = document.createElement("video");
    Object.assign(video, { controls: true, playsInline: true, src: (videoOut.video || videoOut).url });
    video.style.width = "100%"; video.style.borderRadius = "8px";
    const seek = (e) => { video.currentTime = Math.max(0, e[0] - 1); video.play().catch(() => {}); };
    const card = (title, child) => { const c = document.createElement("div"); c.className = "card"; c.style.marginTop = "12px"; c.innerHTML = `<div class="small muted">${title}</div>`; c.append(child); return c; };
    const rows = result.events.map((e) => `<tr><td>${e[0].toFixed(2)}</td><td>${e[1].toFixed(2)}</td><td><span class="pill" style="background:${TW.COLORS[e[2]]}22;color:${TW.COLORS[e[2]]}">${e[2]}</span></td></tr>`).join("");
    const table = document.createElement("div");
    table.className = "table-wrap";
    table.innerHTML = `<table><thead><tr><th>start</th><th>end</th><th>label</th></tr></thead><tbody>${rows || '<tr><td colspan="3" class="muted">no events</td></tr>'}</tbody></table>`;
    const dl = document.createElement("a");
    dl.href = jsonOut.url; dl.textContent = "download events.json"; dl.className = "btn"; dl.download = "events.json";
    out.append(video, card("Event timeline — click to jump", TW.timelineSVG(result.events, duration, { onClick: seek })));
    if (result.risk.length) out.append(card("Accident risk (alarm at 0.5)", TW.riskSVG(result.risk.filter((_, i) => i % 3 === 0), duration)));
    out.append(card("Events", table), dl);
    video.addEventListener("timeupdate", () => TW.moveCursors(out, video.currentTime));
    status(note);
  } catch (err) {
    status(`Demo failed: ${err.message || err}. You can also try it directly on ${spaceUrl}`);
  } finally {
    bar.hidden = true;
    $("#run").disabled = false;
  }
});
