/* TrafficWatch site: renders team, EDA, results and dashboard from data/*.json. No build step, no deps. */
(() => {
  "use strict";
  const S = window.SITE || {};
  const $ = (sel, root = document) => root.querySelector(sel);
  const { COLORS, esc, fmt, el, timelineSVG, riskSVG, moveCursors, lineChart, barChart } = window.TW;
  const getJSON = (url) => fetch(url, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : Promise.reject(r.status)));

  /* ---------- theme + links ---------- */
  $("#theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tw-theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
  });
  try { const t = localStorage.getItem("tw-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
  const repo = S.repo || "#";
  $("#repo-link").href = repo;
  $("#l-repo").href = repo;
  $("#l-weights").href = repo + "/tree/main/weights";
  $("#l-pred").href = repo + "/blob/main/predictions_samples.json";

  /* ---------- team ---------- */
  const team = $("#team-cards");
  (S.members || []).forEach((m) => {
    const links = [["GitHub", m.github], ["LinkedIn", m.linkedin], ["Portfolio", m.portfolio]]
      .filter(([, u]) => u).map(([k, u]) => `<a href="${esc(u)}" target="_blank" rel="noopener">${k}</a>`).join(" · ");
    team.append(el("div", { class: "card" }, `<h3>${esc(m.name)}</h3><div class="muted small">${esc(m.role)}</div>
      <p class="small">${esc(m.did)}</p><p class="small"><b>Proud of:</b> ${esc(m.proud || "")}</p>
      <div class="small">${links || '<span class="muted">links: add in config.js</span>'}</div>`));
  });

  /* ---------- class table ---------- */
  const CLASS_RULES = [
    ["accident", "Two road users reach contact (ground distance < 0.9 body lengths, boxes overlap) after approaching from ≥ 1.5, with an impact-like deceleration or deflection, then stay together.", "first contact → both at rest / leave"],
    ["near_miss", "Time-to-collision < 1.5 s, closest gap 0.9–1.8 body lengths (no contact), with sharp braking (≤ −1.5 bl/s²) or a swerve (≥ 45°/s).", "evasive action → gap > 2.5"],
    ["red_light", "Front of a vehicle crosses a stop line while its approach is red: vehicle head read directly (away-flow); pedestrian WALK or vehicles held at the line (flow towards camera).", "crossing → leaves junction"],
    ["wrong_way", "Moving > 120° against the legal direction of its carriageway for ≥ 1.2 s and ≥ 1.5 body lengths.", "enters → returns / leaves frame"],
    ["illegal_u_turn", "Heading reverses by ≥ 150° within 20 s over ≥ 2 body lengths of path.", "starts turning → settles"],
    ["stopped_vehicle", "Stationary ≥ 10 s on the carriageway while ≥ 3 moving vehicles pass it (a queue moves as a whole). Buses skipped (bus stop).", "stops → moves / leaves"],
    ["jaywalking", "Pedestrian (not a rider or passenger) with feet on the carriageway, outside crosswalks, for ≥ 1 s.", "steps on → leaves road"],
    ["failure_to_yield", "Vehicle footprint crosses a crosswalk at speed while a pedestrian is on it within 4 body lengths.", "enters → leaves crossing"],
    ["illegal_turn", "Movement between zones listed as prohibited in the scene config (none confirmed yet → never predicted).", "starts turning → completes"],
    ["solid_line_crossing", "Ground point changes side of a solid lane divider by ≥ 30 % of the box width within 4 s.", "wheel on line → fully across"],
    ["stop_line", "Stands still past the stop line (inside the box before the junction) for ≥ 2 s during red.", "stops → moves (green)"],
    ["congestion", "Per direction: ≥ 5–6 vehicles, ≥ 80 % crawling, sustained (20 s away-flow, 75 s for the signal queue).", "queue stops → clears"],
    ["road_obstacle", "Animal on the road, or a compact new object that appears, stays ≥ 5 s and is not a detected road user.", "appears → removed"],
    ["fire_smoke", "Flame-coloured blob at a fixed place whose area flickers (CV ≥ 0.2) for ≥ 2 s.", "first flame → clears"],
  ];
  const tbody = $("#class-table tbody");
  CLASS_RULES.forEach(([c, rule, se]) => tbody.append(el("tr", {},
    `<td><span class="pill" style="background:${COLORS[c]}22;color:${COLORS[c]}">${c}</span></td><td class="small">${esc(rule)}</td><td class="small muted">${esc(se)}</td>`)));

  /* ---------- EDA: stills ---------- */
  const FINDINGS = [
    ["Framing drifts between clips", "Registering our stills to the reference gives offsets up to (−51, +32) px and ~1° rotation, so a fixed pixel layout would miss the stop line. → SIFT registration per video."],
    ["Day and dusk", "Mean brightness ranges from ~92 (noon) to ~27 (dusk). Detection holds up at dusk; headlights make LEDs bloom, so signal pixels are counted with loose saturation/value thresholds."],
    ["Two kinds of signal heads", "The left-pole head is a pedestrian signal (walking/standing figure); the median head is the vehicle signal for traffic leaving up the avenue. The queue approaching the camera has its signal facing away."],
    ["Where the queue forms", "Up to ~15 vehicles wait between the gantry and the stop line on 4–5 lanes; stopped-vehicle and congestion rules must not fire on a normal red phase."],
    ["Pedestrians everywhere", "Besides the two zebra crossings, people cut diagonally across the junction (frame 2) — real jaywalking candidates — and stand on the pink islands and the median (excluded from the carriageway)."],
    ["Perspective", "A car is ~200 px wide in the foreground and ~40 px at the far end: speeds are normalised by apparent size (body lengths/s)."],
  ];
  const findings = $("#eda-findings");
  FINDINGS.forEach(([h, p]) => findings.append(el("div", { class: "card" }, `<h3>${esc(h)}</h3><p class="small">${esc(p)}</p>`)));

  getJSON("data/frames.json").then((frames) => {
    const tabs = $("#frame-tabs"), fig = $("#frame-figure"), stats = $("#frame-stats");
    const show = (i) => {
      const f = frames[i];
      tabs.querySelectorAll("button").forEach((b, k) => b.classList.toggle("on", k === i));
      fig.innerHTML = `<img src="${f.image}" alt="${esc(f.name)} with the registered scene layout and detections">
        <div class="cap">Registered layout (carriageways, crosswalks, stop lines, lane lines, signal heads) and YOLO11s detections. <a href="${f.raw}" target="_blank">raw frame</a></div>`;
      const zones = Object.entries(f.by_zone).map(([z, c]) => `<tr><td>${esc(z)}</td><td>${Object.entries(c).map(([k, v]) => `${esc(k)} ${v}`).join(", ")}</td></tr>`).join("");
      stats.innerHTML = `<div class="grid g4">
          <div class="stat"><div class="v">${f.width}×${f.height}</div><div class="k">resolution (still)</div></div>
          <div class="stat"><div class="v">${fmt(f.brightness)}</div><div class="k">mean brightness</div></div>
          <div class="stat"><div class="v">${f.inliers}</div><div class="k">SIFT inliers</div></div>
          <div class="stat"><div class="v">${f.offset_vs_reference.map((v) => fmt(v, 0)).join(", ")}</div><div class="k">offset vs reference (px)</div></div>
        </div>
        <p class="small">Signal heads: ${Object.entries(f.signals).map(([k, v]) => `<b>${esc(k)}</b> ${esc(v)}`).join(" · ")}</p>
        <div class="table-wrap"><table><thead><tr><th>zone</th><th>detections</th></tr></thead><tbody>${zones}</tbody></table></div>`;
    };
    frames.forEach((f, i) => { const b = el("button", {}, esc(f.name.replace(".jpg", ""))); b.onclick = () => show(i); tabs.append(b); });
    if (frames.length) show(0);
  }).catch(() => { $("#frame-figure").innerHTML = '<p class="pending">frames.json not built yet (tools/frame_eda.py).</p>'; });

  /* ---------- videos: EDA + results + dashboard ---------- */
  getJSON("data/videos.json").then(async (list) => {
    if (!list.length) return;
    const videos = await Promise.all(list.map((v) => getJSON(`data/${v.stem}.json`)));
    renderVideoEDA(videos);
    renderResults(videos);
    renderDashboard(videos);
  }).catch(() => { /* no sample data yet: placeholders stay */ });

  function renderVideoEDA(videos) {
    const root = $("#video-eda");
    root.innerHTML = `<h3>Sample videos</h3><div class="table-wrap card" style="padding:4px 12px"><table><thead><tr>
      <th>video</th><th>resolution</th><th>fps</th><th>duration</th><th>brightness</th><th>tracks</th><th>registration</th></tr></thead><tbody>${
      videos.map((v) => { const s = v.eda.summary; return `<tr><td>${esc(v.name)}</td><td>${s.width}×${s.height}</td><td>${s.fps}</td><td>${fmt(s.duration)} s</td><td>${fmt(s.mean_brightness)}</td>
        <td class="small">${Object.entries(s.tracks_by_class).map(([k, n]) => `${k} ${n}`).join(", ")}</td><td>${s.registered ? "✓ " + s.registration_inliers : "✗"}</td></tr>`; }).join("")}</tbody></table></div>`;
    videos.forEach((v) => {
      const e = v.eda, m = v.media;
      const counts = Object.keys(e.counts).filter((k) => k !== "t").map((k) => ({ name: k, values: e.counts[k] }));
      const occ = Object.entries(e.zones.vehicles).map(([z, vals]) => ({ name: z, values: vals }));
      const spd = Object.entries(e.zones.median_speed).map(([z, vals]) => ({ name: z, values: vals }));
      const hist = e.speeds.counts.map((c, i) => ({ label: `${e.speeds.edges[i]}–${e.speeds.edges[i + 1]}`, value: c }));
      const card = el("div", { class: "card", style: "margin-top:16px" }, `<h3>${esc(v.name)}</h3>
        <div class="grid g2">
          <div><div class="small muted">Detections per frame by class</div>${lineChart(e.counts.t, counts)}</div>
          <div><div class="small muted">Vehicles per zone (traffic density)</div>${lineChart(e.zones.t, occ)}</div>
          <div><div class="small muted">Median speed per zone (body lengths / s)</div>${lineChart(e.zones.t, spd)}</div>
          <div><div class="small muted">Lighting (mean grey level)</div>${lineChart(e.lighting.t, [{ name: "brightness", values: e.lighting.brightness }])}</div>
        </div>
        <div class="grid g3" style="margin-top:12px">
          <div class="figure"><img loading="lazy" src="media/${m.heatmap}" alt="motion heat map"><div class="cap">Motion heat map (moving road users)</div></div>
          <div class="figure"><img loading="lazy" src="media/${m.trajectories}" alt="trajectories"><div class="cap">Trajectories (red cars, orange buses/trucks, blue pedestrians, green two-wheelers)</div></div>
          <div class="figure"><img loading="lazy" src="media/${m.directions}" alt="lane directions"><div class="cap">Dominant direction per cell = lane directions</div></div>
        </div>
        <details style="margin-top:10px"><summary class="small">Speed distribution (vehicles), ${fmt(100 * (e.speeds.stationary_fraction || 0), 0)} % of samples stationary</summary>${barChart(hist.filter((d) => d.value > 0))}</details>`);
      root.append(card);
    });
  }

  function renderResults(videos) {
    const tabs = $("#video-tabs"), panel = $("#video-panel");
    const show = (i) => {
      const v = videos[i], dur = v.eda.summary.duration;
      tabs.querySelectorAll("button").forEach((b, k) => b.classList.toggle("on", k === i));
      panel.innerHTML = "";
      const video = el("video", { controls: "", preload: "metadata", playsinline: "", src: `media/${v.media.video}`, poster: `media/${v.media.background}` });
      video.style.width = "100%"; video.style.borderRadius = "8px";
      const seek = (e) => { video.currentTime = Math.max(0, e[0] - 1); video.play().catch(() => {}); };
      const tl = timelineSVG(v.events, dur, { onClick: seek });
      const rk = v.risk && v.risk.length ? riskSVG(v.risk, dur) : el("p", { class: "pending" }, "no risk curve");
      const rows = v.events.map((e, k) => `<tr data-k="${k}" style="cursor:pointer"><td>${fmt(e[0], 2)}</td><td>${fmt(e[1], 2)}</td><td><span class="pill" style="background:${COLORS[e[2]]}22;color:${COLORS[e[2]]}">${e[2]}</span></td></tr>`).join("");
      const box = el("div", { class: "grid g2" });
      const left = el("div", {}); left.append(video);
      const right = el("div", { class: "card" }, `<div class="small muted">${v.events.length} events · click a row to jump</div>
        <div class="table-wrap" style="max-height:360px;overflow:auto"><table><thead><tr><th>start</th><th>end</th><th>label</th></tr></thead><tbody>${rows}</tbody></table></div>`);
      right.querySelectorAll("tr[data-k]").forEach((r) => r.addEventListener("click", () => seek(v.events[+r.dataset.k])));
      box.append(left, right);
      const tlCard = el("div", { class: "card", style: "margin-top:12px" }, '<div class="small muted">Event timeline</div>'); tlCard.append(tl);
      const rkCard = el("div", { class: "card", style: "margin-top:12px" }, '<div class="small muted">Accident risk — P(accident starts within 5 s), alarm at 0.5 (causal: uses past frames only)</div>'); rkCard.append(rk);
      panel.append(box, tlCard, rkCard);
      video.addEventListener("timeupdate", () => moveCursors(panel, video.currentTime));
    };
    tabs.innerHTML = "";
    videos.forEach((v, i) => { const b = el("button", {}, esc(v.name)); b.onclick = () => show(i); tabs.append(b); });
    show(0);
  }

  function renderDashboard(videos) {
    const per = {}, dur = {};
    let total = 0;
    videos.forEach((v) => {
      total += v.eda.summary.duration;
      v.events.forEach(([s, e, lab]) => { per[lab] = (per[lab] || 0) + 1; dur[lab] = (dur[lab] || 0) + (e - s); });
    });
    const items = Object.keys(per).sort((a, b) => per[b] - per[a]).map((k) => ({ label: k, value: per[k], color: COLORS[k], text: `${per[k]} · ${fmt(dur[k], 0)} s` }));
    const n = Object.values(per).reduce((a, b) => a + b, 0);
    const peak = videos.map((v) => ({ label: v.name, value: v.events.length / Math.max(1, v.eda.summary.duration / 60), text: fmt(v.events.length / Math.max(1, v.eda.summary.duration / 60), 1) + " /min" }));
    $("#dash").innerHTML = `<div class="grid g4">
        <div class="card stat"><div class="v">${videos.length}</div><div class="k">videos</div></div>
        <div class="card stat"><div class="v">${fmt(total / 60, 1)} min</div><div class="k">footage</div></div>
        <div class="card stat"><div class="v">${n}</div><div class="k">events</div></div>
        <div class="card stat"><div class="v">${fmt(n / Math.max(1, total / 60), 1)}</div><div class="k">events per minute</div></div>
      </div>
      <div class="grid g2" style="margin-top:16px">
        <div class="card"><h3>Events by class (count · total time)</h3>${items.length ? barChart(items) : '<p class="muted">none</p>'}</div>
        <div class="card"><h3>Event rate per video</h3>${barChart(peak, { color: "#1c7ed6" })}</div>
      </div>`;
  }
})();
