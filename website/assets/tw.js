/* Shared helpers (palette, timeline / risk / line / bar charts) for the site, the demo and the labeler. */
(() => {
  "use strict";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));
  const el = (tag, attrs = {}, html = "") => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (html) n.innerHTML = html;
    return n;
  };

  // Same palette as src/trafficwatch/viz.py (converted from BGR)
  const COLORS = {
    accident: "#e62828", near_miss: "#ff8c00", red_light: "#c83c3c", wrong_way: "#c800c8",
    illegal_u_turn: "#fa50b4", stopped_vehicle: "#ffd700", jaywalking: "#00b4ff", failure_to_yield: "#3c78ff",
    illegal_turn: "#dc3c96", solid_line_crossing: "#00dcdc", stop_line: "#ff5050", congestion: "#b46400",
    road_obstacle: "#64c864", fire_smoke: "#ff4500",
  };
  const SERIES = ["#e8590c", "#1c7ed6", "#2b8a3e", "#ae3ec9", "#f59f00", "#0c8599", "#e03131", "#495057"];


  /* ---------- SVG helpers ---------- */
  function timelineSVG(events, duration, { onClick, width = 900 } = {}) {
    const labels = [...new Set(events.map((e) => e[2]))].sort();
    const rowH = 20, left = 130, right = 10, top = 6;
    const h = Math.max(1, labels.length) * rowH + top + 22;
    const sx = (width - left - right) / Math.max(duration, 1e-6);
    let s = `<svg class="timeline" viewBox="0 0 ${width} ${h}" preserveAspectRatio="none" style="height:auto">`;
    labels.forEach((lab, i) => {
      const y = top + i * rowH;
      s += `<text class="row-label" x="${left - 8}" y="${y + 13}" text-anchor="end">${esc(lab)}</text>`;
      s += `<rect x="${left}" y="${y + 2}" width="${width - left - right}" height="${rowH - 4}" fill="currentColor" opacity="0.05"/>`;
      events.forEach((e, k) => {
        if (e[2] !== lab) return;
        s += `<rect class="ev" data-k="${k}" x="${(left + e[0] * sx).toFixed(1)}" y="${y + 2}" width="${Math.max(2, (e[1] - e[0]) * sx).toFixed(1)}" height="${rowH - 4}" rx="3" fill="${COLORS[lab] || "#888"}"><title>${lab}: ${e[0].toFixed(1)}–${e[1].toFixed(1)} s</title></rect>`;
      });
    });
    if (!labels.length) s += `<text class="axis" x="${width / 2}" y="16" text-anchor="middle">no events</text>`;
    const step = duration > 300 ? 60 : duration > 120 ? 30 : duration > 40 ? 10 : 5;
    for (let t = 0; t <= duration; t += step) s += `<text class="axis" x="${(left + t * sx).toFixed(1)}" y="${h - 6}" text-anchor="middle">${t}s</text>`;
    s += `<line class="cursor" x1="${left}" x2="${left}" y1="0" y2="${h - 18}" data-left="${left}" data-sx="${sx}"/></svg>`;
    const wrap = el("div", {}, s);
    if (onClick) wrap.querySelectorAll("rect.ev").forEach((r) => r.addEventListener("click", () => onClick(events[+r.dataset.k])));
    return wrap;
  }

  function riskSVG(risk, duration, { width = 900, height = 130 } = {}) {
    const left = 34, right = 10, top = 8, bottom = 20;
    const sx = (width - left - right) / Math.max(duration, 1e-6), sy = height - top - bottom;
    const pts = risk.map(([t, r]) => `${(left + t * sx).toFixed(1)},${(top + (1 - r) * sy).toFixed(1)}`).join(" ");
    const yTh = top + 0.5 * sy;
    let s = `<svg class="timeline" viewBox="0 0 ${width} ${height}" style="height:auto">
      <line x1="${left}" x2="${width - right}" y1="${yTh}" y2="${yTh}" stroke="#c92a2a" stroke-dasharray="4 4"/>
      <text class="axis" x="${left - 5}" y="${yTh + 3}" text-anchor="end" fill="#c92a2a">0.5</text>
      <text class="axis" x="${left - 5}" y="${top + 4}" text-anchor="end">1</text>
      <text class="axis" x="${left - 5}" y="${top + sy + 3}" text-anchor="end">0</text>
      <polyline fill="none" stroke="#e8590c" stroke-width="1.6" points="${pts}"/>`;
    const step = duration > 300 ? 60 : duration > 120 ? 30 : duration > 40 ? 10 : 5;
    for (let t = 0; t <= duration; t += step) s += `<text class="axis" x="${(left + t * sx).toFixed(1)}" y="${height - 5}" text-anchor="middle">${t}s</text>`;
    s += `<line class="cursor" x1="${left}" x2="${left}" y1="0" y2="${height - bottom}" data-left="${left}" data-sx="${sx}"/></svg>`;
    return el("div", {}, s);
  }

  function moveCursors(root, t) {
    root.querySelectorAll("line.cursor").forEach((c) => {
      const x = +c.dataset.left + t * +c.dataset.sx;
      c.setAttribute("x1", x); c.setAttribute("x2", x);
    });
  }

  function lineChart(xs, series, { width = 520, height = 180, yLabel = "" } = {}) {
    const left = 36, right = 8, top = 8, bottom = 22;
    const all = series.flatMap((s) => s.values.filter((v) => v != null));
    const ymax = Math.max(1e-6, ...all) * 1.05, xmax = Math.max(1e-6, xs[xs.length - 1] || 1);
    const sx = (width - left - right) / xmax, sy = (height - top - bottom) / ymax;
    let s = `<svg viewBox="0 0 ${width} ${height}" style="width:100%;height:auto;font:10px system-ui">`;
    for (let k = 0; k <= 4; k++) {
      const v = (ymax / 4) * k, y = height - bottom - v * sy;
      s += `<line x1="${left}" x2="${width - right}" y1="${y}" y2="${y}" stroke="currentColor" opacity="0.08"/><text x="${left - 4}" y="${y + 3}" text-anchor="end" fill="currentColor" opacity="0.6">${v.toFixed(v < 10 ? 1 : 0)}</text>`;
    }
    series.forEach((ser, i) => {
      // one polyline per run of non-null values: gaps (e.g. an empty zone) stay gaps
      const runs = [[]];
      xs.forEach((x, k) => {
        const v = ser.values[k];
        if (v == null) { if (runs[runs.length - 1].length) runs.push([]); return; }
        runs[runs.length - 1].push(`${(left + x * sx).toFixed(1)},${(height - bottom - v * sy).toFixed(1)}`);
      });
      runs.filter((r) => r.length).forEach((r) => {
        s += `<polyline fill="none" stroke="${ser.color || SERIES[i % SERIES.length]}" stroke-width="1.5" points="${r.join(" ")}"><title>${esc(ser.name)}</title></polyline>`;
      });
    });
    s += `<text x="${width - right}" y="${height - 6}" text-anchor="end" fill="currentColor" opacity="0.6">time (s)</text>`;
    if (yLabel) s += `<text x="${left}" y="${top + 2}" fill="currentColor" opacity="0.6" dx="4" dy="6">${esc(yLabel)}</text>`;
    s += "</svg>";
    const legend = series.map((ser, i) => `<span><i style="background:${ser.color || SERIES[i % SERIES.length]}"></i>${esc(ser.name)}</span>`).join("");
    return `${s}<div class="legend">${legend}</div>`;
  }

  function barChart(items, { width = 520, rowH = 22, color = "#e8590c" } = {}) {
    const left = 150, max = Math.max(1, ...items.map((d) => d.value));
    const h = items.length * rowH + 6;
    let s = `<svg viewBox="0 0 ${width} ${h}" style="width:100%;height:auto;font:11px system-ui">`;
    items.forEach((d, i) => {
      const y = i * rowH + 3, w = ((width - left - 40) * d.value) / max;
      s += `<text x="${left - 8}" y="${y + 14}" text-anchor="end" fill="currentColor" opacity="0.8">${esc(d.label)}</text>
        <rect x="${left}" y="${y + 3}" width="${Math.max(1, w)}" height="${rowH - 8}" rx="3" fill="${d.color || color}"/>
        <text x="${left + w + 5}" y="${y + 14}" fill="currentColor" opacity="0.7">${esc(d.text ?? d.value)}</text>`;
    });
    return s + "</svg>";
  }

  window.TW = { COLORS, SERIES, esc, fmt, el, timelineSVG, riskSVG, moveCursors, lineChart, barChart };
})();
