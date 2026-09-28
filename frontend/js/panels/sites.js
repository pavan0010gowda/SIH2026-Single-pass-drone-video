import { el, group, kv, fmt, toast, badge, callout, spinnerRow } from "../ui.js";
import { COLORS } from "../viewer.js";

const params = { radius_m: 15, min_building_dist_m: 150, max_slope_deg: 6, max_road_dist_m: 500, top_k: 5 };
let result = null;
const RANK_COLORS = [COLORS.ok, COLORS.accent, COLORS.info, COLORS.caution, COLORS.muted];

// polar chart: distance to the first tree/building along 36 directions (north up)
function radar(site) {
  const R = 54, c = 60, max = site.radius_m + 45;
  const d = site.cover_ray_distances_m;
  const pts = d.map((v, k) => {
    const a = (k / d.length) * Math.PI * 2;          // 0 = east, clockwise towards south (screen +y)
    const r = v == null ? R : R * Math.min(1, v / max);
    return [c + r * Math.cos(a), c + r * Math.sin(a)];
  });
  const poly = pts.map((p) => p.map((x) => x.toFixed(1)).join(",")).join(" ");
  const siteR = (R * site.radius_m) / max;
  const open = d.map((v, k) => {
    if (v != null) return "";
    const a0 = ((k - 0.5) / d.length) * Math.PI * 2, a1 = ((k + 0.5) / d.length) * Math.PI * 2;
    return `<path d="M${c},${c} L${c + R * Math.cos(a0)},${c + R * Math.sin(a0)} A${R},${R} 0 0 1 ${c + R * Math.cos(a1)},${c + R * Math.sin(a1)} Z" fill="rgba(212,101,79,0.22)"/>`;
  }).join("");
  return el("div", { html: `<svg viewBox="0 0 120 120" width="120" height="120">
    <circle cx="${c}" cy="${c}" r="${R}" fill="none" stroke="#3b4147"/>
    ${open}
    <polygon points="${poly}" fill="rgba(120,168,111,0.28)" stroke="#78a86f" stroke-width="1"/>
    <circle cx="${c}" cy="${c}" r="${siteR}" fill="rgba(201,164,92,0.35)" stroke="#c9a45c"/>
    <text x="${c}" y="9" text-anchor="middle" font-size="8" fill="#a8a49b">N</text>
    <text x="${c}" y="118" text-anchor="middle" font-size="8" fill="#a8a49b">S</text>
    <text x="3" y="${c + 3}" font-size="8" fill="#a8a49b">W</text>
    <text x="111" y="${c + 3}" font-size="8" fill="#a8a49b">E</text></svg>` });
}

function clearSites(ctx) {
  result = null;
  ctx.viewer.clearLayer("sites");
  ctx.setAnnotations("sites", []);
}

function draw(ctx) {
  ctx.viewer.clearLayer("sites");
  if (!result) return;
  const ann = [];
  result.sites.forEach((s, i) => {
    const col = RANK_COLORS[i] || COLORS.muted;
    ctx.viewer.disc("sites", s.center, s.radius_m, col, 0.18);
    ctx.viewer.label("sites", [s.center[0], s.center[1] + 3, s.center[2]], `${s.id} · ${s.score.toFixed(0)}`, i === 0 ? "ok" : "accent");
    const ring = [];
    for (let k = 0; k <= 40; k++) {
      const a = (k / 40) * Math.PI * 2;
      const x = s.center[0] + s.radius_m * Math.cos(a), z = s.center[2] + s.radius_m * Math.sin(a);
      ring.push([x, ctx.viewer.groundAt(x, z) + 0.2, z]);
    }
    ann.push({ type: "ring", points: ring, center: s.center, radius: s.radius_m, kind: i === 0 ? "ok" : "accent" });
    ann.push({ type: "point", pos: s.center, label: s.id, kind: i === 0 ? "ok" : "accent" });
  });
  for (const o of result.observers || []) {
    ann.push({ type: "point", pos: [o.x, ctx.viewer.groundAt(o.x, o.z) + 2, o.z], kind: "danger" });
  }
  ctx.setAnnotations("sites", ann);
}

export default {
  id: "sites",
  title: "Base site finder",
  needsMetric: true,
  mount(ctx, body) {
    const field = (key, label, min, max, step, unit) => {
      const v = el("span", { class: "num" }, `${params[key]} ${unit}`);
      const inp = el("input", { type: "range", min: String(min), max: String(max), step: String(step), value: String(params[key]) });
      inp.addEventListener("input", () => { params[key] = parseFloat(inp.value); v.textContent = `${params[key]} ${unit}`; });
      return el("div", { class: "field" }, [el("label", {}, [label, v]), inp]);
    };
    const out = el("div", { class: "group" });
    body.appendChild(group("", [
      el("div", { class: "note" }, "Looks for an open, flat, dry clearing that trees screen on all four sides, far from houses and hidden from them, with a road close enough for supply but not next to it."),
      field("radius_m", "Compound radius", 5, 40, 1, "m"),
      field("min_building_dist_m", "Keep away from buildings", 0, 400, 10, "m"),
      field("max_slope_deg", "Maximum slope", 2, 15, 0.5, "°"),
      field("max_road_dist_m", "Road within", 50, 1500, 50, "m"),
      el("div", { class: "row" }, [
        el("button", { class: "btn primary grow", type: "button", onclick: () => run() }, "Find sites"),
        el("button", { class: "btn", type: "button", onclick: () => { clearSites(ctx); out.innerHTML = ""; } }, "Clear from the model"),
      ]),
    ]));
    body.appendChild(out);

    const run = async () => {
      out.innerHTML = "";
      out.appendChild(spinnerRow("Checking clearings, tree cover in 36 directions and lines of sight from every building…"));
      try {
        result = await ctx.api.post("/api/analysis/sites", { ...params, baseline_id: ctx.bid() });
        ctx.keep("base_sites", result);
        render();
        draw(ctx);
        if (result.sites.length) ctx.viewer.focus(result.sites[0].center, 140);
      } catch (e) { out.innerHTML = ""; out.appendChild(callout(e.message, "danger")); }
    };
    const render = () => {
      out.innerHTML = "";
      if (!result) return;
      for (const r of result.relaxed || []) out.appendChild(callout(r, "caution"));
      if (!result.sites.length) { out.appendChild(el("div", { class: "empty" }, result.message || "No suitable site.")); return; }
      out.appendChild(el("div", { class: "note" }, `${result.observers.length} buildings treated as observers. Scores out of 100.`));
      result.sites.forEach((s, i) => {
        out.appendChild(el("div", { class: `card selectable${i === 0 ? " selected" : ""}`, onclick: () => ctx.viewer.focus(s.center, 110) }, [
          el("div", { class: "card-row" }, [
            el("span", { class: "card-title" }, s.id),
            s.four_sides_covered ? badge("screened on 4 sides", "ok") : badge("partly open", "caution"),
            el("span", { class: "spacer" }),
            el("span", { class: "mono", style: { fontSize: "18px" } }, s.score.toFixed(0)),
          ]),
          el("div", { class: "row", style: { alignItems: "flex-start", marginTop: "6px" } }, [
            radar(s),
            el("div", { class: "grow" }, kv([
              ["Tree cover N/E/S/W", `${s.quadrant_cover_pct.N}/${s.quadrant_cover_pct.E}/${s.quadrant_cover_pct.S}/${s.quadrant_cover_pct.W}%`],
              ["Nearest building", s.nearest_building_m != null ? `${s.nearest_building_m.toFixed(0)} m` : "none"],
              ["Seen from buildings", `${s.detection_probability_pct.toFixed(0)}%`],
              ["Road access", s.nearest_road_m != null ? `${s.nearest_road_m.toFixed(0)} m` : "none"],
              ["Slope · relief", `${s.max_slope_deg.toFixed(1)}° · ${s.relief_m.toFixed(1)} m`],
              ["Ground", s.drainage],
            ])),
          ]),
          el("div", { class: "faint", style: { fontSize: "11px", marginTop: "4px" } }, `${fmt.ll(s.gps_lat, s.gps_lon)} · ${fmt.area(s.area_m2)}`),
        ]));
      });
    };
    if (result) { render(); draw(ctx); }
  },
  clear(ctx) { clearSites(ctx); },
  reset() { result = null; },
};
