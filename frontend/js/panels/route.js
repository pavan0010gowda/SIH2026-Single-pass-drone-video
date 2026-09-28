import { el, group, kv, fmt, toast, badge, callout, spinnerRow } from "../ui.js";
import { COLORS } from "../viewer.js";
import { toLatLon } from "../geo.js";

function where(p) {
  if (!p) return "not set";
  const g = toLatLon(p[0], p[2]);
  return g ? fmt.ll(g.lat, g.lon) : `x ${p[0].toFixed(0)}, z ${p[2].toFixed(0)} m`;
}

const st = { start: null, end: null, posts: [], useBuildings: true, useRoads: false, result: null, selected: "covert" };
const MODE_COLOR = { covert: COLORS.ok, balanced: COLORS.accent, fastest: COLORS.info };
const MODE_KIND = { covert: "ok", balanced: "accent", fastest: "info" };
const MODE_TEXT = {
  covert: "Stays in cover: tree lines, dips and the far side of buildings, even if longer.",
  balanced: "Trades some exposure for a shorter walk.",
  fastest: "Quickest walk (uses roads); shown for comparison.",
};

// exposure 0..1 -> green .. amber .. red
function expoColor(e) {
  const t = Math.min(1, Math.max(0, e / 0.6));
  return t < 0.5 ? [0.47 + 0.38 * t * 2, 0.66, 0.44 - 0.2 * t * 2] : [0.85, 0.66 - 0.4 * (t - 0.5) * 2, 0.24];
}

function clearRoute(ctx) {
  st.start = st.end = null;
  st.posts = [];
  st.result = null;
  ctx.viewer.clearLayer("route");
  ctx.setAnnotations("route", []);
}

function draw(ctx) {
  const v = ctx.viewer;
  v.clearLayer("route");
  const ann = [];
  const marker = (p, text, cls, col) => {
    if (!p) return;
    v.line("route", [p, [p[0], p[1] + 3, p[2]]], col);
    v.sphere("route", [p[0], p[1] + 3, p[2]], 0.45, col);
    v.label("route", [p[0], p[1] + 3.8, p[2]], text, cls);
    ann.push({ type: "point", pos: p, label: text, kind: cls });
  };
  marker(st.start, "Start", "ok", COLORS.ok);
  marker(st.end, "Objective", "danger", COLORS.danger);
  st.posts.forEach((p, i) => marker(p, `Threat post ${i + 1}`, "danger", COLORS.danger));
  const r = st.result;
  if (r) {
    for (const route of r.routes) {
      if (route.mode === st.selected) {
        v.tube("route", route.points, 0.35, route.exposure.map(expoColor));
      } else {
        v.line("route", route.points, MODE_COLOR[route.mode], { opacity: 0.8 });
      }
      ann.push({ type: "polyline", points: route.points, kind: MODE_KIND[route.mode], width: route.mode === st.selected ? 3 : 1.5 });
    }
    for (const o of r.observers || []) {
      if (o.kind === "operator") continue;
      const y = v.groundAt(o.x, o.z);
      v.sphere("route", [o.x, (o.eye_abs ?? y + 1.7), o.z], 0.5, COLORS.danger);
    }
  }
  ctx.setAnnotations("route", ann);
}

export default {
  id: "route",
  title: "Covert route",
  needsMetric: true,
  mount(ctx, body) {
    const out = el("div", { class: "group" });
    const pts = el("div", { class: "group" });
    const pick = (what) => ctx.requestPick({
      prompt: what === "start" ? "Click the start point" : what === "end" ? "Click the objective" : "Click threat posts (enemy observers); Esc when done",
      continuous: what === "post",
      onPick: (p) => {
        if (what === "post") st.posts.push(p); else st[what] = p;
        renderPts();
        draw(ctx);
        if (what === "start" && !st.end) pick("end");
      },
    });
    const renderPts = () => {
      pts.innerHTML = "";
      pts.appendChild(kv([
        ["Start", where(st.start)],
        ["Objective", where(st.end)],
        ["Threat posts", st.posts.length ? `${st.posts.length} placed` : "none (buildings only)"],
      ]));
    };
    const chk = (key, label) => {
      const i = el("input", { type: "checkbox" });
      i.checked = st[key];
      i.addEventListener("change", () => { st[key] = i.checked; });
      return el("label", { class: "check" }, [i, label]);
    };
    body.appendChild(group("", [
      el("div", { class: "note" }, "Plans a walking route that stays out of sight: every cell's chance of being seen is computed from lines of sight (trees and buildings block them) and distance; flat ground is preferred and steep or high ground is used only when nothing else connects."),
      el("div", { class: "grid2" }, [
        el("button", { class: "btn", type: "button", onclick: () => pick("start") }, "Set start"),
        el("button", { class: "btn", type: "button", onclick: () => pick("end") }, "Set objective"),
      ]),
      el("div", { class: "grid2" }, [
        el("button", { class: "btn", type: "button", onclick: () => pick("post") }, "Add threat posts"),
        el("button", { class: "btn", type: "button", onclick: () => { st.posts = []; renderPts(); draw(ctx); } }, "Clear posts"),
      ]),
      chk("useBuildings", "Treat occupied buildings as observers"),
      chk("useRoads", "Treat roads as observed (traffic)"),
      pts,
      el("div", { class: "row" }, [
        el("button", { class: "btn primary grow", type: "button", onclick: () => run() }, "Plan routes"),
        el("button", { class: "btn", type: "button", onclick: () => {
          ctx.cancelPick(); clearRoute(ctx); out.innerHTML = ""; renderPts();
        } }, "Clear route"),
      ]),
    ]));
    body.appendChild(out);

    const run = async () => {
      if (!st.start || !st.end) { toast("Set the start and the objective first."); if (!st.start) pick("start"); else pick("end"); return; }
      out.innerHTML = "";
      out.appendChild(spinnerRow("Computing lines of sight and least-exposure paths…"));
      try {
        st.result = await ctx.api.post("/api/analysis/route", {
          start: st.start, end: st.end, baseline_id: ctx.bid(), use_default_observers: st.useBuildings,
          include_road_observers: st.useRoads, observers: st.posts.map((p) => ({ x: p[0], z: p[2], eye_h: 1.7 })),
        });
        ctx.keep("routes", st.result);
        if (!st.result.routes.length) { out.innerHTML = ""; out.appendChild(callout(st.result.message || "No route found.", "caution")); return; }
        render();
        draw(ctx);
      } catch (e) { out.innerHTML = ""; out.appendChild(callout(e.message, "danger")); }
    };
    const render = () => {
      out.innerHTML = "";
      const r = st.result;
      if (!r || !r.routes) return;
      if (r.start_snap_m > 1 || r.end_snap_m > 1) out.appendChild(callout(`Start/objective moved ${Math.max(r.start_snap_m, r.end_snap_m).toFixed(0)} m to the nearest passable ground.`, "info"));
      out.appendChild(el("div", { class: "note" }, `${(r.observers || []).length} observers considered · grid ${r.grid_res_m} m. Selected route is coloured by exposure (green hidden → red seen).`));
      for (const route of r.routes) {
        const sel = route.mode === st.selected;
        const mix = route.terrain_mix_pct || {};
        out.appendChild(el("div", { class: `card selectable${sel ? " selected" : ""}`, onclick: () => { st.selected = route.mode; render(); draw(ctx); } }, [
          el("div", { class: "card-row" }, [
            el("span", { class: "card-title" }, route.label),
            el("span", { class: "spacer" }),
            badge(`${route.detection_risk_pct.toFixed(0)}% exposed`, route.detection_risk_pct < 25 ? "ok" : route.detection_risk_pct < 55 ? "caution" : "danger"),
          ]),
          el("div", { class: "note", style: { marginTop: "3px" } }, MODE_TEXT[route.mode]),
          kv([
            ["Distance · time", `${fmt.len(route.length_m)} · ${route.time_min.toFixed(0)} min on foot`],
            ["In cover", `${route.concealed_pct.toFixed(0)}% of the way`],
            ["Exposed stretch", `${fmt.len(route.exposed_distance_m)} (${route.exposed_time_min.toFixed(1)} min)`],
            ["Climb · steepest", `${route.climb_m.toFixed(0)} m · ${route.max_slope_deg.toFixed(0)}°`],
            ["Terrain", Object.entries(mix).filter(([, v]) => v > 0).map(([k, v]) => `${k} ${v.toFixed(0)}%`).join(", ")],
          ]),
        ]));
      }
    };
    renderPts();
    render();
    draw(ctx);
    if (!st.start) pick("start");
  },
  unmount(ctx) { ctx.cancelPick(); },
  clear(ctx) { clearRoute(ctx); },
  reset() { st.start = st.end = null; st.posts = []; st.result = null; },
};
