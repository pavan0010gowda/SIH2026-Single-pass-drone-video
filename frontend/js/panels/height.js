import { el, group, kv, fmt, toast, badge, spinnerRow } from "../ui.js";
import { COLORS } from "../viewer.js";

const results = [];

export function drawHeight(ctx, r, layer = "height") {
  ctx.viewer.dimension(layer, r.base, r.top, COLORS.accent, `${r.height_m.toFixed(2)} m`);
  if (r.outline && r.outline.length > 2) ctx.viewer.outline(layer, r.outline, COLORS.accent);
}

function redraw(ctx) {
  ctx.viewer.clearLayer("height");
  const ann = [];
  for (const r of results) {
    drawHeight(ctx, r);
    ann.push({ type: "polyline", points: [r.base, [r.base[0], r.top[1], r.base[2]]], kind: "accent", width: 2 });
    ann.push({ type: "point", pos: r.top, label: `${r.kind} ${r.height_m.toFixed(1)} m`, kind: "accent" });
  }
  ctx.setAnnotations("height", ann);
  ctx.keep("height_measurements", results);
}

export function heightCard(ctx, r, { onRemove } = {}) {
  const b = r.error_budget_m || {};
  return el("div", { class: "card" }, [
    el("div", { class: "card-row" }, [
      el("span", { class: "card-title", style: { textTransform: "capitalize" } }, r.kind),
      el("span", { class: "spacer" }),
      badge(r.method || "", "accent"),
    ]),
    el("div", { class: "hero", style: { margin: "8px 0 2px" }, html: `${r.height_m.toFixed(2)}<small> m ± ${r.uncertainty_m.toFixed(2)}</small>` }),
    el("div", { class: "note" }, `95% interval ${r.interval95_m[0].toFixed(2)} – ${r.interval95_m[1].toFixed(2)} m, above the ground at its base`),
    kv([
      ["Roof", r.roof ? `${r.roof.label}${r.roof.pitch_deg ? `, ${r.roof.pitch_deg.toFixed(0)}° pitch` : ""}` : null],
      ["Eaves", r.eave_height_m != null ? `${r.eave_height_m.toFixed(2)} m` : null],
      ["Highest point (masts, chimneys)", r.max_point_height_m != null ? `${r.max_point_height_m.toFixed(2)} m` : null],
      ["Footprint", `${r.length_m.toFixed(1)} × ${r.width_m.toFixed(1)} m (${fmt.area(r.footprint_m2)})`],
      ["Long axis bearing", `${r.orientation_deg.toFixed(0)}°`],
      ["Ground slope at base", `${r.ground_slope_deg.toFixed(1)}°`],
      ["Location", fmt.ll(r.gps_lat, r.gps_lon)],
    ]),
    el("details", {}, [
      el("summary", { class: "note", style: { cursor: "pointer", marginTop: "6px" } }, "Error budget"),
      kv([
        ["Top (roof / crown fit)", `${(100 * (b.top || 0)).toFixed(1)} cm`],
        ["Base (ground plane)", `${(100 * (b.base || 0)).toFixed(1)} cm`],
        ["Map scale", `${(100 * (b.scale || 0)).toFixed(1)} cm`],
        ["Surface noise", `${(100 * (r.surface_noise_m || 0)).toFixed(1)} cm`],
        ["Points used", `${fmt.n(r.points_used)} object · ${fmt.n(r.ground_points_used)} ground`],
      ]),
    ]),
    el("div", { class: "row", style: { marginTop: "8px" } }, [
      el("button", { class: "btn small ghost", type: "button", onclick: () => ctx.viewer.focus(r.top, 45) }, "Show"),
      el("span", { class: "spacer grow" }),
      onRemove ? el("button", { class: "btn small ghost", type: "button", onclick: onRemove }, "Remove") : null,
    ]),
  ]);
}

export default {
  id: "height",
  title: "Measure height",
  needsMetric: true,
  mount(ctx, body) {
    const status = el("div");
    const list = el("div", { class: "group" });
    const render = () => {
      list.innerHTML = "";
      if (!results.length) list.appendChild(el("div", { class: "empty" }, "Click a building, tree, mast or vehicle."));
      results.slice().reverse().forEach((r) => {
        const i = results.indexOf(r);
        list.appendChild(heightCard(ctx, r, { onRemove: () => { results.splice(i, 1); redraw(ctx); render(); } }));
      });
    };
    const pick = () => ctx.requestPick({
      prompt: "Click a structure to measure its height", continuous: true,
      onPick: async (p) => {
        status.innerHTML = "";
        status.appendChild(spinnerRow("Segmenting the object and fitting roof / crown and ground…"));
        try {
          const r = await ctx.api.post("/api/measure/height", { x: p[0], z: p[2], baseline_id: ctx.bid() });
          status.innerHTML = "";
          if (r.status !== "ok") { toast(r.message || "No structure there.", ""); return; }
          results.push(r);
          redraw(ctx);
          render();
        } catch (e) { status.innerHTML = ""; toast(e.message, "danger"); }
      },
    });
    body.appendChild(group("", [
      el("div", { class: "note" }, "Height is measured from the ground around the object's base to a fitted top: roof planes and their ridge for buildings, a crown-apex fit for trees. The ground is fitted in a ring outside the footprint, so it works on slopes and for oblique footage where the ground under a roof is never seen."),
      el("div", { class: "row" }, [
        el("button", { class: "btn primary grow", type: "button", onclick: pick }, "Pick a structure"),
        el("button", { class: "btn", type: "button", onclick: () => { results.length = 0; redraw(ctx); render(); } }, "Clear"),
      ]),
    ]));
    body.appendChild(status);
    body.appendChild(list);
    render();
    redraw(ctx);
    pick();
  },
  unmount(ctx) { ctx.cancelPick(); },
  clear(ctx) { results.length = 0; redraw(ctx); },
  reset() { results.length = 0; },
};
