import { el, group, fmt, toast } from "../ui.js";
import { COLORS } from "../viewer.js";

// Distance caliper: each picked point is snapped server-side to the median of its surface neighbourhood.
const items = [];
let pending = null;

function redraw(ctx) {
  ctx.viewer.clearLayer("measure");
  const ann = [];
  items.forEach((m, i) => {
    const a = m.p1, b = m.p2;
    ctx.viewer.line("measure", [a, b], COLORS.accent, { depthTest: false });
    ctx.viewer.sphere("measure", a, 0.25, COLORS.accent);
    ctx.viewer.sphere("measure", b, 0.25, COLORS.accent);
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.5, (a[2] + b[2]) / 2];
    ctx.viewer.label("measure", mid, `${m.distance_m.toFixed(2)} m`, "accent");
    ann.push({ type: "polyline", points: [a, b], kind: "accent" }, { type: "point", pos: mid, label: `${m.distance_m.toFixed(1)} m`, kind: "accent" });
  });
  if (pending) ctx.viewer.sphere("measure", pending, 0.25, COLORS.text);
  ctx.setAnnotations("measure", ann);
  ctx.keep("distance_measurements", items);
}

export default {
  id: "measure",
  title: "Measure distance",
  needsMetric: true,
  mount(ctx, body) {
    const list = el("div", { class: "group" });
    const pickLoop = () => {
      ctx.requestPick({
        prompt: pending ? "Click the second point" : "Click the first point",
        continuous: true,
        onPick: async (p) => {
          if (!pending) {
            pending = p;
            redraw(ctx);
            pickLoop();
            return;
          }
          const p1 = pending;
          pending = null;
          try {
            const r = await ctx.api.post("/api/measure/between", { p1, p2: p, baseline_id: ctx.bid() });
            items.push(r);
            redraw(ctx);
            render();
          } catch (e) { toast(e.message, "danger"); }
          pickLoop();
        },
      });
    };
    const render = () => {
      list.innerHTML = "";
      if (!items.length) {
        list.appendChild(el("div", { class: "empty" }, "No measurements yet."));
        return;
      }
      items.slice().reverse().forEach((m, k) => {
        const i = items.length - 1 - k;
        list.appendChild(el("div", { class: "card" }, [
          el("div", { class: "card-row" }, [
            el("span", { class: "card-title" }, `#${i + 1}`),
            el("span", { class: "spacer" }),
            el("span", { class: "mono" }, `${m.distance_m.toFixed(2)} m`),
          ]),
          el("dl", { class: "kv", style: { marginTop: "6px" } }, [
            el("dt", {}, "Horizontal"), el("dd", {}, `${m.horizontal_m.toFixed(2)} ± ${m.horizontal_uncertainty_m.toFixed(2)} m`),
            el("dt", {}, "Vertical"), el("dd", {}, `${m.vertical_m >= 0 ? "+" : ""}${m.vertical_m.toFixed(2)} ± ${m.vertical_uncertainty_m.toFixed(2)} m`),
            el("dt", {}, "Slope / bearing"), el("dd", {}, `${m.slope_deg.toFixed(1)}° · ${m.azimuth_deg.toFixed(0)}°`),
          ]),
          el("div", { class: "row", style: { marginTop: "6px" } }, [
            el("button", { class: "btn small ghost", type: "button", onclick: () => ctx.viewer.focus(m.p1) }, "Show"),
            el("span", { class: "spacer grow" }),
            el("button", { class: "btn small ghost", type: "button", onclick: () => { items.splice(i, 1); redraw(ctx); render(); } }, "Remove"),
          ]),
        ]));
      });
    };
    body.appendChild(group("", [
      el("div", { class: "note" }, "Click two points on the model. Each point is snapped to the median of the surrounding surface, so a stray point cannot throw the result off. Vertical is along true gravity."),
      el("div", { class: "row" }, [
        el("button", { class: "btn primary grow", type: "button", onclick: pickLoop }, "Start measuring"),
        el("button", { class: "btn", type: "button", onclick: () => { items.length = 0; pending = null; redraw(ctx); render(); } }, "Clear"),
      ]),
    ]));
    body.appendChild(list);
    render();
    redraw(ctx);
    pickLoop();
  },
  unmount(ctx) { ctx.cancelPick(); pending = null; },
  clear(ctx) { items.length = 0; pending = null; redraw(ctx); },
  reset() { items.length = 0; pending = null; },
};
