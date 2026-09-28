import { el, group, kv, fmt, toast, spinnerRow } from "../ui.js";

// Height ceiling: everything higher than N metres above its LOCAL ground is marked (a sloping site
// is handled correctly because the reference is the terrain model, not one flat plane).
let ceilingM = 8;

export default {
  id: "ceiling",
  title: "Height ceiling check",
  needsMetric: true,
  mount(ctx, body) {
    const out = el("div");
    const val = el("span", { class: "num" }, `${ceilingM.toFixed(1)} m`);
    const slider = el("input", { type: "range", min: "1", max: "40", step: "0.5", value: String(ceilingM) });
    let timer = null;
    const apply = async () => {
      out.innerHTML = "";
      out.appendChild(spinnerRow("Checking…"));
      try {
        if (!ctx.viewer.hag) {
          await ctx.terrainGrid();
          const buf = await ctx.api.getBinary(`/api/analysis/hag?file=${encodeURIComponent(ctx.state.survey.pointsFile)}${ctx.bq("&")}`);
          ctx.viewer.setHeightAboveGround(new Float32Array(buf));
        }
        ctx.viewer.setCeiling(ceilingM);
        const r = await ctx.api.get(`/api/analysis/height-restriction?max_height_m=${ceilingM}${ctx.bq("&")}`);
        out.innerHTML = "";
        out.appendChild(el("div", { class: `callout ${r.violation_detected ? "danger" : "ok"}` },
          r.violation_detected ? `${fmt.n(r.violating_points)} points (${r.violation_percent}%) are above ${ceilingM} m. Highest: ${r.peak_elevation_m} m (${r.max_breach_m} m over).`
            : `Nothing is higher than ${ceilingM} m above its local ground.`));
        if (r.peak_coordinates) {
          const p = r.peak_coordinates;
          out.appendChild(el("button", { class: "btn small", type: "button", onclick: () => ctx.viewer.focus([p.x, p.y, p.z], 50) }, "Show the highest point"));
        }
        ctx.keep("height_ceiling", r);
      } catch (e) { out.innerHTML = ""; toast(e.message, "danger"); }
    };
    slider.addEventListener("input", () => {
      ceilingM = parseFloat(slider.value);
      val.textContent = `${ceilingM.toFixed(1)} m`;
      clearTimeout(timer);
      timer = setTimeout(apply, 250);
    });
    body.appendChild(group("", [
      el("div", { class: "note" }, "Marks every point higher than the ceiling above the ground directly beneath it (from the bare-earth model), e.g. for helicopter landing zones, antenna masts or construction limits."),
      el("div", { class: "field" }, [el("label", {}, ["Ceiling", val]), slider]),
      el("button", { class: "btn", type: "button", onclick: () => { ctx.viewer.setCeiling(null); out.innerHTML = ""; } }, "Clear marking"),
    ]));
    body.appendChild(out);
    apply();
  },
  unmount(ctx) { ctx.viewer.setCeiling(null); },
  clear(ctx) { ctx.viewer.setCeiling(null); },
};
