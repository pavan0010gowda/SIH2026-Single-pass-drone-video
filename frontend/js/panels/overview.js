import { el, kv, group, callout, fmt, badge, CONFIDENCE_KIND } from "../ui.js";
import { CLASS_COLORS, CLASS_LABELS } from "../geo.js";

const CLASS_KEYS = ["ground", "low_vegetation", "tree", "building", "water", "road", "vehicle"];

export default {
  id: "overview",
  title: "Survey overview",
  mount(ctx, body) {
    const s = ctx.state.survey;
    if (!s || (s.kind === "active" && !s.info)) {
      body.appendChild(group("Get started", [
        el("div", { class: "note" }, "Reconstruct a mission from drone video and its flight log, or import a package processed in Google Colab."),
        el("button", { class: "btn primary", type: "button", "data-action": "new-mission" }, "New mission"),
        el("button", { class: "btn", type: "button", "data-action": "open-survey" }, "Open a saved survey"),
      ]));
      return;
    }
    const cal = s.calibration || {};
    const d = cal.diagnostics || {};
    const rmse = d.gps_rmse_horizontal_m ?? d.final_gps_rmse_m ?? d.rmse_m;
    const o = cal.origin || {};
    const calCard = el("div", { class: "card" }, [
      el("div", { class: "card-row" }, [
        el("span", { class: "card-title" }, "Metric calibration"),
        el("span", { class: "spacer" }),
        cal.status === "ok" ? badge(`${String(cal.confidence || "?").toLowerCase()} confidence`, CONFIDENCE_KIND[cal.confidence] || "") : badge("not calibrated", "danger"),
      ]),
      cal.status === "ok" ? kv([
        ["Scale source", { GPS_SIM3: "GPS track (3-D fit)", GPS_4DOF: "GPS track + image horizon", GPS_LINE: "GPS (straight line)",
          ALTITUDE_AGL: "Logged altitude", ASSUMED_ALTITUDE: "Assumed altitude" }[cal.mode] || cal.mode],
        ["Scale uncertainty", cal.scale_rel_uncertainty != null ? `±${(100 * cal.scale_rel_uncertainty).toFixed(2)} %` : "—"],
        ["GPS fit residual", rmse != null ? `${Number(rmse).toFixed(2)} m` : "—"],
        ["Lat / lon order", cal.latlon_swapped ? "corrected (log had lon, lat)" : (cal.latlon_verified ? "verified" : "as logged")],
        ["Origin", o.lat != null ? fmt.ll(o.lat, o.lon) : "—"],
        ["Frame", "metres · x east · y up · z south"],
      ]) : el("div", { class: "note" }, cal.reason || "This model has no metric scale. Heights and distances are not available."),
    ]);
    body.appendChild(group("Mission", [calCard]));
    if (cal.warnings && cal.warnings.length) body.appendChild(callout(cal.warnings.join(" "), "caution"));
    if (s.kind === "active" && s.info && s.info.has_cameras && cal.source !== "bundle_recalibration" && !cal.latlon_verified) {
      body.appendChild(el("button", { class: "btn", type: "button", "data-action": "recalibrate" }, "Verify scale and north with the flight log"));
    }

    const info = s.info || {};
    body.appendChild(group("Model", [
      kv([
        ["Points", fmt.n(info.vertex_count || (s.meta && s.meta.vertex_count))],
        ["Surface mesh", ctx.viewer.mesh ? `${fmt.n(Math.round(ctx.viewer.mesh.geometry.index ? ctx.viewer.mesh.geometry.index.count / 3 : 0))} faces` : "not built"],
        ["Camera poses", ctx.video.frames.length ? `${ctx.video.frames.length} keyframes` : "none"],
        ["Flight log", s.telemetry && s.telemetry.waypoints && s.telemetry.waypoints.length ? `${fmt.n(s.telemetry.waypoints.length)} fixes` : "none"],
      ]),
      el("button", { class: "btn", type: "button", "data-action": "build-mesh" }, ctx.viewer.mesh ? "Rebuild surface mesh" : "Build gap-free surface mesh"),
    ]));

    const terrainBox = el("div", { class: "group" }, [el("div", { class: "group-title" }, "Terrain"), el("div", { class: "note" }, "Analysing…")]);
    body.appendChild(terrainBox);
    if (!ctx.metric()) { terrainBox.lastChild.textContent = "Needs a metric model."; return; }
    ctx.api.get(`/api/terrain/summary${ctx.bq()}`).then((t) => {
      terrainBox.lastChild.remove();
      const pct = t.class_percent || {};
      const bar = el("div", { class: "stackbar" });
      const legend = el("div", { class: "legend" });
      CLASS_KEYS.forEach((k, i) => {
        const v = pct[k] || 0;
        if (v <= 0) return;
        const c = CLASS_COLORS[i + 1].map((x) => Math.round(x * 255)).join(",");
        bar.appendChild(el("span", { style: { width: `${v}%`, background: `rgb(${c})` }, title: `${CLASS_LABELS[i + 1]} ${v}%` }));
        legend.appendChild(el("span", { html: `<i style="background:rgb(${c})"></i>${CLASS_LABELS[i + 1]} ${v.toFixed(0)}%` }));
      });
      const roads = (t.meta && t.meta.roads) || {};
      terrainBox.appendChild(kv([
        ["Analysed area", fmt.area(t.analysed_area_m2)],
        ["Grid", `${t.res_m} m cells`],
        ["Roads found", roads.total_length_m != null ? `${fmt.len(roads.total_length_m)} in ${roads.segments} segment(s)` : "—"],
      ]));
      terrainBox.appendChild(bar);
      terrainBox.appendChild(legend);
      ctx.keep("terrain", t);
    }).catch((e) => { terrainBox.lastChild.textContent = e.message; });

    body.appendChild(group("Tools", [
      el("div", { class: "grid2" }, [
        el("button", { class: "btn", type: "button", "data-tool": "height" }, "Measure height"),
        el("button", { class: "btn", type: "button", "data-tool": "inventory" }, "Structures"),
        el("button", { class: "btn", type: "button", "data-tool": "roads" }, "Roads & potholes"),
        el("button", { class: "btn", type: "button", "data-tool": "sites" }, "Base sites"),
        el("button", { class: "btn", type: "button", "data-tool": "route" }, "Covert route"),
        el("button", { class: "btn", type: "button", "data-tool": "compare" }, "Change detection"),
      ]),
    ]));
  },
};
