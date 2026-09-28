import { el, group, kv, fmt, toast, badge, callout, spinnerRow, SEVERITY_KIND } from "../ui.js";
import { COLORS } from "../viewer.js";

let report = null;
let hidden = false;           // cleared by the user: not redrawn until asked

function clearRoads(ctx) {
  report = null;
  hidden = true;
  ctx.viewer.clearLayer("roads");
  ctx.setAnnotations("roads", []);
}

const STATUS_KIND = { CRITICAL: "danger", CAUTION: "caution", PASSABLE: "ok", NO_ROAD: "info" };

function draw(ctx) {
  ctx.viewer.clearLayer("roads");
  if (!report) return;
  const ann = [];
  for (const s of report.segments || []) {
    for (const line of s.centerlines || []) {
      const pts = line.map((p) => [p[0], p[1] + 0.35, p[2]]);
      ctx.viewer.tube("roads", pts, 0.28, 0xe0c890);
      ann.push({ type: "polyline", points: line, kind: "accent", width: 2 });
    }
    ctx.viewer.label("roads", [s.centre[0], s.centre[1] + 2.5, s.centre[2]], `${s.id} · ${s.width_median_m.toFixed(1)} m wide`);
  }
  for (const p of report.potholes || []) {
    const col = p.severity === "HIGH" ? COLORS.danger : p.severity === "MEDIUM" ? COLORS.caution : COLORS.info;
    const r = Math.max(0.35, p.diameter_cm / 200);
    ctx.viewer.disc("roads", p.position, r, col, 0.35);
    const top = [p.position[0], p.position[1] + 2.2, p.position[2]];
    ctx.viewer.line("roads", [p.position, top], col);
    ctx.viewer.label("roads", top, `${p.id} · ${p.depth_cm.toFixed(0)} cm`, SEVERITY_KIND[p.severity]);
    ann.push({ type: "point", pos: p.position, label: `${p.id} ${p.depth_cm.toFixed(0)} cm`, kind: SEVERITY_KIND[p.severity] });
  }
  ctx.setAnnotations("roads", ann);
}

export default {
  id: "roads",
  title: "Roads & potholes",
  needsMetric: true,
  mount(ctx, body) {
    const box = el("div", { class: "group" });
    body.appendChild(box);
    const run = async (force) => {
      box.innerHTML = "";
      box.appendChild(spinnerRow(force ? "Finding roads and measuring the surface…" : "Loading…"));
      try {
        const r = force ? await ctx.api.post(`/api/road/audit/compile${ctx.bq()}`) : await ctx.api.get(`/api/road/audit${ctx.bq()}`);
        if (!r.compiled) { showIntro(); return; }
        report = r;
        hidden = false;
        ctx.keep("roads", r);
        render();
        draw(ctx);
      } catch (e) { box.innerHTML = ""; box.appendChild(callout(e.message, "danger")); }
    };
    const showCleared = () => {
      box.innerHTML = "";
      box.appendChild(el("div", { class: "note" }, "Road results are hidden. The analysis is kept, so showing it again is instant."));
      box.appendChild(el("button", { class: "btn primary", type: "button", onclick: () => run(false) }, "Show roads and potholes"));
    };
    const showIntro = () => {
      box.innerHTML = "";
      box.appendChild(el("div", { class: "note" }, "Finds roads in the 3-D model from shape, flatness and colour (not from a fixed part of the image), repairs sections torn by passing vehicles or shadows, then measures potholes as depressions below the intact road surface."));
      box.appendChild(el("button", { class: "btn primary", type: "button", onclick: () => run(true) }, "Analyse roads"));
    };
    const render = () => {
      box.innerHTML = "";
      const st = report.statistics || {};
      const ss = report.surface_stats || {};
      const net = report.network || {};
      box.appendChild(callout(report.tactical_advisory, STATUS_KIND[report.status] || "info"));
      box.appendChild(el("div", { class: "stat-row" }, [
        el("div", { class: "stat" }, [el("div", { class: "k" }, "Road network"), el("div", { class: "v" }, fmt.len(net.total_length_m))]),
        el("div", { class: "stat" }, [el("div", { class: "k" }, "Condition"), el("div", { class: "v", html: report.condition_score != null ? `${report.condition_score}<small>/100</small>` : "—" })]),
        el("div", { class: "stat" }, [el("div", { class: "k" }, "Potholes"), el("div", { class: "v" }, String(st.total_potholes || 0))]),
      ]));
      box.appendChild(kv([
        ["Surface", report.road_classification.surface_label],
        ["Narrowest section", st.narrowest_width_m != null ? `${st.narrowest_width_m.toFixed(1)} m` : "—"],
        ["Torn sections repaired", net.repaired_gap_area_m2 ? `${net.repaired_gap_area_m2} m²` : "none"],
        ["Vehicles on the road", String(net.vehicles_on_road ?? 0)],
        ["Depth detection limit", report.detection_limit_cm != null ? `${report.detection_limit_cm} cm (3σ of surface noise ${(100 * (ss.noise_sigma_m || 0)).toFixed(1)} cm)` : "—"],
      ]));
      if (report.detection_limit_cm > 8) {
        box.appendChild(callout(`The road surface in this model is noisy (flight height / viewing angle), so only depressions deeper than about ${report.detection_limit_cm} cm can be told apart from noise. Fly lower or slower over roads for smaller potholes.`, "caution"));
      }
      const plist = el("div", { class: "group" }, [el("div", { class: "group-title" }, `Potholes (severity per ASTM D6433)`)]);
      if (!(report.potholes || []).length) plist.appendChild(el("div", { class: "empty" }, "No depression deeper than the detection limit."));
      for (const p of report.potholes || []) {
        plist.appendChild(el("div", { class: "card selectable", onclick: () => ctx.viewer.focus(p.position, 18) }, [
          el("div", { class: "card-row" }, [
            el("span", { class: "card-title mono" }, p.id),
            p.kind === "CRATER" ? badge("crater", "danger") : null,
            el("span", { class: "spacer" }),
            badge(p.severity.toLowerCase(), SEVERITY_KIND[p.severity]),
          ]),
          el("div", { class: "row", style: { marginTop: "4px" } }, [
            el("span", { class: "mono", style: { fontSize: "18px" } }, `${p.depth_cm.toFixed(1)} cm`),
            el("span", { class: "muted" }, `± ${p.depth_uncertainty_cm.toFixed(1)} · deepest`),
          ]),
          kv([
            ["Size", `${(p.length_cm / 100).toFixed(2)} × ${(p.width_cm / 100).toFixed(2)} m · ${p.area_m2.toFixed(2)} m²`],
            ["Mean depth · volume", `${p.avg_depth_cm.toFixed(1)} cm · ${p.volume_liters.toFixed(0)} L`],
            ["Location", fmt.ll(p.gps_lat, p.gps_lon)],
          ]),
          el("div", { class: "note", style: { marginTop: "4px" } }, p.convoy_impact),
          el("div", { class: "faint", style: { fontSize: "11px", marginTop: "2px" } }, `Repair: ${p.recommended_action}`),
        ]));
      }
      box.appendChild(plist);
      const segs = report.segments || [];
      if (segs.length) {
        const tb = el("tbody");
        for (const s of segs) {
          tb.appendChild(el("tr", { class: "click", onclick: () => ctx.viewer.focus(s.centre, 60) }, [
            el("td", { class: "mono" }, s.id), el("td", { class: "num" }, fmt.len(s.length_m)),
            el("td", { class: "num" }, `${s.width_median_m.toFixed(1)} m`), el("td", { class: "num" }, `${s.max_grade_pct.toFixed(0)}%`),
            el("td", {}, s.surface.toLowerCase()),
          ]));
        }
        box.appendChild(group("Road segments", [el("table", { class: "data" }, [
          el("thead", {}, el("tr", {}, [el("th", {}, "ID"), el("th", { class: "num" }, "Length"), el("th", { class: "num" }, "Width"), el("th", { class: "num" }, "Max grade"), el("th", {}, "Surface")])),
          tb])]));
      }
      box.appendChild(el("div", { class: "row" }, [
        el("button", { class: "btn grow", type: "button", onclick: () => run(true) }, "Re-run analysis"),
        el("button", { class: "btn", type: "button", onclick: () => { clearRoads(ctx); showCleared(); } }, "Clear from the model"),
      ]));
    };
    if (report) { render(); draw(ctx); } else if (hidden) showCleared(); else run(false);
  },
  clear(ctx) { clearRoads(ctx); },
  reset() { report = null; hidden = false; },
};
