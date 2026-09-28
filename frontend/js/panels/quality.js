// Quality & accuracy: the survey measured against the problem-statement targets, check / control
// points (few-GCP refinement), RTK / PPK re-georeferencing and the printable report.
import { el, group, fmt, badge, callout, stat, kv, toast, spinnerRow } from "../ui.js";

let rep = null;
let cps = null;

const STAGE_COLORS = ["#c9a45c", "#7fa6c9", "#78a86f", "#d9a441", "#a58bc4", "#6fb3a8", "#d4654f", "#a8a49b"];

function metBadge(m) {
  return m === true ? badge("Met", "ok") : m === false ? badge("Not met", "danger") : badge("Unknown");
}

export default {
  id: "quality",
  title: "Quality & accuracy",
  mount(ctx, body) {
    const box = el("div", { class: "group" });
    body.appendChild(box);

    const load = async () => {
      box.innerHTML = "";
      box.appendChild(spinnerRow("Evaluating the survey…"));
      try {
        [rep, cps] = await Promise.all([ctx.api.get(`/api/quality/report${ctx.bq()}`), ctx.api.get(`/api/quality/checkpoints${ctx.bq()}`)]);
        ctx.keep("quality", rep);
        render();
      } catch (e) {
        box.innerHTML = "";
        box.appendChild(callout(e.message, "danger"));
      }
    };

    const render = () => {
      box.innerHTML = "";
      const a = rep.accuracy, c = rep.completeness, p = rep.processing, i = rep.inputs;

      // ---------------- targets
      const tb = el("tbody");
      for (const s of rep.sih) {
        tb.appendChild(el("tr", {}, [el("td", {}, [el("div", {}, s.parameter), el("div", { class: "faint sih-achieved" }, String(s.achieved))]),
          el("td", { class: "num" }, metBadge(s.met))]));
      }
      box.appendChild(group("Problem-statement targets", [el("table", { class: "data" }, [tb])]));

      // ---------------- accuracy
      box.appendChild(group("Accuracy", [
        el("div", { class: "stat-row" }, [
          stat("Scale", a.scale_uncertainty_pct != null ? `±${a.scale_uncertainty_pct}` : "—", "%"),
          stat("Per 100 m", a.relative_error_per_100m_m != null ? `${(100 * a.relative_error_per_100m_m).toFixed(0)}` : "—", "cm"),
          stat("Absolute", a.sih_spatial_accuracy_m != null ? a.sih_spatial_accuracy_m.toFixed(2) : "—", "m"),
        ]),
        kv([
          ["Georeference", `${a.georeference_mode || "—"} (${String(a.confidence || "—").toLowerCase()})`],
          ["GPS fit residual", a.gps_fit_rmse_horizontal_m != null ? `${a.gps_fit_rmse_horizontal_m} m over ${a.gps_fixes_used} fixes` : "—"],
          ["Levelling (camera horizon vs GPS)", a.levelling_disagreement_deg != null ? fmt.deg(a.levelling_disagreement_deg, 2) : "—"],
          ["Building heights, median ±", a.height_precision_median_m != null ? `${(100 * a.height_precision_median_m).toFixed(1)} cm` : "—"],
          ["Absolute basis", a.sih_spatial_accuracy_basis],
        ]),
        el("div", { class: "note" }, a.absolute_note),
      ]));

      // ---------------- check / control points
      box.appendChild(checkpointsBlock(ctx, load));

      // ---------------- completeness
      box.appendChild(group("Completeness", [
        el("div", { class: "stat-row" }, [
          stat("Observed", c.observed_area_ha != null ? c.observed_area_ha : "—", "ha"),
          stat("Measured", c.measured_pct != null ? c.measured_pct : "—", "%"),
          stat("Facades seen", c.facade_coverage_median_pct != null ? c.facade_coverage_median_pct : "—", "%"),
        ]),
        el("div", { class: "note" }, "Measured = surface carrying 3-D points; the rest (water, shadow, occluded ground) is closed by the terrain model. " + (c.facades_note || "")),
      ]));

      // ---------------- processing
      const stages = p.stages || [];
      const tot = stages.reduce((s, x) => s + (x.seconds || 0), 0) || 1;
      const bar = el("div", { class: "stackbar tall" }, stages.map((s, k) => el("span", {
        style: { width: `${(100 * (s.seconds || 0)) / tot}%`, background: STAGE_COLORS[k % STAGE_COLORS.length] },
        title: `${s.stage}: ${s.seconds} s`,
      })));
      const legend = el("div", { class: "legend" }, stages.map((s, k) => el("span", {}, [
        el("i", { style: { background: STAGE_COLORS[k % STAGE_COLORS.length] } }), `${String(s.stage).replace(/^\d+\.\s*/, "")} ${fmt.time(s.seconds)}`])));
      box.appendChild(group("Processing", stages.length ? [
        kv([["Engine", p.engine], ["GPU", (p.hardware || {}).gpu_name], ["Total", p.total_s != null ? `${fmt.time(p.total_s)} for ${fmt.time(p.video_s)} of video` : "—"],
          ["Per 10 min of video", p.minutes_per_10_min_video != null ? `${p.minutes_per_10_min_video} min (target < 15)` : "—"]]),
        bar, legend,
        p.sih_target_met === false ? callout("Use QUALITY = \"sih\" in the Colab notebook: its time budget follows the video length (1.4 x) so the target is met on a free T4.", "caution") : null,
      ] : [el("div", { class: "note" }, "No processing report was stored with this survey.")]));

      // ---------------- inputs
      const cv = i.camera_view;
      const dyn = i.dynamic_objects;
      box.appendChild(group("Inputs", [kv([
        ["Video", i.video.resolution ? `${i.video.resolution} · ${i.video.fps} fps · ${fmt.time(i.video.duration_s)}` : "—"],
        ["Flight log", `${i.flight_log.source || "—"} · ${fmt.n(i.flight_log.fixes)} fixes`],
        ["Positioning", i.position_source + (i.rtk ? ` (${i.rtk.fix_pct}% fixed)` : "")],
        ["Camera view", cv ? `${cv.view}, ${Math.abs(cv.median_pitch_deg).toFixed(1)}° below horizon` : "—"],
        ["IMU / gimbal check", i.imu_gimbal.levelling_check_deg != null ? `${i.imu_gimbal.levelling_check_deg}° vs gimbal` : "no gimbal log"],
        ["Intrinsics", i.camera_intrinsics],
        ["Moving objects removed", dyn ? `${dyn.moving ?? "—"} of ${dyn.detections} detected` : "—"],
      ])]));

      // ---------------- actions
      const acts = [
        el("button", { class: "btn primary", type: "button", onclick: () => window.open(ctx.api.url(`api/quality/report.html${ctx.bq()}`), "_blank") }, "Open printable report"),
        el("button", { class: "btn", type: "button", "data-action": "export-centre" }, "Export centre…"),
      ];
      if (ctx.state.survey && ctx.state.survey.kind === "active") acts.push(rtkButton(ctx, load));
      box.appendChild(group("Deliver", acts));
    };

    if (rep && cps) render(); else load();
  },
  clear(ctx) { ctx.viewer.clearLayer("checkpoints"); ctx.viewer.clearLayer("cp-pick"); },
  reset() { rep = null; cps = null; },
};

// ------------------------------------------------------------------------------------------------ check points
function checkpointsBlock(ctx, reload) {
  ctx.viewer.clearLayer("checkpoints");
  const wrap = group("Check points (surveyed coordinates)", []);
  const st = cps.stats;
  wrap.appendChild(el("div", { class: "note" },
    "Click a marker you surveyed (road paint, corner, GCP target) in the model, then type its coordinates. The residuals show the true map accuracy; "
    + "Apply correction removes the GNSS bias using all points (the leave-one-out figure is the honest accuracy afterwards)."));
  if (cps.checkpoints.length) {
    const tb = el("tbody");
    cps.checkpoints.forEach((c, k) => {
      tb.appendChild(el("tr", {}, [
        el("td", { class: "mono" }, c.name),
        el("td", { class: "num" }, c.d_e != null ? c.d_e.toFixed(2) : "—"),
        el("td", { class: "num" }, c.d_n != null ? c.d_n.toFixed(2) : "—"),
        el("td", { class: "num" }, c.d_h != null ? c.d_h.toFixed(2) : "—"),
        el("td", { class: "num" }, [el("button", { class: "btn ghost small", type: "button", title: "Remove", onclick: async () => {
          try { cps = await ctx.api.del(`/api/quality/checkpoints/${k}${ctx.bq()}`); await reload(); } catch (e) { toast(e.message, "danger"); }
        } }, "×")]),
      ]));
      ctx.viewer.label("checkpoints", [c.model[0], c.model[1] + 1.5, c.model[2]], `${c.name} · ${c.horizontal_m != null ? c.horizontal_m.toFixed(2) : "?"} m`, "accent");
    });
    wrap.appendChild(el("table", { class: "data" }, [
      el("thead", {}, el("tr", {}, [el("th", {}, "Point"), el("th", { class: "num" }, "ΔE m"), el("th", { class: "num" }, "ΔN m"), el("th", { class: "num" }, "ΔH m"), el("th", {}, "")])), tb]));
    if (st) {
      wrap.appendChild(kv([
        ["RMSE horizontal", `${st.rmse_horizontal_m} m`],
        ["Leave-one-out RMSE", st.rmse_horizontal_loo_m != null ? `${st.rmse_horizontal_loo_m} m` : "needs 2+ points"],
        ["RMSE vertical", st.rmse_vertical_m != null ? `${st.rmse_vertical_m} m` : "—"],
        ["Correction applied", cps.correction ? `E ${cps.correction.d_e_m} · N ${cps.correction.d_n_m} · H ${cps.correction.d_h_m} m` : "none"],
      ]));
    }
  }
  const form = el("div", { class: "group hidden cp-form" });
  const lat = el("input", { class: "input", type: "number", step: "any", placeholder: "Latitude (decimal °)" });
  const lon = el("input", { class: "input", type: "number", step: "any", placeholder: "Longitude (decimal °)" });
  const elev = el("input", { class: "input", type: "number", step: "any", placeholder: "Elevation m (optional)" });
  const name = el("input", { class: "input", type: "text", maxlength: "24", placeholder: `CP-${cps.checkpoints.length + 1}` });
  const picked = el("div", { class: "note mono" });
  let model = null;
  const save = el("button", { class: "btn primary small", type: "button", onclick: async () => {
    const la = parseFloat(lat.value), lo = parseFloat(lon.value), ev = parseFloat(elev.value);
    if (!model || !isFinite(la) || !isFinite(lo)) { toast("Pick the point in the model and enter its latitude and longitude."); return; }
    try {
      cps = await ctx.api.post("/api/quality/checkpoints", { name: name.value.trim() || undefined, model, lat: la, lon: lo,
        elev: isFinite(ev) ? ev : null, baseline_id: ctx.bid() });
      await reload();
    } catch (e) { toast(e.message, "danger"); }
  } }, "Save point");
  form.append(picked, el("div", { class: "grid2" }, [lat, lon]), el("div", { class: "grid2" }, [elev, name]), el("div", { class: "row" }, [save]));
  const addBtn = el("button", { class: "btn small", type: "button", onclick: () => {
    ctx.requestPick({ prompt: "Click the surveyed marker in the model", onPick: (p) => {
      model = p;
      picked.textContent = `Model point x ${p[0].toFixed(2)}  y ${p[1].toFixed(2)}  z ${p[2].toFixed(2)}`;
      ctx.viewer.clearLayer("cp-pick");
      ctx.viewer.sphere("cp-pick", p, 0.4, 0xc9a45c);
      form.classList.remove("hidden");
      lat.focus();
    } });
  } }, "Add check point");
  const row = el("div", { class: "row" }, [addBtn]);
  if (cps.checkpoints.length) {
    row.appendChild(el("button", { class: "btn small", type: "button", onclick: async () => {
      if (!confirm("Shift the georeference so the surveyed points fit (removes the GNSS bias)? All georeferenced exports will follow; this can be undone.")) return;
      try {
        await ctx.api.post("/api/quality/checkpoints/apply", { baseline_id: ctx.bid() });
        toast("Georeference corrected with the surveyed points.", "ok");
        await reload();
      } catch (e) { toast(e.message, "danger"); }
    } }, "Apply correction"));
  }
  if (cps.correction) {
    row.appendChild(el("button", { class: "btn ghost small", type: "button", onclick: async () => {
      try { await ctx.api.post("/api/quality/checkpoints/apply", { baseline_id: ctx.bid(), undo: true }); toast("Correction removed."); await reload(); }
      catch (e) { toast(e.message, "danger"); }
    } }, "Undo correction"));
  }
  wrap.append(row, form);
  return wrap;
}

function rtkButton(ctx, reload) {
  const input = el("input", { type: "file", accept: ".pos,.csv,.txt", hidden: true });
  input.addEventListener("change", async () => {
    if (!input.files.length) return;
    const form = new FormData();
    form.append("rtk_file", input.files[0]);
    ctx.setStatus("Re-georeferencing with RTK / PPK positions…", "busy");
    try {
      const r = await ctx.api.upload("/api/model/rtk", form);
      toast(`RTK/PPK applied: ${r.rtk.position_source} (${r.rtk.fix_pct}% fixed), fit ${r.gps_rmse_m} m.`, "ok");
      ctx.reloadActive();
    } catch (e) { toast(`RTK/PPK not applied: ${e.message}`, "danger"); ctx.setStatus("RTK/PPK failed", "danger"); }
  });
  return el("button", { class: "btn", type: "button", onclick: () => input.click() }, ["Apply RTK / PPK positions…", input]);
}
