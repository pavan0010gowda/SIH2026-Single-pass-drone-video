"""
PRISM // survey quality & accuracy report (quality.py)

One place that answers the questions an evaluator asks of a single-pass reconstruction:
  * what went in (video, GNSS log, altitude source, IMU / gimbal, intrinsics, RTK / PPK)
  * how long it took (per stage) against the "10-minute video in < 15 minutes" target
  * how accurate it is: internal (relative) accuracy from the bundle and the GPS fit, absolute accuracy
    from the positioning source, and - when the analyst has entered surveyed check points - the
    measured RMSE on those points
  * how complete it is: observed footprint, measured vs. filled surface, facade coverage
  * what can be delivered (formats) and where the limits are
"""
import html
import json
import math
import os
import time

import numpy as np

SIH_TARGET_RATIO = 1.5            # 10-minute video -> < 15 minutes processing
SIH_ACCURACY_M = 1.0

# typical 1-sigma horizontal / vertical bias of the positioning source (not reduced by averaging)
POSITION_BIAS = {
    "RTK_FIXED": (0.03, 0.05), "PPK_FIXED": (0.03, 0.05), "RTK_FLOAT": (0.4, 0.6), "PPK_FLOAT": (0.4, 0.6),
    "SBAS": (0.8, 1.2), "GNSS": (1.2, 2.0), "NONE": (None, None),
}


def _f(v, nd=2):
    try:
        v = float(v)
        return round(v, nd) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def position_source(telem):
    src = str((telem or {}).get("position_source") or "").upper()
    if src in POSITION_BIAS:
        return src
    if not (telem or {}).get("waypoints"):
        return "NONE"
    return "GNSS"


def checkpoint_stats(cps, correction=None):
    rows = [c for c in (cps or []) if c.get("d_e") is not None]
    if not rows:
        return None
    loo = [c for c in rows if c.get("loo_e") is not None]
    de = np.array([c["d_e"] for c in rows])
    dn = np.array([c["d_n"] for c in rows])
    hz = np.hypot(de, dn)
    dh = np.array([c["d_h"] for c in rows if c.get("d_h") is not None])
    out = {"n": len(rows), "rmse_e_m": _f(np.sqrt(np.mean(de ** 2)), 3), "rmse_n_m": _f(np.sqrt(np.mean(dn ** 2)), 3),
           "rmse_horizontal_m": _f(np.sqrt(np.mean(hz ** 2)), 3), "max_horizontal_m": _f(hz.max(), 3),
           "mean_e_m": _f(de.mean(), 3), "mean_n_m": _f(dn.mean(), 3)}
    if len(dh):
        out.update(rmse_vertical_m=_f(np.sqrt(np.mean(dh ** 2)), 3), mean_vertical_m=_f(dh.mean(), 3),
                   n_vertical=int(len(dh)))
    if len(loo) >= 2:
        le = np.array([c["loo_e"] for c in loo])
        ln_ = np.array([c["loo_n"] for c in loo])
        out["rmse_horizontal_loo_m"] = _f(np.sqrt(np.mean(le ** 2 + ln_ ** 2)), 3)
    if correction:
        out["correction_applied"] = correction
    return out


def build_report(*, name, telem, recon, tm, n_points, mesh_faces, formats, buildings=None, roads=None,
                 checkpoints=None, spacing_m=None, frames=None):
    telem = telem or {}
    recon = recon or {}
    geo = telem.get("georeference") or {}
    diag = geo.get("diagnostics") if isinstance(geo.get("diagnostics"), dict) else {}
    video = recon.get("video") or telem.get("video") or {}
    dur = _f(video.get("duration") or video.get("duration_sec"), 1)
    wps = telem.get("waypoints") or []
    src = position_source(telem)
    try:
        import sensors
        tilt = sensors.camera_tilt_from_frames(frames)
    except Exception:
        tilt = None
    gimbal = telem.get("gimbal_pitch_median_deg")
    imu_check = round(abs(tilt["median_pitch_deg"] - float(gimbal)), 2) if (tilt and gimbal is not None) else None

    # ---------------- inputs
    inputs = {
        "video": {"resolution": f"{video.get('width')}x{video.get('height')}" if video.get("width") else None,
                  "fps": _f(video.get("fps"), 2), "duration_s": dur, "codec": video.get("codec")},
        "flight_log": {"source": telem.get("source"), "fixes": len(wps), "timestamps": bool(telem.get("has_timestamps")),
                       "latlon_order": telem.get("latlon_order"), "latlon_verified": telem.get("latlon_verified")},
        "altitude": {"reference": telem.get("altitude_reference"), "barometer": bool(telem.get("has_barometer")),
                     "estimated": bool(telem.get("altitude_is_estimated"))},
        "imu_gimbal": {"gimbal_pitch_median_deg": gimbal,
                       "attitude_records": telem.get("attitude_records") or 0,
                       "levelling_check_deg": imu_check},
        "camera_view": tilt,
        "dynamic_objects": telem.get("dynamic_objects") or (recon.get("dynamic_objects") or None),
        "camera_intrinsics": telem.get("intrinsics_source") or ("self-calibrated in the bundle adjustment" if recon.get("sfm") else None),
        "position_source": src,
        "rtk": telem.get("rtk_summary"),
    }

    # ---------------- processing
    timings = recon.get("timings") or []
    total = _f(recon.get("total_seconds"), 1)
    ratio = (total / dur) if (total and dur) else None
    processing = {"engine": recon.get("engine"), "hardware": recon.get("hardware"), "stages": timings,
                  "total_s": total, "video_s": dur, "ratio": _f(ratio, 2),
                  "minutes_per_10_min_video": _f(10.0 * ratio, 1) if ratio else None,
                  "sih_target_met": (ratio <= SIH_TARGET_RATIO) if ratio else None,
                  "quality_profile": (recon.get("profile") or {}).get("quality"),
                  "budget_min": _f((recon.get("profile") or {}).get("target_s", 0) / 60.0, 1) if recon.get("profile") else None}

    # ---------------- reconstruction
    sfm = recon.get("sfm") or {}
    area = float(tm.roi.sum()) * tm.res ** 2 if tm is not None else None
    recon_q = {"keyframes": (recon.get("ingest") or {}).get("keyframes") or sfm.get("num_images"),
               "registered": sfm.get("num_registered"), "tie_points": sfm.get("num_points"),
               "mean_reprojection_error_px": _f(sfm.get("mean_reproj_error"), 3),
               "mean_track_length": _f(sfm.get("mean_track_length"), 2),
               "dense_points": n_points, "mesh_faces": mesh_faces,
               "point_density_per_m2": _f(n_points / area, 1) if (area and n_points) else None,
               "point_spacing_cm": _f(100 * spacing_m, 1) if spacing_m else None,
               "camera_model": sfm.get("camera_model")}

    # ---------------- accuracy
    rmse_h = _f(diag.get("gps_rmse_horizontal_m") if diag.get("gps_rmse_horizontal_m") is not None else diag.get("rmse_m"), 3)
    n_gps = diag.get("gps_inliers") or diag.get("gps_matched_cameras") or diag.get("inliers") or 0
    scale_unc = geo.get("scale_rel_uncertainty")
    extent = max(tm.nx, tm.nz) * tm.res if tm is not None else None
    bias_h, bias_v = POSITION_BIAS.get(src, (None, None))
    # random GNSS error averages over the track (correlation length ~10 fixes); the bias does not
    fit_h = (rmse_h / math.sqrt(max(1.0, n_gps / 10.0))) if (rmse_h and n_gps) else None
    abs_h = math.sqrt(fit_h ** 2 + bias_h ** 2) if (fit_h is not None and bias_h is not None) else None
    abs_v = bias_v
    b_unc = [b["uncertainty_m"] for b in (buildings or []) if b.get("uncertainty_m") is not None]
    rel_100 = 100.0 * scale_unc if scale_unc else None
    cp = checkpoint_stats(checkpoints, geo.get("control_correction"))
    accuracy = {
        "georeference_mode": geo.get("mode"), "confidence": geo.get("confidence"), "metric": bool(geo.get("metric")),
        "scale_uncertainty_pct": _f(100 * scale_unc, 3) if scale_unc else None,
        "relative_error_per_100m_m": _f(rel_100, 3),
        "relative_error_over_scene_m": _f(scale_unc * extent, 2) if (scale_unc and extent) else None,
        "scene_extent_m": _f(extent, 0),
        "gps_fit_rmse_horizontal_m": rmse_h, "gps_fit_rmse_3d_m": _f(diag.get("gps_rmse_3d_m"), 3), "gps_fixes_used": n_gps,
        "levelling_disagreement_deg": _f(diag.get("up_disagreement_deg"), 3),
        "height_precision_median_m": _f(float(np.median(b_unc)), 3) if b_unc else None,
        "absolute_horizontal_1sigma_m": _f(abs_h, 2), "absolute_vertical_1sigma_m": _f(abs_v, 2),
        "absolute_note": ({"RTK_FIXED": "RTK fixed positions: centimetre-level georeferencing.",
                           "PPK_FIXED": "PPK fixed positions: centimetre-level georeferencing."}.get(src)
                          or ("Standalone GNSS: the absolute position carries the receiver bias (about 1-2 m); "
                              "distances, areas and heights inside the model are far more accurate than its "
                              "absolute placement. Add RTK/PPK positions or check points to tighten it.")),
        "checkpoints": cp,
    }
    # SIH <= 1 m: measured on check points when available, otherwise the estimate
    if cp and cp.get("correction_applied") and cp.get("rmse_horizontal_loo_m") is not None:
        acc_val, acc_basis = cp["rmse_horizontal_loo_m"], f"leave-one-out RMSE on {cp['n']} control points after correction"
    elif cp and cp.get("correction_applied"):
        acc_val, acc_basis = None, "corrected with 1 control point (add a second point for an independent check)"
    elif cp and cp.get("rmse_horizontal_m") is not None:
        acc_val, acc_basis = cp["rmse_horizontal_m"], f"measured on {cp['n']} check point(s)"
    elif abs_h is not None:
        acc_val, acc_basis = abs_h, "estimated from the GPS fit and the positioning source"
    else:
        acc_val, acc_basis = None, "no georeference"
    accuracy["sih_spatial_accuracy_m"] = _f(acc_val, 2)
    accuracy["sih_spatial_accuracy_basis"] = acc_basis
    accuracy["sih_target_met"] = (acc_val <= SIH_ACCURACY_M) if acc_val is not None else None

    # ---------------- completeness
    comp = {}
    if tm is not None:
        roi = tm.roi
        has = getattr(tm, "has_points", tm.observed) & roi
        comp = {"observed_area_m2": _f(area, 0), "observed_area_ha": _f(area / 1e4, 2) if area else None,
                "measured_pct": _f(100.0 * has.sum() / max(1, roi.sum()), 1),
                "interpolated_pct": _f(100.0 * (roi & ~has).sum() / max(1, roi.sum()), 1),
                "class_pct": tm.summary().get("class_percent")}
    if buildings:
        cov = [b["facade_coverage_pct"] for b in buildings if b.get("facade_coverage_pct") is not None]
        comp["buildings"] = len(buildings)
        comp["facade_coverage_median_pct"] = _f(np.median(cov), 1) if cov else None
        comp["facades_note"] = ("A single pass sees the walls that face the flight line; the other walls are "
                                "closed by the surface model (shown per building in Structures).")
    if roads:
        comp["road_network_km"] = _f((roads.get("network") or {}).get("total_length_m", 0) / 1000.0, 2)

    # ---------------- SIH table
    sih = [
        {"parameter": "Reconstruction type", "target": "3D mesh / point cloud",
         "achieved": ("Textured mesh + point cloud" if mesh_faces else "Point cloud (mesh on demand)"), "met": True},
        {"parameter": "Processing time", "target": "< 15 min for a 10-min video",
         "achieved": (f"{processing['minutes_per_10_min_video']} min per 10 min of video"
                      if processing["minutes_per_10_min_video"] else "not recorded for this survey"),
         "met": processing["sih_target_met"]},
        {"parameter": "Spatial accuracy (measurements inside the model)", "target": "<= 1 m",
         "achieved": (f"{accuracy['relative_error_over_scene_m']} m worst case across the {accuracy['scene_extent_m']:.0f} m scene "
                      f"({accuracy['scale_uncertainty_pct']} % scale); heights +/-{accuracy['height_precision_median_m']} m"
                      if accuracy.get("relative_error_over_scene_m") is not None else "no metric scale"),
         "met": (accuracy["relative_error_over_scene_m"] <= SIH_ACCURACY_M) if accuracy.get("relative_error_over_scene_m") is not None else None},
        {"parameter": "Spatial accuracy (absolute map position)", "target": "<= 1 m",
         "achieved": (f"{accuracy['sih_spatial_accuracy_m']} m ({acc_basis})" if acc_val is not None else acc_basis),
         "met": accuracy["sih_target_met"]},
        {"parameter": "Coverage", "target": "Entire visible scene",
         "achieved": (f"{comp.get('observed_area_ha')} ha observed, {comp.get('measured_pct')} % measured, "
                      f"rest filled" if comp else "n/a"), "met": bool(comp)},
        {"parameter": "Output formats", "target": "OBJ, PLY, LAS, GeoTIFF, glTF/GLB, FBX",
         "achieved": ", ".join(formats), "met": all(k in formats for k in ("OBJ", "PLY", "LAS", "GeoTIFF", "glTF", "GLB", "FBX"))},
        {"parameter": "Visualization", "target": "Web-based or desktop viewer", "achieved": "Web viewer (this dashboard)", "met": True},
    ]
    return {"name": name, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "inputs": inputs, "processing": processing,
            "reconstruction": recon_q, "accuracy": accuracy, "completeness": comp, "sih": sih,
            "warnings": list(geo.get("warnings") or []) + list(recon.get("warnings") or [])}


# =============================================================================
# printable HTML
# =============================================================================
_CSS = """
:root{--ink:#1d2024;--mut:#5d636b;--line:#d9dcdf;--ok:#2f7a45;--bad:#b3412e;--acc:#8a6d2b;--bg:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 'IBM Plex Sans',Segoe UI,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 24px 60px}
h1{font-size:22px;margin:0 0 2px}h2{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--acc);
margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}.sub{color:var(--mut)}
table{width:100%;border-collapse:collapse;margin:6px 0}td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
td.num{font-family:'IBM Plex Mono',Consolas,monospace;text-align:right;white-space:nowrap}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.na{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0 28px}
.note{color:var(--mut);font-size:12px}img{max-width:100%;border:1px solid var(--line)}
@media print{main{padding:0}h2{break-after:avoid}table{break-inside:avoid}}
"""


def _kv(rows):
    out = []
    for k, v in rows:
        if v is None or v == "" or v == {}:
            v = "<span class='na'>n/a</span>"
        elif isinstance(v, bool):
            v = "yes" if v else "no"
        elif isinstance(v, float):
            v = f"{v:,.3f}".rstrip("0").rstrip(".")
        else:
            v = html.escape(str(v))
        out.append(f"<tr><td>{html.escape(k)}</td><td class='num'>{v}</td></tr>")
    return "<table>" + "".join(out) + "</table>"


def render_html(rep, preview_data_url=None):
    a, p, r, c, i = rep["accuracy"], rep["processing"], rep["reconstruction"], rep["completeness"], rep["inputs"]
    sih_rows = "".join(
        f"<tr><td>{html.escape(s['parameter'])}</td><td>{html.escape(s['target'])}</td><td>{html.escape(str(s['achieved']))}</td>"
        f"<td class='{'ok' if s['met'] else ('bad' if s['met'] is False else 'na')}'>"
        f"{'Met' if s['met'] else ('Not met' if s['met'] is False else 'Unknown')}</td></tr>" for s in rep["sih"])
    stages = "".join(f"<tr><td>{html.escape(str(s.get('stage')))}</td><td class='num'>{s.get('seconds')} s</td>"
                     f"<td class='note'>{html.escape(str(s.get('note') or ''))}</td></tr>" for s in p.get("stages") or [])
    cp = a.get("checkpoints")
    corr = (cp or {}).get("correction_applied") or {}
    cp_html = _kv([("Surveyed points", cp["n"]), ("RMSE east (m)", cp["rmse_e_m"]), ("RMSE north (m)", cp["rmse_n_m"]),
                   ("RMSE horizontal (m)", cp["rmse_horizontal_m"]), ("Max horizontal (m)", cp["max_horizontal_m"]),
                   ("Leave-one-out RMSE horizontal (m)", cp.get("rmse_horizontal_loo_m")),
                   ("RMSE vertical (m)", cp.get("rmse_vertical_m")),
                   ("Control correction applied (E, N, H m)",
                    f"{corr.get('d_e_m')}, {corr.get('d_n_m')}, {corr.get('d_h_m')}" if corr else None)]) if cp else \
        "<p class='note'>No check points entered. In the dashboard: Analyze &rsaquo; Quality report &rsaquo; Add check point.</p>"
    classes = c.get("class_pct") or {}
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in rep.get("warnings") or [])
    img = f"<img src='{preview_data_url}' alt='top-down view'>" if preview_data_url else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PRISM quality report</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{_CSS}</style></head><body><main>
<h1>Survey quality &amp; accuracy report</h1>
<div class="sub">{html.escape(rep['name'])} &middot; generated {rep['generated']} &middot; PRISM survey intelligence</div>
<h2>Problem-statement targets</h2>
<table><tr><th>Parameter</th><th>Target</th><th>Achieved</th><th>Status</th></tr>{sih_rows}</table>
<div class="grid"><div><h2>Accuracy</h2>{_kv([
        ("Georeference", f"{a.get('georeference_mode')} ({a.get('confidence')})"),
        ("Scale uncertainty (%)", a.get("scale_uncertainty_pct")),
        ("Relative error per 100 m (m)", a.get("relative_error_per_100m_m")),
        ("Relative error across the scene (m)", a.get("relative_error_over_scene_m")),
        ("GPS fit RMSE, horizontal (m)", a.get("gps_fit_rmse_horizontal_m")),
        ("GPS fixes used", a.get("gps_fixes_used")),
        ("Levelling disagreement (deg)", a.get("levelling_disagreement_deg")),
        ("Height precision, median building (m)", a.get("height_precision_median_m")),
        ("Absolute horizontal, 1 sigma (m)", a.get("absolute_horizontal_1sigma_m")),
        ("Absolute vertical, 1 sigma (m)", a.get("absolute_vertical_1sigma_m"))])}
<p class="note">{html.escape(a.get('absolute_note') or '')}</p></div>
<div><h2>Completeness</h2>{_kv([
        ("Observed area (ha)", c.get("observed_area_ha")), ("Measured surface (%)", c.get("measured_pct")),
        ("Interpolated surface (%)", c.get("interpolated_pct")), ("Buildings", c.get("buildings")),
        ("Facade coverage, median (%)", c.get("facade_coverage_median_pct")), ("Road network (km)", c.get("road_network_km"))]
        + [(f"{k.replace('_', ' ')} (%)", v) for k, v in classes.items()])}</div></div>
<h2>Check points</h2>{cp_html}
<div class="grid"><div><h2>Inputs</h2>{_kv([
        ("Video", f"{i['video'].get('resolution')} @ {i['video'].get('fps')} fps, {i['video'].get('duration_s')} s"),
        ("Flight log", f"{i['flight_log'].get('source')} ({i['flight_log'].get('fixes')} fixes)"),
        ("Lat/lon order verified", i['flight_log'].get('latlon_verified')),
        ("Altitude reference", i['altitude'].get('reference')), ("Barometer", i['altitude'].get('barometer')),
        ("Camera view (from the reconstruction)", (f"{i['camera_view']['view']}, {abs(i['camera_view']['median_pitch_deg']):.1f} deg below the horizon"
                                                    if i.get('camera_view') else None)),
        ("Gimbal pitch, median (deg)", i['imu_gimbal'].get('gimbal_pitch_median_deg')),
        ("IMU levelling check (deg)", i['imu_gimbal'].get('levelling_check_deg')),
        ("Camera intrinsics", i.get('camera_intrinsics')), ("Positioning", i.get('position_source')),
        ("Dynamic objects (detected / moving, not fused)", (f"{i['dynamic_objects'].get('detections')} / {i['dynamic_objects'].get('moving')}"
                                                            if i.get('dynamic_objects') else None))])}</div>
<div><h2>Reconstruction</h2>{_kv([
        ("Keyframes (registered / total)", f"{r.get('registered')} / {r.get('keyframes')}"),
        ("Tie points", r.get("tie_points")), ("Mean reprojection error (px)", r.get("mean_reprojection_error_px")),
        ("Mean track length", r.get("mean_track_length")), ("Dense points", r.get("dense_points")),
        ("Point density (pts/m2)", r.get("point_density_per_m2")), ("Point spacing (cm)", r.get("point_spacing_cm")),
        ("Mesh faces", r.get("mesh_faces")), ("Camera model", r.get("camera_model"))])}</div></div>
<h2>Processing</h2>{_kv([("Engine", p.get('engine')), ("GPU", (p.get('hardware') or {}).get('gpu_name')),
                          ("Quality profile", p.get('quality_profile')), ("Time budget (min)", p.get('budget_min')),
                          ("Total (s)", p.get('total_s')), ("Video length (s)", p.get('video_s')),
                          ("Processing / video time", p.get('ratio'))])}
{('<table><tr><th>Stage</th><th>Time</th><th>Note</th></tr>' + stages + '</table>') if stages else ''}
{('<h2>Warnings</h2><ul>' + warn + '</ul>') if warn else ''}
{('<h2>Survey view</h2>' + img) if img else ''}
</main></body></html>"""
