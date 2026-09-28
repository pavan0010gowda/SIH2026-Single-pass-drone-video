"""
PRISM // Re-georeferencing of reconstructed bundles (recalibrate.py)
====================================================================
A reconstruction that was scaled from a *guessed* altitude (or aligned with a mirrored GPS track)
can be fixed afterwards without re-running photogrammetry, as long as the camera poses are known.
The Colab bundle stores every keyframe pose in frame_index.json; together with the flight log
this is enough to recover the true metric similarity:

  1. Camera centres C_i (current model frame) are paired with GPS positions at the exact frame
     timestamps (staircase-free interpolation of the log).
  2. Latitude/longitude order is tested both ways when the log is ambiguous: swapping them mirrors
     the track, which no proper similarity can fit (Danube sortie: 0.8 m vs 21 m RMSE).
  3. Gravity: the model's current vertical (camera-horizon estimate) and the GPS-implied vertical
     are fused by inverse-variance weighting; a large disagreement keeps the horizon estimate.
  4. A 4-DOF fit (scale, heading, translation) on the levelled frame with iterative outlier
     rejection gives the scale; its uncertainty follows from the residuals and the track spread.
  5. The ground datum is the lowest strong mode of the terrain height histogram, so y = 0 is the
     ground in the output frame; the GPS origin is shifted to that datum.

Every geometry file (points, display cloud, mesh PLY/OBJ/GLB), the camera poses and the
telemetry are rewritten in the new frame. The originals are moved to data/models/_pre_recalibration.
"""

import json
import math
import os
import shutil
import time

import numpy as np

try:
    import georeference as G
    import plyio
    from telemetry_parser import parse_srt_records, _path_length, srt_matches_telemetry
except ImportError:
    from backend import georeference as G
    from backend import plyio
    from backend.telemetry_parser import parse_srt_records, _path_length, srt_matches_telemetry

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
HORIZON_SIGMA_DEG = 0.3          # typical residual roll of a stabilised gimbal
MAX_UP_DISAGREEMENT_DEG = 5.0


# -----------------------------------------------------------------------------
# small maths helpers
# -----------------------------------------------------------------------------
def _umeyama(src, dst):
    """dst ~ s R src + t for any dimension (proper rotation)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    d = src.shape[1]
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.eye(d)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[-1, -1] = -1
    r = u @ sgn @ vt
    var = (a ** 2).sum() / len(src)
    s = float((sv * np.diag(sgn)).sum() / var) if var > 0 else 1.0
    return s, r, md - s * r @ ms


def _quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat_to_quat(r):
    r = np.asarray(r, float)
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _h4(r3=None, t3=None):
    m = np.eye(4)
    if r3 is not None:
        m[:3, :3] = r3
    if t3 is not None:
        m[:3, 3] = t3
    return m


# -----------------------------------------------------------------------------
# inputs
# -----------------------------------------------------------------------------
def load_frame_cameras(frame_index_path):
    """Registered keyframes with a pose: list of dicts (timestamp_sec, position, quaternion_xyzw, ...)."""
    with open(frame_index_path, "r", encoding="utf-8") as f:
        fi = json.load(f)
    entries = fi if isinstance(fi, list) else fi.get("frames") if isinstance(fi, dict) else None
    if not isinstance(entries, list):
        return fi, []
    cams = [e for e in entries if isinstance(e, dict) and e.get("registered", True)
            and e.get("position") is not None and e.get("timestamp_sec") is not None]
    return fi, cams


def _log_arrays(recs, swap, duration_s=None):
    t = np.array([np.nan if r.get("time_s") is None else r["time_s"] for r in recs], float)
    if not np.isfinite(t).all():
        dur = duration_s or float(len(recs) - 1)
        t = np.linspace(0.0, dur, len(recs))
    lat = np.array([r["lat"] for r in recs], float)
    lon = np.array([r["lon"] for r in recs], float)
    if swap:
        lat, lon = lon, lat
    alt = np.array([np.nan if r.get("alt") is None else r["alt"] for r in recs], float)
    return t, lat, lon, alt


def _gps_at(times, recs, swap, duration_s=None):
    t, lat, lon, alt = _log_arrays(recs, swap, duration_s)
    order = np.argsort(t, kind="stable")
    t, lat, lon, alt = t[order], lat[order], lon[order], alt[order]
    q = np.clip(times, t[0], t[-1])
    (ll,) = G._interp_fresh(q, t, np.c_[lat, lon])
    has_alt = np.isfinite(alt).sum() >= max(3, 0.5 * len(alt))
    a = None
    if has_alt:
        good = np.isfinite(alt)
        (a,) = G._interp_fresh(q, t[good], alt[good])
    valid = (times >= t[0] - 0.5) & (times <= t[-1] + 0.5)
    return ll[:, 0], ll[:, 1], a, valid


# -----------------------------------------------------------------------------
# the fit
# -----------------------------------------------------------------------------
def fit_model_to_gps(cam_three, times, recs, ambiguous, duration_s=None, rng=None, log=print):
    """
    cam_three: (N,3) camera centres in the current Y-up model frame.
    Returns dict(T_zup 4x4 in the Z-up model frame -> ENU, origin, swap, rmse, diagnostics, ...).
    """
    rng = rng or np.random.default_rng(0)
    c = np.asarray(cam_three, float) @ G.ENU_FROM_YUP.T          # model frame with z up
    hyps = [False, True] if ambiguous else [False]
    trials = []
    for swap in hyps:
        lat, lon, alt, valid = _gps_at(times, recs, swap, duration_s)
        if valid.sum() < 5:
            continue
        origin = (float(np.median(lat[valid])), float(np.median(lon[valid])),
                  float(np.median(alt[valid])) if alt is not None else 0.0)
        enu = G.geodetic_to_enu(lat, lon, alt if alt is not None else np.zeros_like(lat), origin)
        if alt is not None:
            s, r, t, _, _ = G.ransac_similarity(c[valid], enu[valid], rng, iters=400)
            res = np.linalg.norm(enu[valid] - (s * c[valid] @ r.T + t), axis=1)
        else:
            s, r2, t2 = _umeyama(c[valid][:, :2], enu[valid][:, :2])
            res = np.linalg.norm(enu[valid][:, :2] - (s * c[valid][:, :2] @ r2.T + t2), axis=1)
            r = np.eye(3)
        # score each hypothesis on ALL matched frames (90 % trimmed RMS), not only RANSAC inliers
        r2s = np.sort(res ** 2)
        rmse = float(np.sqrt(np.mean(r2s[:max(3, int(0.9 * len(r2s)))])))
        trials.append(dict(swap=swap, lat=lat, lon=lon, alt=alt, valid=valid, origin=origin, enu=enu,
                           s=s, r=r, rmse=float(rmse)))
    if not trials:
        raise ValueError("fewer than 5 keyframes overlap the flight log in time")
    trials.sort(key=lambda d: d["rmse"])
    best = trials[0]
    decided = len(trials) == 1 or best["rmse"] < 0.6 * trials[1]["rmse"]
    if not decided:            # cannot tell (e.g. straight-line flight): keep the logged order
        best = next(d for d in trials if not d["swap"])
    diag = {"latlon_test_rmse_m": {("swapped" if d["swap"] else "as_logged"): round(d["rmse"], 3) for d in trials},
            "latlon_decided": bool(decided)}
    warnings = []
    if not decided and ambiguous:
        warnings.append("Latitude/longitude order could not be verified (flight path too straight); "
                        "kept the order written in the log.")

    valid, enu = best["valid"], best["enu"]
    cv, ev = c[valid], enu[valid]
    has_alt = best["alt"] is not None
    ez = np.array([0.0, 0.0, 1.0])

    # ---- gravity: fuse the model's horizon-levelled vertical with the GPS vertical
    up = ez.copy()
    if has_alt:
        s7, r7, t7, inl7, rmse7 = G.ransac_similarity(cv, ev, rng, iters=400)
        up_gps = r7.T @ ez
        hs = np.linalg.svd(ev[inl7, :2] - ev[inl7, :2].mean(0), compute_uv=False) / math.sqrt(max(1, inl7.sum()))
        n_eff = min(int(inl7.sum()), 10)
        sig_g = max(math.radians(0.05), math.atan2(max(rmse7, G.GPS_NOISE_FLOOR_M * 0.3),
                                                   max(float(hs[-1]), 1e-6) * math.sqrt(n_eff)))
        sig_h = math.radians(HORIZON_SIGMA_DEG)
        dis = G.angle_deg(up_gps, ez)
        diag.update(up_disagreement_deg=round(dis, 3), gps_up_sigma_deg=round(math.degrees(sig_g), 3))
        if dis <= MAX_UP_DISAGREEMENT_DEG:
            w_h, w_g = 1.0 / sig_h ** 2, 1.0 / sig_g ** 2
            up = G._normalize(w_h * ez + w_g * up_gps)
        else:
            warnings.append(f"GPS vertical disagrees with the image horizon by {dis:.1f} deg; kept the horizon.")
    r_lvl = G.rotation_between(up, ez)
    diag["levelling_correction_deg"] = round(G.angle_deg(up, ez), 3)

    # ---- 4-DOF fit on the levelled frame with iterative outlier rejection
    p = cv @ r_lvl.T
    inl = np.ones(len(p), bool)
    for _ in range(6):
        s, r2, t2 = _umeyama(p[inl, :2], ev[inl, :2])
        pred = s * p[:, :2] @ r2.T + t2
        res_h = np.linalg.norm(ev[:, :2] - pred, axis=1)
        thr = max(2.5 * G.GPS_NOISE_FLOOR_M, 3.0 * float(np.median(res_h[inl])))
        new = res_h < thr
        if new.sum() < 5 or np.array_equal(new, inl):
            break
        inl = new
    tz = float(np.median(ev[inl, 2] - s * p[inl, 2])) if has_alt else 0.0
    r3 = np.eye(3)
    r3[:2, :2] = r2
    r_full = r3 @ r_lvl
    t_full = np.array([t2[0], t2[1], tz])
    res3 = np.linalg.norm(ev - (s * cv @ r_full.T + t_full), axis=1)
    res_h = np.linalg.norm(ev[:, :2] - (s * cv @ r_full.T + t_full)[:, :2], axis=1)
    rmse_h = float(np.sqrt(np.mean(res_h[inl] ** 2)))
    hs = np.linalg.svd(ev[inl, :2] - ev[inl, :2].mean(0), compute_uv=False) / math.sqrt(max(1, inl.sum()))
    spread = float(np.sqrt((hs ** 2).sum()))
    n_eff = min(int(inl.sum()), 10)
    scale_unc = rmse_h / max(spread * math.sqrt(n_eff), 1e-9)
    # pairwise cross-check (independent of the fit)
    i, j = rng.integers(0, len(p), (2, 4000))
    dp = np.linalg.norm(ev[i, :2] - ev[j, :2], axis=1)
    dc = np.linalg.norm(p[i, :2] - p[j, :2], axis=1)
    far = (dp > max(8.0, 0.3 * float(dp.max()))) & (dc > 0)
    if far.sum() >= 10:
        s_pair = float(np.median(dp[far] / dc[far]))
        diag["scale_pairwise_crosscheck"] = round(s_pair, 6)
        if abs(s_pair / s - 1.0) > 0.03:
            warnings.append(f"Pairwise scale differs from the fit by {100 * abs(s_pair / s - 1):.1f}%.")
    if spread < 3.5 * G.GPS_NOISE_FLOOR_M:
        warnings.append(f"GPS track spread only {spread:.1f} m: scale is weakly constrained.")
    conf = "HIGH" if (scale_unc <= 0.01 and rmse_h <= 3.0) else ("MEDIUM" if scale_unc <= 0.05 else "LOW")
    diag.update(gps_rmse_horizontal_m=round(rmse_h, 3), gps_rmse_3d_m=round(float(np.sqrt(np.mean(res3[inl] ** 2))), 3),
                gps_inliers=int(inl.sum()), gps_matched=int(len(p)), track_spread_m=round(spread, 2))
    t_zup = np.eye(4)
    t_zup[:3, :3] = s * r_full
    t_zup[:3, 3] = t_full
    return dict(T_zup=t_zup, s=float(s), r=r_full, origin=best["origin"], swap=best["swap"], has_alt=has_alt,
                rmse=rmse_h, scale_rel_uncertainty=float(scale_unc), confidence=conf, diagnostics=diag,
                warnings=warnings, gps_alt_median=(float(np.nanmedian(best["alt"][valid])) if has_alt else None))


# -----------------------------------------------------------------------------
# applying the transform to a bundle on disk
# -----------------------------------------------------------------------------
def _transform_ply(path, t4, out_path=None):
    d = plyio.read_ply(path)
    xyz = d.xyz @ t4[:3, :3].T + t4[:3, 3]
    nrm = None
    if d.normals is not None and len(d.normals) == len(xyz):
        a = t4[:3, :3] / np.cbrt(np.linalg.det(t4[:3, :3]))
        nrm = d.normals.astype(np.float64) @ a.T
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    plyio.write_ply(out_path or path, xyz, d.rgb, nrm, d.faces)
    return xyz, d


def recalibrate_active_model(data_dir=None, srt_path=None, log=print, force=False, backup=True, rtk_path=None):
    """
    Re-georeferences the deployed model in data/models using data/workspace/frame_index.json and the
    flight log. Returns a summary dict. Raises ValueError when the inputs are insufficient.
    """
    data_dir = data_dir or DATA_DIR
    models = os.path.join(data_dir, "models")
    fi_path = os.path.join(data_dir, "workspace", "frame_index.json")
    telem_path = os.path.join(data_dir, "flight_telemetry.json")
    if not os.path.exists(fi_path):
        raise ValueError("No frame_index.json with camera poses (needs a PRISM-Turbo bundle).")
    fi, cams = load_frame_cameras(fi_path)
    if len(cams) < 5:
        raise ValueError("frame_index.json has fewer than 5 registered camera poses.")
    old = {}
    if os.path.exists(telem_path):
        with open(telem_path, "r", encoding="utf-8") as f:
            old = json.load(f)
    geo_old = old.get("georeference") or {}
    if not force and geo_old.get("source") == "bundle_recalibration" and old.get("latlon_verified"):
        return {"status": "skipped", "reason": "already recalibrated"}

    # ---- flight log
    srt_path = srt_path or next((p for p in (os.path.join(data_dir, "drone_flight.srt"),
                                             os.path.join(data_dir, "raw_videos", "drone_flight.srt"))
                                 if os.path.exists(p)), None)
    recs, meta = [], {"latlon_ambiguous": True, "latlon_order": "unknown", "home": None}
    if srt_path and srt_matches_telemetry(srt_path, old) is False:
        log(f"[Recalibrate] {os.path.basename(srt_path)} belongs to a different flight: using this mission's own telemetry")
        srt_path = None
    if srt_path:
        recs, meta = parse_srt_records(srt_path)
    elif old.get("waypoints") and all(w.get("time_s") is not None for w in old["waypoints"]):
        recs = [{"time_s": w["time_s"], "lat": w["latitude"], "lon": w["longitude"],
                 "alt": w.get("relative_altitude_m"), "alt_kind": "relative" if w.get("relative_altitude_m") is not None else None}
                for w in old["waypoints"] if w.get("latitude") is not None]
        meta = {"latlon_ambiguous": not old.get("latlon_verified", False), "latlon_order": old.get("latlon_order", "lat_lon"),
                "home": old.get("home")}
    if len(recs) < 5:        # fall back to per-keyframe GPS stored in the frame index
        recs = [{"time_s": e["timestamp_sec"], "lat": e["latitude"], "lon": e["longitude"],
                 "alt": e.get("relative_altitude_m", e.get("absolute_altitude_m")), "alt_kind": "gps"}
                for e in cams if e.get("latitude") is not None and e.get("longitude") is not None]
        meta = {"latlon_ambiguous": True, "latlon_order": "lat_lon", "home": None}
    if len(recs) < 5:
        raise ValueError("No usable GPS in the flight log.")
    duration = (old.get("video") or {}).get("duration_sec")
    rtk_summary = None
    if rtk_path:
        # RTK / PPK positions replace the logged GNSS at the same instants (clock aligned by track overlay)
        try:
            import sensors
        except ImportError:
            from backend import sensors
        rtk = sensors.parse_rtk_file(rtk_path)
        clock = sensors.srt_clock_times(srt_path)
        timed = [r for r in recs if r.get("time_s") is not None]
        if len(timed) < 5:
            raise ValueError("The flight log has no timestamps to match with the RTK/PPK file.")
        tries = [timed] + ([[dict(r, lat=r["lon"], lon=r["lat"]) for r in timed]] if meta.get("latlon_ambiguous", True) else [])
        err = None
        for cand in tries:
            try:
                recs, rtk_summary = sensors.apply_rtk(cand, rtk, clock if (clock and len(clock) == len(cand)) else None)
                break
            except ValueError as e:
                err = e
        if rtk_summary is None:
            raise err
        meta = dict(meta, latlon_ambiguous=False, latlon_order="rtk")
        log(f"[Recalibrate] RTK/PPK: {rtk_summary['position_source']} ({rtk_summary['fix_pct']} % fixed), "
            f"clock offset {rtk_summary['clock_offset_s']:.2f} s, tracks agree to {rtk_summary['track_match_median_m']} m")

    times = np.array([float(e["timestamp_sec"]) for e in cams])
    cam_pos = np.array([e["position"] for e in cams], float)
    fit = fit_model_to_gps(cam_pos, times, recs, meta.get("latlon_ambiguous", True), duration, log=log)

    # ---- ground datum from the terrain
    pts_path = os.path.join(models, "actionable_threat_map_points.ply")
    if not os.path.exists(pts_path):
        pts_path = os.path.join(models, "actionable_threat_map.ply")
    if not os.path.exists(pts_path):
        raise ValueError("No deployed point cloud to recalibrate.")
    pts = plyio.read_ply(pts_path).xyz
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 400000), replace=False)]
    zup = (sub @ G.ENU_FROM_YUP.T) @ fit["T_zup"][:3, :3].T + fit["T_zup"][:3, 3]
    ground = G.estimate_ground_height(zup[:, 2])
    t_zup = fit["T_zup"].copy()
    t_zup[2, 3] -= ground
    origin = G.enu_to_geodetic(np.array([0.0, 0.0, ground]), fit["origin"])
    origin = {"lat": float(origin[0]), "lon": float(origin[1]), "alt": float(origin[2])}
    t4 = _h4(G.YUP_FROM_ENU) @ t_zup @ _h4(G.ENU_FROM_YUP)       # old Y-up frame -> new Y-up frame
    rot = t4[:3, :3] / fit["s"]

    # ---- move originals aside, write transformed geometry
    bdir = os.path.join(models, "_pre_recalibration")
    files = ["actionable_threat_map_points.ply", "actionable_threat_map.ply", "actionable_threat_map_cloud.ply",
             "actionable_threat_mesh.ply", "actionable_threat_mesh.glb", "actionable_threat_mesh.obj",
             "preview_topdown.jpg"]
    if backup:
        os.makedirs(bdir, exist_ok=True)
        for fn in files:
            src = os.path.join(models, fn)
            if os.path.exists(src) and not os.path.exists(os.path.join(bdir, fn)):
                shutil.copy2(src, os.path.join(bdir, fn)) if fn.endswith((".json", ".jpg")) else os.replace(src, os.path.join(bdir, fn))
        for extra in (fi_path, telem_path):
            if os.path.exists(extra):
                dst = os.path.join(bdir, os.path.basename(extra))
                if not os.path.exists(dst):
                    shutil.copy2(extra, dst)

    def src_of(fn):
        live = os.path.join(models, fn)
        if os.path.exists(live):
            return live
        b = os.path.join(bdir, fn)
        return b if os.path.exists(b) else None

    t0 = time.time()
    written = []
    new_pts = None
    for fn in ("actionable_threat_map_points.ply", "actionable_threat_map.ply", "actionable_threat_map_cloud.ply"):
        src = src_of(fn)
        if src:
            xyz, d = _transform_ply(src, t4, os.path.join(models, fn))
            written.append(fn)
            if fn == "actionable_threat_map_points.ply":
                new_pts = (xyz, d.rgb)
    mesh_src = src_of("actionable_threat_mesh.ply")
    if mesh_src:
        mxyz, md = _transform_ply(mesh_src, t4, os.path.join(models, "actionable_threat_mesh.ply"))
        written.append("actionable_threat_mesh.ply")
        if md.faces is not None:
            plyio.write_obj(os.path.join(models, "actionable_threat_mesh.obj"), mxyz, md.rgb, md.faces)
            written.append("actionable_threat_mesh.obj")
    glb_src = src_of("actionable_threat_mesh.glb")
    if glb_src:
        plyio.transform_glb(glb_src, os.path.join(models, "actionable_threat_mesh.glb"), t4)
        written.append("actionable_threat_mesh.glb")
    if new_pts is not None:
        try:
            render_topdown_preview(new_pts[0], new_pts[1], os.path.join(models, "preview_topdown.jpg"))
            shutil.copy2(os.path.join(models, "preview_topdown.jpg"), os.path.join(data_dir, "preview_topdown.jpg"))
        except Exception as e:  # preview is cosmetic
            log(f"preview skipped: {e}")

    # ---- camera poses
    entries = fi if isinstance(fi, list) else fi.get("frames")
    lat_k, lon_k, alt_k, _ = _gps_at(np.array([float(e.get("timestamp_sec", 0.0)) for e in entries]),
                                     recs, fit["swap"], duration)
    traj = []
    for k, e in enumerate(entries):
        if e.get("position") is not None:
            pos = np.asarray(e["position"], float)
            e["position"] = [round(float(v), 4) for v in (t4[:3, :3] @ pos + t4[:3, 3])]
            traj.append(e["position"])
        if e.get("quaternion_xyzw") is not None:
            e["quaternion_xyzw"] = [round(float(v), 6) for v in _mat_to_quat(rot @ _quat_to_mat(e["quaternion_xyzw"]))]
        e["latitude"], e["longitude"] = round(float(lat_k[k]), 8), round(float(lon_k[k]), 8)
        if alt_k is not None:
            e["relative_altitude_m"] = round(float(alt_k[k]), 3)
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(fi, f, indent=1)

    # ---- telemetry in the backend format (+ the bundle's own keys)
    t_log, lat_l, lon_l, alt_l = _log_arrays(recs, fit["swap"], duration)
    waypoints = []
    for i in range(len(recs)):
        wp = {"frame_id": i, "latitude": float(lat_l[i]), "longitude": float(lon_l[i]), "time_s": round(float(t_log[i]), 4)}
        if np.isfinite(alt_l[i]):
            wp["relative_altitude_m"] = round(float(alt_l[i]), 3)
        waypoints.append(wp)
    cams_new = np.array([e["position"] for e in cams], float)
    cam_h = float(np.median(cams_new[:, 1]))
    geo = {
        "status": "ok", "version": 3, "source": "bundle_recalibration", "metric": True,
        "mode": "GPS_SIM3" if fit["has_alt"] else "GPS_4DOF", "confidence": fit["confidence"],
        "heading_known": True, "latlon_swapped": bool(fit["swap"]), "origin": origin,
        "scale_m_per_unit": fit["s"], "scale_rel_uncertainty": fit["scale_rel_uncertainty"],
        "display_frame": "METRIC_YUP", "transform_prev_to_metric_yup": t4.tolist(),
        "diagnostics": dict(fit["diagnostics"], median_camera_height_above_datum_m=round(cam_h, 2),
                            logged_altitude_median_m=(round(fit["gps_alt_median"], 2)
                                                      if fit["gps_alt_median"] is not None else None)),
        "warnings": fit["warnings"], "recalibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    order = meta.get("latlon_order", "lat_lon")
    if fit["swap"]:
        order = {"lat_lon": "lon_lat", "lon_lat": "lat_lon"}.get(order, order)
    lat_c, lon_c = float(np.median(lat_l)), float(np.median(lon_l))
    telem = dict(old)
    for stale in ("fixes", "georeferenced", "scale_source", "alignment", "ground_level_offset_m",
                  "assumed_altitude_m", "center_latitude", "center_longitude"):
        telem.pop(stale, None)
    telem.update({
        "source": "DJI_SRT" if srt_path else "FRAME_INDEX_GPS", "is_synthetic": False, "has_gps": True,
        "altitude_reference": "relative" if fit["has_alt"] else "estimated",
        "altitude_is_estimated": not fit["has_alt"],
        "latlon_order": order, "latlon_ambiguous": False, "latlon_verified": bool(fit["diagnostics"]["latlon_decided"]),
        "home": meta.get("home"), "waypoint_count": len(waypoints), "has_timestamps": True,
        "total_distance_meters": round(_path_length(waypoints), 2), "waypoints": waypoints,
        "center_latitude": round(lat_c, 7), "center_longitude": round(lon_c, 7),
        "metric_scale_factor": 1.0, "georeference": geo, "georeferenced": True, "scale_source": "rtk" if rtk_summary else "gps",
        "camera_trajectory": traj,
        "coordinate_frame": {"units": "metres", "up_axis": "+Y", "x_axis": "East", "z_axis": "South (-North)",
                             "origin": "GPS origin projected to the ground datum (y = 0)",
                             "handedness": "right-handed (three.js)"},
    })
    if rtk_summary:
        telem["position_source"] = rtk_summary["position_source"]
        telem["rtk_summary"] = rtk_summary
        for wp, r in zip(waypoints, recs):
            if r.get("abs_alt") is not None:
                wp["absolute_altitude_m"] = round(float(r["abs_alt"]), 3)
    with open(telem_path, "w", encoding="utf-8") as f:
        json.dump(telem, f, indent=1)
    for stale in ("road_pothole_audit.json",):
        p = os.path.join(data_dir, stale)
        if os.path.exists(p):
            os.remove(p)
    summary = {"status": "ok", "scale_correction": round(fit["s"], 5), "latlon_swapped": bool(fit["swap"]),
               "gps_rmse_m": round(fit["rmse"], 3), "confidence": fit["confidence"],
               "scale_rel_uncertainty": round(fit["scale_rel_uncertainty"], 5), "ground_datum_shift_m": round(float(ground), 3),
               "camera_height_above_ground_m": round(cam_h, 2), "files": written,
               "seconds": round(time.time() - t0, 1), "diagnostics": geo["diagnostics"], "warnings": fit["warnings"],
               "rtk": rtk_summary}
    log(f"[Recalibrate] scale x{fit['s']:.4f}, GPS RMSE {fit['rmse']:.2f} m, "
        f"lat/lon {'swapped' if fit['swap'] else 'as logged'}, confidence {fit['confidence']}")
    return summary


def render_topdown_preview(xyz, rgb, path, size=None):
    """North-up orthographic preview (x = East, z = South) with a z-buffer."""
    import cv2
    xyz = np.asarray(xyz)
    if len(xyz) == 0:
        return None
    if rgb is None:
        rgb = np.full((len(xyz), 3), 180, np.uint8)
    if size is None:
        size = int(np.clip(np.sqrt(len(xyz)) * 1.1, 600, 1600))
    x, z, y = xyz[:, 0], xyz[:, 2], xyz[:, 1]
    lo = np.percentile(np.stack([x, z], 1), 0.5, 0)
    hi = np.percentile(np.stack([x, z], 1), 99.5, 0)
    sc = (size - 1) / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    w = int((hi[0] - lo[0]) * sc) + 1
    h = int((hi[1] - lo[1]) * sc) + 1
    u = ((x - lo[0]) * sc).astype(np.int64)
    v = ((z - lo[1]) * sc).astype(np.int64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, y, c = u[ok], v[ok], y[ok], rgb[ok]
    order = np.argsort(y)
    img = np.full((h, w, 3), 24, np.uint8)
    img[v[order], u[order]] = c[order][:, ::-1]
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


if __name__ == "__main__":
    import sys
    print(json.dumps(recalibrate_active_model(force="--force" in sys.argv), indent=1))
