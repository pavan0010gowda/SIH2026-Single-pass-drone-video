"""
PRISM // optional sensor inputs (sensors.py)

The problem statement lists IMU data, barometric altitude, camera intrinsics and RTK / PPK corrections
as optional inputs. This module reads them in the formats that field teams actually have:

  RTK / PPK      RTKLIB .pos (llh or ECEF, GPST or UTC), Emlid / generic CSV (time, lat, lon, height, Q)
                 -> aligned to the video clock by matching the RTK track to the video's GPS track
                    (time-zone / leap-second / unknown-start offsets are searched, then refined to 20 ms)
  intrinsics     OpenCV YAML / JSON (camera_matrix + distortion), COLMAP cameras.txt, plain JSON
                 {fx, fy, cx, cy, k1, k2, p1, p2, width, height}
  IMU / attitude flight-record CSV (AirData, DJI, Litchi style: pitch / roll / yaw, gimbal pitch / yaw,
                 barometric height) -> per-fix attitude, gimbal angles, recording window
"""
import csv
import io
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()
LEAP_S = 18.0                                        # GPST - UTC since 2017


def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# =============================================================================
# RTK / PPK
# =============================================================================
_DT_RE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})[ T]+(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)")


def _dt_to_unix(s):
    m = _DT_RE.search(s)
    if not m:
        return None
    y, mo, d, h, mi, sec = m.groups()
    secf = float(sec)
    dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(secf), tzinfo=timezone.utc)
    return dt.timestamp() + (secf - int(secf))


def parse_rtk_file(path):
    """-> dict(t (s, on the file's own clock), lat, lon, h, q, sd, clock 'GPST'|'UTC'|'unknown', kind)"""
    text = _read_text(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pos" or text.lstrip().startswith("%"):
        return _parse_rtklib(text)
    return _parse_rtk_csv(text)


def _parse_rtklib(text):
    clock, ecef = "GPST", False
    t, lat, lon, h, q, sd = [], [], [], [], [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("%"):
            low = s.lower()
            if "utc" in low and "gpst" not in low:
                clock = "UTC"
            if "x-ecef" in low:
                ecef = True
            continue
        tok = s.split()
        try:
            if "/" in tok[0]:
                tt = _dt_to_unix(tok[0] + " " + tok[1])
                vals = tok[2:]
            else:                                    # GPS week + time of week
                tt = GPS_EPOCH + int(tok[0]) * 604800.0 + float(tok[1])
                vals = tok[2:]
            a, b, c = float(vals[0]), float(vals[1]), float(vals[2])
            qq = int(vals[3]) if len(vals) > 3 else 0
            sdd = [float(v) for v in vals[5:8]] if len(vals) >= 8 else [np.nan] * 3
        except (ValueError, IndexError):
            continue
        if tt is None:
            continue
        t.append(tt)
        if ecef:
            from georeference import ecef_to_geodetic
            la, lo, hh = ecef_to_geodetic(np.array([a, b, c]))
            lat.append(float(la))
            lon.append(float(lo))
            h.append(float(hh))
        else:
            lat.append(a)
            lon.append(b)
            h.append(c)
        q.append(qq)
        sd.append(sdd)
    if len(t) < 5:
        raise ValueError("No RTKLIB solution lines found in the .pos file.")
    return _pack(t, lat, lon, h, q, sd, clock, "PPK")


def _parse_rtk_csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if len(rows) < 6:
        raise ValueError("RTK file has too few rows.")
    head = [c.strip().lower() for c in rows[0]]

    def col(*names, avoid=()):
        for i, hname in enumerate(head):
            if any(n in hname for n in names) and not any(a in hname for a in avoid):
                return i
        return None
    it = col("gpst", "utc", "time", "timestamp", "date")
    ila = col("lat")
    ilo = col("lon", "lng")
    ih = col("ellips", "height", "alt", avoid=("sd", "std"))
    iq = col("q", "fix", "quality", "solution", "status", avoid=("sd", "std", "qual_"))
    if None in (it, ila, ilo):
        raise ValueError("RTK CSV needs time, latitude and longitude columns.")
    clock = "GPST" if "gpst" in head[it] else ("UTC" if "utc" in head[it] else "unknown")
    t, lat, lon, h, q = [], [], [], [], []
    for r in rows[1:]:
        try:
            ts = r[it].strip()
            if re.fullmatch(r"[-+]?\d+(\.\d+)?", ts):
                v = float(ts)
                tt = v / 1000.0 if v > 1e11 else v            # unix ms / s
            else:
                tt = _dt_to_unix(ts)
            if tt is None:
                continue
            t.append(tt)
            lat.append(float(r[ila]))
            lon.append(float(r[ilo]))
            h.append(float(r[ih]) if ih is not None and r[ih].strip() else np.nan)
            qv = r[iq].strip().lower() if iq is not None else ""
            q.append(1 if qv in ("1", "fix", "fixed", "rtk_fixed", "rtk fixed") else
                     2 if qv in ("2", "float", "rtk_float", "rtk float") else
                     (int(qv) if qv.isdigit() else 0))
        except (ValueError, IndexError):
            continue
    if len(t) < 5:
        raise ValueError("No usable rows in the RTK CSV.")
    return _pack(t, lat, lon, h, q, [[np.nan] * 3] * len(t), clock, "RTK")


def _pack(t, lat, lon, h, q, sd, clock, kind):
    o = np.argsort(t)
    return {"t": np.asarray(t, float)[o], "lat": np.asarray(lat, float)[o], "lon": np.asarray(lon, float)[o],
            "h": np.asarray(h, float)[o], "q": np.asarray(q, int)[o], "sd": np.asarray(sd, float)[o],
            "clock": clock, "kind": kind}


def srt_clock_times(srt_path):
    """Absolute wall-clock time of every GPS cue of a DJI SRT (camera clock, time zone unknown) or None."""
    if not srt_path or not os.path.exists(srt_path):
        return None
    text = _read_text(srt_path).replace("\r\n", "\n")
    out = []
    for block in re.split(r"\n\s*\n", text):
        if not re.search(r"GPS|lat", block, re.I):
            continue
        out.append(_dt_to_unix(block))
    if not out or sum(v is not None for v in out) < 0.8 * len(out):
        return None
    return out


def align_rtk(video_t, lat, lon, rtk, video_abs=None):
    """
    Finds the offset `tau` with rtk_time = video_clock + tau that best overlays the RTK track on the
    video GPS track. With camera wall-clock times, candidates are whole quarter-hours (time zones)
    +/- the GPS leap seconds; without them the whole RTK time span is searched. Refined to 20 ms.
    Returns (tau, median distance m, base clock array).
    """
    video_t = np.asarray(video_t, float)
    base = np.asarray(video_abs, float) if video_abs is not None else video_t
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    sel = np.linspace(0, len(base) - 1, min(len(base), 300)).astype(int)
    bt, bla, blo = base[sel], lat[sel], lon[sel]
    rt = rtk["t"]

    def cost(tau):
        tt = bt + tau
        ok = (tt >= rt[0]) & (tt <= rt[-1])
        if ok.sum() < 0.6 * len(tt):
            return np.inf
        la = np.interp(tt[ok], rt, rtk["lat"])
        lo = np.interp(tt[ok], rt, rtk["lon"])
        return float(np.median(_hav(bla[ok], blo[ok], la, lo)))

    tau, best = None, np.inf
    if video_abs is not None:
        cands = [k * 900.0 + d for k in range(-56, 57) for d in (0.0, LEAP_S, -LEAP_S)]
        costs = [cost(c) for c in cands]
        k = int(np.argmin(costs))
        tau, best = cands[k], costs[k]
    if not np.isfinite(best) or best > 12.0:
        # camera clock unset / drifting: search the whole RTK span on the video's own time axis
        base = video_t
        bt = base[sel]
        cands = list(np.arange(rt[0] - bt[0], rt[-1] - bt[-1] + 0.5, 0.5))
        if not cands:
            return 0.0, np.inf, base
        costs = [cost(c) for c in cands]
        k = int(np.argmin(costs))
        tau = cands[k]
    for step, span in ((0.25, 4.0), (0.02, 0.5)):
        grid = np.arange(tau - span, tau + span + 1e-9, step)
        cs = [cost(g) for g in grid]
        tau = float(grid[int(np.argmin(cs))])
    return tau, cost(tau), base


def apply_rtk(recs, rtk, video_abs=None, max_median_m=12.0):
    """
    Replaces the positions of parsed SRT records [{time_s, lat, lon, alt, ...}] by RTK / PPK positions
    at the same instants. Returns (new_recs, summary). Raises ValueError when the tracks do not match.
    """
    t = np.array([r["time_s"] for r in recs], float)
    lat = np.array([r["lat"] for r in recs], float)
    lon = np.array([r["lon"] for r in recs], float)
    tau, med, base = align_rtk(t, lat, lon, rtk, video_abs)
    if not np.isfinite(med) or med > max_median_m:
        raise ValueError(f"The RTK/PPK track does not overlap the video's GPS track (best match {med:.1f} m apart). "
                         "Check that the file belongs to this flight.")
    tt = base + tau
    ok = (tt >= rtk["t"][0]) & (tt <= rtk["t"][-1])
    la = np.interp(tt, rtk["t"], rtk["lat"])
    lo = np.interp(tt, rtk["t"], rtk["lon"])
    h = np.interp(tt, rtk["t"], rtk["h"]) if np.isfinite(rtk["h"]).any() else np.full(len(tt), np.nan)
    # quality of the nearest RTK epoch
    idx = np.clip(np.searchsorted(rtk["t"], tt), 0, len(rtk["t"]) - 1)
    q = rtk["q"][idx]
    h0 = float(np.nanmedian(h[:max(3, len(h) // 50)])) if np.isfinite(h).any() else None
    out = []
    for i, r in enumerate(recs):
        n = dict(r)
        if ok[i]:
            n["lat"], n["lon"] = float(la[i]), float(lo[i])
            if np.isfinite(h[i]):
                n["abs_alt"] = float(h[i])
                if r.get("alt") is None or r.get("alt_kind") != "relative":
                    n["alt"], n["alt_kind"] = float(h[i] - h0), "relative"
            n["rtk_q"] = int(q[i])
        out.append(n)
    qs = q[ok]
    fix = float(np.mean(qs == 1)) if len(qs) else 0.0
    flt = float(np.mean(qs == 2)) if len(qs) else 0.0
    kind = rtk["kind"]
    if fix >= 0.9:
        source = f"{kind}_FIXED"
    elif fix + flt >= 0.9:
        source = f"{kind}_FLOAT"
    else:
        source = "GNSS"
    shift = _hav(lat[ok], lon[ok], la[ok], lo[ok]) if ok.any() else np.array([0.0])
    summary = {"kind": kind, "position_source": source, "epochs": int(len(rtk["t"])), "matched_fixes": int(ok.sum()),
               "fix_pct": round(100 * fix, 1), "float_pct": round(100 * flt, 1),
               "clock_offset_s": round(tau, 3), "clock": rtk["clock"],
               "track_match_median_m": round(med, 2), "mean_shift_from_video_gps_m": round(float(np.mean(shift)), 2),
               "sd_horizontal_median_m": (round(float(np.nanmedian(np.hypot(rtk["sd"][:, 0], rtk["sd"][:, 1]))), 3)
                                          if np.isfinite(rtk["sd"]).any() else None)}
    return out, summary


def rtk_waypoints(telem, rtk, srt_path=None):
    """Applies RTK / PPK positions to a backend telemetry dict (waypoints with time_s)."""
    wps = telem.get("waypoints") or []
    if len(wps) < 5 or not all("time_s" in w for w in wps):
        raise ValueError("The flight log needs timestamps to be matched with RTK/PPK positions.")
    recs = [{"time_s": w["time_s"], "lat": w["latitude"], "lon": w["longitude"], "alt": w.get("relative_altitude_m"),
             "alt_kind": "relative" if w.get("relative_altitude_m") is not None else None} for w in wps]
    clock = srt_clock_times(srt_path)
    video_abs = clock if (clock and len(clock) == len(recs)) else None
    new, summ = apply_rtk(recs, rtk, video_abs)
    for w, r in zip(wps, new):
        w["latitude"], w["longitude"] = r["lat"], r["lon"]
        if r.get("abs_alt") is not None:
            w["absolute_altitude_m"] = round(r["abs_alt"], 3)
        if r.get("alt") is not None:
            w["relative_altitude_m"] = round(r["alt"], 3)
        if "rtk_q" in r:
            w["rtk_q"] = r["rtk_q"]
    telem["waypoints"] = wps
    telem["position_source"] = summ["position_source"]
    telem["rtk_summary"] = summ
    telem["latlon_ambiguous"] = False
    telem["latlon_order"] = "rtk"
    return telem, summ


# =============================================================================
# camera intrinsics
# =============================================================================
def parse_intrinsics(path):
    """-> dict(model 'OPENCV'|'PINHOLE', fx, fy, cx, cy, k1, k2, p1, p2, width, height, source)"""
    text = _read_text(path)
    ext = os.path.splitext(path)[1].lower()
    out = None
    if ext == ".txt" or re.search(r"^\s*\d+\s+(OPENCV|PINHOLE|SIMPLE_RADIAL|RADIAL|SIMPLE_PINHOLE)\s", text, re.M):
        for line in text.splitlines():
            tok = line.split()
            if len(tok) >= 5 and not line.strip().startswith("#") and tok[1].isupper():
                model, w, h = tok[1], int(tok[2]), int(tok[3])
                p = [float(v) for v in tok[4:]]
                if model == "OPENCV" and len(p) >= 8:
                    out = dict(fx=p[0], fy=p[1], cx=p[2], cy=p[3], k1=p[4], k2=p[5], p1=p[6], p2=p[7])
                elif model == "PINHOLE":
                    out = dict(fx=p[0], fy=p[1], cx=p[2], cy=p[3])
                elif model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE"):
                    out = dict(fx=p[0], fy=p[0], cx=p[1], cy=p[2], k1=p[3] if len(p) > 3 else 0.0,
                               k2=p[4] if (model == "RADIAL" and len(p) > 4) else 0.0)
                if out:
                    out.update(width=w, height=h, source=f"COLMAP cameras.txt ({model})")
                    break
    if out is None and ext in (".yaml", ".yml", ".xml") or (out is None and "camera_matrix" in text):
        def arr(name):
            m = re.search(name + r"[\s\S]*?data\s*:\s*\[([^\]]*)\]", text)
            return [float(v) for v in re.split(r"[,\s]+", m.group(1).strip()) if v] if m else None
        K = arr("camera_matrix") or arr("K")
        D = arr("distortion_coefficients") or arr("D") or []
        wm = re.search(r"image_width\s*:\s*(\d+)", text)
        hm = re.search(r"image_height\s*:\s*(\d+)", text)
        if K and len(K) >= 9:
            D = (D + [0.0] * 5)[:5]
            out = dict(fx=K[0], fy=K[4], cx=K[2], cy=K[5], k1=D[0], k2=D[1], p1=D[2], p2=D[3],
                       width=int(wm.group(1)) if wm else None, height=int(hm.group(1)) if hm else None,
                       source="OpenCV calibration (YAML)")
    if out is None:
        try:
            j = json.loads(text)
        except json.JSONDecodeError:
            j = None
        if isinstance(j, dict):
            K = j.get("camera_matrix") or j.get("K")
            if isinstance(K, dict):
                K = K.get("data")
            if K is not None:
                K = np.asarray(K, float).ravel()
                D = list(np.asarray(j.get("distortion_coefficients") or j.get("dist") or j.get("D") or [], float).ravel())
                D = (D + [0.0] * 5)[:5]
                out = dict(fx=K[0], fy=K[4], cx=K[2], cy=K[5], k1=D[0], k2=D[1], p1=D[2], p2=D[3])
            elif "fx" in j:
                out = {k: float(j.get(k, 0.0)) for k in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")}
                out["fy"] = out["fy"] or out["fx"]
            if out is not None:
                out.update(width=j.get("width") or j.get("image_width"), height=j.get("height") or j.get("image_height"),
                           source="JSON calibration")
    if out is None or not (out.get("fx", 0) > 0 and out.get("cx", 0) > 0):
        raise ValueError("Could not read camera intrinsics (expected OpenCV YAML/JSON, COLMAP cameras.txt or "
                         "{fx, fy, cx, cy, k1, k2, p1, p2, width, height}).")
    for k in ("k1", "k2", "p1", "p2"):
        out.setdefault(k, 0.0)
    out["model"] = "OPENCV" if any(abs(out[k]) > 0 for k in ("k1", "k2", "p1", "p2")) else "PINHOLE"
    if not out.get("width") or not out.get("height"):
        out["width"], out["height"] = int(round(2 * out["cx"])), int(round(2 * out["cy"]))
    return out


def intrinsics_for_size(K, width, height):
    """Scales a calibration to the working image size (keyframes are resized). Aspect must match."""
    sx, sy = width / float(K["width"]), height / float(K["height"])
    if abs(sx - sy) > 0.01 * max(sx, sy):
        raise ValueError(f"Calibration is for {K['width']}x{K['height']}, the video is {width}x{height} "
                         "(different aspect ratio: cropped modes need their own calibration).")
    fx, fy, cx, cy = K["fx"] * sx, K["fy"] * sy, K["cx"] * sx, K["cy"] * sy
    if K["model"] == "OPENCV":
        return "OPENCV", [fx, fy, cx, cy, K["k1"], K["k2"], K["p1"], K["p2"]]
    return "PINHOLE", [fx, fy, cx, cy]


# =============================================================================
# IMU / flight-record CSV (attitude, gimbal, barometer)
# =============================================================================
def parse_flight_csv(path):
    """
    AirData / DJI / Litchi-style flight record -> list of dicts with time_s (from the start of video
    recording when the log says when recording started), lat, lon, rel_alt, pitch, roll, yaw,
    gimbal_pitch, gimbal_yaw. Units: feet columns are converted to metres.
    """
    text = _read_text(path)
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r]
    head = [c.strip().lower() for c in rows[0]]

    def col(*pats, avoid=()):
        for i, h in enumerate(head):
            if any(re.search(p, h) for p in pats) and not any(a in h for a in avoid):
                return i
        return None
    c = {
        "t": col(r"time\(millisecond\)", r"^time_?ms", r"offsettime", r"^time\b", r"flytime", r"updatetime"),
        "lat": col(r"latitude", r"^lat"), "lon": col(r"longitude", r"^lon"),
        "rel": col(r"height_above_takeoff", r"osd\.height", r"relative", r"^height", r"altitude\(m\)"),
        "pitch": col(r"^pitch", r"osd\.pitch", r"aircraft.?pitch"), "roll": col(r"^roll", r"osd\.roll"),
        "yaw": col(r"compass_heading", r"^yaw", r"osd\.yaw"),
        "gpitch": col(r"gimbal_pitch", r"gimbal\.pitch", r"gimbalpitch"),
        "gyaw": col(r"gimbal_heading", r"gimbal\.yaw", r"gimbalyaw"),
        "rec": col(r"isvideo", r"is_video", r"recording"),
    }
    if c["lat"] is None or c["lon"] is None:
        raise ValueError("Flight log CSV needs latitude/longitude columns.")
    feet = c["rel"] is not None and ("feet" in head[c["rel"]] or "[ft]" in head[c["rel"]])
    ms = c["t"] is not None and ("millisecond" in head[c["t"]] or head[c["t"]].endswith("ms"))
    out = []
    for k, r in enumerate(rows[1:]):
        try:
            d = {"lat": float(r[c["lat"]]), "lon": float(r[c["lon"]])}
        except (ValueError, IndexError):
            continue
        if abs(d["lat"]) < 1e-9 and abs(d["lon"]) < 1e-9:
            continue
        tv = None
        if c["t"] is not None:
            try:
                tv = float(r[c["t"]]) / (1000.0 if ms else 1.0)
            except ValueError:
                tv = _dt_to_unix(r[c["t"]])
        d["t"] = tv if tv is not None else float(k) * 0.1
        for key, idx in (("rel", c["rel"]), ("pitch", c["pitch"]), ("roll", c["roll"]), ("yaw", c["yaw"]),
                         ("gimbal_pitch", c["gpitch"]), ("gimbal_yaw", c["gyaw"]), ("rec", c["rec"])):
            if idx is not None:
                try:
                    d[key] = float(r[idx])
                except (ValueError, IndexError):
                    pass
        if feet and "rel" in d:
            d["rel"] *= 0.3048
        out.append(d)
    if len(out) < 5:
        raise ValueError("Flight log CSV has too few valid rows.")
    rec = [d for d in out if d.get("rec", 0) >= 1]
    t0 = rec[0]["t"] if rec else out[0]["t"]
    for d in out:
        d["time_s"] = d["t"] - t0
    return out, {"recording_start_found": bool(rec), "rows": len(out),
                 "has_attitude": any("pitch" in d for d in out), "has_gimbal": any("gimbal_pitch" in d for d in out)}


def camera_tilt_from_frames(frames):
    """Median camera pitch (deg below the horizon) and its spread from the reconstructed camera poses."""
    pitches = []
    for f in frames or []:
        q = f.get("quaternion_xyzw") or f.get("q")
        if not q:
            continue
        x, y, z, w = q
        # three.js camera looks along -Z: optical axis = R @ (0, 0, -1)
        ay = -(2 * (y * z - x * w))
        pitches.append(math.degrees(math.asin(max(-1.0, min(1.0, ay)))))
    if not pitches:
        return None
    p = np.array(pitches)
    return {"median_pitch_deg": round(float(np.median(p)), 2), "p10_deg": round(float(np.percentile(p, 10)), 2),
            "p90_deg": round(float(np.percentile(p, 90)), 2),
            "view": "nadir" if np.median(p) < -75 else ("oblique" if np.median(p) < -15 else "low oblique / horizon")}
