"""
PRISM // Flight-log ingestion (telemetry_parser.py, v3)

Supported logs
  * DJI subtitle tracks (.SRT / .txt), every firmware family we know of:
      - Mini / Air / Mavic 3:  "[latitude: 45.31] [longitude: 29.29] [rel_alt: 30.1 abs_alt: 42.3]"
      - Phantom / Mavic Pro:   "HOME(lon,lat) ... GPS(lon,lat,sats) BAROMETER:30.4"
      - Tool / converter style:"HOME(lat,lon)  GPS(lat,lon,alt)"
      - "longtitude" firmware typo, "H 19.9m" height tags, gimbal yaw/pitch, focal length
  * Mid-Air style CSV (GPS columns, or NED position columns)
  * PRISM / generic JSON with a "waypoints" list

v3 fixes (all of them changed heights):
  * GPS(a, b, c): the third value is now read as the flight altitude when it is a real
    altitude (decimal value). v2 dropped it, so every flight silently used a guessed altitude.
  * GPS(a, b, ...) is ambiguous: DJI's legacy firmware writes (lon, lat) but most converters
    write (lat, lon). When both values are inside +/-90 deg the order cannot be decided from
    the log alone. v2 always assumed (lon, lat), which mirrors the flight path and breaks the
    metric scale. v3 records the ambiguity (`latlon_ambiguous`) and the georeferencer decides
    with the camera trajectory from Structure-from-Motion (the wrong order cannot be fitted by
    a proper rotation). Real Danube-delta sortie: correct order RMSE 0.8 m, swapped 21 m.
  * HOME(...) is parsed and used as a consistency hint.
"""

import os
import re
import json
import math
import csv

import numpy as np

try:
    from georeference import enu_to_geodetic, haversine_distance
except ImportError:  # imported as backend.telemetry_parser
    from backend.georeference import enu_to_geodetic, haversine_distance

DEFAULT_BASE_LAT = 12.971600
DEFAULT_BASE_LON = 77.594600

_NUM = r"([-+]?\d+(?:\.\d+)?)"
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
_LAT_RE = re.compile(r"\blat(?:itude)?\s*[:=]\s*" + _NUM, re.I)
_LON_RE = re.compile(r"\blon(?:gt?itude|g)?\s*[:=]\s*" + _NUM, re.I)
_REL_RE = re.compile(r"\brel_alt\s*[:=]\s*" + _NUM, re.I)
_ABS_RE = re.compile(r"\babs_alt\s*[:=]\s*" + _NUM, re.I)
_ALT_RE = re.compile(r"(?<![_a-z])altitude\s*[:=]\s*" + _NUM, re.I)
_BARO_RE = re.compile(r"\bbarometer\s*[:=]\s*" + _NUM, re.I)
_H_RE = re.compile(r"(?<![\w.])H\s+" + _NUM + r"\s*m\b")
_GPS_RE = re.compile(r"\bGPS\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"(?:\s*,\s*" + _NUM + r")?\s*\)", re.I)
_HOME_RE = re.compile(r"\bHOME\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"(?:\s*,\s*" + _NUM + r")?\s*\)", re.I)
_GB_YAW_RE = re.compile(r"\bgb_yaw\s*[:=]\s*" + _NUM, re.I)
_GB_PITCH_RE = re.compile(r"\bgb_pitch\s*[:=]\s*" + _NUM, re.I)
_FOCAL_RE = re.compile(r"\bfocal_len\s*[:=]\s*" + _NUM, re.I)


def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw[:200].count(b"\x00") > 20:
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8-sig", errors="ignore")


def _write(flight_data, output_json_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as out_file:
        json.dump(flight_data, out_file, indent=4)


def _path_length(waypoints):
    """2-D path length over *fresh* fixes only (repeated GPS values do not add jitter)."""
    total, prev = 0.0, None
    for wp in waypoints:
        cur = (wp["latitude"], wp["longitude"])
        if prev is not None and cur != prev:
            total += haversine_distance(prev[0], prev[1], cur[0], cur[1])
        if prev is None or cur != prev:
            prev = cur
    return total


# =============================================================================
# SRT block parsing
# =============================================================================
def _parse_srt_block(block):
    """One subtitle cue -> raw record. GPS(...) pairs are kept raw until the order is decided."""
    rec = {}
    m = _TIME_RE.search(block)
    if m:
        h, mi, se, ms = m.groups()
        rec["time_s"] = int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0
    body = re.sub(r"<[^>]*>", " ", block[m.end():] if m else block)
    lat, lon = _LAT_RE.search(body), _LON_RE.search(body)
    if lat and lon:
        rec["lat"], rec["lon"] = float(lat.group(1)), float(lon.group(1))
    else:
        g = _GPS_RE.search(body)
        if g:
            rec["gps_a"], rec["gps_b"] = float(g.group(1)), float(g.group(2))
            if g.group(3) is not None:
                rec["gps_c"] = float(g.group(3))
                rec["gps_c_is_decimal"] = "." in g.group(3)
    hm = _HOME_RE.search(body)
    if hm:
        rec["home_a"], rec["home_b"] = float(hm.group(1)), float(hm.group(2))
    for key, rx in (("rel", _REL_RE), ("abs", _ABS_RE), ("alt", _ALT_RE), ("baro", _BARO_RE), ("h", _H_RE),
                    ("gb_yaw", _GB_YAW_RE), ("gb_pitch", _GB_PITCH_RE), ("focal", _FOCAL_RE)):
        mm = rx.search(body)
        if mm:
            rec[key] = float(mm.group(1))
    return rec


def _decide_gps_order(recs, text):
    """
    Order of the two numbers inside GPS(a, b, ...).
    Returns (order, ambiguous) with order in {"lat_lon", "lon_lat"}.
    Decided exactly when one value is outside +/-90 deg; otherwise a prior is used and the
    ambiguity is reported so the georeferencer can test both orders against the SfM cameras.
    """
    a = np.array([r["gps_a"] for r in recs])
    b = np.array([r["gps_b"] for r in recs])
    if np.any(np.abs(a) > 90.0):
        return "lon_lat", False
    if np.any(np.abs(b) > 90.0):
        return "lat_lon", False
    # Prior = ISO 6709 order (lat, lon). Every real log checked so far uses it, including the legacy DJI
    # Phantom format "HOME(57.98,-4.00) ... GPS(57.98,-4.00,18) BAROMETER:57.2" (Scotland; the swapped
    # reading would lie in the Indian Ocean). The order is still verified against the camera track.
    return "lat_lon", True


def parse_srt_records(srt_path):
    """
    Parses a subtitle log into clean records:
      [{time_s, lat, lon, alt (m or None), alt_kind, gb_pitch, gb_yaw, focal}], plus a meta dict.
    """
    text = _read_text(srt_path).replace("\r\n", "\n").replace("\r", "\n")
    raw = [_parse_srt_block(b) for b in re.split(r"\n\s*\n", text)]
    raw = [r for r in raw if ("lat" in r and "lon" in r) or ("gps_a" in r and "gps_b" in r)]
    meta = {"latlon_order": "explicit", "latlon_ambiguous": False, "home": None}
    gps_recs = [r for r in raw if "gps_a" in r and "lat" not in r]
    order, ambiguous = ("lat_lon", False)
    if gps_recs:
        order, ambiguous = _decide_gps_order(gps_recs, text)
        meta["latlon_order"], meta["latlon_ambiguous"] = order, ambiguous
    # Does GPS(.., .., c) carry an altitude?  Satellite counts are small integers next to BAROMETER.
    c_vals = [r for r in gps_recs if "gps_c" in r]
    c_is_alt = False
    if c_vals:
        dec = np.mean([r.get("gps_c_is_decimal", False) for r in c_vals]) > 0.5
        vals = np.array([r["gps_c"] for r in c_vals])
        looks_like_sats = (not dec) and np.all((vals >= 0) & (vals <= 40)) and bool(
            re.search(r"\bBAROMETER\s*[:=]", text, re.I))
        c_is_alt = not looks_like_sats
    out = []
    for r in raw:
        if "lat" in r:
            lat, lon = r["lat"], r["lon"]
        else:
            lat, lon = (r["gps_a"], r["gps_b"]) if order == "lat_lon" else (r["gps_b"], r["gps_a"])
        if abs(lat) > 90 or abs(lon) > 180 or (abs(lat) < 1e-9 and abs(lon) < 1e-9):
            continue
        alt, kind = None, None
        for key, k in (("rel", "relative"), ("h", "relative"), ("baro", "relative")):
            if key in r:
                alt, kind = r[key], k
                break
        if alt is None and c_is_alt and "gps_c" in r:
            alt, kind = r["gps_c"], "gps"
        rec = {"time_s": r.get("time_s"), "lat": lat, "lon": lon, "alt": alt, "alt_kind": kind,
               "abs_alt": r.get("abs", r.get("alt"))}
        for k in ("gb_pitch", "gb_yaw", "focal"):
            if k in r:
                rec[k] = r[k]
        out.append(rec)
        if meta["home"] is None and "home_a" in r:
            ha, hb = r["home_a"], r["home_b"]
            meta["home"] = {"lat": ha, "lon": hb} if order == "lat_lon" else {"lat": hb, "lon": ha}
    return out, meta


def srt_matches_telemetry(srt_path, telemetry, max_km=2.0):
    """True when the subtitle log describes the same flight as the telemetry's waypoints (either GPS
    order), None when there is nothing to compare. Guards against a log left over from another mission."""
    wps = [w for w in (telemetry or {}).get("waypoints") or [] if w.get("latitude") is not None]
    if not srt_path or not os.path.exists(srt_path) or len(wps) < 3:
        return None
    try:
        recs, _ = parse_srt_records(srt_path)
    except Exception:
        return False
    if len(recs) < 3:
        return False
    la, lo = float(np.median([r["lat"] for r in recs])), float(np.median([r["lon"] for r in recs]))
    wa, wo = float(np.median([w["latitude"] for w in wps])), float(np.median([w["longitude"] for w in wps]))

    def km(a, b, c, d):
        p1, p2 = math.radians(a), math.radians(c)
        h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
        return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))
    return min(km(la, lo, wa, wo), km(lo, la, wa, wo) if abs(lo) <= 90 else 1e9) <= max_km


def parse_drone_telemetry(srt_path, output_json_path):
    """Parses DJI-style .srt subtitle telemetry (all common firmware formats)."""
    if not os.path.exists(srt_path):
        print(f"Error: Could not find {srt_path}")
        return False
    recs, meta = parse_srt_records(srt_path)
    if not recs:
        print("No GPS data found in the SRT file. Falling back to synthetic baseline.")
        return generate_fallback_telemetry(60, output_json_path)

    n = len(recs)

    def series(values):
        v = np.array([np.nan if x is None else x for x in values], dtype=float)
        good = np.isfinite(v)
        if good.any() and not good.all():
            idx = np.arange(n)
            v = np.interp(idx, idx[good], v[good])
        return v

    kinds = [r["alt_kind"] for r in recs if r["alt_kind"]]
    abs_vals = [r["abs_alt"] for r in recs]
    has_abs = sum(v is not None for v in abs_vals) >= 0.5 * n
    if len(kinds) >= 0.5 * n:
        rel = series([r["alt"] for r in recs])
        dominant = max(set(kinds), key=kinds.count)
        # GPS(.., .., alt) has an unknown datum (take-off or MSL). It is used for the 3-D fit, where a
        # constant offset does not matter, and only as a cross-check for height above ground.
        ref, estimated = ("relative", False) if dominant == "relative" else ("gps_altitude", False)
    elif has_abs:
        a = series(abs_vals)
        rel, ref, estimated = a - a[0], "absolute", False
    else:
        rel, ref, estimated = np.full(n, np.nan), "estimated", True
    absolute = series(abs_vals) if has_abs else None
    has_time = all(r["time_s"] is not None for r in recs)

    waypoints = []
    for i, r in enumerate(recs):
        wp = {"frame_id": i, "latitude": r["lat"], "longitude": r["lon"]}
        if np.isfinite(rel[i]):
            wp["relative_altitude_m"] = round(float(rel[i]), 3)
        if has_time:
            wp["time_s"] = round(float(r["time_s"]), 4)
        if absolute is not None:
            wp["absolute_altitude_m"] = round(float(absolute[i]), 3)
        for k in ("gb_pitch", "gb_yaw"):
            if k in r:
                wp[k] = r[k]
        waypoints.append(wp)

    pitches = [r["gb_pitch"] for r in recs if "gb_pitch" in r]
    focals = [r["focal"] for r in recs if "focal" in r and r["focal"] > 0]
    total_dist = _path_length(waypoints)
    alt_vals = rel[np.isfinite(rel)]
    flight_data = {
        "source": "DJI_SRT", "is_synthetic": False,
        "altitude_reference": ref, "altitude_is_estimated": estimated,
        "latlon_order": meta["latlon_order"], "latlon_ambiguous": meta["latlon_ambiguous"],
        "home": meta["home"],
        "waypoint_count": len(waypoints),
        "total_distance_meters": round(total_dist, 2),
        "initial_altitude_m": waypoints[0].get("relative_altitude_m"),
        "median_altitude_m": round(float(np.median(alt_vals)), 3) if len(alt_vals) else None,
        "gimbal_pitch_median_deg": round(float(np.median(pitches)), 2) if pitches else None,
        "focal_length_35mm": round(float(np.median(focals)), 2) if focals else None,
        "has_timestamps": has_time,
        "waypoints": waypoints,
    }
    _write(flight_data, output_json_path)
    print(f"Parsed {len(waypoints)} telemetry waypoints ({ref} altitude, order {meta['latlon_order']}"
          f"{' - ambiguous, resolved during georeferencing' if meta['latlon_ambiguous'] else ''}, "
          f"timestamps={'yes' if has_time else 'no'}). Distance: {total_dist:.2f} m")
    return True


def swap_latlon_in_telemetry(telemetry):
    """Returns a copy of a telemetry dict with latitude/longitude exchanged in every waypoint."""
    out = dict(telemetry)
    wps = []
    for w in telemetry.get("waypoints") or []:
        w2 = dict(w)
        w2["latitude"], w2["longitude"] = w.get("longitude"), w.get("latitude")
        wps.append(w2)
    out["waypoints"] = wps
    if isinstance(telemetry.get("home"), dict):
        h = telemetry["home"]
        out["home"] = {"lat": h.get("lon"), "lon": h.get("lat")}
    order = telemetry.get("latlon_order")
    out["latlon_order"] = {"lat_lon": "lon_lat", "lon_lat": "lat_lon"}.get(order, order)
    return out


def _to_seconds(t):
    t = np.asarray(t, dtype=float)
    t = t - t[0]
    span = float(t[-1]) if len(t) else 0.0
    for div in (1.0, 1e3, 1e6, 1e9):
        if 1.0 <= span / div <= 36000.0:
            return t / div
    return t


def parse_midair_telemetry(csv_path, output_json_path):
    """
    Mid-Air style trajectory / sensor CSV.
    GPS columns (lat/lon/alt) are used directly; otherwise position_x/y/z in NED metres
    (x = North, y = East, z = Down) are converted exactly to WGS-84 around a local origin.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return False
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else (";" if ";" in sample else ",")
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            return generate_fallback_telemetry(60, output_json_path)
        fm = {name.lower().strip(): name for name in reader.fieldnames}

        def col(*names):
            return next((fm[n] for n in names if n in fm), None)

        lat_c, lon_c = col("gps_latitude", "latitude", "lat"), col("gps_longitude", "longitude", "lon")
        alt_c = col("gps_altitude", "altitude", "alt")
        rel_c = col("relative_altitude_m", "rel_alt", "height_above_takeoff(feet)", "height_above_takeoff")
        x_c, y_c, z_c = col("position_x", "pos_x", "x"), col("position_y", "pos_y", "y"), col("position_z", "pos_z", "z")
        t_c = col("timestamp", "time_s", "time", "t", "timestamp_s", "time(millisecond)")
        rows = list(reader)

    if not rows:
        return generate_fallback_telemetry(60, output_json_path)
    keep = [i for i in range(len(rows)) if i % 10 == 0] + ([len(rows) - 1] if (len(rows) - 1) % 10 else [])
    lat, lon, alt, tt = [], [], [], []
    ref = "relative"
    for i in keep:
        row = rows[i]
        try:
            if lat_c and lon_c:
                la, lo = float(row[lat_c]), float(row[lon_c])
                if rel_c and row.get(rel_c):
                    al = float(row[rel_c]) * (0.3048 if "feet" in rel_c.lower() else 1.0)
                elif alt_c and row.get(alt_c):
                    al, ref = float(row[alt_c]), "absolute"
                else:
                    al, ref = 0.0, "estimated"
            elif x_c and y_c:
                north, east = float(row[x_c]), float(row[y_c])
                up = -float(row[z_c]) if (z_c and row.get(z_c)) else 0.0
                la, lo, _ = enu_to_geodetic(np.array([east, north, 0.0]), (DEFAULT_BASE_LAT, DEFAULT_BASE_LON, 0.0))
                la, lo, al = float(la), float(lo), up
            else:
                continue
            tv = float(row[t_c]) if (t_c and row.get(t_c)) else np.nan
        except (ValueError, TypeError):
            continue
        lat.append(la)
        lon.append(lo)
        alt.append(al)
        tt.append(tv)
    if not lat:
        return generate_fallback_telemetry(60, output_json_path)

    alt = np.asarray(alt, dtype=float)
    rel = alt - alt[0] if ref == "absolute" else alt
    tt = np.asarray(tt, dtype=float)
    has_time = bool(np.isfinite(tt).all() and len(tt) > 1 and tt[-1] > tt[0])
    ts = _to_seconds(tt) if has_time else None
    waypoints = []
    for i in range(len(lat)):
        wp = {"frame_id": i, "latitude": round(lat[i], 9), "longitude": round(lon[i], 9),
              "relative_altitude_m": round(float(rel[i]), 3)}
        if ref == "absolute":
            wp["absolute_altitude_m"] = round(float(alt[i]), 3)
        if has_time and ts is not None:
            wp["time_s"] = round(float(ts[i]), 4)
        waypoints.append(wp)
    total_dist = _path_length(waypoints)
    flight_data = {
        "source": "MidAir_CSV", "is_synthetic": False,
        "altitude_reference": ref, "altitude_is_estimated": ref == "estimated",
        "latlon_order": "explicit", "latlon_ambiguous": False,
        "waypoint_count": len(waypoints), "total_distance_meters": round(total_dist, 2),
        "initial_altitude_m": waypoints[0]["relative_altitude_m"], "has_timestamps": has_time,
        "waypoints": waypoints,
    }
    _write(flight_data, output_json_path)
    print(f"[MidAir] Parsed {len(waypoints)} waypoints. Total distance: {total_dist:.2f} m")
    return True


def generate_fallback_telemetry(frame_count=60, output_json_path=None, total_distance=0.0, altitude=None):
    """
    Placeholder for flights without any log. Flagged is_synthetic=True: it carries NO position and is
    never used as GPS. Without a log the metric scale comes from the altitude fallback chain in the
    georeferencer (explicit altitude if one was given, otherwise a LOW-confidence default).
    """
    flight_data = {
        "source": "NONE", "is_synthetic": True,
        "altitude_reference": "estimated", "altitude_is_estimated": True,
        "assumed_altitude_m": float(altitude) if altitude else None,
        "latlon_order": "explicit", "latlon_ambiguous": False,
        "waypoint_count": 0, "total_distance_meters": 0.0,
        "initial_altitude_m": float(altitude) if altitude else None, "has_timestamps": False,
        "waypoints": [],
        "frame_count": int(frame_count),
    }
    if output_json_path:
        _write(flight_data, output_json_path)
        print("[Telemetry] No flight log: position unknown; metric scale will use the altitude fallback.")
    return True


def auto_detect_and_parse_telemetry(input_path, output_json_path, frame_count=60, assumed_altitude_m=None):
    """Format-agnostic ingestion: .srt / .txt (DJI subtitles), .csv (Mid-Air), .json, or none."""
    if not input_path or not os.path.exists(input_path):
        return generate_fallback_telemetry(frame_count, output_json_path, altitude=assumed_altitude_m)
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".srt", ".txt"):
        return parse_drone_telemetry(input_path, output_json_path)
    if ext == ".csv":
        return parse_midair_telemetry(input_path, output_json_path)
    if ext == ".json":
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("waypoints"):
                for stale in ("georeference", "metric_scale_factor", "camera_trajectory_units"):
                    data.pop(stale, None)          # belonged to a different reconstruction
                data.setdefault("is_synthetic", str(data.get("source", "")).upper().startswith(("SYNTHETIC", "NONE")))
                data.setdefault("altitude_reference", "relative")
                data.setdefault("latlon_ambiguous", False)
                _write(data, output_json_path)
                return True
        except Exception:
            pass
    return generate_fallback_telemetry(frame_count, output_json_path, altitude=assumed_altitude_m)


if __name__ == "__main__":
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
    srt_file = os.path.join(PROJECT_ROOT, "data", "drone_flight.srt")
    out_json = os.path.join(PROJECT_ROOT, "data", "flight_telemetry_parsed.json")
    auto_detect_and_parse_telemetry(srt_file, out_json)
