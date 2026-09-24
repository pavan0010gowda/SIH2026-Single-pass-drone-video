"""
PRISM // Flight-log ingestion (telemetry_parser.py, v2)

Fixes vs v1 (all of them affected heights / scale):
  * DJI Mini / Air / Mavic 3 SRT lines look like  "[rel_alt: 30.100 abs_alt: 450.200]".
    v1's regex required ']' directly after the number, so the altitude NEVER matched and every
    waypoint silently got 25.0 m.
  * Several DJI firmwares spell "longtitude" -> v1 found no GPS and used synthetic data.
  * Older DJI formats (Phantom 4 / Mavic Pro: "GPS(lon,lat,..) BAROMETER:x", "H 19.9m") parsed.
  * Every waypoint now carries its subtitle timestamp (time_s) so video frames can be matched to
    the exact GPS position (required for metric georeferencing).
  * No fabricated values: v1 wrote 125 m of "distance" for any hovering / GPS-less flight.
  * Mid-Air CSV: v1 swapped North/East (a mirror image) and used abs(z) / "20 m" altitude hacks.
  * Synthetic telemetry is flagged `is_synthetic` so it is never mistaken for real GPS.
"""

import os
import re
import json
import csv

from georeference import enu_to_geodetic, haversine_distance

import numpy as np

DEFAULT_BASE_LAT = 12.971600
DEFAULT_BASE_LON = 77.594600

_NUM = r"([-+]?\d+(?:\.\d+)?)"
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
_LAT_RE = re.compile(r"\blat(?:itude)?\s*[:=]\s*" + _NUM, re.I)
_LON_RE = re.compile(r"\blon(?:gt?itude|g)?\s*[:=]\s*" + _NUM, re.I)
_REL_RE = re.compile(r"\brel_alt\s*[:=]\s*" + _NUM, re.I)
_ABS_RE = re.compile(r"\babs_alt\s*[:=]\s*" + _NUM, re.I)
_ALT_RE = re.compile(r"\baltitude\s*[:=]\s*" + _NUM, re.I)
_BARO_RE = re.compile(r"\bbarometer\s*[:=]\s*" + _NUM, re.I)
_H_RE = re.compile(r"(?<![\w.])H\s+" + _NUM + r"\s*m\b")
_GPS_RE = re.compile(r"\bGPS\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"(?:\s*,\s*" + _NUM + r")?\s*\)", re.I)


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


def _parse_srt_block(block):
    rec = {}
    m = _TIME_RE.search(block)
    if m:
        h, mi, se, ms = m.groups()
        rec["time_s"] = int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0
    lat, lon = _LAT_RE.search(block), _LON_RE.search(block)
    if lat and lon:
        rec["lat"], rec["lon"] = float(lat.group(1)), float(lon.group(1))
    else:
        g = _GPS_RE.search(block)
        if g:
            a, b = float(g.group(1)), float(g.group(2))
            rec["lon"], rec["lat"] = (b, a) if (abs(b) > 90 >= abs(a)) else (a, b)   # DJI: GPS(lon, lat, ..)
    for key, rx in (("rel", _REL_RE), ("abs", _ABS_RE), ("alt", _ALT_RE), ("baro", _BARO_RE), ("h", _H_RE)):
        mm = rx.search(block)
        if mm:
            rec[key] = float(mm.group(1))
    return rec


def parse_drone_telemetry(srt_path, output_json_path):
    """Parses DJI-style .srt subtitle telemetry (all common firmware formats)."""
    if not os.path.exists(srt_path):
        print(f"Error: Could not find {srt_path}")
        return False
    text = _read_text(srt_path).replace("\r\n", "\n").replace("\r", "\n")
    recs = []
    for block in re.split(r"\n\s*\n", text):
        r = _parse_srt_block(block)
        if "lat" in r and "lon" in r and not (abs(r["lat"]) < 1e-9 and abs(r["lon"]) < 1e-9) \
                and abs(r["lat"]) <= 90 and abs(r["lon"]) <= 180:
            recs.append(r)
    if not recs:
        print("No GPS data found in the SRT file. Falling back to synthetic baseline.")
        return generate_fallback_telemetry(60, output_json_path)

    n = len(recs)
    rel_key = next((k for k in ("rel", "baro", "h") if sum(k in r for r in recs) >= 0.5 * n), None)
    abs_key = next((k for k in ("abs", "alt") if sum(k in r for r in recs) >= 0.5 * n), None)

    def series(key):
        v = np.array([r.get(key, np.nan) for r in recs], dtype=float)
        good = np.isfinite(v)
        if good.any() and not good.all():
            idx = np.arange(n)
            v = np.interp(idx, idx[good], v[good])
        return v

    if rel_key:
        rel, ref, estimated = series(rel_key), "relative", False
    elif abs_key:
        a = series(abs_key)
        rel, ref, estimated = a - a[0], "absolute", False
    else:
        rel, ref, estimated = np.full(n, 25.0), "estimated", True
    absolute = series(abs_key) if abs_key else None
    has_time = all("time_s" in r for r in recs)

    waypoints = []
    for i, r in enumerate(recs):
        wp = {"frame_id": i, "latitude": r["lat"], "longitude": r["lon"],
              "relative_altitude_m": round(float(rel[i]), 3)}
        if has_time:
            wp["time_s"] = round(r["time_s"], 4)
        if absolute is not None:
            wp["absolute_altitude_m"] = round(float(absolute[i]), 3)
        waypoints.append(wp)

    total_dist = _path_length(waypoints)
    flight_data = {
        "source": "DJI_SRT", "is_synthetic": False,
        "altitude_reference": ref, "altitude_is_estimated": estimated,
        "waypoint_count": len(waypoints),
        "total_distance_meters": round(total_dist, 2),
        "initial_altitude_m": waypoints[0]["relative_altitude_m"],
        "has_timestamps": has_time,
        "waypoints": waypoints,
    }
    _write(flight_data, output_json_path)
    print(f"Successfully extracted {len(waypoints)} telemetry waypoints ({ref} altitude, "
          f"timestamps={'yes' if has_time else 'no'}). Distance: {total_dist:.2f} m")
    return True


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
        rel_c = col("relative_altitude_m", "rel_alt")
        x_c, y_c, z_c = col("position_x", "pos_x", "x"), col("position_y", "pos_y", "y"), col("position_z", "pos_z", "z")
        t_c = col("timestamp", "time_s", "time", "t", "timestamp_s")
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
                    al = float(row[rel_c])
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
        "waypoint_count": len(waypoints), "total_distance_meters": round(total_dist, 2),
        "initial_altitude_m": waypoints[0]["relative_altitude_m"], "has_timestamps": has_time,
        "waypoints": waypoints,
    }
    _write(flight_data, output_json_path)
    print(f"[MidAir] Parsed {len(waypoints)} waypoints. Total distance: {total_dist:.2f} m")
    return True


def generate_fallback_telemetry(frame_count=60, output_json_path=None, total_distance=125.0, altitude=25.0):
    """
    Synthetic HUD track for flights without a log. Flagged is_synthetic=True: it only keeps the
    dashboard alive and is NEVER used as GPS. The metric scale then comes from `altitude`
    (flight height above ground), reported with LOW confidence.
    """
    waypoints = []
    step_lat = (total_distance / 111000.0) / max(frame_count, 1)
    for i in range(max(frame_count, 10)):
        waypoints.append({
            "frame_id": i,
            "latitude": round(DEFAULT_BASE_LAT + i * step_lat, 6),
            "longitude": round(DEFAULT_BASE_LON + i * 0.000003, 6),
            "relative_altitude_m": round(float(altitude), 2),
        })
    flight_data = {
        "source": "SYNTHETIC_ESTIMATE", "is_synthetic": True,
        "altitude_reference": "estimated", "altitude_is_estimated": True,
        "assumed_altitude_m": float(altitude),
        "waypoint_count": len(waypoints), "total_distance_meters": round(total_distance, 2),
        "initial_altitude_m": float(altitude), "has_timestamps": False,
        "waypoints": waypoints,
    }
    if output_json_path:
        _write(flight_data, output_json_path)
        print(f"[Fallback] Generated synthetic HUD telemetry with {len(waypoints)} waypoints "
              f"(assumed flight altitude {altitude} m).")
    return True


def auto_detect_and_parse_telemetry(input_path, output_json_path, frame_count=60, assumed_altitude_m=25.0):
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
                data.setdefault("is_synthetic", str(data.get("source", "")).upper().startswith("SYNTHETIC"))
                data.setdefault("altitude_reference", "relative")
                _write(data, output_json_path)
                return True
        except Exception:
            pass
    return generate_fallback_telemetry(frame_count, output_json_path, altitude=assumed_altitude_m)


if __name__ == "__main__":
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
    srt_file = os.path.join(PROJECT_ROOT, "data", "drone_flight.srt")
    out_json = os.path.join(PROJECT_ROOT, "data", "flight_telemetry.json")
    auto_detect_and_parse_telemetry(srt_file, out_json)