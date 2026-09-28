#!/usr/bin/env python3
# =============================================================================
#  PRISM-TURBO  |  time-budgeted, GPU-first drone-video -> 3D reconstruction
# =============================================================================
#  Input : a drone video (4K, 10+ min is fine) + optional DJI .SRT telemetry
#  Output: prism_colab_bundle.zip (same file names / JSON keys as the original
#          PRISM Colab pipeline, plus a few extras)
#
#  Pipeline (every heavy stage is on the GPU or strictly time-budgeted):
#    1. INGEST  one ffmpeg pass: NVDEC decode -> scale_cuda -> {NVENC web video,
#               keyframe candidates}.  Streaming keyframe selector chooses the
#               sharpest frame per motion-adaptive window (KLT parallax).
#    2. SfM     pycolmap-cuda12: GPU SIFT, GPU matching (sequential + loop /
#               GPS-spatial pairs), view-graph focal calibration, GLOBAL mapper
#               (GLOMAP) with incremental fallback.
#    3. MVS     coverage-greedy reference-view ordering, baseline-aware source
#               views, CUDA PatchMatch (photometric -> geometric) run in chunks
#               against a hard deadline, then depth-map fusion.
#    4. GEO     gravity from camera horizon / ground plane, GPS Sim3 (7-DOF or
#               4-DOF) -> metric ENU, ground at y = 0, Three.js axes.
#    5. MESH    screened Poisson (Kazhdan 18.75 via COLMAP) on a uniformly
#               resampled cloud, support-distance trimming, island removal,
#               QEM simplification with colour interpolation.
#    6. BUNDLE  PLY / OBJ / GLB + telemetry / frame index JSON + web video.
#
#  Heavy C++ stages run in child processes: a crash or OOM there cannot kill
#  the notebook kernel, and the orchestrator can fall back gracefully.
# =============================================================================
import argparse
import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

ENGINE_VERSION = "prism-turbo 1.1 (pycolmap 4.2)"
ENGINE_PATH = os.path.abspath(__file__)
_T0 = time.time()


# -----------------------------------------------------------------------------
# small utilities
# -----------------------------------------------------------------------------
def log(msg):
    print(f"[PRISM {time.time() - _T0:7.1f}s] {msg}", flush=True)


def warn(msg):
    log(f"WARNING: {msg}")


def fmt_s(sec):
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}m{sec % 60:04.1f}s" if sec >= 60 else f"{sec:.1f}s"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_cmd(cmd, timeout=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "timeout"
    except FileNotFoundError as e:
        return -2, "", str(e)


def even(x):
    return max(2, int(round(x / 2.0)) * 2)


def fit_long_side(w, h, long_side):
    if long_side <= 0 or max(w, h) <= long_side:
        return even(w), even(h)
    s = long_side / float(max(w, h))
    return even(w * s), even(h * s)


def json_dump(obj, path, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


class Budget:
    """Wall-clock budget for the whole run (processing only, not installs)."""

    def __init__(self, total_s):
        self.t0 = time.time()
        self.total = float(total_s)
        self.marks = []

    def elapsed(self):
        return time.time() - self.t0

    def remaining(self):
        return self.total - self.elapsed()

    def at(self, frac):
        return self.t0 + frac * self.total


class StageClock:
    def __init__(self):
        self.rows = []
        self._cur = None

    def start(self, name):
        self._cur = (name, time.time())
        log(f"==== {name} ====")

    def stop(self, note=""):
        if self._cur is None:
            return 0.0
        name, t = self._cur
        dt = time.time() - t
        self.rows.append((name, dt, note))
        log(f"---- {name} done in {fmt_s(dt)} {note}")
        self._cur = None
        return dt

    def table(self, total_target):
        lines = ["", "  stage                                time      note",
                 "  " + "-" * 72]
        tot = 0.0
        for name, dt, note in self.rows:
            tot += dt
            lines.append(f"  {name:<30} {fmt_s(dt):>9}   {note}")
        lines.append("  " + "-" * 72)
        lines.append(f"  {'TOTAL':<30} {fmt_s(tot):>9}   (target {fmt_s(total_target)})")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# configuration & hardware profiles
# -----------------------------------------------------------------------------
@dataclass
class Config:
    video_path: str = ""
    srt_path: str = ""                      # DJI .SRT (or .csv); "" -> auto-detect next to video
    workdir: str = "/content/prism_workspace"
    bundle_zip: str = "/content/prism_colab_bundle.zip"
    quality: str = "balanced"               # fast | balanced | max
    target_minutes: float = 0.0             # 0 -> 12 / 18 / 24 depending on quality
    # ingest
    keyframe_long_side: int = 0             # 0 -> auto (1920 T4, 2560 bigger GPUs)
    max_keyframes: int = 0                  # 0 -> auto
    candidate_fps: float = 0.0              # 0 -> auto
    kf_spacing: float = 0.12                # target parallax between keyframes (fraction of width)
    web_video: bool = True
    web_video_long_side: int = 1920
    # camera / scale
    camera_model: str = "RADIAL"            # SIMPLE_RADIAL | RADIAL | OPENCV | OPENCV_FISHEYE
    focal_35mm: float = 0.0                 # override 35mm-equivalent focal (0 -> SRT / auto)
    assumed_altitude_m: float = 0.0         # 0 = automatic: GPS track, else the altitude in the log; only
                                            # a flight with NO log at all falls back to 30 m (LOW confidence)
    # dense
    mvs_max_size: int = 0                   # 0 -> auto
    mvs_num_sources: int = 0                # 0 -> auto
    geom_consistency: bool = True
    # outputs
    max_export_points: int = 0              # 0 -> auto
    max_mesh_faces: int = 0                 # 0 -> auto
    poisson_depth: int = 0                  # 0 -> auto
    export_glb: bool = True
    keep_intermediate: bool = False         # keep depth maps / undistorted images after fusion
    resume: bool = False                    # reuse finished stages in workdir
    verbose: bool = False


PROFILES = {
    # tier: T4 / P100 / unknown small GPUs; usually 2 vCPUs on Colab
    "t4": dict(kf_long=1920, max_kf=420, cand_fps=6.0, sift_feats=5000, seq_overlap=8, loop_imgs=25,
               max_pairs=9000, keep_tracks=60000, ba_rounds=2, ba_iters=40, mvs_size=1600, mvs_src=10,
               sec_per_ref=1.8, fusion_px=130e6, export_pts=3_000_000, mesh_faces=1_000_000,
               poisson_max_depth=11),
    # tier: L4 / A10 / V100 / RTX 3090-4090
    "mid": dict(kf_long=2560, max_kf=600, cand_fps=8.0, sift_feats=6500, seq_overlap=10, loop_imgs=30,
                max_pairs=16000, keep_tracks=160000, ba_rounds=3, ba_iters=60, mvs_size=2048, mvs_src=12,
                sec_per_ref=1.7,
                fusion_px=320e6, export_pts=5_000_000, mesh_faces=1_500_000, poisson_max_depth=12),
    # tier: A100 / H100 / L40
    "high": dict(kf_long=2560, max_kf=800, cand_fps=8.0, sift_feats=8192, seq_overlap=10, loop_imgs=35,
                 max_pairs=24000, keep_tracks=250000, ba_rounds=3, ba_iters=80, mvs_size=2560, mvs_src=14,
                 sec_per_ref=1.6,
                 fusion_px=600e6, export_pts=6_000_000, mesh_faces=2_000_000, poisson_max_depth=12),
}

QUALITY = {
    # PatchMatch settings (COLMAP 'medium' style: window step 2 => ~5x faster than default)
    "fast": dict(minutes=12.0, win_radius=4, num_samples=8, it_photo=4, it_geo=2, scale=0.85),
    "balanced": dict(minutes=18.0, win_radius=4, num_samples=10, it_photo=5, it_geo=3, scale=1.0),
    "max": dict(minutes=24.0, win_radius=5, num_samples=12, it_photo=5, it_geo=3, scale=1.15),
}


def _ram_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return 12.0


def probe_hardware():
    info = dict(cpu=os.cpu_count() or 2, ram_gb=round(_ram_gb(), 1), gpu_name="", gpu_mem_gb=0.0,
                gpu_count=0, driver="")
    rc, out, _ = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader,nounits"], timeout=20)
    if rc == 0 and out.strip():
        rows = [r.split(",") for r in out.strip().splitlines()]
        info["gpu_count"] = len(rows)
        info["gpu_name"] = rows[0][0].strip()
        try:
            info["gpu_mem_gb"] = round(float(rows[0][1]) / 1024.0, 1)
        except Exception:
            pass
        info["driver"] = rows[0][2].strip() if len(rows[0]) > 2 else ""
    return info


def gpu_tier(hw):
    n = hw.get("gpu_name", "").upper()
    if any(k in n for k in ("A100", "H100", "H200", "B100", "B200", "GH200", "L40", "A6000", "RTX 6000", "RTX PRO")):
        return "high"
    if any(k in n for k in ("L4", "A10", "A30", "A40", "V100", "4090", "3090", "5090", "4080", "5080")):
        return "mid"
    return "t4"


def make_profile(cfg, hw):
    tier = gpu_tier(hw)
    p = dict(PROFILES[tier])
    q = dict(QUALITY.get(cfg.quality, QUALITY["balanced"]))
    p.update(tier=tier, quality=cfg.quality, **{k: v for k, v in q.items() if k != "minutes"})
    p["target_s"] = 60.0 * (cfg.target_minutes if cfg.target_minutes > 0 else q["minutes"])
    s = q["scale"]
    p["max_kf"] = int(p["max_kf"] * s)
    p["export_pts"] = int(p["export_pts"] * s)
    p["mesh_faces"] = int(p["mesh_faces"] * s)
    # CPU-bound stages scale with vCPUs (Colab T4 has only 2)
    cpu = hw.get("cpu", 2)
    if cpu >= 8 and tier == "t4":
        p["fusion_px"] *= 2.0
    if cpu <= 2:
        p["fusion_px"] = min(p["fusion_px"], 130e6)
    # user overrides
    if cfg.keyframe_long_side > 0:
        p["kf_long"] = cfg.keyframe_long_side
    if cfg.max_keyframes > 0:
        p["max_kf"] = cfg.max_keyframes
    if cfg.candidate_fps > 0:
        p["cand_fps"] = cfg.candidate_fps
    if cfg.mvs_max_size > 0:
        p["mvs_size"] = cfg.mvs_max_size
    if cfg.mvs_num_sources > 0:
        p["mvs_src"] = cfg.mvs_num_sources
    if cfg.max_export_points > 0:
        p["export_pts"] = cfg.max_export_points
    if cfg.max_mesh_faces > 0:
        p["mesh_faces"] = cfg.max_mesh_faces
    p["mvs_size"] = min(p["mvs_size"], p["kf_long"])
    ram = hw.get("ram_gb", 12.0)
    p["cache_gb"] = float(max(2.0, min(16.0, 0.25 * ram)))
    return p


# -----------------------------------------------------------------------------
# video probing
# -----------------------------------------------------------------------------
def _parse_rate(s):
    try:
        if not s or s in ("0/0", "N/A"):
            return 0.0
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) != 0 else 0.0
        return float(s)
    except Exception:
        return 0.0


def probe_video(path):
    rc, out, err = run_cmd(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                            "-show_streams", path], timeout=120)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {err.strip()[:400]}")
    j = json.loads(out)
    vs = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), None)
    if vs is None:
        raise RuntimeError("no video stream found")
    fmt = j.get("format", {})
    fps = _parse_rate(vs.get("avg_frame_rate")) or _parse_rate(vs.get("r_frame_rate")) or 30.0
    duration = float(vs.get("duration") or fmt.get("duration") or 0.0)
    nb = int(vs.get("nb_frames") or 0) if str(vs.get("nb_frames", "")).isdigit() else 0
    if nb <= 0:
        nb = int(round(duration * fps))
    rotation = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(round(float(sd["rotation"])))
            except Exception:
                pass
    if not rotation:
        try:
            rotation = int((vs.get("tags") or {}).get("rotate", 0))
        except Exception:
            rotation = 0
    return dict(
        path=path, width=int(vs["width"]), height=int(vs["height"]), fps=fps, frames=nb, duration=duration,
        codec=vs.get("codec_name", ""), pix_fmt=vs.get("pix_fmt", ""), bit_rate=int(vs.get("bit_rate") or fmt.get("bit_rate") or 0),
        color_space=vs.get("color_space", ""), color_range=vs.get("color_range", ""),
        color_transfer=vs.get("color_transfer", ""), rotation=rotation % 360,
        has_audio=any(s.get("codec_type") == "audio" for s in j.get("streams", [])),
        size_mb=round(float(fmt.get("size") or 0) / 1e6, 1))


# -----------------------------------------------------------------------------
# telemetry (DJI SRT and generic CSV)
# -----------------------------------------------------------------------------
_NUM = r"(-?\d+(?:\.\d+)?)"
_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_PATTERNS = {
    "lat": re.compile(r"\blat(?:itude)?\s*[:=]\s*" + _NUM, re.I),
    "lon": re.compile(r"\blon(?:g(?:t)?(?:itude)?)?\s*[:=]\s*" + _NUM, re.I),
    "rel_alt": re.compile(r"\brel_alt\s*[:=]\s*" + _NUM, re.I),
    "abs_alt": re.compile(r"\babs_alt\s*[:=]\s*" + _NUM, re.I),
    "alt": re.compile(r"(?<![_a-z])altitude\s*[:=]\s*" + _NUM, re.I),
    "baro": re.compile(r"\bBAROMETER\s*[:=]\s*" + _NUM, re.I),
    "height": re.compile(r"\bH\s+" + _NUM + r"\s*m\b"),
    "focal": re.compile(r"\bfocal_len\s*[:=]\s*" + _NUM, re.I),
    "yaw": re.compile(r"\bgb_yaw\s*[:=]\s*" + _NUM, re.I),
    "pitch": re.compile(r"\bgb_pitch\s*[:=]\s*" + _NUM, re.I),
    "roll": re.compile(r"\bgb_roll\s*[:=]\s*" + _NUM, re.I),
    "gps": re.compile(r"\bGPS\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"(?:\s*,\s*" + _NUM + r")?\s*\)", re.I),
    "home": re.compile(r"\bHOME\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"(?:\s*,\s*" + _NUM + r")?\s*\)", re.I),
    "datetime": re.compile(r"(\d{4}[-./]\d{2}[-./]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,:]\d+)*)"),
}


def _find(key, text):
    m = _PATTERNS[key].search(text)
    return float(m.group(1)) if m else None


def parse_srt(path):
    """
    DJI / converter subtitle logs. GPS(a, b[, c]) is ambiguous: DJI's legacy firmware writes
    (lon, lat, satellites) next to BAROMETER, most converters write (lat, lon, altitude). When both
    numbers are inside +/-90 deg the order is decided later from the SfM camera track (the wrong
    order mirrors the path and cannot be fitted by a rotation). The third value is the altitude
    whenever it is a decimal number (satellite counts are small integers).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<[^>]*>", " ", text)
    legacy = bool(re.search(r"\bBAROMETER\s*[:=]", text, re.I))
    recs = []
    for block in re.split(r"\n\s*\n", text):
        m = _TIME_RE.search(block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / (1000.0 if len(m.group(4)) == 3 else 10 ** len(m.group(4)))
        t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / (1000.0 if len(m.group(8)) == 3 else 10 ** len(m.group(8)))
        body = block[m.end():]
        lat, lon = _find("lat", body), _find("lon", body)
        pair, gps_alt = None, None
        if lat is None or lon is None:
            gm = _PATTERNS["gps"].search(body)
            if gm:
                pair = (float(gm.group(1)), float(gm.group(2)))
                if gm.group(3) is not None and "." in gm.group(3) and not legacy:
                    gps_alt = float(gm.group(3))
        rel = _find("rel_alt", body)
        if rel is None:
            rel = _find("height", body)
        if rel is None:
            rel = _find("baro", body)
        if rel is None:
            rel = _find("alt", body)
        if rel is None:
            rel = gps_alt
        dm = _PATTERNS["datetime"].search(body)
        recs.append(dict(t=0.5 * (t0 + t1) if t1 > t0 else t0, t0=t0, lat=lat, lon=lon, pair=pair, rel_alt=rel,
                         abs_alt=_find("abs_alt", body), focal=_find("focal", body), yaw=_find("yaw", body),
                         pitch=_find("pitch", body), roll=_find("roll", body),
                         datetime=dm.group(1) if dm else None))
    recs.sort(key=lambda r: r["t0"])
    # use frame-start times so frame n (t = n / fps) lines up with its SRT entry
    for r in recs:
        r["t"] = r["t0"]
    pairs = [r["pair"] for r in recs if r["pair"] is not None]
    order, ambiguous = "explicit", False
    if pairs:
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        if np.any(np.abs(a) > 90):
            order = "lon_lat"
        elif np.any(np.abs(b) > 90):
            order = "lat_lon"
        else:
            ambiguous = True
            order = "lat_lon"      # ISO 6709; real DJI logs (incl. legacy Phantom) write latitude first
        for r in recs:
            if r["pair"] is not None:
                r["lat"], r["lon"] = (r["pair"] if order == "lat_lon" else r["pair"][::-1])
    for r in recs:
        r.pop("pair", None)
    return recs, {"latlon_order": order, "latlon_ambiguous": ambiguous}


def parse_csv_telemetry(path, duration):
    import csv
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    keys = {k.lower().strip(): k for k in rows[0].keys()}

    def pick(*cands):
        for c in cands:
            for k, orig in keys.items():
                if k.startswith(c):
                    return orig
        return None

    klat, klon = pick("latitude", "lat"), pick("longitude", "lon", "lng")
    kalt = pick("height_above_takeoff", "rel_alt", "altitude(m)", "altitude", "alt", "height")
    ktime = pick("time(millisecond)", "time_ms", "timestamp", "time(s)", "time")
    recs = []
    for i, r in enumerate(rows):
        try:
            lat, lon = float(r[klat]), float(r[klon])
        except Exception:
            continue
        alt = None
        if kalt:
            try:
                alt = float(r[kalt])
                if "feet" in kalt.lower() or "(ft" in kalt.lower():
                    alt *= 0.3048
            except Exception:
                alt = None
        t = None
        if ktime:
            try:
                t = float(r[ktime])
                if "milli" in ktime.lower() or "_ms" in ktime.lower():
                    t /= 1000.0
            except Exception:
                t = None
        recs.append(dict(t=t, t0=t, lat=lat, lon=lon, rel_alt=alt, abs_alt=None, focal=None, yaw=None,
                         pitch=None, roll=None, datetime=None))
    if recs and any(r["t"] is None for r in recs):
        n = len(recs)
        for i, r in enumerate(recs):
            r["t"] = r["t0"] = duration * i / max(1, n - 1)
    elif recs:
        t0 = recs[0]["t"]
        for r in recs:
            r["t"] -= t0
            r["t0"] = r["t"]
    return recs


class Telemetry:
    def __init__(self, recs, source="", meta=None):
        self.source = source
        self.meta = dict(meta or {"latlon_order": "explicit", "latlon_ambiguous": False})
        recs = sorted(recs, key=lambda r: r["t"])
        self.recs = recs
        good = [r for r in recs if r["lat"] is not None and r["lon"] is not None
                and abs(r["lat"]) <= 90 and abs(r["lon"]) <= 180 and (abs(r["lat"]) + abs(r["lon"])) > 1e-3]
        self.has_gps = len(good) >= 3
        self.g = good
        self.t = np.array([r["t"] for r in good]) if good else np.zeros(0)
        self.lat = np.array([r["lat"] for r in good]) if good else np.zeros(0)
        self.lon = np.array([r["lon"] for r in good]) if good else np.zeros(0)
        rel = [r["rel_alt"] for r in recs if r["rel_alt"] is not None]
        self.has_alt = len(rel) >= 3
        self.alt_t = np.array([r["t"] for r in recs if r["rel_alt"] is not None])
        self.alt = np.array(rel, dtype=float) if rel else np.zeros(0)
        absa = [(r["t"], r["abs_alt"]) for r in recs if r["abs_alt"] is not None]
        self.abs_t = np.array([a[0] for a in absa]) if absa else np.zeros(0)
        self.abs_alt = np.array([a[1] for a in absa]) if absa else np.zeros(0)
        foc = [r["focal"] for r in recs if r["focal"] is not None and r["focal"] > 0]
        f = float(np.median(foc)) if foc else 0.0
        if f > 100:  # some DJI models write 280 for 28.0 mm
            f /= 10.0
        self.focal35 = f
        yaw = [(r["t"], r["yaw"], r["pitch"]) for r in recs if r["yaw"] is not None and r["pitch"] is not None]
        self.gimbal = np.array(yaw) if yaw else np.zeros((0, 3))
        self.gps_spread_m = 0.0
        if self.has_gps:
            e, n = latlon_to_en(self.lat, self.lon, float(np.mean(self.lat)), float(np.mean(self.lon)))
            self.gps_spread_m = float(np.hypot(np.ptp(e), np.ptp(n)))

    def at(self, times):
        times = np.asarray(times, dtype=float)
        out = dict(lat=None, lon=None, rel_alt=None, abs_alt=None, gimbal_yaw=None, gimbal_pitch=None)
        if self.has_gps:
            out["lat"] = np.interp(times, self.t, self.lat)
            out["lon"] = np.interp(times, self.t, _unwrap_lon(self.lon))
            out["lon"] = ((out["lon"] + 180.0) % 360.0) - 180.0
        if self.has_alt:
            out["rel_alt"] = np.interp(times, self.alt_t, self.alt)
        if len(self.abs_alt) >= 2:
            out["abs_alt"] = np.interp(times, self.abs_t, self.abs_alt)
        if len(self.gimbal) >= 2:
            out["gimbal_yaw"] = np.interp(times, self.gimbal[:, 0], np.degrees(np.unwrap(np.radians(self.gimbal[:, 1]))))
            out["gimbal_pitch"] = np.interp(times, self.gimbal[:, 0], self.gimbal[:, 2])
        return out

    def swapped(self):
        """Same log with latitude and longitude exchanged (to test the other GPS(a, b) order)."""
        recs = [dict(r, lat=r["lon"], lon=r["lat"]) for r in self.recs]
        meta = dict(self.meta)
        meta["latlon_order"] = {"lat_lon": "lon_lat", "lon_lat": "lat_lon"}.get(meta.get("latlon_order"), "swapped")
        return Telemetry(recs, self.source, meta)


def _unwrap_lon(lon):
    return np.degrees(np.unwrap(np.radians(lon)))


def load_telemetry(path, duration):
    if not path or not os.path.exists(path):
        return Telemetry([], "")
    try:
        if path.lower().endswith(".csv"):
            recs, meta = parse_csv_telemetry(path, duration), None
        else:
            recs, meta = parse_srt(path)
        tel = Telemetry(recs, os.path.basename(path), meta)
        log(f"telemetry: {len(recs)} records, gps={tel.has_gps} (spread {tel.gps_spread_m:.0f} m), "
            f"alt={tel.has_alt} (median {np.median(tel.alt):.1f} m)" if tel.has_alt else
            f"telemetry: {len(recs)} records, gps={tel.has_gps} (spread {tel.gps_spread_m:.0f} m), alt=False")
        if tel.meta.get("latlon_ambiguous"):
            log("telemetry: GPS(a, b) order is ambiguous -> will be decided from the camera track after SfM")
        return tel
    except Exception as e:
        warn(f"could not parse telemetry {path}: {e}")
        return Telemetry([], "")


# WGS84 geodesy ---------------------------------------------------------------
_WGS_A = 6378137.0
_WGS_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = _WGS_A / np.sqrt(1 - _WGS_E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - _WGS_E2) + h) * np.sin(lat)
    return np.stack([x, y, z], -1)


def ecef_to_enu(xyz, lat0, lon0, h0=0.0):
    o = geodetic_to_ecef(np.array(lat0), np.array(lon0), np.array(h0))
    la, lo = math.radians(lat0), math.radians(lon0)
    R = np.array([[-math.sin(lo), math.cos(lo), 0],
                  [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    return (np.asarray(xyz) - o) @ R.T


def latlon_to_en(lat, lon, lat0, lon0):
    enu = ecef_to_enu(geodetic_to_ecef(np.asarray(lat), np.asarray(lon), np.zeros_like(np.asarray(lat, float))),
                      lat0, lon0, 0.0)
    return enu[..., 0], enu[..., 1]


def en_to_latlon(e, n, lat0, lon0):
    # local inverse (accurate to mm over a few km)
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(lat0)) + 1.175 * math.cos(4 * math.radians(lat0))
    m_per_deg_lon = (math.pi / 180) * _WGS_A * math.cos(math.radians(lat0)) / math.sqrt(1 - _WGS_E2 * math.sin(math.radians(lat0)) ** 2)
    return lat0 + np.asarray(n) / m_per_deg_lat, lon0 + np.asarray(e) / m_per_deg_lon


# -----------------------------------------------------------------------------
# colour conversion (NV12 -> BGR with the correct matrix / range)
# -----------------------------------------------------------------------------
def yuv_to_bgr_matrix(matrix="bt709", full_range=False):
    kr, kb = (0.2126, 0.0722) if matrix == "bt709" else ((0.2627, 0.0593) if matrix == "bt2020" else (0.299, 0.114))
    kg = 1.0 - kr - kb
    if full_range:
        ys, cs, y0 = 1.0, 1.0, 0.0
    else:
        ys, cs, y0 = 255.0 / 219.0, 255.0 / 224.0, 16.0
    cr_r = 2 * (1 - kr) * cs
    cb_b = 2 * (1 - kb) * cs
    cb_g = -2 * (1 - kb) * kb / kg * cs
    cr_g = -2 * (1 - kr) * kr / kg * cs
    rows = {
        "B": [ys, cb_b, 0.0],
        "G": [ys, cb_g, cr_g],
        "R": [ys, 0.0, cr_r],
    }
    M = np.zeros((3, 4), np.float32)
    for i, ch in enumerate("BGR"):
        a, b, c = rows[ch]
        M[i, :3] = (a, b, c)
        M[i, 3] = -a * y0 - (b + c) * 128.0
    return M


def nv12_to_bgr(buf, w, h, M):
    import cv2
    buf = np.asarray(buf, dtype=np.uint8)
    y = buf[: w * h].reshape(h, w)
    uv = buf[w * h: w * h * 3 // 2].reshape(h // 2, w // 2, 2)
    uv = cv2.resize(uv, (w, h), interpolation=cv2.INTER_LINEAR)
    yuv = np.empty((h, w, 3), np.uint8)
    yuv[..., 0] = y
    yuv[..., 1:] = uv
    return cv2.transform(yuv, M)


def color_matrix_for(vinfo):
    cs = (vinfo.get("color_space") or "").lower()
    if "2020" in cs:
        m = "bt2020"
    elif cs in ("smpte170m", "bt470bg", "bt601", "fcc"):
        m = "bt601"
    elif cs in ("bt709",):
        m = "bt709"
    else:
        m = "bt709" if vinfo.get("height", 0) >= 720 else "bt601"
    full = (vinfo.get("color_range") or "").lower() in ("pc", "jpeg", "full") or vinfo.get("pix_fmt", "").startswith("yuvj")
    return m, full


# -----------------------------------------------------------------------------
# INGEST: single-pass decode + web video + streaming keyframe selection
# -----------------------------------------------------------------------------
class _Cand:
    __slots__ = ("n", "sharp", "small", "data", "disp", "surv")

    def __init__(self, n, sharp, small, data):
        self.n, self.sharp, self.small, self.data = n, sharp, small, data
        self.disp, self.surv = 0.0, 1.0


class _KLT:
    def __init__(self, long_side):
        self.long = float(long_side)
        self.p0 = self.p = None
        self.alive = None
        self.prev = None
        self.n0 = 0

    def reset(self, small):
        import cv2
        try:
            pts = cv2.goodFeaturesToTrack(small, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7)
        except Exception:        # OpenCV API drift -> time-based selection still works
            pts = None
        self.prev = small
        if pts is None or len(pts) < 25:
            self.p0 = self.p = None
            self.n0 = 0
            return False
        self.p0 = pts.reshape(-1, 2).astype(np.float32)
        self.p = self.p0.copy()
        self.alive = np.ones(len(self.p0), bool)
        self.n0 = len(self.p0)
        return True

    @property
    def ok(self):
        return self.n0 >= 25

    def update(self, small):
        """Track into `small`; returns (median parallax as fraction of the long side, survival)."""
        import cv2
        if not self.ok:
            self.prev = small
            return None, 0.0
        idx = np.nonzero(self.alive)[0]
        if len(idx) < 8:
            self.prev = small
            return 1.0, 0.0
        try:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, small, self.p[idx].reshape(-1, 1, 2), None,
                                                 winSize=(21, 21), maxLevel=3,
                                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        except Exception:
            self.n0 = 0
            self.prev = small
            return None, 0.0
        p1 = p1.reshape(-1, 2)
        st = st.reshape(-1).astype(bool)
        h, w = small.shape[:2]
        inside = (p1[:, 0] >= 0) & (p1[:, 1] >= 0) & (p1[:, 0] < w) & (p1[:, 1] < h)
        good = st & inside
        self.alive[idx[~good]] = False
        self.p[idx[good]] = p1[good]
        self.prev = small
        a = self.alive
        if a.sum() < 8:
            return 1.0, float(a.sum()) / self.n0
        d = np.linalg.norm(self.p[a] - self.p0[a], axis=1)
        return float(np.median(d)) / self.long, float(a.sum()) / self.n0


class KeyframeSelector:
    """Streaming keyframe selection.

    One keyframe per motion window: a window closes when the median KLT
    parallax since the last keyframe exceeds `D` (fraction of image width),
    too many tracks are lost, or too much time passed.  From the late part of
    the window the sharpest frame (variance of Laplacian) is kept.  `D` adapts
    online so the total count stays within [min_kf, max_kf].
    """

    def __init__(self, w, h, fps, total_frames, out_dir, max_kf, spacing, color_M, jpeg_quality=95,
                 writer_threads=2, min_kf=60, t_max=5.0):
        self.w, self.h, self.fps = w, h, fps
        self.total = total_frames if total_frames and total_frames > 0 else 10 ** 9
        self.out_dir = ensure_dir(out_dir)
        self.max_kf, self.min_kf = max_kf, min_kf
        self.D = float(spacing)
        self.D_min, self.D_max = 0.03, 0.40
        self.t_max, self.t_fallback = t_max, 1.0
        self.M = color_M
        self.q = jpeg_quality
        s = 480.0 / max(w, h)
        self.aw, self.ah = even(w * s), even(h * s)
        self.klt = _KLT(max(self.aw, self.ah))
        self.window = []
        self.startup = []
        self.kf = None
        self.kfs = []
        self.pool = ThreadPoolExecutor(max_workers=max(1, writer_threads))
        self.futs = []
        self.n_cand = 0
        self.n_skipped = 0

    # -- analysis -------------------------------------------------------------
    def _analyse(self, n, buf):
        import cv2
        y = np.frombuffer(buf, np.uint8, count=self.w * self.h).reshape(self.h, self.w)
        small = cv2.resize(y, (self.aw, self.ah), interpolation=cv2.INTER_AREA)
        mean, std = float(small.mean()), float(small.std())
        if mean < 10 or mean > 245 or std < 3.0:
            return None
        half = cv2.resize(y, (self.w // 2, self.h // 2), interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(half, cv2.CV_16S, ksize=3)
        _, sd = cv2.meanStdDev(lap)
        return _Cand(n, float(sd[0, 0]) ** 2, small, buf)

    def push(self, n, buf):
        self.n_cand += 1
        c = self._analyse(n, buf)
        if c is None:
            self.n_skipped += 1
            return
        if self.kf is None:
            self.startup.append(c)
            if len(self.startup) >= 3 or (c.n - self.startup[0].n) / self.fps >= 0.5:
                best = max(self.startup, key=lambda k: k.sharp)
                self._commit(best, self.startup[-1])
                self.startup = []
            return
        d, s = self.klt.update(c.small)
        c.disp, c.surv = (d if d is not None else 0.0), s
        self.window.append(c)
        self._maybe_commit()

    def _trigger(self):
        c = self.window[-1]
        dt = (c.n - self.kf.n) / self.fps
        if not self.klt.ok:
            return dt >= self.t_fallback
        return c.disp >= self.D or c.surv < 0.45 or dt >= self.t_max

    def _maybe_commit(self):
        for _ in range(3):
            if not self.window or not self._trigger():
                return
            cur = self.window[-1]
            if self.klt.ok:
                elig = [k for k in self.window if k.disp >= 0.5 * self.D]
            else:
                elig = list(self.window)
            if not elig:
                elig = [cur]
            best = max(elig, key=lambda k: k.sharp * (1.0 + 0.15 * min(1.0, k.disp / max(1e-6, self.D))))
            self._commit(best, cur)

    def _commit(self, best, cur):
        self._emit(best)
        self.kf = best
        if best is cur:
            self.klt.reset(cur.small)
            self.window = []
        else:
            self.klt.reset(best.small)
            d, s = self.klt.update(cur.small)
            cur.disp, cur.surv = (d if d is not None else 0.0), s
            self.window = [cur]
        self._adapt()

    def _emit(self, c):
        k = len(self.kfs)
        name = f"frame_{k:05d}.jpg"
        meta = dict(index=k, frame_number=int(c.n), timestamp_sec=round(c.n / self.fps, 3), filename=name,
                    blur_score=round(c.sharp, 1), parallax=round(float(c.disp), 4))
        self.kfs.append(meta)
        data = c.data
        c.data = None
        self.futs.append(self.pool.submit(self._write, os.path.join(self.out_dir, name), data))
        if len(self.futs) > 64:
            pending = []
            for f in self.futs:
                if f.done():
                    f.result()          # surface write errors early
                else:
                    pending.append(f)
            self.futs = pending

    def _write(self, path, data):
        import cv2
        bgr = nv12_to_bgr(data, self.w, self.h, self.M)
        if not cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self.q]):
            raise IOError(f"failed to write {path}")

    def _adapt(self):
        prog = self.kf.n / self.total
        if prog < 0.06:
            return
        proj = len(self.kfs) / prog
        if proj > self.max_kf * 1.02 and self.D < self.D_max:
            self.D = min(self.D_max, self.D * min(1.25, (proj / self.max_kf) ** 0.5))
        elif proj < self.min_kf and self.D > self.D_min:
            self.D = max(self.D_min, self.D / 1.15)

    def finish(self):
        if self.kf is None and self.startup:
            best = max(self.startup, key=lambda k: k.sharp)
            self._emit(best)
            self.kf = best
        elif self.window:
            late = [k for k in self.window if k.disp >= 0.35 * self.D] if self.klt.ok else list(self.window)
            if late:
                self._emit(max(late, key=lambda k: k.sharp))
        for f in self.futs:
            f.result()
        self.pool.shutdown(wait=True)
        return self.kfs


def _nvenc_args(fps):
    g = str(int(round(2 * fps)))
    return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "25", "-b:v", "0",
            "-maxrate", "12M", "-bufsize", "24M", "-profile:v", "high", "-g", g, "-bf", "2"]


def _x264_args(fps):
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-g", str(int(round(2 * fps)))]


def build_ingest_cmd(vinfo, variant, kf_wh, step, web=None, t_limit=None):
    """variant: cuda_full | cuda_dl | nvdec_sw | cpu.  web: dict(path, wh) or None (NVENC inline)."""
    kw, kh = kf_wh
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-y"]
    cmd += ["-loglevel", "info"] if variant == "cpu_keyframes" else ["-loglevel", "error"]
    if variant in ("cuda_full", "cuda_dl"):
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "24"]
    elif variant == "nvdec_sw":
        cmd += ["-hwaccel", "cuda"]
    elif variant == "cpu_keyframes":
        cmd += ["-skip_frame", "nokey", "-threads", "0"]   # decode I-frames only (non-NVDEC codecs, e.g. 10-bit H.264)
    else:
        cmd += ["-threads", "0"]
    if t_limit:
        cmd += ["-t", str(t_limit)]
    cmd += ["-i", vinfo["path"]]
    fs = f"framestep={step}" if step > 1 else "null"
    g = []
    if variant in ("cuda_full", "cuda_dl"):
        sk = f"scale_cuda=w={kw}:h={kh}:format=nv12:interp_algo=bicubic"
        if variant == "cuda_full":
            kf_chain = f"{fs},{sk},hwdownload,format=nv12"
        else:
            kf_chain = f"{sk},hwdownload,format=nv12,{fs}"
        if web:
            ww, wh = web["wh"]
            if variant == "cuda_full":
                g.append(f"[0:v]split=2[a][b];[a]{kf_chain}[kf];[b]scale_cuda=w={ww}:h={wh}:format=nv12:interp_algo=bicubic[web]")
            else:
                g.append(f"[0:v]scale_cuda=w={kw}:h={kh}:format=nv12:interp_algo=bicubic,split=2[a][b];"
                         f"[a]hwdownload,format=nv12,{fs}[kf];[b]scale_cuda=w={ww}:h={wh}:format=nv12[web]")
        else:
            g.append(f"[0:v]{kf_chain}[kf]")
    elif variant == "cpu_keyframes":
        g.append(f"[0:v]showinfo,scale={kw}:{kh}:flags=area,format=nv12[kf]")
        web = None
    else:
        kf_chain = f"{fs},scale={kw}:{kh}:flags=area,format=nv12"
        if web:
            ww, wh = web["wh"]
            g.append(f"[0:v]split=2[a][b];[a]{kf_chain}[kf];[b]scale={ww}:{wh}:flags=bilinear,format=nv12[web]")
        else:
            g.append(f"[0:v]{kf_chain}[kf]")
    cmd += ["-filter_complex", ";".join(g)]
    cmd += ["-map", "[kf]", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
    if web:
        cmd += ["-map", "[web]"]
        if vinfo.get("has_audio"):
            cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
        cmd += _nvenc_args(vinfo["fps"]) + ["-movflags", "+faststart", web["path"]]
    return cmd


_PTS_RE = re.compile(r"Parsed_showinfo.*?\bn:\s*(\d+).*?pts_time:\s*(-?[\d.]+)")


def _drain(stream, sink, maxlen=200, times=None):
    for line in iter(stream.readline, b""):
        try:
            txt = line.decode("utf-8", "ignore").rstrip()
        except Exception:
            continue
        if times is not None:
            m = _PTS_RE.search(txt)
            if m:
                times.append(float(m.group(2)))
                continue
        sink.append(txt)
        if len(sink) > maxlen:
            sink.popleft()


def _probe_ingest(vinfo, variant, kf_wh, step, web, tmpdir):
    web_probe = None
    if web:
        web_probe = dict(path=os.path.join(tmpdir, "probe_web.mp4"), wh=web["wh"])
    cmd = build_ingest_cmd(vinfo, variant, kf_wh, step, web_probe, t_limit=1.5)
    frame_bytes = kf_wh[0] * kf_wh[1] * 3 // 2
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    ok = p.returncode == 0 and len(p.stdout) >= frame_bytes
    if ok and web_probe:
        ok = os.path.exists(web_probe["path"]) and os.path.getsize(web_probe["path"]) > 1000
    return ok, p.stderr.decode("utf-8", "ignore")[-400:]


def detect_nvenc():
    rc, _, _ = run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=s=640x360:d=0.3", "-c:v", "h264_nvenc", "-f", "null", "-"], timeout=60)
    return rc == 0


def ingest_video(cfg, prof, vinfo, dirs):
    """Returns (keyframes metadata, ingest info dict)."""
    kf_wh = fit_long_side(vinfo["width"], vinfo["height"], prof["kf_long"])
    web_wh = fit_long_side(vinfo["width"], vinfo["height"], cfg.web_video_long_side)
    fps = vinfo["fps"]
    cand_fps = min(prof["cand_fps"], fps)
    step = max(1, int(round(fps / cand_fps)))
    web_path = os.path.join(dirs["work"], "drone_flight.mp4")
    M = yuv_to_bgr_matrix(*color_matrix_for(vinfo))
    nvenc = cfg.web_video and detect_nvenc()
    copy_web = (cfg.web_video and vinfo["codec"] == "h264" and max(vinfo["width"], vinfo["height"]) <= 1920
                and vinfo["rotation"] == 0 and (vinfo["bit_rate"] == 0 or vinfo["bit_rate"] < 20e6))
    # full CPU decode of long 4K clips on 2 vCPUs is far too slow -> decode I-frames only in that case
    cpu_decode_s = vinfo["frames"] * (vinfo["width"] * vinfo["height"] / 2.07e6) / (55.0 * max(1, prof.get("cpu", 2)) / 2)
    cpu_order = ["cpu_keyframes", "cpu"] if cpu_decode_s > 0.2 * prof["target_s"] else ["cpu", "cpu_keyframes"]
    variants = ["cuda_full", "cuda_dl", "nvdec_sw"] + cpu_order
    if vinfo["rotation"]:
        variants = ["nvdec_sw"] + cpu_order  # let ffmpeg auto-rotate software frames
    chosen, web_inline = None, None
    for v in variants:
        for want_web in ([True, False] if (nvenc and not copy_web and v != "cpu_keyframes") else [False]):
            web = dict(path=web_path, wh=web_wh) if want_web else None
            ok, err = _probe_ingest(vinfo, v, kf_wh, step, web, dirs["tmp"])
            if ok:
                chosen, web_inline = v, web
                break
            log(f"ingest probe {v}{' +nvenc' if want_web else ''} failed: {err.strip()[-200:]}")
        if chosen:
            break
    if not chosen:
        raise RuntimeError("ffmpeg could not decode the video with any backend")
    log(f"ingest backend: {chosen}{' + inline NVENC web video' if web_inline else ''}; keyframes at "
        f"{kf_wh[0]}x{kf_wh[1]}, candidates every {step} frames ({fps / step:.1f}/s)")

    cmd = build_ingest_cmd(vinfo, chosen, kf_wh, step, web_inline)
    sel = KeyframeSelector(kf_wh[0], kf_wh[1], fps, vinfo["frames"], dirs["images"], prof["max_kf"],
                           cfg.kf_spacing, M, writer_threads=2 if prof.get("cpu", 2) <= 4 else 4)
    frame_bytes = kf_wh[0] * kf_wh[1] * 3 // 2
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    errq = deque()
    ptimes = deque() if chosen == "cpu_keyframes" else None
    th = threading.Thread(target=_drain, args=(proc.stderr, errq), kwargs=dict(times=ptimes), daemon=True)
    th.start()
    k = 0
    next_report = 0.1
    if ptimes is not None:
        warn("NVDEC cannot decode this stream; decoding I-frames only on the CPU (fewer keyframe candidates)")
    try:
        while True:
            buf = bytearray(frame_bytes)
            mv = memoryview(buf)
            got = 0
            while got < frame_bytes:
                r = proc.stdout.readinto(mv[got:])
                if not r:
                    break
                got += r
            if got < frame_bytes:
                break
            if ptimes is None:
                n = k * step
            else:
                t_wait = time.time()
                while len(ptimes) <= k and time.time() - t_wait < 3.0 and th.is_alive():
                    time.sleep(0.002)
                n = int(round(ptimes[k] * fps)) if len(ptimes) > k else (k * step)
            sel.push(n, buf)
            k += 1
            prog = n / max(1, vinfo["frames"])
            if prog >= next_report:
                el = time.time() - t0
                log(f"  ingest {100 * prog:5.1f}%  {n / max(el, 1e-3):6.0f} fps  keyframes={len(sel.kfs)}  "
                    f"spacing={sel.D:.3f}  eta {fmt_s(el / max(prog, 1e-3) - el)}")
                next_report += 0.1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
        th.join(timeout=5)
    kfs = sel.finish()
    el = time.time() - t0
    if rc != 0 and k < 10:
        raise RuntimeError("ffmpeg ingest failed: " + " | ".join(list(errq)[-5:]))
    if rc != 0:
        warn(f"ffmpeg exited with code {rc} after {k} candidates (continuing): {' | '.join(list(errq)[-3:])}")
    decoded = (kfs[-1]["frame_number"] if kfs else 0) if ptimes is not None else k * step
    info = dict(backend=chosen, kf_size=list(kf_wh), candidates=k, skipped=sel.n_skipped, keyframes=len(kfs),
                decode_fps=round(decoded / max(el, 1e-3), 1), spacing_final=round(sel.D, 4),
                web_video=web_path if web_inline else None, web_mode="nvenc_inline" if web_inline else None,
                copy_web=copy_web, nvenc=nvenc, seconds=round(el, 1))
    log(f"ingest: {k} candidates -> {len(kfs)} keyframes in {fmt_s(el)} "
        f"({info['decode_fps']:.0f} decoded fps)")
    return kfs, info


def start_web_video_job(cfg, vinfo, info, work):
    """Background web-video encode when NVENC could not run inline."""
    if not cfg.web_video:
        return None
    out = os.path.join(work, "drone_flight.mp4")
    if info.get("web_video"):
        return None
    if info.get("copy_web"):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", vinfo["path"], "-map", "0:v:0",
               "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", out]
    else:
        ww, wh = fit_long_side(vinfo["width"], vinfo["height"], cfg.web_video_long_side)
        dec = ["-hwaccel", "cuda"] if info.get("backend") in ("cuda_full", "cuda_dl", "nvdec_sw") else ["-threads", "0"]
        enc = _nvenc_args(vinfo["fps"]) if info.get("nvenc") else _x264_args(vinfo["fps"])
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + dec + ["-i", vinfo["path"], "-map", "0:v:0",
                                                                             "-map", "0:a:0?", "-vf", f"scale={ww}:{wh}"] + enc + \
              ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out]
    errlog = open(os.path.join(ensure_dir(os.path.join(work, "logs")), "web_video.log"), "w")
    try:
        proc = subprocess.Popen(["nice", "-n", "10"] + cmd, stdout=subprocess.DEVNULL, stderr=errlog)
    except FileNotFoundError:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errlog)
    log("web video: background encode started")
    return dict(proc=proc, path=out, t0=time.time(), log=errlog)


def finish_web_video_job(job, timeout):
    if job is None:
        return None
    try:
        job["proc"].wait(timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        warn("web video encode did not finish in time; bundle will not contain drone_flight.mp4")
        job["proc"].kill()
        return None
    try:
        job["log"].close()
    except Exception:
        pass
    if job["proc"].returncode == 0 and os.path.exists(job["path"]):
        return job["path"]
    warn("web video encode failed")
    return None


# -----------------------------------------------------------------------------
# sky masks for oblique video
# -----------------------------------------------------------------------------
def sky_mask(bgr):
    """
    255 = use, 0 = sky. Sky = bright, smooth, blue-or-grey region connected to the top border; the
    horizon is taken per column (first non-sky pixel from the top) and median-smoothed. Returns None
    when the frame shows no horizon (nadir / steep views): nothing is masked then.
    """
    import cv2
    h, w = bgr.shape[:2]
    s = 480.0 / max(h, w)
    small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.GaussianBlur(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)), (7, 7), 0)
    skyish = (V > 0.45) & (grad < 18.0) & ((S < 0.30) | ((H >= 88) & (H <= 132) & (S < 0.75)))
    sh, sw = skyish.shape
    top = skyish[: max(2, sh // 30)].mean()
    if top < 0.7:
        return None
    n, lab = cv2.connectedComponents(skyish.astype(np.uint8))
    top_labels = set(np.unique(lab[0][skyish[0]])) - {0}
    region = np.isin(lab, list(top_labels))
    profile = np.argmin(np.vstack([region, np.zeros((1, sw), bool)]), axis=0)     # first non-sky row
    # a drone horizon is a straight line: robust line fit through the columns where the sky reaches it
    # (textured cloud edges stop the per-column profile early; they become outliers here)
    xs = np.arange(sw, dtype=np.float64)
    ok = profile > 0.02 * sh
    if ok.sum() < 0.3 * sw:
        return None
    a, b = np.polyfit(xs[ok], profile[ok], 1)
    for _ in range(4):
        res = profile - (a * xs + b)
        inl = ok & (np.abs(res) <= max(3.0, 2.5 * 1.4826 * np.median(np.abs(res[ok]))))
        if inl.sum() < 0.2 * sw:
            break
        a, b = np.polyfit(xs[inl], profile[inl], 1)
    line = a * xs + b
    if line.min() < 0 or line.max() > 0.7 * sh or abs(a) > 0.35:
        return None
    horizon = np.maximum(0, line - 3)                                               # keep the tree line
    rows = np.arange(sh)[:, None]
    keep_small = (rows >= horizon[None, :]).astype(np.uint8) * 255
    return cv2.resize(keep_small, (w, h), interpolation=cv2.INTER_NEAREST)


def write_sky_masks(image_dir, names, mask_dir):
    """COLMAP convention: mask of 'frame_00001.jpg' is 'frame_00001.jpg.png'. Returns share of masked frames."""
    import cv2
    ensure_dir(mask_dir)
    masked = 0
    for n in names:
        img = cv2.imread(os.path.join(image_dir, n))
        if img is None:
            continue
        m = sky_mask(img)
        if m is None:
            m = np.full(img.shape[:2], 255, np.uint8)
        else:
            masked += 1
        cv2.imwrite(os.path.join(mask_dir, n + ".png"), m)
    return masked / max(1, len(names))


# -----------------------------------------------------------------------------
# worker processes (all pycolmap work happens in children)
# -----------------------------------------------------------------------------
class WorkerError(RuntimeError):
    pass


_ECHO_RE = re.compile(r"(error|exception|traceback|fatal|check failed|out of memory|killed|segmentation)", re.I)


def run_worker(name, args, dirs, timeout=None, heartbeat=45.0, verbose=False):
    logdir = ensure_dir(os.path.join(dirs["work"], "logs"))
    args = dict(args)
    args["result_path"] = os.path.join(logdir, f"{name}_result.json")
    args_path = os.path.join(logdir, f"{name}_args.json")
    log_path = os.path.join(logdir, f"{name}.log")
    json_dump(args, args_path, indent=1)
    if os.path.exists(args["result_path"]):
        os.remove(args["result_path"])
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("GLOG_minloglevel", "0" if verbose else "1")
    proc = subprocess.Popen([sys.executable, ENGINE_PATH, "worker", name, args_path], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, bufsize=1, text=True, errors="replace")
    tail = deque(maxlen=60)

    def pump():
        with open(log_path, "w") as lf:
            for line in proc.stdout:
                lf.write(line)
                line = line.rstrip()
                tail.append(line)
                if line.startswith("[PRISM") or verbose or (_ECHO_RE.search(line) and "WARNING: Logging before" not in line):
                    print(line if line.startswith("[PRISM") else f"    | {line}", flush=True)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    t0 = time.time()
    last = t0
    while True:
        try:
            proc.wait(timeout=5.0)
            break
        except subprocess.TimeoutExpired:
            now = time.time()
            if timeout and now - t0 > timeout:
                proc.kill()
                th.join(timeout=5)
                raise WorkerError(f"{name} timed out after {fmt_s(now - t0)}")
            if now - last > heartbeat:
                log(f"  ... {name} running ({fmt_s(now - t0)})")
                last = now
    th.join(timeout=10)
    if proc.returncode != 0 or not os.path.exists(args["result_path"]):
        raise WorkerError(f"{name} failed (exit {proc.returncode}). Last log lines:\n    " + "\n    ".join(list(tail)[-15:]))
    with open(args["result_path"]) as fh:
        return json.load(fh)


def _import_pycolmap(verbose=False):
    import pycolmap
    try:
        pycolmap.logging.minloglevel = 0 if verbose else 1
        pycolmap.logging.logtostderr = True
    except Exception:
        pass
    return pycolmap


def _set(obj, name, value):
    """setattr that tolerates API drift (logs instead of crashing)."""
    try:
        setattr(obj, name, value)
        return True
    except Exception as e:
        log(f"  (option {type(obj).__name__}.{name} not set: {e})")
        return False


def _device(pycolmap, want_gpu=True):
    if want_gpu and getattr(pycolmap, "has_cuda", False):
        return pycolmap.Device.cuda
    return pycolmap.Device.cpu


# ---- SfM worker --------------------------------------------------------------
def worker_sfm(a):
    pycolmap = _import_pycolmap(a.get("verbose", False))
    db, img_dir, out_dir = a["database_path"], a["image_dir"], a["output_dir"]
    names = a["image_names"]
    dev = _device(pycolmap)
    gpu = dev == pycolmap.Device.cuda
    log(f"SfM worker: pycolmap {pycolmap.__version__}, cuda={gpu}, images={len(names)}")
    timings = {}
    if os.path.exists(db):
        os.remove(db)
    # 1) features --------------------------------------------------------------
    t = time.time()
    ro = pycolmap.ImageReaderOptions()
    _set(ro, "camera_model", a["camera_model"])
    if a.get("mask_path"):
        _set(ro, "mask_path", a["mask_path"])
    if a.get("camera_params"):
        _set(ro, "camera_params", a["camera_params"])
    else:
        _set(ro, "default_focal_length_factor", float(a["focal_factor"]))
    eo = pycolmap.FeatureExtractionOptions()
    _set(eo, "max_image_size", int(a["sift_max_size"]))
    _set(eo, "num_threads", -1)
    sift = eo.sift
    _set(sift, "max_num_features", int(a["sift_feats"]))
    _set(sift, "first_octave", int(a.get("first_octave", 0)))
    try:
        eo.sift = sift
    except Exception:
        pass
    pycolmap.extract_features(db, img_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, extraction_options=eo, device=dev)
    timings["features"] = time.time() - t
    log(f"  features: {fmt_s(timings['features'])}")
    # 2) matching ---------------------------------------------------------------
    t = time.time()
    mo = pycolmap.FeatureMatchingOptions()
    _set(mo, "max_num_matches", 16384)
    _set(mo, "num_threads", -1)
    vo = pycolmap.TwoViewGeometryOptions()
    po = pycolmap.SequentialPairingOptions()
    _set(po, "overlap", int(a["seq_overlap"]))
    _set(po, "quadratic_overlap", True)
    loop = bool(a.get("vocab_tree_path"))
    if loop:
        _set(po, "loop_detection", True)
        _set(po, "loop_detection_period", int(a.get("loop_period", 10)))
        _set(po, "loop_detection_num_images", int(a["loop_imgs"]))
        _set(po, "loop_detection_min_index_distance", int(a.get("loop_min_dist", 12)))
        _set(po, "vocab_tree_path", a["vocab_tree_path"])
    try:
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=po, verification_options=vo, device=dev)
    except Exception as e:
        if not loop:
            raise
        log(f"  loop detection failed ({e}); retrying sequential matching without it")
        _set(po, "loop_detection", False)
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=po, verification_options=vo, device=dev)
    if a.get("pairs_path") and os.path.exists(a["pairs_path"]):
        ip = pycolmap.ImportedPairingOptions()
        _set(ip, "match_list_path", a["pairs_path"])
        pycolmap.match_image_pairs(db, matching_options=mo, pairing_options=ip, verification_options=vo, device=dev)
    timings["matching"] = time.time() - t
    log(f"  matching: {fmt_s(timings['matching'])}")
    # 3) mapping ----------------------------------------------------------------
    t = time.time()
    best, mapper = None, None
    try:
        try:
            ok = pycolmap.calibrate_view_graph(db, pycolmap.ViewGraphCalibrationOptions())
            log(f"  view-graph focal calibration: {'ok' if ok else 'not converged'}")
        except Exception as e:
            log(f"  view-graph calibration skipped: {e}")
        go = pycolmap.GlobalPipelineOptions()
        _set(go, "random_seed", 0)
        _set(go, "num_threads", -1)
        _set(go, "multiple_models", True)
        _set(go, "min_model_size", 3)
        gm = go.mapper
        _set(gm, "random_seed", 0)
        _set(gm, "keep_max_num_tracks", int(a["keep_tracks"]))
        _set(gm, "ba_num_iterations", int(a.get("ba_rounds", 3)))
        try:
            gm.bundle_adjustment.ceres.solver_options.max_num_iterations = int(a.get("ba_iters", 60))
        except Exception as e:
            log(f"  (BA iteration cap not applied: {e})")
        go.mapper = gm
        gdir = ensure_dir(os.path.join(out_dir, "global"))
        recs = pycolmap.global_mapping(db, img_dir, gdir, go)
        if recs:
            best = max(recs.values(), key=lambda r: r.num_reg_images())
            mapper = "global"
            log(f"  global mapper: {len(recs)} model(s), best registers {best.num_reg_images()}/{len(names)}")
    except Exception as e:
        log(f"  global mapper failed: {e}")
    timings["global_mapping"] = time.time() - t
    budget = float(a.get("incremental_budget_s", 300))
    if (best is None or best.num_reg_images() < 0.6 * len(names)) and budget > 45:
        t = time.time()
        log(f"  running incremental mapper (budget {fmt_s(budget)})")
        io = pycolmap.IncrementalPipelineOptions()
        _set(io, "max_runtime_seconds", int(budget))
        _set(io, "random_seed", 0)
        _set(io, "num_threads", -1)
        _set(io, "min_model_size", 10)
        _set(io, "ba_global_max_num_iterations", 30)
        _set(io, "ba_local_max_num_iterations", 15)
        _set(io, "ba_global_frames_ratio", 1.4)
        _set(io, "ba_global_points_ratio", 1.4)
        idir = ensure_dir(os.path.join(out_dir, "incremental"))
        try:
            recs = pycolmap.incremental_mapping(db, img_dir, idir, io)
            if recs:
                inc = max(recs.values(), key=lambda r: r.num_reg_images())
                log(f"  incremental mapper registers {inc.num_reg_images()}/{len(names)}")
                if best is None or inc.num_reg_images() > best.num_reg_images():
                    best, mapper = inc, "incremental"
        except Exception as e:
            log(f"  incremental mapper failed: {e}")
        timings["incremental_mapping"] = time.time() - t
    if best is None or best.num_reg_images() < 3:
        raise RuntimeError("Structure-from-Motion could not register the video frames")
    model_dir = ensure_dir(os.path.join(out_dir, "model"))
    best.write(model_dir)
    try:
        best.export_PLY(os.path.join(out_dir, "sparse.ply"))
    except Exception:
        pass
    summary = _export_sfm_summary(best, os.path.join(out_dir, "summary.npz"))
    res = dict(mapper=mapper, num_images=len(names), num_registered=int(best.num_reg_images()),
               num_points=int(best.num_points3D()), timings=timings, model_dir=model_dir,
               summary=os.path.join(out_dir, "summary.npz"), **summary)
    try:
        res["mean_reproj_error"] = float(best.compute_mean_reprojection_error())
        res["mean_track_length"] = float(best.compute_mean_track_length())
    except Exception:
        pass
    return res


def _export_sfm_summary(rec, path):
    reg = sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name)
    idx = {iid: k for k, iid in enumerate(reg)}
    names, R, T, K, wh, cam_params = [], [], [], [], [], []
    for iid in reg:
        im = rec.images[iid]
        m = np.asarray(im.cam_from_world().matrix(), dtype=np.float64)
        cam = rec.cameras[im.camera_id]
        names.append(im.name)
        R.append(m[:, :3])
        T.append(m[:, 3])
        K.append(np.asarray(cam.calibration_matrix(), dtype=np.float64))
        wh.append((int(cam.width), int(cam.height)))
        cam_params.append(np.asarray(cam.params, dtype=np.float64))
    xyz, rgb, err, tptr, timg = [], [], [], [0], []
    for pid, p in rec.points3D.items():
        ims = [idx[el.image_id] for el in p.track.elements if el.image_id in idx]
        if len(ims) < 2:
            continue
        ims = sorted(set(ims))
        xyz.append(np.asarray(p.xyz, dtype=np.float64))
        rgb.append(np.asarray(p.color, dtype=np.uint8))
        err.append(float(p.error))
        timg.extend(ims)
        tptr.append(len(timg))
    R = np.asarray(R).reshape(-1, 3, 3)
    T = np.asarray(T).reshape(-1, 3)
    C = -np.einsum("nji,nj->ni", R, T)
    cam0 = rec.cameras[rec.images[reg[0]].camera_id]
    model_name = getattr(cam0.model, "name", str(cam0.model)).split(".")[-1]
    np.savez_compressed(path, names=np.asarray(names), R=R, T=T, C=C, K=np.asarray(K), wh=np.asarray(wh),
                        cam_params=np.asarray(cam_params), cam_model=model_name,
                        xyz=np.asarray(xyz).reshape(-1, 3), rgb=np.asarray(rgb, np.uint8).reshape(-1, 3),
                        err=np.asarray(err), track_ptr=np.asarray(tptr, np.int64), track_img=np.asarray(timg, np.int32))
    return dict(camera_model=model_name, camera_params=[float(v) for v in np.asarray(cam0.params)],
                camera_size=[int(cam0.width), int(cam0.height)])


# ---- MVS worker --------------------------------------------------------------
def worker_mvs(a):
    pycolmap = _import_pycolmap(a.get("verbose", False))
    if not getattr(pycolmap, "has_cuda", False):
        raise RuntimeError("PatchMatch stereo needs pycolmap built with CUDA (pycolmap-cuda12)")
    dense, sparse, images = a["dense_dir"], a["sparse_dir"], a["image_dir"]
    with open(a["plan_path"]) as fh:
        plan = json.load(fh)
    refs = plan["refs"]
    deadline = float(a["deadline"])
    K = int(plan["num_sources"])
    stereo = os.path.join(dense, "stereo")
    res = dict(photo_done=[], geo_done=[], chunks=[])
    # 1) undistortion ----------------------------------------------------------
    t = time.time()
    need = plan["undistort"]
    uo = pycolmap.UndistortCameraOptions()
    _set(uo, "max_image_size", int(a["mvs_size"]))
    pycolmap.undistort_images(dense, sparse, images, image_names=need, output_type="COLMAP",
                              undistort_options=uo, jpeg_quality=95, num_threads=-1)
    res["undistort_s"] = time.time() - t
    have = {os.path.relpath(p, os.path.join(dense, "images")) for p in glob.glob(os.path.join(dense, "images", "*"))}
    log(f"  undistorted {len(have)} images in {fmt_s(res['undistort_s'])}")

    def opts(geom, filt, iters):
        o = pycolmap.PatchMatchOptions()
        _set(o, "max_image_size", -1)
        _set(o, "gpu_index", str(a.get("gpu_index", "-1")))
        _set(o, "window_radius", int(a["win_radius"]))
        _set(o, "window_step", 2)
        _set(o, "num_samples", int(a["num_samples"]))
        _set(o, "num_iterations", int(iters))
        _set(o, "geom_consistency", bool(geom))
        _set(o, "filter", bool(filt))
        _set(o, "allow_missing_files", True)
        _set(o, "cache_size", float(a["cache_gb"]))
        _set(o, "num_threads", -1)
        return o

    def srcs_for(ref, pool, k):
        out = [c for c, _s in ref["cands"] if c in pool and c != ref["name"]]
        return out[:k]

    def write_cfg(path, items):
        with open(path, "w") as fh:
            for name, srcs in items:
                fh.write(name + "\n" + ", ".join(srcs) + "\n")

    def exists(name, kind):
        return (os.path.exists(os.path.join(stereo, "depth_maps", f"{name}.{kind}.bin")) and
                os.path.exists(os.path.join(stereo, "normal_maps", f"{name}.{kind}.bin")))

    geom = bool(a.get("geom_consistency", True))
    refs = [r for r in refs if r["name"] in have]
    geo_ratio = 1.1 * float(a["it_geo"]) / float(a["it_photo"])   # geometric pass is warm-started
    t_p = float(a["sec_per_ref"]) / (1.0 + (geo_ratio if geom else 0.0))
    t_g = t_p * geo_ratio
    n_active = len(refs)
    # 2) photometric pass ---------------------------------------------------------
    done_p = [r["name"] for r in refs if exists(r["name"], "photometric")]
    k = len(done_p)
    pos = 0
    chunk_n = 8
    fails = 0
    while pos < len(refs):
        now = time.time()
        R = deadline - now
        t_g = t_p * geo_ratio
        n_aff = int((R + k * t_p) / (t_p + (t_g if geom else 0.0)))
        n_active = min(len(refs), max(k, n_aff))
        if k >= n_active or R < 3 * t_p:
            break
        chunk = [r for r in refs[pos:pos + chunk_n] if not exists(r["name"], "photometric")]
        pos += chunk_n
        if not chunk:
            continue
        chunk = chunk[:max(1, n_active - k)]
        items = [(r["name"], srcs_for(r, have, K)) for r in chunk]
        items = [it for it in items if len(it[1]) >= 2]
        if not items:
            continue
        cfg = os.path.join(stereo, f"pm_photo_{pos:05d}.cfg")
        write_cfg(cfg, items)
        t = time.time()
        try:
            pycolmap.patch_match_stereo(dense, workspace_format="COLMAP", pmvs_option_name="option-all",
                                        options=opts(False, not geom, a["it_photo"]), config_path=cfg)
            fails = 0
        except Exception as e:
            fails += 1
            log(f"  PatchMatch chunk failed ({str(e).splitlines()[0][:160]})")
            if fails >= 2:
                break
        dt = time.time() - t
        ok = [n for n, _ in items if exists(n, "photometric")]
        k += len(ok)
        done_p.extend(ok)
        if ok:
            t_p = 0.5 * t_p + 0.5 * dt / len(items) if len(res["chunks"]) else dt / len(items)
        res["chunks"].append(dict(phase="photometric", n=len(items), ok=len(ok), s=round(dt, 1)))
        log(f"  MVS photometric {k}/{n_active} refs ({dt / max(1, len(items)):.2f} s/ref), "
            f"{fmt_s(deadline - time.time())} left")
        chunk_n = int(np.clip(round(45.0 / max(t_p, 0.05)), 8, 48))
    res["photo_done"] = done_p
    res["t_photo_per_ref"] = t_p
    # 3) geometric pass -----------------------------------------------------------
    if geom and done_p:
        pool = set(done_p)
        order = [r for r in refs if r["name"] in pool]
        done_g = [r["name"] for r in order if exists(r["name"], "geometric")]
        todo = [r for r in order if r["name"] not in set(done_g)]
        chunk_n = int(np.clip(round(45.0 / max(t_g, 0.05)), 8, 48))
        first = True
        gfails = 0
        while todo:
            R = deadline - time.time()
            n_fit = int(R / max(t_g, 0.05))
            if n_fit < 2:
                break
            chunk, todo = todo[:min(chunk_n, n_fit)], todo[min(chunk_n, n_fit):]
            items = [(r["name"], srcs_for(r, pool, K)) for r in chunk]
            items = [it for it in items if len(it[1]) >= 2]
            if not items:
                continue
            cfg = os.path.join(stereo, f"pm_geo_{len(res['chunks']):05d}.cfg")
            write_cfg(cfg, items)
            t = time.time()
            try:
                pycolmap.patch_match_stereo(dense, workspace_format="COLMAP", pmvs_option_name="option-all",
                                            options=opts(True, True, a["it_geo"]), config_path=cfg)
                gfails = 0
            except Exception as e:
                gfails += 1
                log(f"  PatchMatch chunk failed ({str(e).splitlines()[0][:160]})")
                if gfails >= 2:
                    break
            dt = time.time() - t
            ok = [n for n, _ in items if exists(n, "geometric")]
            done_g.extend(ok)
            if ok:
                t_g = dt / len(items) if first else 0.5 * t_g + 0.5 * dt / len(items)
                first = False
            res["chunks"].append(dict(phase="geometric", n=len(items), ok=len(ok), s=round(dt, 1)))
            log(f"  MVS geometric {len(done_g)}/{len(order)} refs ({dt / max(1, len(items)):.2f} s/ref), "
                f"{fmt_s(deadline - time.time())} left")
        res["geo_done"] = done_g
        res["t_geo_per_ref"] = t_g
    res["input_type"] = "geometric" if geom and res["geo_done"] else "photometric"
    fused_names = res["geo_done"] if res["input_type"] == "geometric" else res["photo_done"]
    res["fusion_images"] = fused_names
    return res


# ---- fusion / meshing workers ------------------------------------------------
def worker_fusion(a):
    pycolmap = _import_pycolmap(a.get("verbose", False))
    dense = a["dense_dir"]
    with open(os.path.join(dense, "stereo", "fusion.cfg"), "w") as fh:
        fh.write("\n".join(a["image_names"]) + "\n")
    fo = pycolmap.StereoFusionOptions()
    _set(fo, "max_image_size", int(a["fusion_size"]))
    _set(fo, "min_num_pixels", int(a["min_num_pixels"]))
    # slightly more tolerant than COLMAP's defaults: geometric-consistency filtering already removed the
    # outliers, and low-texture ground (roads, fields, roofs) otherwise fuses too few points -> gaps
    _set(fo, "max_reproj_error", 2.0)
    _set(fo, "max_depth_error", 0.012)
    _set(fo, "max_normal_error", 12.0)
    _set(fo, "check_num_images", 50)
    _set(fo, "num_threads", -1)
    _set(fo, "use_cache", False)
    t = time.time()
    pycolmap.stereo_fusion(a["output_path"], dense, workspace_format="COLMAP", pmvs_option_name="option-all",
                           input_type=a["input_type"], options=fo, output_type="ply")
    if not os.path.exists(a["output_path"]):
        raise RuntimeError("fusion produced no output")
    return dict(seconds=time.time() - t, bytes=os.path.getsize(a["output_path"]))


def worker_poisson(a):
    pycolmap = _import_pycolmap(a.get("verbose", False))
    po = pycolmap.PoissonMeshingOptions()
    _set(po, "depth", int(a["depth"]))
    _set(po, "point_weight", float(a.get("point_weight", 1.0)))
    _set(po, "color", True)
    _set(po, "trim", float(a.get("trim", 0.0)))
    _set(po, "num_threads", -1)
    t = time.time()
    pycolmap.poisson_meshing(a["input_path"], a["output_path"], po)
    if not os.path.exists(a["output_path"]):
        raise RuntimeError("poisson produced no output")
    return dict(seconds=time.time() - t)


def worker_simplify(a):
    pycolmap = _import_pycolmap(a.get("verbose", False))
    so = pycolmap.MeshSimplificationOptions()
    _set(so, "target_face_ratio", float(a["ratio"]))
    _set(so, "interpolate_colors", True)
    _set(so, "num_threads", -1)
    t = time.time()
    pycolmap.simplify_mesh(a["input_path"], a["output_path"], so)
    return dict(seconds=time.time() - t)


WORKERS = dict(sfm=worker_sfm, mvs=worker_mvs, fusion=worker_fusion, poisson=worker_poisson, simplify=worker_simplify)


def _worker_entry(name, args_path):
    with open(args_path) as fh:
        a = json.load(fh)
    try:
        res = WORKERS[name](a)
        json_dump(res, a["result_path"])
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(3)


# -----------------------------------------------------------------------------
# PLY I/O (numpy, binary little endian)
# -----------------------------------------------------------------------------
_PLY_T = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
          "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
          "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply(path):
    """Returns dict(vertex=structured array, faces=(F,3) int array or None)."""
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("bad PLY header")
            line = line.decode("ascii", "ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        body_off = fh.tell()
    fmt = next((h.split()[1] for h in header if h.startswith("format")), "ascii")
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt}")
    elements = []
    for h in header:
        p = h.split()
        if not p:
            continue
        if p[0] == "element":
            elements.append(dict(name=p[1], count=int(p[2]), props=[]))
        elif p[0] == "property" and elements:
            if p[1] == "list":
                elements[-1]["props"].append(("list", p[2], p[3], p[4]))
            else:
                elements[-1]["props"].append(("scalar", p[1], p[2]))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=body_off)
    off = 0
    out = dict(vertex=None, faces=None)
    for el in elements:
        if all(pr[0] == "scalar" for pr in el["props"]):
            dt = np.dtype([(pr[2], "<" + _PLY_T[pr[1]]) for pr in el["props"]])
            n = el["count"] * dt.itemsize
            arr = np.frombuffer(data[off:off + n].tobytes(), dtype=dt, count=el["count"])
            off += n
            if el["name"] == "vertex":
                out["vertex"] = arr
        else:
            # assume "list <count> <index>" + optional scalars; fast path for triangles
            props = el["props"]
            li = next(i for i, pr in enumerate(props) if pr[0] == "list")
            ct, it = _PLY_T[props[li][1]], _PLY_T[props[li][2]]
            pre = [(pr[2], "<" + _PLY_T[pr[1]]) for pr in props[:li]]
            post = [(pr[2], "<" + _PLY_T[pr[1]]) for pr in props[li + 1:] if pr[0] == "scalar"]
            if any(pr[0] == "list" for pr in props[li + 1:]):
                raise ValueError("unsupported PLY face layout")
            dt = np.dtype(pre + [("n", "<" + ct), ("v", "<" + it, (3,))] + post)
            n = el["count"] * dt.itemsize
            arr = np.frombuffer(data[off:off + n].tobytes(), dtype=dt, count=el["count"])
            if el["count"] and not np.all(arr["n"] == 3):
                raise ValueError("non-triangular faces in PLY")
            off += n
            if el["name"] == "face":
                out["faces"] = np.ascontiguousarray(arr["v"]).astype(np.int64)
    del data
    return out


def ply_xyz(v):
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def ply_rgb(v):
    names = v.dtype.names
    for r, g, b in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if r in names and g in names and b in names:
            c = np.stack([v[r], v[g], v[b]], 1)
            if c.dtype.kind == "f":
                c = c * (255.0 if c.max() <= 1.0 + 1e-6 else 1.0)
            return np.clip(c, 0, 255).astype(np.uint8)
    return None


def ply_normals(v):
    if all(k in v.dtype.names for k in ("nx", "ny", "nz")):
        return np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float32)
    return None


def write_ply(path, xyz, rgb=None, normals=None, faces=None):
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    v = np.empty(n, dtype=np.dtype(fields))
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if normals is not None:
        v["nx"], v["ny"], v["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if rgb is not None:
        v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hdr = ["ply", "format binary_little_endian 1.0", f"comment {ENGINE_VERSION}", f"element vertex {n}"]
    hdr += [f"property {'float' if t == '<f4' else 'uchar'} {nme}" for nme, t in fields]
    if faces is not None:
        hdr += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    hdr += ["end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(hdr) + "\n").encode("ascii"))
        fh.write(v.tobytes())
        if faces is not None:
            f = np.empty(len(faces), dtype=np.dtype([("n", "u1"), ("v", "<i4", (3,))]))
            f["n"] = 3
            f["v"] = faces
            fh.write(f.tobytes())


def write_obj(path, xyz, rgb=None, faces=None, chunk=200000):
    with open(path, "w") as fh:
        fh.write(f"# {ENGINE_VERSION}\n# vertices {len(xyz)} faces {0 if faces is None else len(faces)}\n")
        if rgb is not None:
            data = np.hstack([xyz.astype(np.float64), rgb.astype(np.float64) / 255.0])
            fmt = "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
        else:
            data = xyz.astype(np.float64)
            fmt = "v %.4f %.4f %.4f\n"
        for i in range(0, len(data), chunk):
            blk = data[i:i + chunk]
            fh.write((fmt * len(blk)) % tuple(blk.ravel()))
        if faces is not None:
            f1 = faces.astype(np.int64) + 1
            for i in range(0, len(f1), chunk):
                blk = f1[i:i + chunk]
                fh.write(("f %d %d %d\n" * len(blk)) % tuple(blk.ravel()))


def write_glb(path, xyz, rgb, faces, normals=None, unlit=True):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    parts, views, accessors = [], [], []
    off = 0

    def add(buf, target):
        nonlocal off
        b = buf.tobytes()
        pad = (-len(b)) % 4
        parts.append(b + b"\x00" * pad)
        views.append(dict(buffer=0, byteOffset=off, byteLength=len(b), target=target))
        off += len(b) + pad
        return len(views) - 1

    attrs = {}
    v = add(xyz, 34962)
    accessors.append(dict(bufferView=v, componentType=5126, count=len(xyz), type="VEC3",
                          min=[float(x) for x in xyz.min(0)], max=[float(x) for x in xyz.max(0)]))
    attrs["POSITION"] = len(accessors) - 1
    if normals is not None:
        v = add(np.ascontiguousarray(normals, dtype=np.float32), 34962)
        accessors.append(dict(bufferView=v, componentType=5126, count=len(xyz), type="VEC3"))
        attrs["NORMAL"] = len(accessors) - 1
    if rgb is not None:
        c = np.empty((len(xyz), 4), np.uint8)
        c[:, :3] = rgb
        c[:, 3] = 255
        v = add(c, 34962)
        accessors.append(dict(bufferView=v, componentType=5121, normalized=True, count=len(xyz), type="VEC4"))
        attrs["COLOR_0"] = len(accessors) - 1
    v = add(faces.reshape(-1), 34963)
    accessors.append(dict(bufferView=v, componentType=5125, count=int(faces.size), type="SCALAR"))
    mat = dict(name="photogrammetry", doubleSided=True,
               pbrMetallicRoughness=dict(baseColorFactor=[1, 1, 1, 1], metallicFactor=0.0, roughnessFactor=1.0))
    gltf = dict(asset=dict(version="2.0", generator=ENGINE_VERSION), scene=0, scenes=[dict(nodes=[0])],
                nodes=[dict(mesh=0, name="actionable_threat_mesh")],
                meshes=[dict(primitives=[dict(attributes=attrs, indices=len(accessors) - 1, material=0, mode=4)])],
                materials=[mat], accessors=accessors, bufferViews=views, buffers=[dict(byteLength=off)])
    if unlit:
        mat["extensions"] = {"KHR_materials_unlit": {}}
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    binb = b"".join(parts)
    total = 12 + 8 + len(js) + 8 + len(binb)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(binb), 0x004E4942))
        fh.write(binb)


# -----------------------------------------------------------------------------
# geometry helpers
# -----------------------------------------------------------------------------
def voxel_downsample(xyz, rgb=None, nrm=None, voxel=None, target=None, max_iter=6):
    """Uniform re-sampling: average of all points in each occupied voxel.
    If `target` is given the voxel size is searched so that ~target points remain."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return xyz, rgb, nrm, voxel or 1.0
    lo = xyz.min(0)
    ext = np.maximum(xyz.max(0) - lo, 1e-9)
    if voxel is None:
        if target is None or len(xyz) <= target:
            # nothing to thin out: keep every point, just report the sample spacing
            return xyz, rgb, nrm, sample_spacing(xyz)
        # surfaces: N ~ area / v^2 -> start from that estimate, refine below
        area = 2 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2]) / 3.0
        voxel = math.sqrt(area / target)
    keys = inv = None
    for _ in range(max_iter):
        q = np.floor((xyz - lo) / voxel).astype(np.int64)
        dims = q.max(0) + 1
        keys = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        uniq, inv = np.unique(keys, return_inverse=True)
        n = len(uniq)
        if target is None or abs(n - target) / target < 0.12:
            break
        voxel *= (n / target) ** 0.5
    inv = inv.reshape(-1)
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    out = np.stack([np.bincount(inv, xyz[:, i], n) for i in range(3)], 1) / cnt[:, None]
    orgb = onrm = None
    if rgb is not None:
        orgb = (np.stack([np.bincount(inv, rgb[:, i].astype(np.float64), n) for i in range(3)], 1) / cnt[:, None])
        orgb = np.clip(np.round(orgb), 0, 255).astype(np.uint8)
    if nrm is not None:
        onrm = np.stack([np.bincount(inv, nrm[:, i].astype(np.float64), n) for i in range(3)], 1)
        ln = np.linalg.norm(onrm, axis=1, keepdims=True)
        onrm = (onrm / np.maximum(ln, 1e-12)).astype(np.float32)
    return out, orgb, onrm, float(voxel)


def sample_spacing(xyz, n=20000):
    """Median nearest-neighbour distance (robust point spacing for surfaces)."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, float)
    if len(xyz) < 3:
        return 1.0
    sub = xyz[np.random.default_rng(0).choice(len(xyz), min(n, len(xyz)), replace=False)]
    d, _ = cKDTree(xyz).query(sub, k=2, workers=-1)
    return float(max(np.median(d[:, 1]), 1e-6))


def remove_isolated(xyz, voxel, min_neighbors=4):
    """Drop points with fewer than `min_neighbors` other points in their 3x3x3 voxel neighbourhood."""
    q = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64) + 1
    dims = q.max(0) + 2
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    order = np.argsort(key, kind="stable")
    sk = key[order]
    uk, start, counts = np.unique(sk, return_index=True, return_counts=True)
    total = np.zeros(len(uk), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nk = uk + (dx * dims[1] + dy) * dims[2] + dz
                pos = np.searchsorted(uk, nk)
                pos = np.clip(pos, 0, len(uk) - 1)
                hit = uk[pos] == nk
                total += np.where(hit, counts[pos], 0)
    per_voxel_keep = (total - 1) >= min_neighbors
    keep_sorted = np.repeat(per_voxel_keep, counts)
    keep = np.empty(len(xyz), bool)
    keep[order] = keep_sorted
    return keep


def rot_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def umeyama(src, dst, with_scale=True):
    """dst ~ s R src + t (least squares)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if with_scale and var > 1e-12 else 1.0
    t = md - s * R @ ms
    return s, R, t


def ransac_similarity(src, dst, thresh, iters=1500, seed=0, min_sample_spread=1e-3):
    rng = np.random.default_rng(seed)
    n, d = src.shape
    k = d if d >= 3 else 2
    best_inl, best = None, None
    if n < k + 1:
        s, R, t = umeyama(src, dst)
        return s, R, t, np.ones(n, bool)
    span = np.linalg.norm(src.max(0) - src.min(0)) + 1e-12
    for _ in range(iters):
        idx = rng.choice(n, k, replace=False)
        p = src[idx]
        if d == 3:
            if np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) < (min_sample_spread * span) ** 2:
                continue
        elif np.linalg.norm(p[1] - p[0]) < min_sample_spread * span:
            continue
        s, R, t = umeyama(p, dst[idx])
        if not np.isfinite(s) or s <= 0:
            continue
        r = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        inl = r < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best = inl, (s, R, t)
    if best is None:
        s, R, t = umeyama(src, dst)
        return s, R, t, np.ones(n, bool)
    inl = best_inl
    for _ in range(3):
        s, R, t = umeyama(src[inl], dst[inl])
        r = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        new = r < thresh
        if new.sum() < k + 1:
            break
        inl = new
    return s, R, t, inl


def fit_plane_ransac(pts, thresh, iters=400, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n < 10:
        return None, None
    best_cnt, best = 0, None
    for _ in range(iters):
        p = pts[rng.choice(n, 3, replace=False)]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm /= ln
        d = np.abs((pts - p[0]) @ nrm)
        cnt = int((d < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nrm, p[0])
    if best is None:
        return None, None
    nrm, p0 = best
    inl = np.abs((pts - p0) @ nrm) < thresh
    c = pts[inl].mean(0)
    _, _, Vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    return Vt[2], inl


def estimate_up(Rw2c, C, pts=None):
    """Gravity 'up' in the SfM frame.

    Gimbal-stabilised drone video has ~zero roll, so every camera x-axis is
    horizontal: up = null-vector of sum(x x^T).  Falls back to the ground
    plane (oriented towards the cameras) and finally to the mean view."""
    X, Y, Z = Rw2c[:, 0, :], Rw2c[:, 1, :], Rw2c[:, 2, :]
    M = X.T @ X / len(X)
    w, V = np.linalg.eigh(M)
    u = V[:, 0]
    if (-(Y + Z) @ u).sum() < 0:
        u = -u
    resid = np.degrees(np.arcsin(np.clip(np.abs(X @ u), 0, 1)))
    roll_ok = np.median(resid) < 3.0 and np.percentile(resid, 90) < 8.0
    yaw_diverse = w[1] > 0.03 * w[2]
    n_plane = None
    if pts is not None and len(pts) >= 50:
        sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
        ext = np.linalg.norm(np.percentile(sub, 95, 0) - np.percentile(sub, 5, 0))
        nrm, inl = fit_plane_ransac(sub, 0.01 * ext)
        if nrm is not None and inl.mean() > 0.2:
            if (C.mean(0) - sub[inl].mean(0)) @ nrm < 0:
                nrm = -nrm
            n_plane = nrm
    if roll_ok and yaw_diverse:
        return u, "camera_horizon", float(np.median(resid))
    if roll_ok and n_plane is not None:
        xm = V[:, 2]
        u2 = n_plane - (n_plane @ xm) * xm
        if np.linalg.norm(u2) > 0.5:
            u2 /= np.linalg.norm(u2)
            if (-(Y + Z) @ u2).sum() < 0:
                u2 = -u2
            return u2, "camera_horizon+ground_plane", float(np.median(resid))
    if n_plane is not None:
        return n_plane, "ground_plane", float(np.median(resid))
    d = -(Y + Z).mean(0)
    return d / np.linalg.norm(d), "view_direction", float(np.median(resid))


def estimate_ground_height(h):
    """Lowest strong mode of the height histogram (robust 'ground' level)."""
    h = np.asarray(h, float)
    if len(h) < 20:
        return float(np.percentile(h, 5)) if len(h) else 0.0
    lo, hi = np.percentile(h, [0.5, 75])
    if hi - lo < 1e-9:
        return float(lo)
    hist, edges = np.histogram(h[(h >= lo) & (h <= hi)], bins=240, range=(lo, hi))
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    sm = np.convolve(hist.astype(float), k / k.sum(), mode="same")
    i = int(np.argmax(sm >= 0.3 * sm.max()))
    while i + 1 < len(sm) and sm[i + 1] >= sm[i]:
        i += 1
    return float(0.5 * (edges[i] + edges[i + 1]))


def quat_xyzw(R):
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


# ENU (east, north, up) -> Three.js (x = east, y = up, z = south)
ENU_TO_THREE = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


# -----------------------------------------------------------------------------
# SfM summary & georeferencing
# -----------------------------------------------------------------------------
class SfmSummary:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.names = [str(n) for n in z["names"]]
        self.R, self.T, self.C, self.K = z["R"], z["T"], z["C"], z["K"]
        self.wh = z["wh"]
        self.xyz, self.rgb, self.err = z["xyz"], z["rgb"], z["err"]
        self.track_ptr, self.track_img = z["track_ptr"], z["track_img"]
        self.cam_model = str(z["cam_model"])
        self.index = {n: i for i, n in enumerate(self.names)}

    @property
    def n(self):
        return len(self.names)


def _gps_enu(tel, times):
    g = tel.at(times)
    if g["lat"] is None:
        return None, None, None
    lat0, lon0 = float(np.mean(g["lat"])), float(np.mean(g["lon"]))
    e, n = latlon_to_en(g["lat"], g["lon"], lat0, lon0)
    if g["rel_alt"] is not None:
        u = g["rel_alt"]
    elif g["abs_alt"] is not None:
        u = g["abs_alt"] - np.min(g["abs_alt"])
    else:
        u = None
    return np.stack([e, n], 1), u, (lat0, lon0)


def _trimmed_rmse(res, keep=0.9):
    r2 = np.sort(np.asarray(res) ** 2)
    return float(np.sqrt(np.mean(r2[:max(3, int(keep * len(r2)))])))


def resolve_latlon_order(summ, times, tel, R_lvl):
    """Fits the levelled camera track to both GPS(a, b) orders; the mirrored (wrong) one cannot be fitted."""
    P = summ.C @ R_lvl.T
    scores = {}
    for name, t in (("as_logged", tel), ("swapped", tel.swapped())):
        en, _, _ = _gps_enu(t, times)
        if en is None or len(en) < 6:
            scores[name] = np.inf
            continue
        thr = max(2.5, 0.02 * float(np.linalg.norm(np.ptp(en, axis=0))))
        s2, R2, t2, _ = ransac_similarity(P[:, :2], en, thr)
        scores[name] = _trimmed_rmse(np.linalg.norm(en - (s2 * P[:, :2] @ R2.T + t2), axis=1))
    a, b = scores["as_logged"], scores["swapped"]
    info = {"latlon_test_rmse_m": {k: (round(v, 3) if np.isfinite(v) else None) for k, v in scores.items()}}
    if np.isfinite(b) and b < 0.6 * a:
        log(f"GPS order: log is GPS(lon, lat) - swapped (RMSE {a:.1f} m -> {b:.1f} m)")
        info.update(latlon_order_resolved="swapped", latlon_verified=True)
        return tel.swapped(), info
    decided = np.isfinite(a) and a < 0.6 * b
    log(f"GPS order: as logged ({'verified' if decided else 'undecided: straight flight'}; RMSE {a:.1f} m vs {b:.1f} m)")
    info.update(latlon_order_resolved="as_logged" if decided else "undecided", latlon_verified=bool(decided))
    return tel, info


def compute_georef(summ, kf_by_name, tel, dense_xyz, cfg):
    """Returns dict with 4x4 similarity 'T' (SfM frame -> output frame, metres, Three.js axes)."""
    C, Rw2c = summ.C, summ.R
    info = {}
    pts_for_plane = dense_xyz if dense_xyz is not None and len(dense_xyz) > 1000 else summ.xyz
    up, up_method, roll_med = estimate_up(Rw2c, C, pts_for_plane)
    info.update(up_method=up_method, camera_roll_median_deg=round(roll_med, 2))
    R_lvl = rot_between(up, np.array([0.0, 0.0, 1.0]))
    times = np.array([kf_by_name[n]["timestamp_sec"] for n in summ.names])
    if tel is not None and tel.has_gps:
        if tel.meta.get("latlon_ambiguous"):
            tel, oi = resolve_latlon_order(summ, times, tel, R_lvl)
            info.update(oi)
        else:
            info.update(latlon_verified=True, latlon_order_resolved="explicit")
    info["telemetry"] = tel
    method, s, R, t = None, 1.0, R_lvl, np.zeros(3)
    lat0 = lon0 = None
    en, u, ll0 = _gps_enu(tel, times) if (tel is not None and tel.has_gps) else (None, None, None)
    gps = tel.at(times) if tel is not None else dict(lat=None, rel_alt=None, abs_alt=None)
    if en is not None:
        lat0, lon0 = ll0
        hen = en - en.mean(0)
        sv = np.linalg.svd(hen, compute_uv=False) / math.sqrt(max(1, len(hen)))
        spread_ok = len(en) >= 6 and sv[1] > max(4.0, 0.05 * sv[0])
        line_ok = len(en) >= 4 and sv[0] > 8.0
        thr = max(2.5, 0.02 * float(sv[0]))
        if u is not None and spread_ok:
            enu = np.c_[en, u]
            s7, R7, t7, inl = ransac_similarity(C, enu, thr)
            up7 = R7.T @ np.array([0, 0, 1.0])
            ang = math.degrees(math.acos(np.clip(up7 @ up, -1, 1)))
            info["gps_vs_horizon_up_deg"] = round(ang, 2)
            if ang < 15.0 or up_method != "camera_horizon":
                method, s, R, t = "GPS_SIM3", s7, R7, t7
                res = np.linalg.norm(enu - (s * C @ R.T + t), axis=1)
                info.update(inliers=int(inl.sum()), rmse_m=round(float(np.sqrt(np.mean(res[inl] ** 2))), 3))
        if method is None and line_ok:
            # horizontal GPS alone fixes scale and heading once gravity is known (image horizon); this
            # works with no altitude in the log at all
            P = C @ R_lvl.T
            s2, R2, t2, inl = ransac_similarity(P[:, :2], en, thr)
            tz = float(np.median(u[inl] - s2 * P[inl, 2])) if u is not None else 0.0
            R3 = np.eye(3)
            R3[:2, :2] = R2
            method, s, R, t = "GPS_4DOF", s2, R3 @ R_lvl, np.array([t2[0], t2[1], tz])
            res = np.linalg.norm(en - (s * C @ R.T + t)[:, :2], axis=1)
            info.update(inliers=int(inl.sum()), rmse_m=round(float(np.sqrt(np.mean(res[inl] ** 2))), 3))
        if method is not None:
            n_eff = max(1, min(int(info.get("inliers", 1)), 10))
            info["scale_rel_uncertainty"] = float(info["rmse_m"] / max(float(np.sqrt((sv ** 2).sum())) * math.sqrt(n_eff), 1e-6))
    # output frame before ground shift
    if method is None:
        # no usable GPS: level + scale by the logged height above ground; a guessed altitude is the last resort
        P = C @ R_lvl.T
        R = R_lvl
        s = 1.0
        if dense_xyz is not None and len(dense_xyz) > 100:
            zg = estimate_ground_height((dense_xyz @ R.T)[:, 2])
        else:
            zg = estimate_ground_height((summ.xyz @ R.T)[:, 2])
        cam_h = float(np.median(P[:, 2] - zg))
        alt_src, alt = "assumed_altitude", float(cfg.assumed_altitude_m or 30.0)
        if gps.get("rel_alt") is not None and np.median(gps["rel_alt"]) > 2.0:
            alt_src, alt = "logged_altitude", float(np.median(gps["rel_alt"]))
        if cam_h > 1e-9 and alt > 0:
            s = alt / cam_h
        t = np.array([-s * P[:, 0].mean(), -s * P[:, 1].mean(), 0.0])
        method = "ALTITUDE_AGL" if alt_src == "logged_altitude" else "ASSUMED_ALTITUDE"
        info["scale_source"] = alt_src
        info["scale_rel_uncertainty"] = 0.05 if alt_src == "logged_altitude" else 0.25
        if alt_src == "assumed_altitude":
            warn(f"no GPS and no altitude in the flight log: scale assumes {alt:.0f} m above ground "
                 "(heights are LOW confidence - supply the DJI .SRT)")
    else:
        info["scale_source"] = "gps"
    A = ENU_TO_THREE
    Rt = A @ R
    tt = A @ t
    # ground level from dense (or sparse) points in the output frame
    pts = dense_xyz if dense_xyz is not None and len(dense_xyz) > 100 else summ.xyz
    y = (s * pts @ Rt.T + tt)[:, 1]
    ground = estimate_ground_height(y)
    tt = tt - np.array([0.0, ground, 0.0])
    T = np.eye(4)
    T[:3, :3] = s * Rt
    T[:3, 3] = tt
    unc = info.get("scale_rel_uncertainty", 0.25)
    conf = ("HIGH" if unc <= 0.01 and info.get("rmse_m", 99) <= 3.0 else "MEDIUM" if unc <= 0.05 else "LOW")
    info.update(method=method, scale=float(s), ground_offset_m=round(float(ground), 3),
                lat0=lat0, lon0=lon0, T=T, confidence=conf,
                origin=({"lat": lat0, "lon": lon0, "alt": float(ground)} if lat0 is not None else None))
    log(f"georef: {method} ({conf}), up from {up_method}, scale {s:.4f}"
        + (f", GPS RMSE {info.get('rmse_m')} m ({info.get('inliers')} inliers)" if "rmse_m" in info else ""))
    return info


def apply_T(T, xyz):
    return np.asarray(xyz, float) @ T[:3, :3].T + T[:3, 3]


def viewing_angle_gate(xyz, cams, min_grazing_deg=5.0, above_margin=5.0):
    """Points seen from the flight path at a usable grazing angle and not above the cameras (Y-up frame)."""
    from scipy.spatial import cKDTree
    d, k = cKDTree(cams[:, [0, 2]]).query(xyz[:, [0, 2]], workers=-1)
    ang = np.degrees(np.arctan2(cams[k, 1] - xyz[:, 1], np.maximum(d, 1e-6)))
    return (ang >= min_grazing_deg) & (xyz[:, 1] <= cams[:, 1].max() + above_margin)


def rotate_normals(T, n):
    Rn = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
    return (np.asarray(n, np.float64) @ Rn.T).astype(np.float32)


# -----------------------------------------------------------------------------
# MVS planning
# -----------------------------------------------------------------------------
def _gauss_angle_weight(theta_deg, t0=7.0, s_lo=3.0, s_hi=14.0):
    w = np.where(theta_deg < t0, np.exp(-0.5 * ((theta_deg - t0) / s_lo) ** 2),
                 np.exp(-0.5 * ((theta_deg - t0) / s_hi) ** 2))
    return np.where(theta_deg < 1.0, 0.0, w)


def pair_scores(summ):
    """Baseline-aware covisibility scores S[i, j] (MVSNet-style angle prior) and shared counts."""
    N = summ.n
    S = np.zeros(N * N)
    Nsh = np.zeros(N * N)
    ptr, img = summ.track_ptr, summ.track_img
    lens = np.diff(ptr)
    for L in np.unique(lens):
        if L < 2:
            continue
        pid = np.nonzero(lens == L)[0]
        step = int(max(1, min(20000, 2e7 // (L * L))))   # bound the (n, L, L) temporaries
        for s0 in range(0, len(pid), step):
            pp = pid[s0:s0 + step]
            ii = ptr[pp][:, None] + np.arange(L)[None, :]
            im = img[ii]                                  # (n, L)
            rays = summ.xyz[pp][:, None, :] - summ.C[im]  # (n, L, 3)
            rays /= np.linalg.norm(rays, axis=2, keepdims=True) + 1e-12
            cos = np.einsum("nid,njd->nij", rays, rays)
            iu, ju = np.triu_indices(L, 1)
            th = np.degrees(np.arccos(np.clip(cos[:, iu, ju], -1, 1)))
            w = _gauss_angle_weight(th)
            a, b = im[:, iu].ravel(), im[:, ju].ravel()
            S += np.bincount(a * N + b, w.ravel(), N * N) + np.bincount(b * N + a, w.ravel(), N * N)
            Nsh += np.bincount(a * N + b, minlength=N * N) + np.bincount(b * N + a, minlength=N * N)
    return S.reshape(N, N), Nsh.reshape(N, N)


def plan_mvs(summ, kf_by_name, prof, budget_s, cfg):
    from scipy.sparse import csr_matrix
    N, M = summ.n, len(summ.xyz)
    S, Nsh = pair_scores(summ)
    lens = np.diff(summ.track_ptr)
    obs_pt = np.repeat(np.arange(M), lens)
    A = csr_matrix((np.ones(len(obs_pt)), (summ.track_img, obs_pt)), shape=(N, M))
    blur = np.array([kf_by_name.get(n, {}).get("blur_score", 1.0) for n in summ.names], float)
    sharp_w = np.clip((blur / max(np.median(blur), 1e-6)) ** 0.3, 0.6, 1.3)
    gain = np.array([1.0, 0.75, 0.45, 0.0])
    cov = np.zeros(M, np.int64)
    avail = np.ones(N, bool)
    order = []
    for _ in range(N):
        g = gain[np.minimum(cov, 3)]
        sc = (A @ g) * sharp_w
        sc[~avail] = -1.0
        i = int(np.argmax(sc))
        if sc[i] <= 1e-9:
            break
        order.append(i)
        avail[i] = False
        cov[A.indices[A.indptr[i]:A.indptr[i + 1]]] += 1
    rest = list(np.nonzero(avail)[0])
    if rest:  # farthest-point fill in time for views that add no new coverage
        tix = np.arange(N, dtype=float)
        picked = np.array(order if order else [rest[0]], float)
        rest = np.array(rest)
        dmin = np.min(np.abs(tix[rest][:, None] - picked[None, :]), axis=1)
        while len(rest):
            j = int(np.argmax(dmin))
            order.append(int(rest[j]))
            dnew = np.abs(tix[rest] - rest[j])
            dmin = np.minimum(dmin, dnew)
            rest = np.delete(rest, j)
            dmin = np.delete(dmin, j)
    n_plan = int(min(N, max(20, math.ceil(1.35 * budget_s / prof["sec_per_ref"]))))
    refs_idx = order[:n_plan]
    K = int(prof["mvs_src"])
    refs, need = [], set()
    for i in refs_idx:
        sc = S[i].copy()
        sc[i] = -1
        cand = [j for j in np.argsort(-sc)[:3 * K] if sc[j] > 0]
        if len(cand) < 4:
            extra = [j for j in np.argsort(-Nsh[i])[:3 * K] if j != i and Nsh[i, j] > 0 and j not in cand]
            cand += extra[:4 - len(cand) + 4]
        refs.append(dict(name=summ.names[i], cands=[[summ.names[j], round(float(sc[j]), 3)] for j in cand]))
        need.add(summ.names[i])
        need.update(summ.names[j] for j in cand[:2 * K])
    return dict(refs=refs, num_sources=K, undistort=sorted(need), order=[summ.names[i] for i in order])


# -----------------------------------------------------------------------------
# mesh utilities
# -----------------------------------------------------------------------------
def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        for c in range(3):
            n[:, c] += np.bincount(f[:, k], fn[:, c], len(v))
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(ln, 1e-20)).astype(np.float32)


def compact_mesh(v, f, extra=()):
    used = np.zeros(len(v), bool)
    used[f.ravel()] = True
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(used.sum())
    return v[used], remap[f], [e[used] if e is not None else None for e in extra]


def drop_small_components(v, f, min_faces):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(f) == 0:
        return f
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(len(v), len(v)))
    _, lab = connected_components(g, directed=False)
    fl = lab[f[:, 0]]
    cnt = np.bincount(fl)
    return f[cnt[fl] >= min_faces]


def heightfield_mesh(xyz, rgb, cell):
    """Robust 2.5D fallback surface (y is up): median height per XZ cell, triangulated grid."""
    x, z = xyz[:, 0], xyz[:, 2]
    i = np.floor((x - x.min()) / cell).astype(np.int64)
    j = np.floor((z - z.min()) / cell).astype(np.int64)
    W, H = i.max() + 1, j.max() + 1
    key = j * W + i
    order = np.lexsort((xyz[:, 1], key))          # by cell, then by height
    k_sorted = key[order]
    uk, st, cnt = np.unique(k_sorted, return_index=True, return_counts=True)
    med = xyz[order, 1][st + cnt // 2]             # per-cell median height
    grp = np.repeat(np.arange(len(uk)), cnt)
    col = np.stack([np.bincount(grp, rgb[order, c].astype(float)) / cnt for c in range(3)], 1)
    grid = -np.ones(W * H, np.int64)
    grid[uk] = np.arange(len(uk))
    gi, gj = uk % W, uk // W
    verts = np.stack([x.min() + (gi + 0.5) * cell, med, z.min() + (gj + 0.5) * cell], 1)
    g = grid.reshape(H, W)
    a, b, c, d = g[:-1, :-1], g[:-1, 1:], g[1:, :-1], g[1:, 1:]
    m1 = (a >= 0) & (b >= 0) & (c >= 0)
    m2 = (b >= 0) & (d >= 0) & (c >= 0)
    faces = np.concatenate([np.stack([a[m1], c[m1], b[m1]], 1), np.stack([b[m2], c[m2], d[m2]], 1)])
    return verts, faces, np.clip(col, 0, 255).astype(np.uint8)


def render_preview(xyz, rgb, path, size=None):
    """Top-down orthographic preview (y up) with a simple z-buffer."""
    import cv2
    if len(xyz) == 0:
        return None
    if size is None:
        size = int(np.clip(np.sqrt(len(xyz)) * 1.1, 600, 1600))
    x, z, y = xyz[:, 0], xyz[:, 2], xyz[:, 1]
    lo = np.percentile(np.stack([x, z], 1), 0.5, 0)
    hi = np.percentile(np.stack([x, z], 1), 99.5, 0)
    sc = (size - 1) / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    W = int((hi[0] - lo[0]) * sc) + 1
    H = int((hi[1] - lo[1]) * sc) + 1
    u = ((x - lo[0]) * sc).astype(np.int64)
    v = ((z - lo[1]) * sc).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, y, c = u[ok], v[ok], y[ok], rgb[ok]
    order = np.argsort(y)  # later (higher) points overwrite
    img = np.full((H, W, 3), 24, np.uint8)
    img[v[order], u[order]] = c[order][:, ::-1]
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# -----------------------------------------------------------------------------
# helpers for the orchestrator
# -----------------------------------------------------------------------------
VOCAB_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin"


def fetch_vocab_tree(cache_dir):
    path = os.path.join(ensure_dir(cache_dir), "vocab_tree_faiss_flickr100K_words32K.bin")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    try:
        import urllib.request
        tmp = path + ".part"
        with urllib.request.urlopen(VOCAB_URL, timeout=60) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        os.replace(tmp, path)
        return path
    except Exception as e:
        warn(f"vocab tree download failed ({e}); loop detection disabled")
        return ""


def spatial_pairs(names, enu, overlap, k=10, radius=None):
    """GPS neighbours that sequential matching would not already cover."""
    from scipy.spatial import cKDTree
    n = len(names)
    if n < 3:
        return []
    if radius is None:
        steps = np.linalg.norm(np.diff(enu, axis=0), axis=1)
        radius = max(25.0, 12.0 * float(np.median(steps)) if len(steps) else 25.0)
    tree = cKDTree(enu)
    seq = set()
    for i in range(n):
        for d in range(1, overlap + 1):
            seq.add((i, i + d))
        for p in range(overlap):
            seq.add((i, i + (1 << p)))
    pairs = set()
    dist, idx = tree.query(enu, k=min(n, k + overlap + 1), distance_upper_bound=radius)
    for i in range(n):
        for d, j in zip(dist[i], idx[i]):
            if not np.isfinite(d) or j >= n or j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seq:
                pairs.add((a, b))
    return [(names[a], names[b]) for a, b in sorted(pairs)]


def focal_prior(cfg, tel, kf_w):
    f35 = cfg.focal_35mm if cfg.focal_35mm > 0 else (tel.focal35 if tel is not None else 0.0)
    if f35 and 8.0 <= f35 <= 300.0:
        return f35 * kf_w / 34.62, f35
    return None, None


def camera_params_string(model, f, w, h):
    cx, cy = w / 2.0, h / 2.0
    model = model.upper()
    if model == "SIMPLE_PINHOLE":
        return f"{f:.3f},{cx:.3f},{cy:.3f}"
    if model == "PINHOLE":
        return f"{f:.3f},{f:.3f},{cx:.3f},{cy:.3f}"
    if model == "SIMPLE_RADIAL":
        return f"{f:.3f},{cx:.3f},{cy:.3f},0"
    if model == "RADIAL":
        return f"{f:.3f},{cx:.3f},{cy:.3f},0,0"
    if model in ("OPENCV", "FULL_OPENCV"):
        extra = ",0,0,0,0" if model == "OPENCV" else ",0,0,0,0,0,0,0,0"
        return f"{f:.3f},{f:.3f},{cx:.3f},{cy:.3f}" + extra
    if model == "OPENCV_FISHEYE":
        return f"{f:.3f},{f:.3f},{cx:.3f},{cy:.3f},0,0,0,0"
    return ""


def pycolmap_preflight():
    """Import pycolmap in a child process (never in the notebook kernel) and report CUDA support."""
    code = ("import json,pycolmap;print(json.dumps(dict(version=pycolmap.__version__,"
            "cuda=bool(getattr(pycolmap,'has_cuda',False)),"
            "gpus=int(pycolmap.get_num_cuda_devices()) if hasattr(pycolmap,'get_num_cuda_devices') else 0)))")
    rc, out, err = run_cmd([sys.executable, "-c", code], timeout=180)
    if rc != 0:
        raise RuntimeError("pycolmap is not importable. Run the setup cell first "
                           "(pip install pycolmap-cuda12==4.2.0).\n" + err.strip()[-800:])
    return json.loads(out.strip().splitlines()[-1])


def _dirs(cfg):
    w = cfg.workdir
    d = dict(work=w, images=os.path.join(w, "images"), sfm=os.path.join(w, "sfm"), dense=os.path.join(w, "dense"),
             models=os.path.join(w, "models"), tmp=os.path.join(w, "tmp"), bundle=os.path.join(w, "bundle"),
             cache=os.path.join(os.path.dirname(w.rstrip("/")) or "/content", "prism_cache"))
    for k, v in d.items():
        ensure_dir(v)
    return d


def _state(dirs, update=None):
    p = os.path.join(dirs["work"], "state.json")
    st = {}
    if os.path.exists(p):
        try:
            with open(p) as fh:
                st = json.load(fh)
        except Exception:
            st = {}
    if update:
        st.update(update)
        json_dump(st, p, indent=1)
    return st


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------
def run(cfg: Config):
    clock = StageClock()
    hw = probe_hardware()
    prof = make_profile(cfg, hw)
    prof["cpu"] = hw["cpu"]
    budget = Budget(prof["target_s"])
    dirs = _dirs(cfg)
    st = _state(dirs) if cfg.resume else {}
    if not cfg.resume:
        for k in ("images", "sfm", "dense", "models", "bundle", "tmp"):
            shutil.rmtree(dirs[k], ignore_errors=True)
            ensure_dir(dirs[k])
        _state(dirs, {"_reset": time.time()})
    pc = pycolmap_preflight()
    if not pc["cuda"]:
        # CPU-only pycolmap: keep SIFT affordable, dense MVS is impossible without CUDA
        prof.update(max_kf=min(prof["max_kf"], 220), sift_feats=3000, kf_long=min(prof["kf_long"], 1600))
        warn("pycolmap has NO CUDA support here -> CPU SIFT, no dense MVS (sparse model + mesh only). "
             "Use a GPU runtime and pycolmap-cuda12 for full quality.")
    prof["pycolmap_cuda"] = pc["cuda"]
    log(f"{ENGINE_VERSION} | pycolmap {pc['version']} cuda={pc['cuda']} | GPU {hw['gpu_name'] or 'none'} "
        f"({hw['gpu_mem_gb']} GB, tier {prof['tier']}) | {hw['cpu']} vCPU | {hw['ram_gb']} GB RAM | "
        f"quality {prof['quality']} | budget {fmt_s(budget.total)}")
    report = dict(engine=ENGINE_VERSION, hardware=hw, profile={k: v for k, v in prof.items()},
                  config=asdict(cfg), warnings=[])

    # ---- probe inputs ----------------------------------------------------------
    clock.start("0. probe inputs")
    if not cfg.video_path or not os.path.exists(cfg.video_path):
        raise FileNotFoundError(f"video not found: {cfg.video_path}")
    vinfo = probe_video(cfg.video_path)
    srt = cfg.srt_path
    if not srt:
        base = os.path.splitext(cfg.video_path)[0]
        for ext in (".SRT", ".srt", ".csv", ".CSV"):
            if os.path.exists(base + ext):
                srt = base + ext
                break
    tel = load_telemetry(srt, vinfo["duration"])
    report["video"] = vinfo
    clock.stop(f"{vinfo['width']}x{vinfo['height']} {vinfo['codec']} {vinfo['fps']:.2f} fps, "
               f"{vinfo['duration'] / 60:.1f} min, {vinfo['size_mb']:.0f} MB")

    # ---- 1. ingest -------------------------------------------------------------
    clock.start("1. ingest + keyframes")
    if cfg.resume and st.get("ingest"):
        kfs, ing = st["kf"], st["ingest"]
        log("resume: reusing keyframes")
    else:
        kfs, ing = ingest_video(cfg, prof, vinfo, dirs)
        _state(dirs, {"kf": kfs, "ingest": ing})
    web_job = start_web_video_job(cfg, vinfo, ing, dirs["work"]) if not ing.get("web_video") else None
    if len(kfs) < 8:
        raise RuntimeError(f"only {len(kfs)} usable keyframes; is the video mostly static/black?")
    times = np.array([k["timestamp_sec"] for k in kfs])
    tk = tel.at(times)
    for i, k in enumerate(kfs):
        if tk["lat"] is not None:
            k["latitude"], k["longitude"] = round(float(tk["lat"][i]), 8), round(float(tk["lon"][i]), 8)
        if tk["rel_alt"] is not None:
            k["relative_altitude_m"] = round(float(tk["rel_alt"][i]), 2)
        if tk["abs_alt"] is not None:
            k["absolute_altitude_m"] = round(float(tk["abs_alt"][i]), 2)
    kf_by_name = {k["filename"]: k for k in kfs}
    report["ingest"] = ing
    clock.stop(f"{len(kfs)} keyframes ({ing['backend']}, {ing['decode_fps']:.0f} fps)")

    # ---- 2. SfM ----------------------------------------------------------------
    clock.start("2. SfM (GPU SIFT + global)")
    kf_w, kf_h = ing["kf_size"]
    names = [k["filename"] for k in kfs]
    f_px, f35 = focal_prior(cfg, tel, kf_w)
    sfm_args = dict(database_path=os.path.join(dirs["sfm"], "database.db"), image_dir=dirs["images"],
                    output_dir=dirs["sfm"], image_names=names, camera_model=cfg.camera_model,
                    camera_params=camera_params_string(cfg.camera_model, f_px, kf_w, kf_h) if f_px else "",
                    focal_factor=0.72, sift_max_size=max(kf_w, kf_h), sift_feats=prof["sift_feats"],
                    first_octave=0,
                    seq_overlap=prof["seq_overlap"], loop_imgs=prof["loop_imgs"],
                    keep_tracks=prof["keep_tracks"], ba_rounds=prof["ba_rounds"], ba_iters=prof["ba_iters"],
                    verbose=cfg.verbose)
    n = len(names)
    # oblique video: keep sky / clouds out of feature matching
    try:
        mdir = os.path.join(dirs["work"], "masks")
        share = write_sky_masks(dirs["images"], names, mdir)
        if share > 0.05:
            sfm_args["mask_path"] = mdir
            log(f"sky masks: horizon visible in {100 * share:.0f}% of keyframes -> sky excluded from features")
        report["sky_masked_share"] = round(share, 3)
    except Exception as e:
        warn(f"sky masks skipped: {e}")
    est_pairs = n * (prof["seq_overlap"] + 4)
    if tel.has_gps and tel.gps_spread_m > 20:
        en_lat, en_lon = tk["lat"], tk["lon"]
        e, nn = latlon_to_en(en_lat, en_lon, float(np.mean(en_lat)), float(np.mean(en_lon)))
        up = tk["rel_alt"] if tk["rel_alt"] is not None else np.zeros(n)
        pr = spatial_pairs(names, np.stack([e, nn, up], 1), prof["seq_overlap"])
        room = max(0, prof["max_pairs"] - est_pairs)
        if len(pr) > room:
            pr = [pr[i] for i in np.linspace(0, len(pr) - 1, room).astype(int)] if room else []
        if pr:
            pp = os.path.join(dirs["sfm"], "gps_pairs.txt")
            with open(pp, "w") as fh:
                fh.write("\n".join(f"{a} {b}" for a, b in pr) + "\n")
            sfm_args["pairs_path"] = pp
        log(f"GPS spatial pairs: {len(pr)}")
    else:
        vt = fetch_vocab_tree(dirs["cache"])
        if vt:
            sfm_args["vocab_tree_path"] = vt
            sfm_args["loop_period"] = max(5, int(round(n * prof["loop_imgs"] / max(1, prof["max_pairs"] - est_pairs))))
    sfm_args["incremental_budget_s"] = max(60.0, 0.35 * budget.remaining())
    if cfg.resume and st.get("sfm"):
        sfm = st["sfm"]
        log("resume: reusing SfM")
    else:
        sfm = run_worker("sfm", sfm_args, dirs, verbose=cfg.verbose)
        _state(dirs, {"sfm": sfm})
    summ = SfmSummary(sfm["summary"])
    report["sfm"] = {k: v for k, v in sfm.items() if k not in ("summary",)}
    clock.stop(f"{sfm['num_registered']}/{sfm['num_images']} registered, {sfm['num_points']} pts, "
               f"{sfm['mapper']} mapper, reproj {sfm.get('mean_reproj_error', float('nan')):.2f}px")

    # ---- 3. dense MVS ----------------------------------------------------------
    clock.start("3. dense MVS (CUDA PatchMatch)")
    reserve = _post_mvs_reserve(prof, hw)
    mvs_deadline = budget.t0 + budget.total - reserve
    mvs_budget = mvs_deadline - time.time()
    fused_path = os.path.join(dirs["dense"], "fused.ply")
    mvs = None
    dense_ok = False
    try:
        if not prof.get("pycolmap_cuda", True):
            raise RuntimeError("pycolmap without CUDA cannot run PatchMatch stereo")
        if mvs_budget < 40:
            raise RuntimeError(f"no time left for MVS ({fmt_s(mvs_budget)}); increase target_minutes")
        plan = plan_mvs(summ, kf_by_name, prof, mvs_budget, cfg)
        plan_path = os.path.join(dirs["dense"], "mvs_plan.json")
        json_dump(plan, plan_path)
        log(f"MVS plan: {len(plan['refs'])} candidate refs (priority ordered), {plan['num_sources']} sources each, "
            f"budget {fmt_s(mvs_budget)}")
        mvs = run_worker("mvs", dict(dense_dir=dirs["dense"], sparse_dir=sfm["model_dir"], image_dir=dirs["images"],
                                     plan_path=plan_path, deadline=mvs_deadline, mvs_size=prof["mvs_size"],
                                     win_radius=prof["win_radius"], num_samples=prof["num_samples"],
                                     it_photo=prof["it_photo"], it_geo=prof["it_geo"], sec_per_ref=prof["sec_per_ref"],
                                     cache_gb=prof["cache_gb"], geom_consistency=cfg.geom_consistency,
                                     verbose=cfg.verbose), dirs, verbose=cfg.verbose)
    except Exception as e:
        msg = f"dense MVS problem: {str(e).splitlines()[0][:300]}"
        warn(msg)
        report["warnings"].append(msg)
        mvs = _salvage_depth_maps(dirs["dense"])
        if mvs:
            log(f"salvaged {len(mvs['fusion_images'])} finished {mvs['input_type']} depth maps from disk")
    n_f = len(mvs["fusion_images"]) if mvs else 0
    clock.stop(f"{len(mvs['photo_done']) if mvs else 0} photometric / "
               f"{len(mvs.get('geo_done', [])) if mvs else 0} geometric depth maps")
    # ---- 4. fusion ---------------------------------------------------------------
    clock.start("4. depth-map fusion")
    try:
        if n_f < 3:
            raise RuntimeError("too few depth maps for fusion")
        W0, H0 = sfm["camera_size"]
        aspect = W0 / float(H0)
        px = prof["fusion_px"] / n_f
        long_side = int(min(prof["mvs_size"], max(640, math.sqrt(px * max(aspect, 1 / aspect)))))
        fus = run_worker("fusion", dict(dense_dir=dirs["dense"], image_names=mvs["fusion_images"],
                                        input_type=mvs["input_type"], fusion_size=long_side,
                                        min_num_pixels=3 if n_f < 250 else 4, output_path=fused_path,
                                        verbose=cfg.verbose), dirs, verbose=cfg.verbose)
        dense_ok = os.path.exists(fused_path) and os.path.getsize(fused_path) > 10000
        clock.stop(f"{n_f} maps at {long_side}px, {fus['bytes'] / 27e6:.1f}M points")
    except Exception as e:
        msg = f"dense reconstruction unavailable ({str(e).splitlines()[0][:300]}); falling back to sparse model"
        warn(msg)
        report["warnings"].append(msg)
        clock.stop("FAILED -> sparse fallback")
    if not cfg.keep_intermediate:
        for sub in ("stereo/depth_maps", "stereo/normal_maps", "images"):
            shutil.rmtree(os.path.join(dirs["dense"], sub), ignore_errors=True)
    report["mvs"] = {k: v for k, v in (mvs or {}).items() if k not in ("photo_done", "geo_done", "fusion_images")}
    if mvs:
        report["mvs"].update(n_photo=len(mvs["photo_done"]), n_geo=len(mvs.get("geo_done", [])))

    # ---- 5. georeference + point cloud -------------------------------------------
    clock.start("5. georef + point cloud")
    if dense_ok:
        ply = read_ply(fused_path)
        v = ply["vertex"]
        xyz, rgb, nrm = ply_xyz(v), ply_rgb(v), ply_normals(v)
        del ply, v
    else:
        xyz, rgb = summ.xyz.copy(), summ.rgb.copy()
        nrm = _sparse_normals(summ)
    log(f"input cloud: {len(xyz):,} points")
    geo = compute_georef(summ, kf_by_name, tel, xyz if dense_ok else None, cfg)
    tel = geo.pop("telemetry", tel)
    if tel is not None and tel.has_gps:           # keyframe GPS in the verified lat/lon order
        tk = tel.at(np.array([k["timestamp_sec"] for k in kfs]))
        for i, k in enumerate(kfs):
            k["latitude"], k["longitude"] = round(float(tk["lat"][i]), 8), round(float(tk["lon"][i]), 8)
    report["georef"] = {k: v for k, v in geo.items() if k != "T"}
    T = geo["T"]
    xyz = apply_T(T, xyz)
    if nrm is not None:
        nrm = rotate_normals(T, nrm)
    # viewing-geometry gate (oblique video): keep surface seen at >= 5 deg grazing from the flight path
    # and below the cameras - removes sky, cloud and horizon floaters that stretch the Poisson octree
    cam_out = apply_T(T, summ.C)
    keep = viewing_angle_gate(xyz, cam_out, min_grazing_deg=5.0)
    log(f"viewing-angle gate: kept {100 * keep.mean():.1f}% of points")
    xyz, rgb = xyz[keep], rgb[keep]
    nrm = nrm[keep] if nrm is not None else None
    # robust crop (drop far floaters) + uniform resampling + isolated-point removal
    lo, hi = np.percentile(xyz, 0.2, 0), np.percentile(xyz, 99.8, 0)
    pad = 0.15 * (hi - lo)
    keep = np.all((xyz >= lo - pad) & (xyz <= hi + pad), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    nrm = nrm[keep] if nrm is not None else None
    target_pts = prof["export_pts"] if dense_ok else len(xyz)
    xyz, rgb, nrm, voxel = voxel_downsample(xyz, rgb, nrm, target=target_pts if len(xyz) > target_pts else None)
    if dense_ok:
        k2 = remove_isolated(xyz, 2.0 * voxel, min_neighbors=4)
        xyz, rgb = xyz[k2], rgb[k2]
        nrm = nrm[k2] if nrm is not None else None
    points_ply = os.path.join(dirs["models"], "actionable_threat_map_points.ply")
    write_ply(points_ply, xyz, rgb, nrm)
    render_preview(xyz, rgb, os.path.join(dirs["models"], "preview_topdown.jpg"))
    clock.stop(f"{len(xyz):,} points, voxel {voxel * 100:.1f} cm, {geo['method']}")

    # ---- 6. mesh ----------------------------------------------------------------
    clock.start("6. mesh (Poisson+trim+QEM)")
    mesh = _build_mesh(xyz, rgb, nrm, voxel, dense_ok, prof, cfg, dirs, budget)
    clock.stop(f"{len(mesh['f']):,} faces, {len(mesh['v']):,} vertices ({mesh['method']})")

    # ---- 7. export + bundle ---------------------------------------------------------
    clock.start("7. export + bundle")
    web = ing.get("web_video")
    if web_job is not None:
        web = finish_web_video_job(web_job, max(30.0, budget.remaining()))
    out = _write_bundle(cfg, dirs, vinfo, tel, srt, kfs, summ, geo, mesh, points_ply, web, report, prof, sfm, f35)
    clock.stop(f"{out['zip_mb']:.0f} MB zip")
    report["timings"] = [dict(stage=a, seconds=round(b, 1), note=c) for a, b, c in clock.rows]
    total = sum(r[1] for r in clock.rows)
    report["total_seconds"] = round(total, 1)
    with zipfile.ZipFile(cfg.bundle_zip, "a") as z:
        z.writestr("recon_report.json", json.dumps(report, indent=1, default=_json_default))
    print(clock.table(budget.total), flush=True)
    log(f"DONE: {cfg.bundle_zip} ({os.path.getsize(cfg.bundle_zip) / 1e6:.0f} MB) in {fmt_s(total)}")
    return dict(zip=cfg.bundle_zip, report=report, preview=os.path.join(dirs["models"], "preview_topdown.jpg"))


def _salvage_depth_maps(dense):
    """After an MVS crash, reuse whatever depth maps were finished."""
    dm = os.path.join(dense, "stereo", "depth_maps")
    nm = os.path.join(dense, "stereo", "normal_maps")

    def names(kind):
        out = []
        for p in glob.glob(os.path.join(dm, f"*.{kind}.bin")):
            n = os.path.basename(p)[: -len(f".{kind}.bin")]
            if os.path.exists(os.path.join(nm, f"{n}.{kind}.bin")):
                out.append(n)
        return sorted(out)

    geo, pho = names("geometric"), names("photometric")
    if len(geo) >= 10:
        return dict(photo_done=pho, geo_done=geo, fusion_images=geo, input_type="geometric", salvaged=True)
    if len(pho) >= 10:
        return dict(photo_done=pho, geo_done=[], fusion_images=pho, input_type="photometric", salvaged=True)
    return None


def _post_mvs_reserve(prof, hw):
    """Seconds kept for fusion, meshing, export and zipping."""
    cpu = max(2, hw.get("cpu", 2))
    base = {"t4": 290.0, "mid": 180.0, "high": 150.0}[prof["tier"]]
    return base * (2.0 / cpu) ** 0.35 + 25.0


def _sparse_normals(summ):
    from scipy.spatial import cKDTree
    xyz = summ.xyz
    if len(xyz) < 10:
        return np.tile(np.array([[0, 0, 1.0]], np.float32), (len(xyz), 1))
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(16, len(xyz)), workers=-1)
    nb = xyz[idx] - xyz[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, V = np.linalg.eigh(cov)
    n = V[:, :, 0]
    # orient towards the mean of the observing cameras
    ptr, img = summ.track_ptr, summ.track_img
    cam_mean = np.stack([np.add.reduceat(summ.C[img][:, i], ptr[:-1]) for i in range(3)], 1) / np.diff(ptr)[:, None]
    flip = np.einsum("ni,ni->n", n, cam_mean - xyz) < 0
    n[flip] *= -1
    return n.astype(np.float32)


def _edt_close(mask, r_cells):
    from scipy import ndimage as ndi
    pad = int(math.ceil(r_cells)) + 2
    m = np.pad(mask, pad)
    dil = ndi.distance_transform_edt(~m) <= r_cells
    return (ndi.distance_transform_edt(dil) > r_cells)[pad:-pad, pad:-pad]


def _harmonic_fill(a, known, iters=250):
    from scipy import ndimage as ndi
    idx = ndi.distance_transform_edt(~known, return_distances=False, return_indices=True)
    out = np.where(known, a, a[tuple(idx)]).astype(np.float64)
    unk = ~known
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    for _ in range(iters):
        sm = ndi.convolve(out, k, mode="nearest")
        out[unk] = sm[unk]
    return out


def gap_fill_samples(xyz, rgb, spacing, res=0.5, max_hole_m2=400.0):
    """
    Terrain-aware gap filling for the mesh (the exported point cloud stays purely measured):
    a bare-earth model is built by progressive morphological opening; empty cells enclosed by dense
    data are filled with samples on it (roads torn by traffic, occlusion shadows, missing patches);
    large elongated no-return bodies (canals, ponds - water gives no stereo matches) are closed as
    flat water at the lowest bank level. Fill colours come from an in-painted orthophoto.
    """
    import cv2
    from scipy import ndimage as ndi
    x0, z0 = xyz[:, 0].min() - 2 * res, xyz[:, 2].min() - 2 * res
    nx = int(math.ceil((xyz[:, 0].max() - x0) / res)) + 2
    nz = int(math.ceil((xyz[:, 2].max() - z0) / res)) + 2
    ci = np.floor((xyz[:, 2] - z0) / res).astype(np.int64)
    cj = np.floor((xyz[:, 0] - x0) / res).astype(np.int64)
    key = ci * nx + cj
    order = np.lexsort((xyz[:, 1], key))
    uk, st, cnt = np.unique(key[order], return_index=True, return_counts=True)
    zlo = np.full(nx * nz, np.nan)
    zlo[uk] = xyz[order, 1][st + np.floor(0.1 * (cnt - 1)).astype(np.int64)]
    zlo = zlo.reshape(nz, nx)
    has = np.zeros(nx * nz, bool)
    has[uk] = True
    has = has.reshape(nz, nx)
    col = np.zeros((nx * nz, 3))
    grp = np.repeat(np.arange(len(uk)), cnt)
    for c in range(3):
        col[uk, c] = np.bincount(grp, rgb[order, c].astype(np.float64)) / cnt
    ortho = np.clip(col.reshape(nz, nx, 3), 0, 255).astype(np.uint8)
    fill_ratio = ndi.uniform_filter(has.astype(np.float32), size=max(3, int(round(5.0 / res))) | 1)
    enclosed = _edt_close(fill_ratio >= 0.45, 15.0 / res)
    # bare earth: SMRF-style progressive opening of the minimum surface
    idx = ndi.distance_transform_edt(~has, return_distances=False, return_indices=True)
    zmin = np.where(has, zlo, zlo[tuple(idx)])
    nonground = ~has
    prev = zmin
    rmax = max(2, int(round(18.0 / res)))
    for r in sorted({int(round(v)) for v in np.geomspace(1, rmax, 8)}):
        opened = ndi.grey_opening(prev, size=(2 * r + 1, 2 * r + 1))
        nonground |= (prev - opened) > max(0.25, 0.15 * r * res)
        prev = opened
    dtm = _harmonic_fill(zlo, has & ~nonground)
    holes = enclosed & ~has
    lab, n = ndi.label(holes, np.ones((3, 3)))
    if not n:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    area = np.bincount(lab.ravel(), minlength=n + 1) * res * res
    objs = ndi.find_objects(lab)
    ortho = cv2.inpaint(ortho, (~has).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    sub = max(1, int(round(res / (1.5 * max(spacing, 0.05)))))
    off = (np.arange(sub) + 0.5) / sub - 0.5
    oi, oj = [a.ravel() for a in np.meshgrid(off, off, indexing="ij")]
    pts, cols = [], []
    water_area = ground_area = 0.0
    for k in range(1, n + 1):
        sl = objs[k - 1]
        m = lab[sl] == k
        ii, jj = np.nonzero(m)
        cov = np.cov(np.c_[ii, jj].T) if len(ii) > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))
        elong = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
        gi, gj = ii + sl[0].start, jj + sl[1].start
        if (area[k] >= 150.0 and elong >= 4.0) or area[k] >= 800.0:
            ring = ndi.binary_dilation(m, iterations=3) & ~m & has[sl]
            if ring.sum() < 5:
                continue
            level = float(np.percentile(zlo[sl][ring], 5)) - 0.15
            x, z = x0 + (gj + 0.5) * res, z0 + (gi + 0.5) * res
            pts.append(np.c_[x, np.full(len(x), level), z])
            cols.append(np.tile([52, 66, 72], (len(x), 1)))
            water_area += area[k]
        elif area[k] <= max_hole_m2:
            fi = (gi[:, None] + 0.5 + oi[None, :]).ravel()
            fj = (gj[:, None] + 0.5 + oj[None, :]).ravel()
            from scipy.ndimage import map_coordinates
            y = map_coordinates(dtm, [fi - 0.5, fj - 0.5], order=1, mode="nearest")
            pts.append(np.c_[x0 + fj * res, y, z0 + fi * res])
            cols.append(np.repeat(ortho[gi, gj], sub * sub, axis=0))
            ground_area += area[k]
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    p, c = np.vstack(pts), np.clip(np.vstack(cols), 0, 255).astype(np.uint8)
    log(f"gap filling: {ground_area:.0f} m^2 ground holes, {water_area:.0f} m^2 water surface, {len(p):,} samples")
    return p, c


def _build_mesh(xyz, rgb, nrm, voxel, dense_ok, prof, cfg, dirs, budget):
    from scipy.spatial import cKDTree
    method = "poisson"
    if dense_ok and nrm is not None:
        try:
            fp, fc = gap_fill_samples(xyz, rgb, voxel)
            if len(fp):
                xyz = np.vstack([xyz, fp])
                rgb = np.vstack([rgb, fc])
                nrm = np.vstack([nrm, np.tile(np.array([[0.0, 1.0, 0.0]], np.float32), (len(fp), 1))]).astype(np.float32)
        except Exception as e:
            warn(f"gap filling skipped: {e}")
    ext = float(np.max(xyz.max(0) - xyz.min(0)))
    depth = cfg.poisson_depth or int(np.clip(math.ceil(math.log2(1.1 * ext / max(voxel, 1e-9))), 8, prof["poisson_max_depth"]))
    if budget.remaining() < 120 and depth > 10:
        depth -= 1
    pin = os.path.join(dirs["models"], "poisson_input.ply")
    pout = os.path.join(dirs["models"], "poisson_raw.ply")
    v = f = c = None
    try:
        if nrm is None:
            raise RuntimeError("no normals")
        write_ply(pin, xyz, rgb, nrm)
        # screened Poisson with point weight 4 (Kazhdan's default): the surface follows the samples
        # closely, so roof edges and walls stay crisp instead of melting
        run_worker("poisson", dict(input_path=pin, output_path=pout, depth=depth, trim=0.0,
                                   point_weight=4.0 if dense_ok else 1.0, verbose=cfg.verbose), dirs, verbose=cfg.verbose)
        m = read_ply(pout)
        v, f = ply_xyz(m["vertex"]), m["faces"]
        c = ply_rgb(m["vertex"])
        log(f"poisson depth {depth}: {len(f):,} raw faces")
        # support trimming: remove surface that is not backed by samples.  The
        # tolerance follows the coarser of sample spacing and Poisson cell size.
        cell = 1.1 * ext / float(2 ** depth)
        tol = (2.5 if dense_ok else 4.0) * max(voxel, cell)
        tree = cKDTree(xyz)
        d, nn = tree.query(v, k=1, workers=-1, distance_upper_bound=2.0 * tol)
        far = ~(d < tol)
        f = f[~far[f].any(1)]
        if c is None:
            c = rgb[np.minimum(nn, len(rgb) - 1)]
        f = drop_small_components(v, f, max(200, int(0.002 * len(f))))
        v, f, (c,) = compact_mesh(v, f, (c,))
        if len(f) < 100:
            raise RuntimeError("poisson surface empty after trimming")
    except Exception as e:
        warn(f"Poisson meshing failed ({str(e).splitlines()[0][:200]}); using 2.5D height-field mesh")
        method = "heightfield"
        v, f, c = heightfield_mesh(xyz, rgb, max(voxel * 4.0, ext / 3000.0, 1e-6))
    # simplification
    max_f = prof["mesh_faces"]
    if len(f) > max_f * 1.05:
        sin = os.path.join(dirs["models"], "mesh_full.ply")
        sout = os.path.join(dirs["models"], "mesh_simplified.ply")
        write_ply(sin, v, c, None, f)
        try:
            run_worker("simplify", dict(input_path=sin, output_path=sout, ratio=max_f / float(len(f)),
                                        verbose=cfg.verbose), dirs, verbose=cfg.verbose)
            m = read_ply(sout)
            v2, f2, c2 = ply_xyz(m["vertex"]), m["faces"], ply_rgb(m["vertex"])
            if f2 is not None and len(f2) > 100:
                v, f, c = v2, f2, (c2 if c2 is not None else c[:len(v2)] if c is not None else None)
                method += "+qem"
        except Exception as e:
            warn(f"QEM simplification failed ({str(e).splitlines()[0][:200]}); decimating by face sampling")
            keep = np.linspace(0, len(f) - 1, max_f).astype(np.int64)
            v, f, (c,) = compact_mesh(v, f[keep], (c,))
    if c is None:
        tree = cKDTree(xyz)
        _, nn = tree.query(v, k=1, workers=-1)
        c = rgb[nn]
    n = vertex_normals(v, f)
    return dict(v=v, f=f, c=c, n=n, method=method, depth=depth)


def _write_bundle(cfg, dirs, vinfo, tel, srt, kfs, summ, geo, mesh, points_ply, web, report, prof, sfm, f35):
    md = dirs["models"]
    T = geo["T"]
    # --- mesh files
    mesh_ply = os.path.join(md, "actionable_threat_mesh.ply")
    mesh_obj = os.path.join(md, "actionable_threat_mesh.obj")
    mesh_glb = os.path.join(md, "actionable_threat_mesh.glb")
    write_ply(mesh_ply, mesh["v"], mesh["c"], mesh["n"], mesh["f"])
    write_obj(mesh_obj, mesh["v"], mesh["c"], mesh["f"])
    if cfg.export_glb:
        write_glb(mesh_glb, mesh["v"], mesh["c"], mesh["f"], mesh["n"])
    map_ply = os.path.join(md, "actionable_threat_map.ply")
    shutil.copy2(points_ply, map_ply)
    # --- camera poses in the output frame
    Rs = T[:3, :3] / geo["scale"]
    cams = {}
    flip = np.diag([1.0, -1.0, -1.0])   # COLMAP camera (x right, y down, z fwd) -> three.js camera
    for i, name in enumerate(summ.names):
        Rwc = summ.R[i].T
        pos = apply_T(T, summ.C[i][None])[0]
        Rthree = Rs @ Rwc @ flip
        K = summ.K[i]
        w, h = summ.wh[i]
        cams[name] = dict(position=[round(float(x), 4) for x in pos],
                          quaternion_xyzw=[round(float(x), 6) for x in quat_xyzw(Rthree)],
                          fov_y_deg=round(float(2 * math.degrees(math.atan(h / (2 * K[1, 1])))), 3))
    frame_index = []
    for k in kfs:
        e = dict(index=k["index"], frame_number=k["frame_number"], timestamp_sec=k["timestamp_sec"],
                 filename=k["filename"], blur_score=k["blur_score"])
        for key in ("latitude", "longitude", "relative_altitude_m", "absolute_altitude_m"):
            if key in k:
                e[key] = k[key]
        c = cams.get(k["filename"])
        e["registered"] = c is not None
        if c:
            e.update(c)
        frame_index.append(e)
    json_dump(frame_index, os.path.join(dirs["work"], "frame_index.json"), indent=1)
    # --- telemetry json (original keys first)
    lat0, lon0 = geo.get("lat0"), geo.get("lon0")
    rel = [r["rel_alt"] for r in tel.recs if r["rel_alt"] is not None] if tel else []
    gps_used = str(geo["method"]).startswith("GPS")
    telem = dict(has_gps=bool(tel and tel.has_gps),
                 center_latitude=round(lat0, 7) if lat0 is not None else None,
                 center_longitude=round(lon0, 7) if lon0 is not None else None,
                 logged_altitude_median_m=round(float(np.median(rel)), 2) if rel else None)
    # PRISM dashboard format: waypoints + a georeference record (metric, verified lat/lon order)
    if tel and tel.has_gps:
        telem["waypoints"] = [dict(frame_id=i, latitude=r["lat"], longitude=r["lon"], time_s=round(r["t"], 3),
                                   **({"relative_altitude_m": r["rel_alt"]} if r["rel_alt"] is not None else {}))
                              for i, r in enumerate(tel.g)]
        telem["waypoint_count"] = len(telem["waypoints"])
        telem["source"] = "DJI_SRT"
        telem["latlon_order"] = tel.meta.get("latlon_order")
        telem["latlon_ambiguous"] = False
        telem["latlon_verified"] = bool(geo.get("latlon_verified", not tel.meta.get("latlon_ambiguous")))
        e, n = latlon_to_en(tel.lat, tel.lon, float(np.mean(tel.lat)), float(np.mean(tel.lon)))
        telem["total_distance_meters"] = round(float(np.sum(np.hypot(np.diff(e), np.diff(n)))), 1)
        telem["altitude_reference"] = "relative" if tel.has_alt else "estimated"
    telem["metric_scale_factor"] = 1.0
    telem["georeference"] = dict(
        status="ok", version=3, source="prism_turbo", metric=True, mode=geo["method"], confidence=geo.get("confidence"),
        heading_known=gps_used, origin=geo.get("origin"), scale_rel_uncertainty=geo.get("scale_rel_uncertainty"),
        display_frame="METRIC_YUP", latlon_swapped=geo.get("latlon_order_resolved") == "swapped",
        diagnostics={k: geo.get(k) for k in ("rmse_m", "inliers", "up_method", "camera_roll_median_deg",
                                             "gps_vs_horizon_up_deg", "latlon_test_rmse_m", "scale_source")},
        warnings=[] if gps_used else ["No usable GPS track: scale from altitude; heading unknown."])
    telem.update(
        georeferenced=gps_used,
        scale_source=geo.get("scale_source"), alignment=dict(method=geo["method"], rmse_m=geo.get("rmse_m"),
                                                             inliers=geo.get("inliers"), up_from=geo.get("up_method")),
        coordinate_frame=dict(units="metres", up_axis="+Y", x_axis="East (or arbitrary without GPS)",
                              z_axis="South (-North)", origin="GPS centre of the flight projected to the ground (y = 0)",
                              handedness="right-handed (three.js)"),
        ground_level_offset_m=geo.get("ground_offset_m"),
        focal_35mm_equiv=f35,
        camera_trajectory=[cams[n]["position"] for n in summ.names if n in cams],
        video=dict(width=vinfo["width"], height=vinfo["height"], fps=vinfo["fps"], duration_sec=vinfo["duration"],
                   codec=vinfo["codec"]),
        source_telemetry=os.path.basename(srt) if srt else None)
    json_dump(telem, os.path.join(dirs["work"], "flight_telemetry.json"), indent=1)
    # --- zip
    if os.path.exists(cfg.bundle_zip):
        os.remove(cfg.bundle_zip)
    members = [(points_ply, "actionable_threat_map_points.ply"), (map_ply, "actionable_threat_map.ply"),
               (mesh_ply, "actionable_threat_mesh.ply"), (mesh_obj, "actionable_threat_mesh.obj"),
               (os.path.join(dirs["work"], "flight_telemetry.json"), "flight_telemetry.json"),
               (os.path.join(dirs["work"], "frame_index.json"), "frame_index.json")]
    if cfg.export_glb and os.path.exists(mesh_glb):
        members.append((mesh_glb, "actionable_threat_mesh.glb"))
    prev = os.path.join(md, "preview_topdown.jpg")
    if os.path.exists(prev):
        members.append((prev, "preview_topdown.jpg"))
    if web and os.path.exists(web):
        members.append((web, "drone_flight.mp4"))
    if srt and os.path.exists(srt) and srt.lower().endswith(".srt"):
        members.append((srt, "drone_flight.srt"))
    with zipfile.ZipFile(cfg.bundle_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for src, arc in members:
            ct = zipfile.ZIP_STORED if arc.endswith((".mp4", ".jpg", ".glb")) else zipfile.ZIP_DEFLATED
            z.write(src, arc, compress_type=ct)
    report["outputs"] = dict(points=int(_ply_count(points_ply)), mesh_faces=int(len(mesh["f"])),
                             mesh_vertices=int(len(mesh["v"])), mesh_method=mesh["method"],
                             poisson_depth=mesh.get("depth"), georef=geo["method"],
                             files=[a for _, a in members])
    return dict(zip_mb=os.path.getsize(cfg.bundle_zip) / 1e6)


def _ply_count(path):
    with open(path, "rb") as fh:
        for _ in range(40):
            line = fh.readline().decode("ascii", "ignore")
            if line.startswith("element vertex"):
                return int(line.split()[2])
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[0] == "worker":
        _worker_entry(argv[1], argv[2])
        return
    ap = argparse.ArgumentParser(description="PRISM-TURBO drone video -> 3D bundle")
    ap.add_argument("video")
    ap.add_argument("--srt", default="")
    ap.add_argument("--workdir", default="/content/prism_workspace")
    ap.add_argument("--zip", default="/content/prism_colab_bundle.zip")
    ap.add_argument("--quality", default="balanced", choices=list(QUALITY))
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ns = ap.parse_args(argv)
    cfg = Config(video_path=ns.video, srt_path=ns.srt, workdir=ns.workdir, bundle_zip=ns.zip, quality=ns.quality,
                 target_minutes=ns.minutes, resume=ns.resume, verbose=ns.verbose)
    run(cfg)


if __name__ == "__main__":
    main()