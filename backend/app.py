"""
PRISM // API server (app.py)

Serves the dashboard (frontend/), the data folder, and the analysis API:
  model & calibration   /api/model/info, /api/model/recalibrate, /api/model/meshify(+/status), downloads
  terrain & measurement /api/terrain/*, /api/measure/height, /api/measure/between, /api/analysis/hag
  roads & potholes      /api/road/audit, /api/road/audit/compile
  planning              /api/analysis/sites, /api/analysis/route, /api/analysis/observers
  change detection      /api/baseline/*, /api/baselines, /api/diff/*
  buildings & exports   /api/analysis/buildings, /api/export/catalog, /api/export/file/{product}
  quality & accuracy    /api/quality/report(.html), /api/quality/checkpoints
  missions              /api/pipeline/start|status|cancel|import-colab, /api/cameras, /api/telemetry
"""
import base64
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from pipeline_manager import pipeline_mgr, generate_blender_surface_mesh  # noqa: E402
from change_detector import change_detector, extract_telemetry_bounds    # noqa: E402
from georeference import load_point_cloud                                 # noqa: E402
import terrain as T                                                        # noqa: E402
import height_engine as HE                                                 # noqa: E402
import road_engine as RE                                                   # noqa: E402
import tactical as TA                                                      # noqa: E402
import recalibrate as RC                                                   # noqa: E402
import plyio                                                               # noqa: E402
import buildings as BL                                                     # noqa: E402
import exports as EX                                                       # noqa: E402

app = FastAPI(title="PRISM API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def revalidate_dashboard(request, call_next):
    """The dashboard's ES modules must never be served stale after an update (ETag keeps it cheap)."""
    response = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith(("/js/", "/css/", "/data/")):
        response.headers["Cache-Control"] = "no-cache"
    return response

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
MODELS_DIR = os.path.join(DATA_DIR, "models")
BASELINES_DIR = os.path.join(DATA_DIR, "baselines")
for _d in (UPLOADS_DIR, MODELS_DIR, BASELINES_DIR):
    os.makedirs(_d, exist_ok=True)

app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
for _sub in ("css", "js", "assets"):
    if os.path.exists(os.path.join(FRONTEND_DIR, _sub)):
        app.mount(f"/{_sub}", StaticFiles(directory=os.path.join(FRONTEND_DIR, _sub)), name=_sub)


# =============================================================================
# helpers
# =============================================================================
def clean(o):
    """numpy-safe JSON conversion (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def jr(obj, status_code=200):
    return JSONResponse(content=clean(obj), status_code=status_code)


def _read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


_terrain_lock = threading.Lock()


def terrain_for(baseline_id=None):
    with _terrain_lock:
        try:
            return T.get_terrain(DATA_DIR, baseline_id=baseline_id, log=pipeline_mgr.add_log)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))


def _calibration(telem):
    geo = (telem or {}).get("georeference") or {}
    if not geo:
        return None
    return {"status": geo.get("status"), "metric": bool(geo.get("metric")), "mode": geo.get("mode"),
            "confidence": geo.get("confidence"), "scale_rel_uncertainty": geo.get("scale_rel_uncertainty"),
            "display_frame": geo.get("display_frame"), "units": "m" if geo.get("metric") else "model units",
            "origin": geo.get("origin"), "source": geo.get("source"), "latlon_swapped": geo.get("latlon_swapped"),
            "latlon_verified": (telem or {}).get("latlon_verified"), "warnings": geo.get("warnings", []),
            "reason": geo.get("reason"), "diagnostics": geo.get("diagnostics")}


def _read_calibration(telemetry_path):
    return _calibration(_read_json(telemetry_path, {}))


# =============================================================================
# dashboard, telemetry, cameras
# =============================================================================
@app.get("/")
def serve_dashboard():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, headers={"Cache-Control": "no-cache"})
    return {"status": "PRISM API is running."}


@app.get("/api/telemetry")
def get_telemetry():
    path = os.path.join(DATA_DIR, "flight_telemetry.json")
    data = _read_json(path)
    if data is None:
        return jr({"error": "No telemetry yet. Ingest a mission first.", "waypoints": []}, 404)
    return jr(data)


@app.get("/api/cameras")
def get_cameras():
    """Keyframe camera poses in the model frame (for the video / 3-D overlay)."""
    fi = _read_json(os.path.join(DATA_DIR, "workspace", "frame_index.json"))
    frames = fi if isinstance(fi, list) else []
    frames = [{"t": f.get("timestamp_sec"), "position": f.get("position"), "q": f.get("quaternion_xyzw"),
               "fov_y": f.get("fov_y_deg"), "name": f.get("filename")}
              for f in frames if f.get("position") and f.get("quaternion_xyzw") and f.get("timestamp_sec") is not None]
    telem = _read_json(os.path.join(DATA_DIR, "flight_telemetry.json"), {})
    return jr({"count": len(frames), "frames": frames, "video": telem.get("video")})


# =============================================================================
# model info, recalibration, meshing, downloads
# =============================================================================
@app.get("/api/model/info")
def get_model_info():
    pts = os.path.join(MODELS_DIR, "actionable_threat_map_points.ply")
    if not os.path.exists(pts):
        pts = os.path.join(MODELS_DIR, "actionable_threat_map.ply")
    if not os.path.exists(pts):
        return {"exists": False}
    v, _ = plyio.ply_counts(pts)
    mesh = os.path.join(MODELS_DIR, "actionable_threat_mesh.ply")
    mv, mf = plyio.ply_counts(mesh) if os.path.exists(mesh) else (0, 0)
    telem = _read_json(os.path.join(DATA_DIR, "flight_telemetry.json"), {})
    report = _read_json(os.path.join(DATA_DIR, "recon_report.json"), {}) or {}
    return clean({
        "exists": True, "points_file": os.path.relpath(pts, PROJECT_ROOT).replace("\\", "/"),
        "vertex_count": v, "size_mb": round(os.path.getsize(pts) / 1e6, 1), "last_modified": os.path.getmtime(pts),
        "has_mesh": os.path.exists(mesh), "mesh_vertices": mv, "mesh_faces": mf,
        "mesh_last_modified": os.path.getmtime(mesh) if os.path.exists(mesh) else None,
        "has_obj": os.path.exists(os.path.join(MODELS_DIR, "actionable_threat_mesh.obj")),
        "has_glb": os.path.exists(os.path.join(MODELS_DIR, "actionable_threat_mesh.glb")),
        "has_cameras": isinstance(_read_json(os.path.join(DATA_DIR, "workspace", "frame_index.json")), list),
        "calibration": _calibration(telem), "telemetry_source": telem.get("source"),
        "center_latitude": telem.get("center_latitude"), "center_longitude": telem.get("center_longitude"),
        "recon": {k: report.get(k) for k in ("engine", "total_seconds") if report}, "sfm": report.get("sfm"),
    })


class RecalibrateRequest(BaseModel):
    force: bool = True


@app.post("/api/model/recalibrate")
def recalibrate(req: Optional[RecalibrateRequest] = None):
    try:
        with _terrain_lock:
            out = RC.recalibrate_active_model(DATA_DIR, force=bool(req.force if req else True), log=pipeline_mgr.add_log)
            T._MEM.clear()
        return jr(out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/model/rtk")
async def apply_rtk_positions(rtk_file: UploadFile = File(...)):
    """Re-georeferences the current mission with RTK / PPK positions (RTKLIB .pos or CSV)."""
    ext = os.path.splitext(rtk_file.filename or "")[1].lower() or ".pos"
    path = os.path.join(UPLOADS_DIR, f"rtk_input{ext}")
    with open(path, "wb") as buf:
        shutil.copyfileobj(rtk_file.file, buf)
    try:
        with _terrain_lock:
            out = RC.recalibrate_active_model(DATA_DIR, force=True, log=pipeline_mgr.add_log, rtk_path=path)
            T._MEM.clear()
        _he_cache.clear()
        _bld_cache.clear()
        return jr(out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class MeshifyRequest(BaseModel):
    baseline_id: Optional[str] = None


_mesh_job = {"status": "idle", "percent": 0, "message": "", "result": None, "error": None, "started": None}
_mesh_lock = threading.Lock()


def _mesh_paths(baseline_id):
    if baseline_id:
        bdir = os.path.join(BASELINES_DIR, baseline_id)
        if not os.path.isdir(bdir):
            raise HTTPException(status_code=404, detail=f"Baseline '{baseline_id}' not found.")
        src = os.path.join(bdir, "model_cloud.ply") if os.path.exists(os.path.join(bdir, "model_cloud.ply")) else os.path.join(bdir, "model.ply")
        return src, os.path.join(bdir, "mesh.ply"), os.path.join(bdir, "mesh.obj"), os.path.join(bdir, "mesh.glb"), \
            f"data/baselines/{baseline_id}/mesh.ply"
    src = os.path.join(MODELS_DIR, "actionable_threat_map_points.ply")
    if not os.path.exists(src):
        src = os.path.join(MODELS_DIR, "actionable_threat_map.ply")
    return src, os.path.join(MODELS_DIR, "actionable_threat_mesh.ply"), os.path.join(MODELS_DIR, "actionable_threat_mesh.obj"), \
        os.path.join(MODELS_DIR, "actionable_threat_mesh.glb"), "data/models/actionable_threat_mesh.ply"


def _run_mesh_job(baseline_id):
    try:
        src, ply, obj, glb, url = _mesh_paths(baseline_id)

        def prog(p, msg):
            with _mesh_lock:
                _mesh_job.update(percent=int(p), message=msg)

        stats = generate_blender_surface_mesh(src, ply, obj, baseline_id=baseline_id, output_glb_path=glb,
                                              progress=prog, log=pipeline_mgr.add_log)
        if baseline_id:
            meta_path = os.path.join(BASELINES_DIR, baseline_id, "metadata.json")
            meta = _read_json(meta_path)
            if meta is not None:
                meta.update(has_mesh=True, mesh_vertices=stats["vertices"], mesh_triangles=stats["triangles"])
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
        with _mesh_lock:
            _mesh_job.update(status="completed", percent=100, message="Mesh ready",
                             result=clean(dict(stats, mesh_ply_url=url, baseline_id=baseline_id)))
    except Exception as e:
        with _mesh_lock:
            _mesh_job.update(status="failed", error=f"{type(e).__name__}: {e}", message="Mesh synthesis failed")


@app.post("/api/model/meshify")
def start_meshify(req: Optional[MeshifyRequest] = None):
    baseline_id = req.baseline_id if req else None
    src, *_ = _mesh_paths(baseline_id)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="No point cloud to mesh.")
    with _mesh_lock:
        if _mesh_job["status"] == "running":
            raise HTTPException(status_code=409, detail="A mesh is already being built.")
        _mesh_job.update(status="running", percent=1, message="Starting", result=None, error=None, started=time.time())
    threading.Thread(target=_run_mesh_job, args=(baseline_id,), daemon=True).start()
    return {"success": True, "status": "running"}


@app.get("/api/model/meshify/status")
def meshify_status():
    with _mesh_lock:
        job = dict(_mesh_job)
    if job.get("started"):
        job["elapsed_s"] = round(time.time() - job["started"], 1)
    return jr(job)


@app.get("/api/model/download/{fmt}")
def download_model(fmt: str, baseline_id: Optional[str] = None):
    fmt = fmt.lower().lstrip(".")
    if baseline_id:
        bdir = os.path.join(BASELINES_DIR, baseline_id)
        files = {"obj": ("mesh.obj", "text/plain"), "ply": ("mesh.ply", "application/octet-stream"),
                 "glb": ("mesh.glb", "model/gltf-binary"), "points": ("model.ply", "application/octet-stream")}
        if fmt not in files:
            raise HTTPException(status_code=400, detail="Use obj, ply, glb or points.")
        path = os.path.join(bdir, files[fmt][0])
        name = f"{baseline_id}_{files[fmt][0]}"
    else:
        files = {"obj": ("actionable_threat_mesh.obj", "text/plain"), "ply": ("actionable_threat_mesh.ply", "application/octet-stream"),
                 "glb": ("actionable_threat_mesh.glb", "model/gltf-binary"),
                 "points": ("actionable_threat_map_points.ply", "application/octet-stream")}
        if fmt not in files:
            raise HTTPException(status_code=400, detail="Use obj, ply, glb or points.")
        path = os.path.join(MODELS_DIR, files[fmt][0])
        name = "prism_" + files[fmt][0].replace("actionable_threat_", "")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No {fmt.upper()} file yet. Build the mesh first.")
    return FileResponse(path, filename=name, media_type=files[fmt][1])


# =============================================================================
# terrain, measurement, inventory
# =============================================================================
@app.get("/api/terrain/summary")
def terrain_summary(baseline_id: Optional[str] = None):
    tm, _, telem = terrain_for(baseline_id)
    s = tm.summary()
    s["grid"] = {"x0": tm.x0, "z0": tm.z0, "res": tm.res, "nx": tm.nx, "nz": tm.nz}
    s["calibration"] = _calibration(telem)
    return jr(s)


def _png(img_rgba):
    import cv2
    bgra = cv2.cvtColor(np.ascontiguousarray(img_rgba), cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _colormap(values, vmin, vmax, alpha_mask, cmap="turbo"):
    import cv2
    v = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    cm = {"turbo": cv2.COLORMAP_TURBO, "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS}[cmap]
    rgb = cv2.applyColorMap((v * 255).astype(np.uint8), cm)[..., ::-1]
    a = np.where(alpha_mask, 190, 0).astype(np.uint8)
    return np.dstack([rgb, a])


class ObserverIn(BaseModel):
    x: float
    z: float
    eye_h: float = 1.7


@app.get("/api/terrain/layer/{name}")
def terrain_layer(name: str, baseline_id: Optional[str] = None):
    """Top-down (north-up) analysis layer as a PNG data URL plus its placement in the model frame."""
    tm, _, _ = terrain_for(baseline_id)
    grid = tm
    if name == "classes":
        img = tm.class_image()
    elif name == "heights":
        img = _colormap(tm.ndsm, 0.0, 20.0, tm.observed & (np.nan_to_num(tm.ndsm) > 0.5))
    elif name == "slope":
        img = _colormap(tm.slope, 0.0, 30.0, tm.roi, "magma")
    elif name == "ortho":
        img = np.dstack([np.clip(tm.rgb, 0, 255).astype(np.uint8), np.where(tm.observed, 255, 0).astype(np.uint8)])
    elif name == "exposure":
        grid = tm.coarsen(max(1, int(round(1.0 / tm.res))))
        expo, _ = TA.exposure_map(grid, TA.default_observers(grid))
        img = _colormap(expo, 0.0, 1.0, grid.roi, "turbo")
    else:
        raise HTTPException(status_code=404, detail="Unknown layer. Use classes, heights, slope, ortho or exposure.")
    return jr({"name": name, "image": _png(img), "x0": grid.x0, "z0": grid.z0, "res": grid.res,
               "nx": grid.nx, "nz": grid.nz, "legend": {"classes": T.CLASS_NAMES}})


@app.get("/api/terrain/grid")
def terrain_grid(baseline_id: Optional[str] = None, max_cells: int = 200):
    """Down-sampled bare-earth heights (for draping analysis layers on the terrain)."""
    tm, _, _ = terrain_for(baseline_id)
    f = max(1, int(math.ceil(max(tm.nx, tm.nz) / max(16, min(max_cells, 512)))))
    dtm = tm.dtm[::f, ::f]
    return jr({"x0": tm.x0 + 0.5 * tm.res, "z0": tm.z0 + 0.5 * tm.res, "dx": tm.res * f, "nx": dtm.shape[1],
               "nz": dtm.shape[0], "heights": np.round(dtm, 2).ravel().tolist(),
               "extent": [tm.nx * tm.res, tm.nz * tm.res]})


class PointReq(BaseModel):
    x: float
    z: float
    y: Optional[float] = None
    baseline_id: Optional[str] = None


class TwoPointReq(BaseModel):
    p1: List[float]
    p2: List[float]
    baseline_id: Optional[str] = None


_he_cache = {}


def height_engine(baseline_id=None):
    tm, pts, telem = terrain_for(baseline_id)
    key = (id(tm), pts["path"])
    if key not in _he_cache:
        _he_cache.clear()
        _he_cache[key] = HE.HeightEngine(tm, pts["xyz"], pts["rgb"], telem)
    return _he_cache[key]


@app.post("/api/measure/height")
def measure_height(req: PointReq):
    return jr(height_engine(req.baseline_id).measure_at(req.x, req.z))


@app.post("/api/measure/between")
def measure_between(req: TwoPointReq):
    return jr(height_engine(req.baseline_id).measure_between(req.p1, req.p2))


@app.get("/api/analysis/inventory")
def inventory(baseline_id: Optional[str] = None):
    return jr(height_engine(baseline_id).inventory())


@app.get("/api/analysis/hag")
def height_above_ground(file: Optional[str] = None, baseline_id: Optional[str] = None):
    """Float32 height above the local bare ground for every vertex of `file` (same order as the PLY)."""
    tm, pts, _ = terrain_for(baseline_id)
    xyz = pts["xyz"]
    if file:
        path = os.path.normpath(os.path.join(PROJECT_ROOT, file))
        if not path.startswith(DATA_DIR) or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Unknown model file.")
        if os.path.normcase(path) != os.path.normcase(pts["path"]):
            xyz = plyio.read_ply(path).xyz
    hag = (xyz[:, 1] - tm.sample(tm.dtm, xyz[:, 0], xyz[:, 2])).astype("<f4")
    return Response(content=hag.tobytes(), media_type="application/octet-stream")


# =============================================================================
# roads & potholes
# =============================================================================
ROAD_CACHE = os.path.join(DATA_DIR, "road_pothole_audit.json")


def build_road_report(baseline_id=None):
    tm, pts, telem = terrain_for(baseline_id)
    t0 = time.time()
    roads = RE.detect_roads(tm, log=pipeline_mgr.add_log)
    potholes, st = RE.detect_potholes(tm, pts["xyz"], roads["mask"], telem, log=pipeline_mgr.add_log)
    summ = roads["summary"]
    score = RE.condition_score(potholes, summ["road_area_m2"], st.get("noise_sigma_m"))
    segs = roads["segments"]
    area_by = {}
    for s in segs:
        area_by[s["surface"]] = area_by.get(s["surface"], 0.0) + s["area_m2"]
    surface = max(area_by, key=area_by.get) if area_by else None
    n_high = sum(p["severity"] == "HIGH" for p in potholes)
    n_med = sum(p["severity"] == "MEDIUM" for p in potholes)
    if not segs:
        status, advisory = "NO_ROAD", "No road surface was found in the surveyed area."
    elif n_high >= 2:
        status, advisory = "CRITICAL", "Several deep potholes: wheeled convoys should slow to 10-15 km/h and steer around marked hazards."
    elif n_high or n_med:
        status, advisory = "CAUTION", "Isolated potholes: reduce speed near marked hazards."
    else:
        status, advisory = "PASSABLE", "No potholes deeper than the detection limit on the measured road surface."
    min_w = min((s["width_min_m"] for s in segs), default=None)
    report = {
        "compiled": True, "status": status, "tactical_advisory": advisory,
        "road_classification": {"surface": surface, "surface_label": {"PAVED": "Paved (asphalt / concrete)",
                                                                        "UNPAVED": "Unpaved (earth / gravel)",
                                                                        "MIXED": "Mixed surface"}.get(surface, "Unknown"),
                                "area_by_surface_m2": area_by},
        "network": summ, "segments": segs, "potholes": potholes, "surface_stats": st,
        "condition_score": score,
        "statistics": {"total_potholes": len(potholes), "high": n_high, "medium": n_med,
                       "low": sum(p["severity"] == "LOW" for p in potholes),
                       "craters": sum(p.get("kind") == "CRATER" for p in potholes),
                       "max_depth_cm": max((p["depth_cm"] for p in potholes), default=0.0),
                       "total_damaged_area_m2": round(sum(p["area_m2"] for p in potholes), 2),
                       "narrowest_width_m": min_w},
        "detection_limit_cm": round(100 * st["lod_m"], 1) if st.get("lod_m") else None,
        "seconds": round(time.time() - t0, 1),
        "model_key": _model_key(baseline_id),
    }
    return clean(report)


def _model_key(baseline_id=None):
    p, _, _ = T._active_paths(DATA_DIR, baseline_id)
    return f"{p}:{int(os.path.getmtime(p))}" if p and os.path.exists(p) else None


@app.get("/api/road/audit")
def get_road_audit(force: bool = False, baseline_id: Optional[str] = None):
    if not force:
        cached = _read_json(ROAD_CACHE)
        if cached and cached.get("model_key") == _model_key(baseline_id):
            return jr(cached)
        return jr({"compiled": False, "status": "not_compiled"})
    return compile_road_audit(baseline_id)


@app.post("/api/road/audit/compile")
@app.post("/api/road/audit/recompute")
def compile_road_audit(baseline_id: Optional[str] = None):
    report = build_road_report(baseline_id)
    try:
        with open(ROAD_CACHE, "w", encoding="utf-8") as f:
            json.dump(report, f)
    except OSError:
        pass
    return jr(report)


# =============================================================================
# buildings (rooftops, facades, digital-twin blocks)
# =============================================================================
_bld_cache = {}


def buildings_for(baseline_id=None):
    key = _model_key(baseline_id)
    if key in _bld_cache:
        return _bld_cache[key]
    sv = EX.Survey(DATA_DIR, baseline_id)
    disk = os.path.join(sv.dir, "buildings_cache.json")
    cached = _read_json(disk)
    if cached and cached.get("model_key") == key and cached.get("v") == 2:
        _bld_cache.clear()
        _bld_cache[key] = cached["buildings"]
        return cached["buildings"]
    he = height_engine(baseline_id)
    t0 = time.time()
    b = clean(BL.extract_buildings(he, log=pipeline_mgr.add_log))
    pipeline_mgr.add_log(f"[Buildings] {len(b)} buildings analysed in {time.time() - t0:.1f}s")
    try:
        with open(disk, "w", encoding="utf-8") as f:
            json.dump({"model_key": key, "v": 2, "buildings": b}, f)
    except OSError:
        pass
    _bld_cache.clear()
    _bld_cache[key] = b
    return b


def roads_for(baseline_id=None):
    cached = _read_json(ROAD_CACHE)
    if cached and cached.get("model_key") == _model_key(baseline_id):
        return cached
    return json.loads(compile_road_audit(baseline_id).body)


@app.get("/api/analysis/buildings")
def analysis_buildings(baseline_id: Optional[str] = None):
    b = buildings_for(baseline_id)
    return jr({"summary": BL.summary(b), "buildings": b})


# =============================================================================
# exports & quality report
# =============================================================================
EXPORTS = EX.ExportService(DATA_DIR, terrain_for, height_engine, roads_for, buildings_for,
                           lambda bid: height_engine(bid).inventory(), log=pipeline_mgr.add_log)


def _survey_errors(fn):
    try:
        return fn()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/export/catalog")
def export_catalog(baseline_id: Optional[str] = None):
    return jr(_survey_errors(lambda: EXPORTS.catalog(baseline_id)))


@app.get("/api/export/file/{product}")
def export_file(product: str, baseline_id: Optional[str] = None, vertical_offset: Optional[float] = None):
    if product not in EX.PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Unknown export '{product}'.")
    if product in ("report_html", "report_json", "package"):
        # reports change when check points are added: always rebuilt
        try:
            sv = EX.Survey(DATA_DIR, baseline_id)
            stale = os.path.join(sv.export_dir, f"{sv.slug}_{EX.PRODUCTS[product][2]}")
            if os.path.exists(stale):
                os.remove(stale)
        except (FileNotFoundError, OSError):
            pass
    path, name, media = _survey_errors(lambda: EXPORTS.build(product, baseline_id, vertical_offset))
    return FileResponse(path, filename=name, media_type=media)


@app.get("/api/quality/report")
def quality_report(baseline_id: Optional[str] = None):
    return jr(_survey_errors(lambda: EXPORTS.report(baseline_id)))


@app.get("/api/quality/report.html")
def quality_report_html(baseline_id: Optional[str] = None):
    import quality as Q
    from fastapi.responses import HTMLResponse
    rep = _survey_errors(lambda: EXPORTS.report(baseline_id))
    sv = EX.Survey(DATA_DIR, baseline_id)
    return HTMLResponse(Q.render_html(clean(rep), EXPORTS._preview_data_url(sv, baseline_id)))


class CheckpointReq(BaseModel):
    name: Optional[str] = None
    model: List[float]
    lat: float
    lon: float
    elev: Optional[float] = None
    vertical_offset: Optional[float] = None
    baseline_id: Optional[str] = None


def _cp_payload(baseline_id, cps):
    import quality as Q
    telem = _read_json(EX.Survey(DATA_DIR, baseline_id).telem, {})
    corr = (telem.get("georeference") or {}).get("control_correction")
    return {"checkpoints": cps, "stats": Q.checkpoint_stats(cps, corr), "correction": corr}


@app.get("/api/quality/checkpoints")
def list_checkpoints(baseline_id: Optional[str] = None):
    cps = _survey_errors(lambda: EXPORTS.checkpoints(baseline_id))
    return jr(_cp_payload(baseline_id, cps))


class ControlReq(BaseModel):
    baseline_id: Optional[str] = None
    undo: bool = False


@app.post("/api/quality/checkpoints/apply")
def apply_control_points(req: ControlReq):
    out = _survey_errors(lambda: EXPORTS.apply_control(req.baseline_id, undo=req.undo))
    with _terrain_lock:                    # cached telemetry (lat/lon of every result) must follow
        T._MEM.clear()
    _he_cache.clear()
    return jr(dict(out, **_cp_payload(req.baseline_id, out["checkpoints"])))


@app.post("/api/quality/checkpoints")
def add_checkpoint(req: CheckpointReq):
    import quality as Q
    if not (-90 <= req.lat <= 90 and -180 <= req.lon <= 180) or len(req.model) != 3:
        raise HTTPException(status_code=400, detail="Give latitude/longitude in decimal degrees and a picked model point.")
    cp = _survey_errors(lambda: EXPORTS.add_checkpoint(req.baseline_id, req.name, req.model, req.lat, req.lon,
                                                       req.elev, req.vertical_offset))
    return jr(dict(_cp_payload(req.baseline_id, EXPORTS.checkpoints(req.baseline_id)), checkpoint=cp))


@app.delete("/api/quality/checkpoints/{index}")
def delete_checkpoint(index: int, baseline_id: Optional[str] = None):
    import quality as Q
    cps = _survey_errors(lambda: EXPORTS.delete_checkpoint(baseline_id, index))
    return jr(_cp_payload(baseline_id, cps))


# =============================================================================
# planning: base sites & covert routes
# =============================================================================
class SitesRequest(BaseModel):
    radius_m: float = 15.0
    min_building_dist_m: float = 150.0
    max_slope_deg: float = 6.0
    max_road_dist_m: float = 500.0
    top_k: int = 5
    observers: Optional[List[ObserverIn]] = None
    use_default_observers: bool = True
    baseline_id: Optional[str] = None


class RouteRequest(BaseModel):
    start: List[float]
    end: List[float]
    observers: Optional[List[ObserverIn]] = None
    use_default_observers: bool = True
    include_road_observers: bool = False
    max_slope_deg: float = 35.0
    baseline_id: Optional[str] = None


def _observers(grid, custom, use_default, include_roads=False):
    obs = TA.default_observers(grid, include_roads=include_roads) if use_default else []
    for o in custom or []:
        obs.append({"x": o.x, "z": o.z, "eye_h": o.eye_h, "kind": "operator"})
    return obs


@app.post("/api/analysis/sites")
def base_sites(req: SitesRequest):
    tm, _, telem = terrain_for(req.baseline_id)
    grid = tm.coarsen(max(1, int(round(1.0 / tm.res))))
    obs = _observers(grid, req.observers, req.use_default_observers)
    out = TA.find_base_sites(tm, telem, radius_m=req.radius_m, min_building_dist_m=req.min_building_dist_m,
                             max_slope_deg=req.max_slope_deg, max_road_dist_m=req.max_road_dist_m,
                             top_k=max(1, min(req.top_k, 10)), observers=obs, log=pipeline_mgr.add_log)
    return jr(out)


@app.post("/api/analysis/route")
def covert_route(req: RouteRequest):
    if len(req.start) < 2 or len(req.end) < 2:
        raise HTTPException(status_code=400, detail="start and end need [x, y, z] model coordinates.")
    tm, _, telem = terrain_for(req.baseline_id)
    grid = TA._grid_for_routes(tm)
    obs = _observers(grid, req.observers, req.use_default_observers, req.include_road_observers)
    out = TA.plan_routes(tm, telem, req.start, req.end, observers=obs, max_slope_deg=req.max_slope_deg,
                         log=pipeline_mgr.add_log)
    return jr(out)


@app.get("/api/analysis/observers")
def list_observers(baseline_id: Optional[str] = None, include_roads: bool = False):
    tm, _, _ = terrain_for(baseline_id)
    grid = tm.coarsen(max(1, int(round(1.0 / tm.res))))
    obs = TA.default_observers(grid, include_roads=include_roads)
    for o in obs:
        o["y"] = float(grid.sample(grid.dtm, o["x"], o["z"]))
    return jr({"observers": obs})


# =============================================================================
# baselines & change detection
# =============================================================================
class SaveBaselineRequest(BaseModel):
    name: Optional[str] = None
    keep_video: bool = True


class CompareRequest(BaseModel):
    earlier: Optional[str] = None            # day 1: a saved survey id or "active"
    later: Optional[str] = None              # day 2: a saved survey id or "active" (default: the current mission)
    min_height_m: float = 0.5
    min_area_m2: float = 2.0
    baseline_id: Optional[str] = None        # legacy form: baseline (day 1) vs the current mission
    height_threshold: Optional[float] = None


active_diff_state = {"status": "idle", "data": None}


def _write_srt_from_waypoints(telem, path):
    wps = telem.get("waypoints") or []
    if not wps:
        return False
    with open(path, "w", encoding="utf-8") as out:
        for i, wp in enumerate(wps):
            t0 = wp.get("time_s", float(i))
            t1 = wps[i + 1].get("time_s", t0 + 1.0) if i + 1 < len(wps) else t0 + 1.0

            def ts(t):
                h, r = divmod(t, 3600)
                m, s = divmod(r, 60)
                return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"
            out.write(f"{i + 1}\n{ts(t0)} --> {ts(t1)}\n[latitude: {wp.get('latitude', 0):.8f}] "
                      f"[longitude: {wp.get('longitude', 0):.8f}] [rel_alt: {wp.get('relative_altitude_m', 0) or 0:.3f}]\n\n")
    return True


@app.get("/api/baseline/{baseline_id}/srt")
def get_baseline_srt(baseline_id: str):
    bdir = os.path.join(BASELINES_DIR, baseline_id)
    if not os.path.isdir(bdir):
        raise HTTPException(status_code=404, detail="Baseline not found.")
    srt = os.path.join(bdir, "telemetry.srt")
    if not os.path.exists(srt):
        telem = _read_json(os.path.join(bdir, "telemetry.json"), {})
        if not _write_srt_from_waypoints(telem, srt):
            raise HTTPException(status_code=404, detail="No telemetry for this baseline.")
    return FileResponse(srt, filename=f"{baseline_id}_telemetry.srt", media_type="text/plain")


@app.post("/api/baseline/save")
def save_active_as_baseline(req: Optional[SaveBaselineRequest] = None):
    from datetime import datetime
    pts = os.path.join(MODELS_DIR, "actionable_threat_map_points.ply")
    if not os.path.exists(pts):
        pts = os.path.join(MODELS_DIR, "actionable_threat_map.ply")
    if not os.path.exists(pts):
        raise HTTPException(status_code=404, detail="No active 3D model to archive.")
    bid = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bdir = os.path.join(BASELINES_DIR, bid)
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(pts, os.path.join(bdir, "model.ply"))
    cloud = os.path.join(MODELS_DIR, "actionable_threat_map_cloud.ply")
    has_cloud = os.path.exists(cloud)
    if has_cloud:
        shutil.copy2(cloud, os.path.join(bdir, "model_cloud.ply"))
    has_mesh = False
    for src, dst in (("actionable_threat_mesh.ply", "mesh.ply"), ("actionable_threat_mesh.obj", "mesh.obj"),
                     ("actionable_threat_mesh.glb", "mesh.glb")):
        if os.path.exists(os.path.join(MODELS_DIR, src)):
            shutil.copy2(os.path.join(MODELS_DIR, src), os.path.join(bdir, dst))
            has_mesh = has_mesh or dst == "mesh.ply"
    telem_src = os.path.join(DATA_DIR, "flight_telemetry.json")
    telem = _read_json(telem_src, {})
    bounds = None
    if os.path.exists(telem_src):
        shutil.copy2(telem_src, os.path.join(bdir, "telemetry.json"))
        bounds = extract_telemetry_bounds(telem_src)
    fi = os.path.join(DATA_DIR, "workspace", "frame_index.json")
    if os.path.exists(fi):
        shutil.copy2(fi, os.path.join(bdir, "frame_index.json"))
    video = os.path.join(DATA_DIR, "raw_videos", "drone_flight.mp4")
    if (req is None or req.keep_video) and os.path.exists(video):
        shutil.copy2(video, os.path.join(bdir, "video.mp4"))
    from telemetry_parser import srt_matches_telemetry
    srt_src = next((p for p in (os.path.join(DATA_DIR, "drone_flight.srt"), os.path.join(DATA_DIR, "raw_videos", "drone_flight.srt"))
                    if os.path.exists(p) and srt_matches_telemetry(p, telem) is not False), None)
    has_srt = False
    if srt_src:
        shutil.copy2(srt_src, os.path.join(bdir, "telemetry.srt"))
        has_srt = True
    else:
        has_srt = _write_srt_from_waypoints(telem, os.path.join(bdir, "telemetry.srt"))
    v, _ = plyio.ply_counts(os.path.join(bdir, "model.ply"))
    rep = _read_json(os.path.join(DATA_DIR, "recon_report.json"))
    if rep and int((rep.get("outputs") or {}).get("points") or v) == int(v):
        shutil.copy2(os.path.join(DATA_DIR, "recon_report.json"), os.path.join(bdir, "recon_report.json"))
    for extra in ("checkpoints.json", "preview_topdown.jpg"):
        if os.path.exists(os.path.join(MODELS_DIR, extra)):
            shutil.copy2(os.path.join(MODELS_DIR, extra), os.path.join(bdir, extra))
    name = (req.name if (req and req.name) else f"Survey {datetime.now().strftime('%d %b %Y, %H:%M')}")
    meta = {"id": bid, "name": name, "created_at": datetime.now().isoformat(),
            "size_mb": round(os.path.getsize(os.path.join(bdir, "model.ply")) / 1e6, 2), "vertex_count": v,
            "telemetry_bounds": bounds, "has_metric_cloud": has_cloud, "has_mesh": has_mesh, "has_srt": has_srt,
            "calibration": _calibration(telem)}
    with open(os.path.join(bdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(clean(meta), f, indent=2)
    cal = meta["calibration"] or {}
    msg = f"Saved '{name}' ({v:,} points)."
    if not (cal.get("metric") and cal.get("status") == "ok"):
        msg += " Warning: this model has no metric calibration, so it cannot be used for height comparison."
    elif cal.get("confidence") == "LOW":
        msg += " Note: metric scale confidence is LOW (no usable GPS)."
    return jr({"success": True, "message": msg, "baseline": meta})


@app.get("/api/baselines")
def list_baselines():
    out = []
    for folder in sorted(os.listdir(BASELINES_DIR), reverse=True):
        bdir = os.path.join(BASELINES_DIR, folder)
        if not os.path.isdir(bdir):
            continue
        item = _read_json(os.path.join(bdir, "metadata.json"))
        if not item:
            ply = os.path.join(bdir, "model.ply")
            if not os.path.exists(ply):
                continue
            item = {"id": folder, "name": folder, "created_at": "", "size_mb": round(os.path.getsize(ply) / 1e6, 2),
                    "vertex_count": plyio.ply_counts(ply)[0]}
        item["has_mesh"] = os.path.exists(os.path.join(bdir, "mesh.ply"))
        # file versions: the dashboard adds them to the URLs so a rebuilt model is never taken from a cache
        item["model_version"] = int(os.path.getmtime(os.path.join(bdir, "model.ply"))) if os.path.exists(os.path.join(bdir, "model.ply")) else 0
        item["mesh_version"] = int(os.path.getmtime(os.path.join(bdir, "mesh.ply"))) if item["has_mesh"] else 0
        item["has_srt"] = os.path.exists(os.path.join(bdir, "telemetry.srt"))
        item["has_cameras"] = os.path.exists(os.path.join(bdir, "frame_index.json"))
        vid = next((v for v in ("video.mp4", "video.webm") if os.path.exists(os.path.join(bdir, v))), None)
        if vid:
            item["video_url"] = f"data/baselines/{folder}/{vid}"
        elif item.get("video_url") and not os.path.exists(os.path.join(PROJECT_ROOT, item["video_url"])):
            item["video_url"] = None                  # referenced video was moved or deleted
        item["has_video"] = bool(item.get("video_url"))
        out.append(item)
    return jr({"count": len(out), "baselines": out})


@app.delete("/api/baseline/{baseline_id}")
def delete_baseline(baseline_id: str):
    global active_diff_state
    bdir = os.path.normpath(os.path.join(BASELINES_DIR, baseline_id))
    if not bdir.startswith(BASELINES_DIR) or not os.path.isdir(bdir):
        raise HTTPException(status_code=404, detail="Baseline not found.")
    shutil.rmtree(bdir)
    if (active_diff_state.get("data") or {}).get("baseline_id") == baseline_id:
        active_diff_state = {"status": "idle", "data": None}
    return {"success": True, "message": f"Baseline '{baseline_id}' deleted."}


def _survey_for_change(sid):
    """Terrain, points and telemetry of a saved survey or of the current mission ("active")."""
    bid = None if (not sid or sid == "active") else sid
    tm, pts, telem = terrain_for(bid)
    if bid:
        meta = _read_json(os.path.join(BASELINES_DIR, bid, "metadata.json"), {}) or {}
        name, date = meta.get("name") or bid, meta.get("created_at")
    else:
        name, date = "Current mission", None
    return {"tm": tm, "xyz": pts["xyz"], "telem": telem, "id": bid or "active", "name": name, "date": date}


@app.post("/api/diff/compare")
def compare_surveys(req: CompareRequest):
    """Day-1 vs day-2 change detection between any two surveys of the same place."""
    import change_engine as CE
    global active_diff_state
    earlier = req.earlier or req.baseline_id
    later = req.later or "active"
    if not earlier:
        raise HTTPException(status_code=400, detail="Choose the earlier survey (day 1).")
    if earlier == later:
        raise HTTPException(status_code=400, detail="Choose two different surveys.")
    min_h = req.height_threshold if (req.height_threshold is not None and req.earlier is None) else req.min_height_m
    min_h = float(min(max(min_h, 0.2), 10.0))
    sa = _survey_for_change(earlier)
    sb = _survey_for_change(later)
    result = CE.compare(sa, sb, min_height_m=min_h, min_area_m2=float(max(0.5, req.min_area_m2)), log=pipeline_mgr.add_log)
    img = result.pop("change_image", None)
    if img is not None:
        result["change_layer"] = dict(result.pop("grid"), image=_png(img), name="change")
    result["earlier"]["date"], result["later"]["date"] = sa["date"], sb["date"]
    result = clean(result)
    active_diff_state = {"status": result["status"], "data": {k: v for k, v in result.items() if k != "change_layer"}}
    return jr(result)


@app.get("/api/diff/active")
def get_active_diff():
    return jr(active_diff_state)


# =============================================================================
# height ceiling audit (height above the LOCAL ground, not a global percentile)
# =============================================================================
@app.get("/api/analysis/height-restriction")
@app.post("/api/analysis/height-restriction")
def analyze_height_restriction(max_height_m: float = 5.0, baseline_id: Optional[str] = None):
    tm, pts, _ = terrain_for(baseline_id)
    xyz = pts["xyz"]
    hag = xyz[:, 1] - tm.sample(tm.dtm, xyz[:, 0], xyz[:, 2])
    breach = hag > max_height_m
    k = int(np.argmax(hag))
    return jr({"status": "completed", "max_height_threshold_m": max_height_m, "datum_mode": "local_ground",
               "total_points": len(xyz), "violating_points": int(breach.sum()),
               "violation_percent": round(100.0 * float(breach.mean()), 2),
               "peak_elevation_m": round(float(hag[k]), 2), "max_breach_m": round(max(0.0, float(hag[k]) - max_height_m), 2),
               "violation_detected": bool(breach.any()),
               "peak_coordinates": {"x": xyz[k, 0], "y": xyz[k, 1], "z": xyz[k, 2]} if breach.any() else None})


# =============================================================================
# missions: local pipeline and Colab bundles
# =============================================================================
@app.post("/api/pipeline/start")
async def start_reconstruction(
    input_type: str = Form("video"),
    has_telemetry: bool = Form(False),
    quality: str = Form("medium"),
    media_file: UploadFile = File(...),
    telemetry_file: Optional[UploadFile] = File(None),
    flight_altitude_m: Optional[float] = Form(None),
    rtk_file: Optional[UploadFile] = File(None),
    intrinsics_file: Optional[UploadFile] = File(None),
    imu_file: Optional[UploadFile] = File(None),
    dynamic_masks: bool = Form(True),
):
    if pipeline_mgr.get_status()["status"] == "running":
        raise HTTPException(status_code=400, detail="A reconstruction is already running.")
    if flight_altitude_m is not None and not (1.0 <= flight_altitude_m <= 1000.0):
        raise HTTPException(status_code=400, detail="flight_altitude_m must be between 1 and 1000 metres.")
    ext = os.path.splitext(media_file.filename or "")[1].lower()
    # one file per input kind: repeated uploads replace each other instead of piling up gigabytes
    media_path = os.path.join(UPLOADS_DIR, f"mission_input{ext}")
    with open(media_path, "wb") as buf:
        shutil.copyfileobj(media_file.file, buf, length=16 * 1024 * 1024)
    telem_path = None
    if has_telemetry and telemetry_file is not None and telemetry_file.filename:
        telem_path = os.path.join(UPLOADS_DIR, f"telemetry_input{os.path.splitext(telemetry_file.filename)[1].lower()}")
        with open(telem_path, "wb") as buf:
            shutil.copyfileobj(telemetry_file.file, buf)
    extras = {"dynamic_masks": bool(dynamic_masks)}
    for key, up in (("rtk_path", rtk_file), ("intrinsics_path", intrinsics_file), ("imu_path", imu_file)):
        if up is not None and up.filename:
            p = os.path.join(UPLOADS_DIR, f"{key.replace('_path', '')}_input{os.path.splitext(up.filename)[1].lower()}")
            with open(p, "wb") as buf:
                shutil.copyfileobj(up.file, buf)
            extras[key] = p
    if ext == ".zip":
        try:
            with zipfile.ZipFile(media_path) as z:
                names = [n.lower() for n in z.namelist()]
            if any("actionable_threat" in n or "flight_telemetry.json" in n for n in names):
                result = deploy_colab_bundle(media_path)
                return jr(dict(result, success=True, job_id="colab_bundle", input_type="colab"))
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="The .zip file is damaged.")
    ok, job_id = pipeline_mgr.start_pipeline(input_type=input_type.lower(), has_telemetry=bool(telem_path),
                                             quality=quality.lower(), media_path=media_path, telemetry_path=telem_path,
                                             flight_altitude_m=flight_altitude_m, extras=extras)
    if not ok:
        raise HTTPException(status_code=400, detail=job_id)
    return {"success": True, "job_id": job_id, "input_type": input_type, "quality": quality,
            "has_telemetry": bool(telem_path)}


@app.get("/api/pipeline/status")
def get_pipeline_status():
    return jr(pipeline_mgr.get_status())


@app.post("/api/pipeline/cancel")
def cancel_pipeline():
    ok = pipeline_mgr.cancel()
    return {"success": ok, "message": "Pipeline cancelled" if ok else "No running job"}


def deploy_colab_bundle(zip_path):
    """Unpacks a PRISM-Turbo bundle into the active model slots and verifies its georeferencing."""
    extract_dir = tempfile.mkdtemp(prefix="prism_bundle_")
    deployed = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        found = {}
        for root, _, files in os.walk(extract_dir):
            for f in files:
                found[f.lower()] = os.path.join(root, f)
        raw_dir = os.path.join(DATA_DIR, "raw_videos")
        ws_dir = os.path.join(DATA_DIR, "workspace")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(ws_dir, exist_ok=True)
        # the previous mission's derived files must not survive
        for stale in ("actionable_threat_mesh.ply", "actionable_threat_mesh.obj", "actionable_threat_mesh.glb",
                      "actionable_threat_map_diff.ply", "actionable_threat_map_cloud.ply", "terrain_cache.npz",
                      "checkpoints.json", "buildings_cache.json"):
            p = os.path.join(MODELS_DIR, stale)
            if os.path.exists(p):
                os.remove(p)
        old_backup = os.path.join(MODELS_DIR, "_pre_recalibration")
        if os.path.isdir(old_backup):
            os.replace(old_backup, old_backup + time.strftime("_%Y%m%d_%H%M%S"))
        for p in (ROAD_CACHE,):
            if os.path.exists(p):
                os.remove(p)
        pts_src = found.get("actionable_threat_map_points.ply") or found.get("actionable_threat_map.ply")
        if pts_src:
            shutil.copy2(pts_src, os.path.join(MODELS_DIR, "actionable_threat_map_points.ply"))
            shutil.copy2(pts_src, os.path.join(MODELS_DIR, "actionable_threat_map.ply"))
            deployed.append("point cloud")
        for name in ("actionable_threat_mesh.ply", "actionable_threat_mesh.obj", "actionable_threat_mesh.glb"):
            if name in found:
                shutil.copy2(found[name], os.path.join(MODELS_DIR, name))
                deployed.append(name.split(".")[-1].upper() + " mesh")
        if "flight_telemetry.json" in found:
            shutil.copy2(found["flight_telemetry.json"], os.path.join(DATA_DIR, "flight_telemetry.json"))
            deployed.append("telemetry")
        srt = next((found[k] for k in found if k.endswith(".srt")), None)
        if srt:
            shutil.copy2(srt, os.path.join(DATA_DIR, "drone_flight.srt"))
            shutil.copy2(srt, os.path.join(raw_dir, "drone_flight.srt"))
            deployed.append("SRT log")
        if "frame_index.json" in found:
            shutil.copy2(found["frame_index.json"], os.path.join(ws_dir, "frame_index.json"))
            deployed.append("camera poses")
        for v in ("drone_flight.mp4", "flight.mp4", "video.mp4"):
            if v in found:
                shutil.copy2(found[v], os.path.join(raw_dir, "drone_flight.mp4"))
                deployed.append("video")
                break
        for name, dst in (("recon_report.json", os.path.join(DATA_DIR, "recon_report.json")),
                          ("preview_topdown.jpg", os.path.join(MODELS_DIR, "preview_topdown.jpg"))):
            if name in found:
                shutil.copy2(found[name], dst)
        # verify / fix metric scale and orientation with the flight log + camera poses
        recal = None
        telem = _read_json(os.path.join(DATA_DIR, "flight_telemetry.json"), {})
        verified = (telem.get("georeference") or {}).get("status") == "ok" and telem.get("latlon_verified")
        if not verified:
            try:
                with _terrain_lock:
                    recal = RC.recalibrate_active_model(DATA_DIR, force=True, log=pipeline_mgr.add_log)
                deployed.append("GPS recalibration")
            except Exception as e:
                recal = {"status": "skipped", "reason": str(e)}
        T._MEM.clear()
        v, _ = plyio.ply_counts(os.path.join(MODELS_DIR, "actionable_threat_map_points.ply"))
        _, fcount = plyio.ply_counts(os.path.join(MODELS_DIR, "actionable_threat_mesh.ply"))
        with pipeline_mgr.lock:
            pipeline_mgr.status, pipeline_mgr.progress_percent, pipeline_mgr.current_stage = "completed", 100, "COMPLETED"
            pipeline_mgr.end_time = time.time()
        pipeline_mgr.add_log(f"Cloud bundle deployed: {', '.join(deployed)} ({v:,} points, {fcount:,} mesh faces).")
        return {"message": f"Deployed {len(deployed)} items from the cloud bundle.", "deployed_files": deployed,
                "vertex_count": v, "face_count": fcount, "recalibration": recal,
                "has_video": "video" in deployed, "has_telemetry": "telemetry" in deployed}
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


@app.post("/api/pipeline/import-colab")
async def import_colab_bundle(bundle_file: UploadFile = File(...)):
    if not (bundle_file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Choose the prism_colab_bundle.zip produced by the Colab notebook.")
    path = os.path.join(UPLOADS_DIR, "colab_bundle.zip")
    with open(path, "wb") as buf:
        shutil.copyfileobj(bundle_file.file, buf, length=16 * 1024 * 1024)
    try:
        return jr(dict(deploy_colab_bundle(path), success=True))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The .zip file is damaged.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not deploy the bundle: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    print(f" PRISM dashboard: http://127.0.0.1:{port}   (frontend: {FRONTEND_DIR})")
    uvicorn.run(app, host="127.0.0.1", port=port)
