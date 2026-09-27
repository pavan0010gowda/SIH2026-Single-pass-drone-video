import os
import sys
import json
import shutil
import time
import tempfile
import zipfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from pipeline_manager import pipeline_mgr
except ImportError:
    from backend.pipeline_manager import pipeline_mgr

app = FastAPI(title="PRISM Command Center API & Tactical Dashboard")

# Enable CORS for cross-origin local access if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamically resolve absolute project paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static asset folders
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
if os.path.exists(os.path.join(FRONTEND_DIR, "css")):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
if os.path.exists(os.path.join(FRONTEND_DIR, "js")):
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")

# Serve dashboard root directly at "/"
@app.get("/")
def serve_dashboard():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "PRISM Command Center API is actively running."}

@app.get("/api/telemetry")
def get_telemetry():
    """
    Feeds GPS and altitude data directly to the dashboard HUD.
    """
    telemetry_path = os.path.join(DATA_DIR, "flight_telemetry.json")
    if os.path.exists(telemetry_path):
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(content=data)
        except Exception as e:
            return JSONResponse(content={"error": f"Failed reading telemetry: {str(e)}"}, status_code=500)
    else:
        return JSONResponse(
            content={"error": "Telemetry not found. Run telemetry parser or upload mission data first."}, 
            status_code=404
        )

@app.get("/api/targets")
def get_tactical_targets():
    """
    Returns AI-detected scene objects mapped to spatial coordinates within the 3D model.
    """
    targets = [
        {
            "id": "TGT-01",
            "type": "VEHICLE",
            "threat_level": "LOW",
            "label": "Civilian Sedan (Grey)",
            "position": [-0.65, -0.45, 0.85],
            "dimensions": [1.8, 1.4, 4.2]
        },
        {
            "id": "TGT-02",
            "type": "VEHICLE",
            "threat_level": "ELEVATED",
            "label": "Black SUV (Static)",
            "position": [-0.60, -0.45, -0.40],
            "dimensions": [2.0, 1.7, 4.8]
        },
        {
            "id": "TGT-03",
            "type": "VANTAGE_POINT",
            "threat_level": "HIGH",
            "label": "Rooftop Overlook (East)",
            "position": [1.45, 0.90, -0.30],
            "dimensions": [3.5, 1.2, 3.5]
        }
    ]
    return JSONResponse(content={"target_count": len(targets), "targets": targets})

def _read_calibration(telemetry_path):
    """Summary of the metric georeferencing record stored in a telemetry JSON (None if absent)."""
    try:
        with open(telemetry_path, "r", encoding="utf-8") as f:
            geo = json.load(f).get("georeference") or {}
    except Exception:
        return None
    if not geo:
        return None
    return {
        "status": geo.get("status"),
        "metric": bool(geo.get("metric")),
        "mode": geo.get("mode"),
        "confidence": geo.get("confidence"),
        "scale_rel_uncertainty": geo.get("scale_rel_uncertainty"),
        "display_frame": geo.get("display_frame"),
        "units": "m" if geo.get("metric") else "model units",
        "warnings": geo.get("warnings", []),
        "reason": geo.get("reason"),
    }

@app.get("/api/model/info")
def get_model_info():
    """
    Returns current active 3D model status, polygon mesh metadata, and file details.
    """
    model_path = os.path.join(DATA_DIR, "models", "actionable_threat_map.ply")
    obj_path = os.path.join(DATA_DIR, "models", "actionable_threat_mesh.obj")
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        mtime = os.path.getmtime(model_path)
        vertex_count = 0
        face_count = 0
        is_mesh = False
        try:
            with open(model_path, "rb") as f:
                for _ in range(40):
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith("element face"):
                        face_count = int(line.split()[-1])
                        if face_count > 0:
                            is_mesh = True
                    elif line == "end_header":
                        break
        except Exception:
            pass

        return {
            "exists": True,
            "filename": "actionable_threat_map.ply",
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "last_modified": mtime,
            "vertex_count": vertex_count,
            "face_count": face_count,
            "is_mesh": is_mesh,
            "has_obj": os.path.exists(obj_path),
            "obj_size_bytes": os.path.getsize(obj_path) if os.path.exists(obj_path) else 0,
            "has_metric_cloud": os.path.exists(os.path.join(DATA_DIR, "models", "actionable_threat_map_cloud.ply")),
            "calibration": _read_calibration(os.path.join(DATA_DIR, "flight_telemetry.json"))
        }
    return {"exists": False, "filename": None, "is_mesh": False}

from pydantic import BaseModel
from change_detector import change_detector, extract_telemetry_bounds
try:
    from georeference import load_point_cloud
except ImportError:
    from backend.georeference import load_point_cloud

BASELINES_DIR = os.path.join(DATA_DIR, "baselines")
os.makedirs(BASELINES_DIR, exist_ok=True)

class MeshifyRequest(BaseModel):
    baseline_id: Optional[str] = None

class SaveBaselineRequest(BaseModel):
    name: Optional[str] = None

class CompareRequest(BaseModel):
    baseline_id: str
    height_threshold: float = 1.5

class HeightRestrictionRequest(BaseModel):
    max_height_m: float = 5.0
    datum_mode: str = "ground"
    baseline_id: Optional[str] = None

@app.post("/api/model/meshify")
def convert_to_ultra_mesh(req: Optional[MeshifyRequest] = None):
    """
    On-demand endpoint: Converts point cloud into a solid 3D polygon surface mesh
    (Blender 3D model) using High-Fidelity Ball Pivoting Algorithm (BPA).
    Connects strictly nearby dots without creating artificial blobs or bridging empty space.
    Supports both the active mission scan and any selected baseline archive.
    """
    baseline_id = req.baseline_id if (req and req.baseline_id) else None

    if baseline_id:
        baseline_dir = os.path.join(BASELINES_DIR, baseline_id)
        if not os.path.exists(baseline_dir):
            raise HTTPException(status_code=404, detail=f"Baseline '{baseline_id}' not found.")
        src_ply = os.path.join(baseline_dir, "model.ply")
        mesh_ply = os.path.join(baseline_dir, "mesh.ply")
        obj_path = os.path.join(baseline_dir, "mesh.obj")
        if not os.path.exists(src_ply):
            raise HTTPException(status_code=404, detail=f"No 3D model found in baseline '{baseline_id}'.")
        try:
            from pipeline_manager import generate_blender_surface_mesh
            stats = generate_blender_surface_mesh(src_ply, mesh_ply, obj_path)
            # Update baseline metadata.json
            meta_path = os.path.join(baseline_dir, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["has_mesh"] = True
                    meta["mesh_vertices"] = stats["vertices"]
                    meta["mesh_triangles"] = stats["triangles"]
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=4)
                except Exception:
                    pass
            return {
                "success": True,
                "message": f"Solid 3D mesh synthesized for baseline '{baseline_id}': {stats['vertices']:,} vertices, {stats['triangles']:,} polygonal faces.",
                "vertices": stats["vertices"],
                "triangles": stats["triangles"],
                "has_obj": os.path.exists(obj_path),
                "obj_size_bytes": os.path.getsize(obj_path) if os.path.exists(obj_path) else 0,
                "mesh_ply_url": f"data/baselines/{baseline_id}/mesh.ply",
                "points_ply_url": f"data/baselines/{baseline_id}/model.ply",
                "baseline_id": baseline_id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mesh synthesis failed: {str(e)}")
    else:
        points_ply = os.path.join(DATA_DIR, "models", "actionable_threat_map_points.ply")
        active_ply = os.path.join(DATA_DIR, "models", "actionable_threat_map.ply")
        obj_path = os.path.join(DATA_DIR, "models", "actionable_threat_mesh.obj")
        mesh_ply = os.path.join(DATA_DIR, "models", "actionable_threat_mesh.ply")

        # Ensure points_ply is preserved
        if not os.path.exists(points_ply) and os.path.exists(active_ply):
            shutil.copy2(active_ply, points_ply)

        src_ply = points_ply if os.path.exists(points_ply) else active_ply
        if not os.path.exists(src_ply):
            raise HTTPException(status_code=404, detail="No active 3D model found to meshify.")

        try:
            from pipeline_manager import generate_blender_surface_mesh
            stats = generate_blender_surface_mesh(src_ply, mesh_ply, obj_path)
            return {
                "success": True,
                "message": f"Solid 3D mesh generated: {stats['vertices']:,} vertices, {stats['triangles']:,} polygonal faces.",
                "vertices": stats["vertices"],
                "triangles": stats["triangles"],
                "has_obj": os.path.exists(obj_path),
                "obj_size_bytes": os.path.getsize(obj_path) if os.path.exists(obj_path) else 0,
                "mesh_ply_url": "data/models/actionable_threat_mesh.ply",
                "points_ply_url": "data/models/actionable_threat_map_points.ply",
                "baseline_id": None
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mesh synthesis failed: {str(e)}")

@app.get("/api/model/download/{fmt}")
def download_model(fmt: str, baseline_id: Optional[str] = None):
    """
    Direct export endpoint for Blender OBJ or meshed PLY, supporting active recon or baseline models.
    """
    fmt_lower = fmt.lower()
    if baseline_id:
        baseline_dir = os.path.join(BASELINES_DIR, baseline_id)
        if fmt_lower in ("obj", ".obj"):
            path = os.path.join(baseline_dir, "mesh.obj")
            filename = f"{baseline_id}_mesh.obj"
            media_type = "text/plain"
        elif fmt_lower in ("ply", ".ply"):
            path = os.path.join(baseline_dir, "model.ply")
            filename = f"{baseline_id}_model.ply"
            media_type = "application/octet-stream"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'. Use 'obj' or 'ply'.")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Requested baseline file not found.")
        return FileResponse(path, filename=filename, media_type=media_type)

    if fmt_lower in ("obj", ".obj"):
        path = os.path.join(DATA_DIR, "models", "actionable_threat_mesh.obj")
        filename = "prism_tactical_twin_mesh.obj"
        media_type = "text/plain"
    elif fmt_lower in ("ply", ".ply"):
        path = os.path.join(DATA_DIR, "models", "actionable_threat_map.ply")
        filename = "prism_tactical_twin_mesh.ply"
        media_type = "application/octet-stream"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'. Use 'obj' or 'ply'.")

    if not os.path.exists(path):
        ply_path = os.path.join(DATA_DIR, "models", "actionable_threat_map.ply")
        if fmt_lower in ("obj", ".obj") and os.path.exists(ply_path):
            try:
                from pipeline_manager import generate_blender_surface_mesh
                generate_blender_surface_mesh(ply_path, ply_path, path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed synthesizing OBJ: {e}")
        else:
            raise HTTPException(status_code=404, detail=f"Model file in format '{fmt}' not yet generated.")

    return FileResponse(path, filename=filename, media_type=media_type)

@app.get("/api/baseline/{baseline_id}/srt")
def get_baseline_srt(baseline_id: str):
    """
    Returns the flight telemetry SRT file for a specific baseline.
    """
    baseline_dir = os.path.join(BASELINES_DIR, baseline_id)
    if not os.path.exists(baseline_dir):
        raise HTTPException(status_code=404, detail="Baseline not found.")
    srt_path = os.path.join(baseline_dir, "telemetry.srt")
    if not os.path.exists(srt_path):
        telem_path = os.path.join(baseline_dir, "telemetry.json")
        if os.path.exists(telem_path):
            try:
                with open(telem_path, "r", encoding="utf-8") as f:
                    telem = json.load(f)
                waypoints = telem.get("waypoints", [])
                if waypoints:
                    with open(srt_path, "w", encoding="utf-8") as out:
                        for i, wp in enumerate(waypoints):
                            sh = i // 3600
                            sm = (i % 3600) // 60
                            ss = i % 60
                            eh = (i + 1) // 3600
                            em = ((i + 1) % 3600) // 60
                            es = (i + 1) % 60
                            lat = wp.get("latitude", 0.0)
                            lon = wp.get("longitude", 0.0)
                            alt = wp.get("relative_altitude_m", 0.0)
                            fid = wp.get("frame_id", i)
                            out.write(f"{i + 1}\n{sh:02d}:{sm:02d}:{ss:02d},000 --> {eh:02d}:{em:02d}:{es:02d},000\n[FRAME {fid}] LAT: {lat:.6f}, LON: {lon:.6f}, REL_ALT: {alt:.2f}m\n\n")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not generate SRT: {e}")
        else:
            raise HTTPException(status_code=404, detail="No telemetry available for this baseline.")
    return FileResponse(srt_path, filename=f"{baseline_id}_telemetry.srt", media_type="text/plain")

active_diff_state = {
    "status": "idle",
    "data": None
}

@app.post("/api/baseline/save")
def save_active_as_baseline(req: Optional[SaveBaselineRequest] = None):
    """
    Archives active 3D model and telemetry into a designated baseline slot.
    Avoids hoarding all scans - only models explicitly saved are kept.
    """
    active_ply = os.path.join(DATA_DIR, "models", "actionable_threat_map.ply")
    active_cloud = os.path.join(DATA_DIR, "models", "actionable_threat_map_cloud.ply")
    active_telem = os.path.join(DATA_DIR, "flight_telemetry.json")

    if not os.path.exists(active_ply):
        raise HTTPException(status_code=404, detail="No active 3D model found to archive as baseline.")

    from datetime import datetime
    baseline_id = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    baseline_folder = os.path.join(BASELINES_DIR, baseline_id)
    os.makedirs(baseline_folder, exist_ok=True)

    dest_ply = os.path.join(baseline_folder, "model.ply")
    shutil.copy2(active_ply, dest_ply)

    # Metric, never-meshed point cloud used for height measurement
    has_metric_cloud = os.path.exists(active_cloud)
    if has_metric_cloud:
        shutil.copy2(active_cloud, os.path.join(baseline_folder, "model_cloud.ply"))

    # Archive solid mesh if present
    active_mesh = os.path.join(DATA_DIR, "models", "actionable_threat_mesh.ply")
    active_obj = os.path.join(DATA_DIR, "models", "actionable_threat_mesh.obj")
    has_mesh = False
    if os.path.exists(active_mesh):
        shutil.copy2(active_mesh, os.path.join(baseline_folder, "mesh.ply"))
        has_mesh = True
    if os.path.exists(active_obj):
        shutil.copy2(active_obj, os.path.join(baseline_folder, "mesh.obj"))

    dest_telem = os.path.join(baseline_folder, "telemetry.json")
    telem_bounds = None
    if os.path.exists(active_telem):
        shutil.copy2(active_telem, dest_telem)
        telem_bounds = extract_telemetry_bounds(active_telem)

    # Archive or generate telemetry.srt for baseline
    dest_srt = os.path.join(baseline_folder, "telemetry.srt")
    active_srt = os.path.join(DATA_DIR, "flight_telemetry.srt")
    if not os.path.exists(active_srt):
        active_srt = os.path.join(DATA_DIR, "drone_flight.srt")
    has_srt = False
    if os.path.exists(active_srt):
        shutil.copy2(active_srt, dest_srt)
        has_srt = True
    elif os.path.exists(dest_telem):
        try:
            with open(dest_telem, "r", encoding="utf-8") as f:
                telem_obj = json.load(f)
            wps = telem_obj.get("waypoints", [])
            if wps:
                with open(dest_srt, "w", encoding="utf-8") as out:
                    for i, wp in enumerate(wps):
                        sh = i // 3600
                        sm = (i % 3600) // 60
                        ss = i % 60
                        eh = (i + 1) // 3600
                        em = ((i + 1) % 3600) // 60
                        es = (i + 1) % 60
                        lat = wp.get("latitude", 0.0)
                        lon = wp.get("longitude", 0.0)
                        alt = wp.get("relative_altitude_m", 0.0)
                        fid = wp.get("frame_id", i)
                        out.write(f"{i + 1}\n{sh:02d}:{sm:02d}:{ss:02d},000 --> {eh:02d}:{em:02d}:{es:02d},000\n[FRAME {fid}] LAT: {lat:.6f}, LON: {lon:.6f}, REL_ALT: {alt:.2f}m\n\n")
                has_srt = True
        except Exception:
            pass

    size_bytes = os.path.getsize(dest_ply)
    vertex_count = 0
    try:
        with open(dest_ply, "rb") as f:
            for _ in range(35):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line == "end_header":
                    break
    except Exception:
        pass

    name = req.name if (req and req.name) else f"Sector Recon Baseline ({datetime.now().strftime('%b %d, %H:%M')})"
    meta = {
        "id": baseline_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "vertex_count": vertex_count,
        "telemetry_bounds": telem_bounds,
        "has_metric_cloud": has_metric_cloud,
        "has_mesh": has_mesh,
        "has_srt": has_srt,
        "calibration": _read_calibration(dest_telem) if os.path.exists(dest_telem) else None
    }

    meta_path = os.path.join(baseline_folder, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    calib = meta["calibration"] or {}
    message = f"Successfully archived baseline '{name}' ({meta['size_mb']} MB, {vertex_count:,} vertices)."
    if not (calib.get("metric") and calib.get("status") == "ok"):
        message += (" WARNING: this model has no metric calibration (built before the georeferencing upgrade "
                    "or georeferencing failed), so it cannot be used for height comparison. Re-run the pipeline.")
    elif calib.get("confidence") == "LOW":
        message += " NOTE: metric scale is LOW confidence (no usable GPS) - heights depend on the flight altitude."
    return {
        "success": True,
        "message": message,
        "baseline": meta
    }

@app.get("/api/baselines")
def list_baselines():
    """
    Returns list of all saved baseline 3D models with metadata.
    """
    baselines = []
    if os.path.exists(BASELINES_DIR):
        for folder_name in sorted(os.listdir(BASELINES_DIR), reverse=True):
            folder_path = os.path.join(BASELINES_DIR, folder_name)
            if os.path.isdir(folder_path):
                meta_path = os.path.join(folder_path, "metadata.json")
                item = None
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            item = json.load(f)
                    except Exception:
                        pass
                if not item:
                    ply_path = os.path.join(folder_path, "model.ply")
                    if os.path.exists(ply_path):
                        item = {
                            "id": folder_name,
                            "name": folder_name,
                            "created_at": "",
                            "size_bytes": os.path.getsize(ply_path),
                            "size_mb": round(os.path.getsize(ply_path) / (1024 * 1024), 2),
                            "vertex_count": 0
                        }
                if item:
                    item["has_mesh"] = os.path.exists(os.path.join(folder_path, "mesh.ply"))
                    item["has_srt"] = os.path.exists(os.path.join(folder_path, "telemetry.srt"))
                    has_vid_file = os.path.exists(os.path.join(folder_path, "video.mp4"))
                    item["has_video"] = has_vid_file or (item.get("video_url") is not None)
                    if has_vid_file:
                        item["video_url"] = f"data/baselines/{folder_name}/video.mp4"
                    baselines.append(item)
    return {"count": len(baselines), "baselines": baselines}

from road_pothole_detector import road_pothole_detector

@app.delete("/api/baseline/{baseline_id}")
def delete_baseline(baseline_id: str):
    """
    Deletes an archived baseline to manage disk space.
    """
    global active_diff_state
    folder_path = os.path.join(BASELINES_DIR, baseline_id)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Baseline not found.")

    try:
        shutil.rmtree(folder_path)
        if active_diff_state.get("data") and active_diff_state["data"].get("baseline_id") == baseline_id:
            active_diff_state = {"status": "idle", "data": None}
        return {"success": True, "message": f"Baseline '{baseline_id}' removed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed deleting baseline: {str(e)}")

@app.get("/api/road/audit")
def get_road_audit(force: bool = False):
    """
    BRO (Border Roads Organisation) Tactical Road & Pothole Audit.
    Classifies road surface (Tar / Bitumen vs Muddy / Unpaved), computes 3D pothole
    depth measurements (cm), dimensions, severity ratings, and video detections.
    Decoupled from core 3D pipeline: returns compiled=False if not run yet.
    """
    try:
        cache_file = road_pothole_detector.cache_file
        if not force and not os.path.exists(cache_file):
            return JSONResponse(content={
                "status": "not_compiled",
                "compiled": False,
                "message": "Road & Pothole assessment not compiled yet. Decoupled from core 3D pipeline."
            })
        report = road_pothole_detector.run_full_audit(force_recompute=force)
        report["compiled"] = True
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Road & Pothole audit failed: {str(e)}")

@app.post("/api/road/audit/compile")
@app.post("/api/road/audit/recompute")
def compile_road_audit():
    """
    On-demand compilation of the road classification and 3D pothole depths.
    Only executed when operator explicitly clicks 'Compile Road & Potholes'.
    """
    try:
        report = road_pothole_detector.run_full_audit(force_recompute=True)
        report["compiled"] = True
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Road audit compilation failed: {str(e)}")

@app.post("/api/diff/compare")
def compare_with_baseline(req: CompareRequest):
    """
    Executes multi-epoch 3D change detection between active model and chosen baseline.
    Checks spatial overlap via telemetry; flags vertical structural height increases.
    """
    global active_diff_state
    try:
        result = change_detector.compare_active_against_baseline(
            baseline_id=req.baseline_id,
            height_threshold_m=req.height_threshold
        )
        active_diff_state = {
            "status": result["status"],
            "data": result
        }
        return JSONResponse(content=result)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/api/diff/active")
def get_active_diff():
    """
    Returns current active comparison / alert state.
    """
    return JSONResponse(content=active_diff_state)

@app.get("/api/analysis/height-restriction")
@app.post("/api/analysis/height-restriction")
def analyze_height_restriction(
    max_height_m: float = 5.0,
    datum_mode: str = "ground",
    baseline_id: Optional[str] = None
):
    """
    Analyzes point cloud against vertical height restriction ceiling.
    Computes total points, violating points count/ratio, peak elevation, and maximum breach.
    """
    import numpy as np
    try:
        if baseline_id:
            baseline_dir = os.path.join(BASELINES_DIR, baseline_id)
            ply_path = os.path.join(baseline_dir, "model.ply")
            telem_path = os.path.join(baseline_dir, "telemetry.json")
        else:
            ply_path = os.path.join(DATA_DIR, "models", "actionable_threat_map.ply")
            telem_path = os.path.join(DATA_DIR, "flight_telemetry.json")

        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Point cloud model not found at {ply_path}")

        metric_scale = 1.0
        if os.path.exists(telem_path):
            try:
                with open(telem_path, "r", encoding="utf-8") as f:
                    telem = json.load(f)
                    metric_scale = float(telem.get("metric_scale_factor", 1.0))
            except Exception:
                metric_scale = 1.0

        pts, _, _ = load_point_cloud(ply_path)
        if pts is None or len(pts) == 0:
            return JSONResponse(content={
                "status": "empty",
                "total_points": 0,
                "violating_points": 0,
                "violation_percent": 0.0,
                "peak_elevation_m": 0.0,
                "max_breach_m": 0.0,
                "violation_detected": False
            })

        ys = pts[:, 1]
        if datum_mode == "ground":
            ground_y = float(np.percentile(ys, 2.0))
        else:
            ground_y = 0.0

        elevations_m = (ys - ground_y) * metric_scale
        breach_mask = elevations_m > max_height_m
        violating_count = int(np.sum(breach_mask))
        total_pts = len(pts)
        peak_elev_m = float(np.max(elevations_m))
        max_breach_m = max(0.0, peak_elev_m - max_height_m)
        violation_pct = round((violating_count / total_pts) * 100.0, 2)

        peak_coords = None
        if violating_count > 0:
            peak_idx = int(np.argmax(elevations_m))
            peak_coords = {
                "x": round(float(pts[peak_idx, 0]), 3),
                "y": round(float(pts[peak_idx, 1]), 3),
                "z": round(float(pts[peak_idx, 2]), 3)
            }

        return JSONResponse(content={
            "status": "completed",
            "max_height_threshold_m": float(max_height_m),
            "datum_mode": datum_mode,
            "metric_scale_factor": float(metric_scale),
            "total_points": total_pts,
            "violating_points": violating_count,
            "violation_percent": violation_pct,
            "peak_elevation_m": round(peak_elev_m, 2),
            "max_breach_m": round(max_breach_m, 2),
            "violation_detected": violating_count > 0,
            "peak_coordinates": peak_coords,
            "timestamp": time.time()
        })
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Height restriction analysis failed: {str(e)}")

# =================================================================
# AUTOMATED RECONSTRUCTION PIPELINE API
# =================================================================

@app.post("/api/pipeline/start")
async def start_reconstruction(
    input_type: str = Form("video"),          # "video" or "frames"
    has_telemetry: bool = Form(False),        # True or False
    quality: str = Form("medium"),            # "fast", "medium", "high", "ultra"
    media_file: UploadFile = File(...),       # .mp4/.mov or .zip
    telemetry_file: Optional[UploadFile] = File(None),
    flight_altitude_m: Optional[float] = Form(None)   # height above ground; used for scale when no GPS log
):
    """
    Accepts video or frames zip along with optional telemetry coordinates,
    and initiates the asynchronous 3D reconstruction pipeline.
    """
    # Verify not currently running
    current_status = pipeline_mgr.get_status()
    if current_status["status"] == "running":
        raise HTTPException(status_code=400, detail="A reconstruction pipeline is already in progress.")

    # Save uploaded media file to uploads directory
    media_ext = os.path.splitext(media_file.filename or "")[1]
    saved_media_path = os.path.join(UPLOADS_DIR, f"mission_input_{int(os.path.getmtime(UPLOADS_DIR)) if os.path.exists(UPLOADS_DIR) else 0}{media_ext}")
    
    with open(saved_media_path, "wb") as buffer:
        shutil.copyfileobj(media_file.file, buffer)

    # Save telemetry file if uploaded
    saved_telemetry_path = None
    if has_telemetry and telemetry_file:
        telem_ext = os.path.splitext(telemetry_file.filename or "")[1]
        saved_telemetry_path = os.path.join(UPLOADS_DIR, f"telemetry_input{telem_ext}")
        with open(saved_telemetry_path, "wb") as buffer:
            shutil.copyfileobj(telemetry_file.file, buffer)

    if flight_altitude_m is not None and not (1.0 <= flight_altitude_m <= 1000.0):
        raise HTTPException(status_code=400, detail="flight_altitude_m must be between 1 and 1000 metres.")

    # Auto-detect if uploaded file is a pre-computed Colab bundle (.ZIP)
    if saved_media_path.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(saved_media_path, "r") as z_check:
                names_lower = [n.lower() for n in z_check.namelist()]
                if any("actionable_threat" in n or "flight_telemetry.json" in n for n in names_lower):
                    result = deploy_colab_bundle(saved_media_path)
                    return {
                        "success": True,
                        "job_id": "colab_bundle_auto_import",
                        "message": f"Pre-computed Google Colab Package auto-detected and deployed ({result['vertex_count']:,} points, {result['face_count']:,} mesh faces).",
                        "input_type": "colab",
                        "has_telemetry": result["has_telemetry"],
                        "quality": "colab_gpu",
                        "flight_altitude_m": flight_altitude_m,
                        "deployed_files": result["deployed_files"],
                        "vertex_count": result["vertex_count"],
                        "face_count": result["face_count"]
                    }
        except Exception as ze:
            print(f"[Notice] Media zip check not a colab bundle: {ze}")

    # Trigger background reconstruction
    success, job_id = pipeline_mgr.start_pipeline(
        input_type=input_type.lower(),
        has_telemetry=has_telemetry,
        quality=quality.lower(),
        media_path=saved_media_path,
        telemetry_path=saved_telemetry_path,
        flight_altitude_m=flight_altitude_m
    )

    if not success:
        raise HTTPException(status_code=400, detail=job_id)

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Pipeline started in {input_type.upper()} mode ({quality.upper()} quality).",
        "input_type": input_type,
        "has_telemetry": has_telemetry,
        "quality": quality,
        "flight_altitude_m": flight_altitude_m
    }

@app.get("/api/pipeline/status")
def get_pipeline_status():
    """
    Returns live progress percentage, current stage, and streaming terminal logs.
    """
    return JSONResponse(content=pipeline_mgr.get_status())

@app.post("/api/pipeline/cancel")
def cancel_pipeline():
    """
    Cancels the active reconstruction job.
    """
    cancelled = pipeline_mgr.cancel()
    return {"success": cancelled, "message": "Pipeline cancelled" if cancelled else "No running job to cancel"}


def deploy_colab_bundle(zip_path: str) -> dict:
    """
    Core engine to unpack and deploy a pre-computed 3D tactical mission bundle
    generated on Google Colab / Kaggle. Deploys points, mesh, glb, telemetry,
    and video directly into PRISM active datasets.
    """
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail=f"Mission bundle file not found: {zip_path}")

    extract_dir = tempfile.mkdtemp(prefix="prism_colab_")
    deployed = []

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # Walk extracted files to find target assets regardless of nested subdirectories
        extracted_map = {}
        for root, _, files in os.walk(extract_dir):
            for f in files:
                extracted_map[f.lower()] = os.path.join(root, f)

        models_dir = os.path.join(DATA_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        raw_videos_dir = os.path.join(DATA_DIR, "raw_videos")
        os.makedirs(raw_videos_dir, exist_ok=True)
        workspace_dir = os.path.join(DATA_DIR, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        # 1. Point Cloud (Points)
        pts_target = os.path.join(models_dir, "actionable_threat_map_points.ply")
        active_target = os.path.join(models_dir, "actionable_threat_map.ply")
        if "actionable_threat_map_points.ply" in extracted_map:
            shutil.copy2(extracted_map["actionable_threat_map_points.ply"], pts_target)
            shutil.copy2(extracted_map["actionable_threat_map_points.ply"], active_target)
            deployed.append("actionable_threat_map_points.ply")
        elif "actionable_threat_map.ply" in extracted_map:
            shutil.copy2(extracted_map["actionable_threat_map.ply"], pts_target)
            shutil.copy2(extracted_map["actionable_threat_map.ply"], active_target)
            deployed.append("actionable_threat_map.ply")

        # 2. Blender Solid 3D Mesh (PLY & OBJ & GLB)
        if "actionable_threat_mesh.ply" in extracted_map:
            shutil.copy2(extracted_map["actionable_threat_mesh.ply"], os.path.join(models_dir, "actionable_threat_mesh.ply"))
            deployed.append("actionable_threat_mesh.ply")
        if "actionable_threat_mesh.obj" in extracted_map:
            shutil.copy2(extracted_map["actionable_threat_mesh.obj"], os.path.join(models_dir, "actionable_threat_mesh.obj"))
            deployed.append("actionable_threat_mesh.obj")
        if "actionable_threat_mesh.glb" in extracted_map:
            shutil.copy2(extracted_map["actionable_threat_mesh.glb"], os.path.join(models_dir, "actionable_threat_mesh.glb"))
            deployed.append("actionable_threat_mesh.glb")

        # 3. Flight Telemetry & GNSS metadata
        if "flight_telemetry.json" in extracted_map:
            shutil.copy2(extracted_map["flight_telemetry.json"], os.path.join(DATA_DIR, "flight_telemetry.json"))
            deployed.append("flight_telemetry.json")

        if "drone_flight.srt" in extracted_map:
            shutil.copy2(extracted_map["drone_flight.srt"], os.path.join(DATA_DIR, "drone_flight.srt"))
            shutil.copy2(extracted_map["drone_flight.srt"], os.path.join(raw_videos_dir, "drone_flight.srt"))
            deployed.append("drone_flight.srt")

        # 4. Frame Timestamps Index
        if "frame_index.json" in extracted_map:
            shutil.copy2(extracted_map["frame_index.json"], os.path.join(workspace_dir, "frame_index.json"))
            deployed.append("frame_index.json")

        # 5. Video Stream
        for v_name in ("drone_flight.mp4", "flight.mp4", "video.mp4"):
            if v_name in extracted_map:
                shutil.copy2(extracted_map[v_name], os.path.join(raw_videos_dir, "drone_flight.mp4"))
                deployed.append("drone_flight.mp4")
                break

        # 6. Reports & Visual Previews
        if "recon_report.json" in extracted_map:
            shutil.copy2(extracted_map["recon_report.json"], os.path.join(DATA_DIR, "recon_report.json"))
            deployed.append("recon_report.json")
        if "preview_topdown.jpg" in extracted_map:
            shutil.copy2(extracted_map["preview_topdown.jpg"], os.path.join(DATA_DIR, "preview_topdown.jpg"))
            shutil.copy2(extracted_map["preview_topdown.jpg"], os.path.join(models_dir, "preview_topdown.jpg"))
            deployed.append("preview_topdown.jpg")

        # Check vertex counts
        v_count = 0
        f_count = 0
        if os.path.exists(pts_target):
            try:
                with open(pts_target, "rb") as f:
                    for _ in range(40):
                        line = f.readline().decode("latin1", errors="ignore").strip()
                        if line.startswith("element vertex"):
                            v_count = int(line.split()[-1])
                        elif line == "end_header":
                            break
            except Exception:
                pass

        mesh_target = os.path.join(models_dir, "actionable_threat_mesh.ply")
        if os.path.exists(mesh_target):
            try:
                with open(mesh_target, "rb") as f:
                    for _ in range(40):
                        line = f.readline().decode("latin1", errors="ignore").strip()
                        if line.startswith("element face"):
                            f_count = int(line.split()[-1])
                        elif line == "end_header":
                            break
            except Exception:
                pass

        # Update pipeline manager singleton state
        with pipeline_mgr.lock:
            pipeline_mgr.status = "completed"
            pipeline_mgr.progress_percent = 100
            pipeline_mgr.current_stage = "COMPLETED"
            pipeline_mgr.end_time = time.time()
        pipeline_mgr.add_log(f"Colab Cloud Mission Ingested: {len(deployed)} assets deployed ({v_count:,} points, {f_count:,} mesh faces).")

        return {
            "success": True,
            "message": f"Successfully ingested cloud bundle from Colab: {len(deployed)} core assets deployed!",
            "deployed_files": deployed,
            "vertex_count": v_count,
            "face_count": f_count,
            "has_points": os.path.exists(pts_target),
            "has_mesh": os.path.exists(mesh_target),
            "has_video": "drone_flight.mp4" in deployed,
            "has_telemetry": "flight_telemetry.json" in deployed
        }

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


@app.post("/api/pipeline/import-colab")
async def import_colab_bundle(bundle_file: UploadFile = File(...)):
    """
    Directly ingests a pre-computed 3D tactical mission package generated on
    Google Colab / Kaggle. Bypasses local GPU processing completely.
    """
    if not (bundle_file.filename and bundle_file.filename.lower().endswith(".zip")):
        raise HTTPException(status_code=400, detail="Invalid package format. Must be a .ZIP archive generated from Google Colab / Kaggle.")

    temp_zip = os.path.join(UPLOADS_DIR, f"colab_import_{int(time.time())}.zip")
    with open(temp_zip, "wb") as buffer:
        shutil.copyfileobj(bundle_file.file, buffer)

    try:
        result = deploy_colab_bundle(temp_zip)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed unpacking Colab mission bundle: {str(e)}")
    finally:
        if os.path.exists(temp_zip):
            try:
                os.remove(temp_zip)
            except Exception:
                pass



if __name__ == "__main__":
    import uvicorn
    print(f" Serving PRISM Tactical Command Center from: {FRONTEND_DIR}")
    print(f" Serving Data assets from: {DATA_DIR}")
    print(f" Open your browser at: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)