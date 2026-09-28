"""
PRISM // export service (exports.py)

Builds and caches every deliverable of a survey (the active mission or a saved baseline):

  3-D         LAS 1.2 (UTM, ASPRS classes), PLY points, OBJ / PLY / GLB / glTF / FBX mesh
  rasters     GeoTIFF DSM, DTM, heights above ground, true orthophoto, land cover (UTM)
  vectors     GeoJSON (roads, potholes, buildings, trees, vehicles, flight track, survey footprint),
              CityJSON LoD 1.2 buildings
  documents   quality & accuracy report (HTML / JSON), georeference sheet
  package     one ZIP with all of the above

Files live in data/exports/<survey>/ and are rebuilt only when the model changes (or the vertical
offset requested for elevations changes).
"""
import json
import os
import shutil
import threading
import time
import zipfile

import numpy as np

import geo_export as G
import plyio

PRODUCTS = {
    # key: (group, label, file name, media type)
    "las": ("Point cloud", "LAS 1.2 point cloud (UTM, colour, ASPRS classes)", "points.las", "application/vnd.las"),
    "points_ply": ("Point cloud", "PLY point cloud (local metres)", "points.ply", "application/octet-stream"),
    "mesh_obj": ("Mesh", "OBJ mesh (vertex colours)", "mesh.obj", "text/plain"),
    "mesh_ply": ("Mesh", "PLY mesh (vertex colours)", "mesh.ply", "application/octet-stream"),
    "mesh_glb": ("Mesh", "GLB (binary glTF 2.0)", "mesh.glb", "model/gltf-binary"),
    "mesh_gltf": ("Mesh", "glTF 2.0 (.gltf + .bin, zipped)", "mesh_gltf.zip", "application/zip"),
    "mesh_fbx": ("Mesh", "FBX 7.4 binary (vertex colours)", "mesh.fbx", "application/octet-stream"),
    "dsm": ("Raster", "GeoTIFF surface model (DSM)", "DSM.tif", "image/tiff"),
    "dtm": ("Raster", "GeoTIFF bare-earth terrain (DTM)", "DTM.tif", "image/tiff"),
    "ndsm": ("Raster", "GeoTIFF heights above ground (nDSM)", "heights_above_ground.tif", "image/tiff"),
    "ortho": ("Raster", "GeoTIFF true orthophoto (RGBA)", "orthophoto.tif", "image/tiff"),
    "landcover": ("Raster", "GeoTIFF land cover classes", "land_cover.tif", "image/tiff"),
    "features": ("Vector", "GeoJSON features (roads, potholes, buildings, trees, track)", "features.geojson",
                 "application/geo+json"),
    "cityjson": ("Vector", "CityJSON 1.1 buildings (LoD 1.2 digital twin)", "buildings.city.json", "application/city+json"),
    "report_html": ("Document", "Quality & accuracy report (HTML, printable)", "quality_report.html", "text/html"),
    "report_json": ("Document", "Quality & accuracy report (JSON)", "quality_report.json", "application/json"),
    "georef": ("Document", "Georeference sheet (CRS, origin, axes, datum)", "georeference.json", "application/json"),
    "package": ("Package", "Everything above in one ZIP", "package.zip", "application/zip"),
}
MESH_PRODUCTS = {"mesh_obj", "mesh_ply", "mesh_glb", "mesh_gltf", "mesh_fbx"}
GEO_PRODUCTS = {"las", "dsm", "dtm", "ndsm", "ortho", "landcover", "features", "cityjson", "georef"}
FORMAT_NAMES = ["OBJ", "PLY", "LAS", "GeoTIFF", "glTF", "GLB", "FBX", "GeoJSON", "CityJSON"]


class Survey:
    """File locations of the active mission or of one saved baseline."""

    def __init__(self, data_dir, baseline_id=None):
        self.data_dir, self.baseline_id = data_dir, baseline_id
        if baseline_id:
            d = os.path.join(data_dir, "baselines", baseline_id)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Saved survey '{baseline_id}' not found.")
            self.dir = d
            self.points = next((p for p in (os.path.join(d, "model_cloud.ply"), os.path.join(d, "model.ply")) if os.path.exists(p)), None)
            self.mesh_ply, self.mesh_obj, self.mesh_glb = (os.path.join(d, n) for n in ("mesh.ply", "mesh.obj", "mesh.glb"))
            self.telem = os.path.join(d, "telemetry.json")
            self.recon = os.path.join(d, "recon_report.json")
            self.checkpoints = os.path.join(d, "checkpoints.json")
            self.frames = os.path.join(d, "frame_index.json")
            meta = _read(os.path.join(d, "metadata.json"), {})
            self.name = meta.get("name") or baseline_id
            self.export_dir = os.path.join(data_dir, "exports", baseline_id)
            self.preview = os.path.join(d, "preview_topdown.jpg")
        else:
            m = os.path.join(data_dir, "models")
            self.dir = m
            self.points = next((p for p in (os.path.join(m, "actionable_threat_map_points.ply"),
                                            os.path.join(m, "actionable_threat_map_cloud.ply"),
                                            os.path.join(m, "actionable_threat_map.ply")) if os.path.exists(p)), None)
            self.mesh_ply, self.mesh_obj, self.mesh_glb = (os.path.join(m, n) for n in (
                "actionable_threat_mesh.ply", "actionable_threat_mesh.obj", "actionable_threat_mesh.glb"))
            self.telem = os.path.join(data_dir, "flight_telemetry.json")
            self.recon = os.path.join(data_dir, "recon_report.json")
            self.checkpoints = os.path.join(m, "checkpoints.json")
            self.frames = os.path.join(data_dir, "workspace", "frame_index.json")
            self.name = "Current mission"
            self.export_dir = os.path.join(data_dir, "exports", "active")
            self.preview = os.path.join(m, "preview_topdown.jpg")
        if not self.points:
            raise FileNotFoundError("No 3D model is deployed yet.")

    @property
    def slug(self):
        return self.baseline_id or "prism_survey"

    @property
    def has_mesh(self):
        return os.path.exists(self.mesh_ply)

    def key(self):
        st = os.stat(self.points)
        k = [os.path.basename(self.points), st.st_size, int(st.st_mtime)]
        if self.has_mesh:
            ms = os.stat(self.mesh_ply)
            k += [ms.st_size, int(ms.st_mtime)]
        if os.path.exists(self.telem):
            k.append(int(os.stat(self.telem).st_mtime))
        return k

    def recon_report(self, n_points=None):
        """Processing report of THIS model (a report left over from another mission is ignored)."""
        r = _read(self.recon, None)
        if not r:
            return {}
        pts = (r.get("outputs") or {}).get("points")
        if n_points is not None and pts and int(pts) != int(n_points):
            return {}
        return r


def _read(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


class ExportService:
    def __init__(self, data_dir, terrain_fn, he_fn, roads_fn, buildings_fn, inventory_fn, log=print):
        self.data_dir = data_dir
        self.terrain_fn, self.he_fn, self.roads_fn = terrain_fn, he_fn, roads_fn
        self.buildings_fn, self.inventory_fn = buildings_fn, inventory_fn
        self.log = log
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ cache
    def _dir(self, sv, v_offset):
        d = sv.export_dir
        man_path = os.path.join(d, "manifest.json")
        key = {"model": sv.key(), "v_offset": v_offset}
        man = _read(man_path, {})
        if man.get("key") != key:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump({"key": key, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        return d

    def catalog(self, baseline_id=None):
        sv = Survey(self.data_dir, baseline_id)
        telem = _read(sv.telem, {})
        geo_ok = bool((telem.get("georeference") or {}).get("metric")) and ((telem.get("georeference") or {}).get("origin") or {}).get("lat") is not None
        d = sv.export_dir
        items = []
        for k, (group, label, fname, _) in PRODUCTS.items():
            avail, why = True, None
            if k in MESH_PRODUCTS and not sv.has_mesh:
                avail, why = False, "Build the mesh first (Mission > Build surface mesh)."
            if k in GEO_PRODUCTS and not geo_ok:
                avail, why = False, "Needs a GPS-georeferenced metric model."
            p = os.path.join(d, f"{sv.slug}_{fname}")
            items.append({"key": k, "group": group, "label": label, "file": f"{sv.slug}_{fname}", "available": avail,
                          "reason": why, "ready": os.path.exists(p), "size_mb": round(os.path.getsize(p) / 1e6, 2) if os.path.exists(p) else None})
        gf = None
        if geo_ok:
            try:
                gf = G.GeoFrame(telem).describe()
            except ValueError:
                gf = None
        return {"survey": sv.name, "baseline_id": baseline_id, "georeference": gf, "has_mesh": sv.has_mesh,
                "altitude_reference": telem.get("altitude_reference"), "products": items}

    # ------------------------------------------------------------------ build
    def build(self, product, baseline_id=None, v_offset=None):
        if product not in PRODUCTS:
            raise KeyError(product)
        sv = Survey(self.data_dir, baseline_id)
        with self.lock:
            d = self._dir(sv, v_offset)
            out = os.path.join(d, f"{sv.slug}_{PRODUCTS[product][2]}")
            if not os.path.exists(out):
                t0 = time.time()
                self._make(product, sv, out, d, v_offset)
                self.log(f"[Export] {product} -> {os.path.basename(out)} ({os.path.getsize(out) / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
        return out, os.path.basename(out), PRODUCTS[product][3]

    def _gf(self, sv, v_offset):
        return G.GeoFrame(_read(sv.telem, {}), vertical_offset_m=v_offset)

    def _make(self, product, sv, out, d, v_offset):
        bid = sv.baseline_id
        if product == "points_ply":
            shutil.copy2(sv.points, out)
        elif product in ("mesh_obj", "mesh_ply", "mesh_glb", "mesh_gltf", "mesh_fbx"):
            if not sv.has_mesh:
                raise FileNotFoundError("No mesh yet: build the surface mesh first.")
            if product == "mesh_ply":
                shutil.copy2(sv.mesh_ply, out)
            elif product == "mesh_obj":
                if os.path.exists(sv.mesh_obj):
                    shutil.copy2(sv.mesh_obj, out)
                else:
                    m = plyio.read_ply(sv.mesh_ply)
                    plyio.write_obj(out, m.xyz, m.rgb, m.faces)
            elif product in ("mesh_glb", "mesh_gltf"):
                glb = sv.mesh_glb
                if not os.path.exists(glb):
                    glb = os.path.join(d, "_tmp_mesh.glb")
                    m = plyio.read_ply(sv.mesh_ply)
                    plyio.write_glb(glb, m.xyz, m.rgb, m.faces, normals=m.normals)
                if product == "mesh_glb":
                    shutil.copy2(glb, out)
                else:
                    extra = {}
                    try:
                        extra["georeference.json"] = json.dumps(self._gf(sv, v_offset).describe(), indent=2)
                    except ValueError:
                        pass
                    G.glb_to_gltf_zip(glb, out, extra, stem=sv.slug)
            else:
                m = plyio.read_ply(sv.mesh_ply)
                G.write_fbx(out, m.xyz, m.faces, m.rgb, name=sv.slug)
        elif product == "las":
            tm, pts, telem = self.terrain_fn(bid)
            gf = self._gf(sv, v_offset)
            u = gf.model_to_utm(pts["xyz"])
            G.write_las(out, u, pts["rgb"], G.classify_points(pts["xyz"], tm), gf.epsg)
        elif product in ("dsm", "dtm", "ndsm", "landcover"):
            tm, _, _ = self.terrain_fn(bid)
            gf = self._gf(sv, v_offset)
            paths, _ = G.terrain_rasters(gf, tm, d, sv.slug)
            names = {"dsm": "dsm", "dtm": "dtm", "ndsm": "ndsm", "landcover": "classes"}
            src = paths[names[product]]
            if os.path.normcase(src) != os.path.normcase(out):
                os.replace(src, out)
            # the sibling rasters were written in the same pass: give them their catalogue names
            for k, v in (("dsm", "dsm"), ("dtm", "dtm"), ("ndsm", "ndsm"), ("landcover", "classes")):
                tgt = os.path.join(d, f"{sv.slug}_{PRODUCTS[k][2]}")
                if k != product and os.path.exists(paths[v]) and os.path.normcase(paths[v]) != os.path.normcase(tgt):
                    os.replace(paths[v], tgt)
        elif product == "ortho":
            _, pts, _ = self.terrain_fn(bid)
            G.orthophoto(self._gf(sv, v_offset), pts["xyz"], pts["rgb"], out)
        elif product == "features":
            fc = self.features(sv, v_offset)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(fc, f)
        elif product == "cityjson":
            import buildings as B
            gf = self._gf(sv, v_offset)
            cj = B.to_cityjson(self.buildings_fn(bid), gf, title=sv.name)
            if cj is None:
                raise FileNotFoundError("No buildings were found in this survey.")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(cj, f)
        elif product in ("report_html", "report_json"):
            import quality as Q
            rep = self.report(bid)
            if product == "report_json":
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(rep, f, indent=1, default=float)
            else:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(Q.render_html(rep, self._preview_data_url(sv, bid)))
        elif product == "georef":
            gf = self._gf(sv, v_offset)
            desc = gf.describe()
            desc["files"] = {"LAS / GeoTIFF / GeoJSON / CityJSON": f"georeferenced in {gf.crs_name} (EPSG:{gf.epsg}); GeoJSON in WGS 84",
                             "OBJ / PLY / GLB / glTF / FBX": "local metric frame described by model_axes and origin_wgs84"}
            with open(out, "w", encoding="utf-8") as f:
                json.dump(desc, f, indent=2)
        elif product == "package":
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                for k in PRODUCTS:
                    if k == "package":
                        continue
                    try:
                        p, name, _ = self._build_unlocked(k, sv, d, v_offset)
                    except Exception as e:           # a product that cannot be made is simply left out
                        self.log(f"[Export] package: {k} skipped ({e})")
                        continue
                    z.write(p, name)
        else:
            raise KeyError(product)

    def _build_unlocked(self, product, sv, d, v_offset):
        out = os.path.join(d, f"{sv.slug}_{PRODUCTS[product][2]}")
        if not os.path.exists(out):
            self._make(product, sv, out, d, v_offset)
        return out, os.path.basename(out), PRODUCTS[product][3]

    # ------------------------------------------------------------------ vectors
    def features(self, sv, v_offset=None):
        bid = sv.baseline_id
        gf = self._gf(sv, v_offset)
        tm, _, telem = self.terrain_fn(bid)
        feats = []
        # survey footprint (analysed area)
        try:
            import cv2
            cs, _ = cv2.findContours(tm.roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cs, key=cv2.contourArea, reverse=True)[:3]:
                c = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
                if len(c) < 3:
                    continue
                x, z = tm.centre(c[:, 1], c[:, 0])
                ring = G.lonlat(gf, np.c_[x, z])
                ring.append(ring[0])
                feats.append(G.feature("Polygon", [ring], {"layer": "survey_footprint",
                                                          "area_m2": round(float(cv2.contourArea(c)) * tm.res ** 2, 0)}))
        except Exception as e:
            self.log(f"[Export] footprint skipped: {e}")
        # flight track (GNSS as logged, verified order)
        wps = [w for w in (telem.get("waypoints") or []) if w.get("latitude") is not None]
        if len(wps) >= 2:
            step = max(1, len(wps) // 2000)
            line = [[round(w["longitude"], 7), round(w["latitude"], 7)] for w in wps[::step]]
            feats.append(G.feature("LineString", line, {"layer": "flight_track", "fixes": len(wps),
                                                        "source": telem.get("source")}))
        # roads and potholes
        try:
            rr = self.roads_fn(bid)
            for s in rr.get("segments") or []:
                for cl in s.get("centerlines") or []:
                    a = np.asarray(cl)
                    if len(a) < 2:
                        continue
                    feats.append(G.feature("LineString", G.lonlat(gf, a[:, [0, 2]]), {
                        "layer": "road", "id": s.get("id"), "width_m": s.get("width_median_m"),
                        "width_min_m": s.get("width_min_m"), "surface": s.get("surface"),
                        "max_grade_pct": s.get("max_grade_pct"), "length_m": s.get("length_m")}))
            for p in rr.get("potholes") or []:
                pos = p.get("position")
                if not pos:
                    continue
                feats.append(G.feature("Point", G.lonlat(gf, [pos])[0], {
                    "layer": "pothole", "id": p.get("id"), "depth_cm": p.get("depth_cm"), "severity": p.get("severity"),
                    "kind": p.get("kind"), "area_m2": p.get("area_m2"), "volume_l": p.get("volume_l")}))
        except Exception as e:
            self.log(f"[Export] roads skipped: {e}")
        # buildings (footprints with heights) + trees / vehicles (points)
        try:
            import buildings as B
            feats.extend(B.to_geojson_features(self.buildings_fn(bid), gf))
        except Exception as e:
            self.log(f"[Export] buildings skipped: {e}")
        try:
            inv = self.inventory_fn(bid)
            for it in inv.get("items") or []:
                if it["kind"] == "building":
                    continue
                feats.append(G.feature("Point", G.lonlat(gf, [it["position"]])[0], {
                    "layer": it["kind"], "id": it.get("id"), "height_m": it.get("approx_height_m"),
                    "footprint_m2": it.get("footprint_m2")}))
        except Exception as e:
            self.log(f"[Export] inventory skipped: {e}")
        return G.feature_collection(feats, f"PRISM {sv.name}", gf)

    # ------------------------------------------------------------------ report
    def report(self, baseline_id=None):
        import quality as Q
        sv = Survey(self.data_dir, baseline_id)
        tm, pts, telem = self.terrain_fn(baseline_id)
        n = len(pts["xyz"])
        recon = sv.recon_report(n)
        mesh_faces = plyio.ply_counts(sv.mesh_ply)[1] if sv.has_mesh else (recon.get("outputs") or {}).get("mesh_faces")
        try:
            blds = self.buildings_fn(baseline_id)
        except Exception:
            blds = None
        try:
            roads = self.roads_fn(baseline_id)
        except Exception:
            roads = None
        he = self.he_fn(baseline_id)
        frames = _read(sv.frames, [])
        return Q.build_report(name=sv.name, telem=telem, recon=recon, tm=tm, n_points=n, mesh_faces=mesh_faces,
                              formats=FORMAT_NAMES, buildings=blds, roads=roads,
                              checkpoints=self.checkpoints(baseline_id), spacing_m=getattr(he, "spacing", None),
                              frames=frames if isinstance(frames, list) else [])

    def _preview_data_url(self, sv, bid):
        import base64
        import cv2
        try:
            tm, _, _ = self.terrain_fn(bid)
            img = np.dstack([np.clip(tm.rgb, 0, 255).astype(np.uint8)])[..., ::-1].copy()
            img[~tm.observed] = (242, 242, 242)
            ii, jj = np.nonzero(tm.roi)
            if len(ii):
                img = img[ii.min():ii.max() + 1, jj.min():jj.max() + 1]
            h, w = img.shape[:2]
            s = min(1.0, 900.0 / max(h, w))
            if s < 1.0:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        except Exception:
            return None

    # ------------------------------------------------------------------ check / control points
    def checkpoints(self, baseline_id=None, v_offset=None):
        """Stored points with residuals (model - surveyed) under the CURRENT georeference, plus the
        leave-one-out residual (what the point would show if the others alone had corrected the model)."""
        sv = Survey(self.data_dir, baseline_id)
        cps = _read(sv.checkpoints, [])
        if not cps:
            return []
        try:
            gf = self._gf(sv, v_offset)
        except ValueError:
            return cps
        m = gf.model_to_utm(np.asarray([c["model"] for c in cps], float))
        for c, mu in zip(cps, m):
            e, n = G.utm_forward(c["lat"], c["lon"], gf.zone, gf.north)
            c["d_e"], c["d_n"] = round(float(mu[0] - e), 3), round(float(mu[1] - n), 3)
            c["d_h"] = round(float(mu[2] - c["elev"]), 3) if c.get("elev") is not None else None
            c["horizontal_m"] = round(float(np.hypot(c["d_e"], c["d_n"])), 3)
        if len(cps) >= 2:
            de = np.array([c["d_e"] for c in cps])
            dn = np.array([c["d_n"] for c in cps])
            for k, c in enumerate(cps):
                o = np.arange(len(cps)) != k
                c["loo_e"] = round(float(de[k] - de[o].mean()), 3)
                c["loo_n"] = round(float(dn[k] - dn[o].mean()), 3)
        return cps

    def add_checkpoint(self, baseline_id, name, model_xyz, lat, lon, elev=None, v_offset=None):
        """A surveyed point (lat, lon[, elevation]) and the same feature picked in the model."""
        sv = Survey(self.data_dir, baseline_id)
        self._gf(sv, v_offset)                     # must be georeferenced
        cps = _read(sv.checkpoints, [])
        cps.append({"name": name or f"CP-{len(cps) + 1}", "lat": float(lat), "lon": float(lon),
                    "elev": float(elev) if elev is not None else None,
                    "model": [round(float(v), 3) for v in model_xyz], "added": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(sv.checkpoints, "w", encoding="utf-8") as f:
            json.dump(cps, f, indent=1)
        return self.checkpoints(baseline_id, v_offset)[-1]

    def delete_checkpoint(self, baseline_id, index):
        sv = Survey(self.data_dir, baseline_id)
        cps = _read(sv.checkpoints, [])
        if 0 <= index < len(cps):
            cps.pop(index)
            with open(sv.checkpoints, "w", encoding="utf-8") as f:
                json.dump(cps, f, indent=1)
        return self.checkpoints(baseline_id)

    def apply_control(self, baseline_id=None, v_offset=None, undo=False):
        """Few-GCP georeference refinement: shifts the model origin so that the mean residual of the
        surveyed points becomes zero (GNSS bias is a translation; the bundle already fixes shape and
        scale). The point cloud is not touched; every georeferenced export follows the new origin."""
        sv = Survey(self.data_dir, baseline_id)
        telem = _read(sv.telem, {})
        geo = telem.get("georeference") or {}
        if undo:
            prev = geo.get("origin_before_control")
            if not prev:
                raise ValueError("No control-point correction to undo.")
            geo["origin"] = prev
            if geo.get("altitude_reference_before_control"):
                telem["altitude_reference"] = geo.pop("altitude_reference_before_control")
            for k in ("origin_before_control", "control_correction", "vertical_datum_note"):
                geo.pop(k, None)
        else:
            cps = self.checkpoints(baseline_id, v_offset)
            if not cps:
                raise ValueError("Enter at least one surveyed point first.")
            de = float(np.mean([c["d_e"] for c in cps]))
            dn = float(np.mean([c["d_n"] for c in cps]))
            dhs = [c["d_h"] for c in cps if c.get("d_h") is not None]
            dh = float(np.mean(dhs)) if dhs else 0.0
            gf = self._gf(sv, v_offset)
            o = geo["origin"]
            e0, n0 = G.utm_forward(o["lat"], o["lon"], gf.zone, gf.north)
            la, lo = G.utm_inverse(float(e0) - de, float(n0) - dn, gf.zone, gf.north)
            if "origin_before_control" not in geo:
                geo["origin_before_control"] = dict(o)
            geo["origin"] = {"lat": float(la), "lon": float(lo), "alt": float(o.get("alt") or 0.0) - dh}
            prev = geo.get("control_correction") or {}
            geo["control_correction"] = {
                "d_e_m": round(de + prev.get("d_e_m", 0.0), 3), "d_n_m": round(dn + prev.get("d_n_m", 0.0), 3),
                "d_h_m": round(dh + prev.get("d_h_m", 0.0), 3), "points": len(cps), "vertical_points": len(dhs),
                "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if dhs:
                # elevations now come from the surveyed points: the model sits on their vertical datum
                geo.setdefault("altitude_reference_before_control", telem.get("altitude_reference"))
                telem["altitude_reference"] = "absolute"
                geo["vertical_datum_note"] = "vertical datum of the surveyed control points"
        telem["georeference"] = geo
        tmp = sv.telem + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(telem, f, indent=2)
        os.replace(tmp, sv.telem)
        return {"correction": geo.get("control_correction"), "origin": geo["origin"],
                "checkpoints": self.checkpoints(baseline_id, v_offset)}
