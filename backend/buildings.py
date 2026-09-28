"""
PRISM // buildings, rooftops, facades and digital-twin blocks (buildings.py)

For every building in the survey:
  footprint     outline from the building cells; squared to a rectangle when the shape is rectangular
                (rectangularity >= 0.85), otherwise a simplified polygon
  height        the Ring-Base Robust Height of height_engine (roof planes / ridge line, ring ground plane)
  rooftop       multi-plane RANSAC on the roof points -> plane count, slopes, aspects ->
                FLAT / SHED (mono-pitch) / GABLE / HIPPED / COMPLEX, roof pitch, roof surface area
  facades       wall area (perimeter x eaves height) and FACADE COVERAGE: the share of the wall surface
                (0.5 m x 0.5 m wall cells) that actually carries 3-D points. A single oblique pass
                sees only the walls facing the flight line, so this number tells the analyst which
                facades are measured and which are inferred.
  volume        integral of the height-above-ground over the footprint
  storeys       eaves height / 3.0 m (residential storey convention)
Outputs: JSON for the dashboard, GeoJSON (extrudable in QGIS / Cesium / Mapbox) and CityJSON 1.1
(LoD 1.2 solids with Ground / Wall / Roof semantics) in UTM.
"""
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    import terrain as T
    from height_engine import ransac_plane
except ImportError:                                    # pragma: no cover
    from backend import terrain as T
    from backend.height_engine import ransac_plane

ROOF_CODES = {"FLAT": "1000", "SHED": "1010", "GABLE": "1030", "HIPPED": "1040", "COMPLEX": "1130", "IRREGULAR": "1130"}
ROOF_LABEL = {"FLAT": "Flat", "SHED": "Mono-pitch (shed)", "GABLE": "Gable", "HIPPED": "Hipped", "COMPLEX": "Complex / combined",
              "IRREGULAR": "Irregular (not planar)"}


# =============================================================================
# roof planes
# =============================================================================
def roof_planes(pts, noise, rng, max_planes=6, max_slope_deg=65.0):
    """Sequential RANSAC + least-squares refit. Returns planes sorted by support."""
    if len(pts) < 25:
        return []
    thr = max(2.5 * noise, 0.05)
    cos_lim = math.cos(math.radians(max_slope_deg))
    rem = np.ones(len(pts), bool)
    out = []
    for _ in range(max_planes):
        idx = np.flatnonzero(rem)
        if len(idx) < max(15, 0.06 * len(pts)):
            break
        q = pts[idx]
        s0, n0 = ransac_plane(q, thr, cos_lim, rng)
        if n0 is None:
            break
        best = np.abs((q - s0) @ n0) < thr
        if best.sum() < max(12, 0.06 * len(pts)):
            break
        qi = q[best]
        c = qi.mean(0)
        _, _, vt = np.linalg.svd(qi - c, full_matrices=False)
        n = vt[2] if vt[2][1] >= 0 else -vt[2]
        inl = np.abs((q - c) @ n) < thr
        qi = q[inl]
        slope = math.degrees(math.acos(min(1.0, abs(n[1]))))
        # downslope azimuth (x East, z South): horizontal part of the normal points downhill
        aspect = (math.degrees(math.atan2(n[0], -n[2])) + 360.0) % 360.0
        out.append(dict(normal=n, centre=c, n=int(inl.sum()), slope_deg=slope, aspect_deg=aspect,
                        share=float(inl.sum()) / len(pts), points=qi))
        rem[idx[inl]] = False
    out.sort(key=lambda p: -p["n"])
    return out


def classify_roof(planes, flat_deg=7.0):
    """Roof form from the fitted planes. A single oblique pass sees the roof side facing the flight line
    far better than the other, so a pitched plane counts from 8 % support, and opposing planes are
    tested before the one-plane (shed) rule."""
    cands = [p for p in planes if p["share"] >= 0.08]
    if not cands:
        return "IRREGULAR", None
    covered = sum(p["share"] for p in cands)
    sloped = [p for p in cands if p["slope_deg"] >= flat_deg]
    flat_share = sum(p["share"] for p in cands if p["slope_deg"] < flat_deg)
    slope_share = sum(p["share"] for p in sloped)
    pitch = round(float(np.average([p["slope_deg"] for p in sloped], weights=[p["n"] for p in sloped])), 1) if sloped else 0.0
    if covered < 0.45:
        return "IRREGULAR", pitch
    if not sloped or flat_share >= 0.75:
        return "FLAT", 0.0
    if flat_share >= 0.25 and slope_share >= 0.25:
        return "COMPLEX", pitch

    def adiff(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    opposing = any(adiff(a["aspect_deg"], b["aspect_deg"]) >= 140.0 and abs(a["slope_deg"] - b["slope_deg"]) < 12.0
                   for i, a in enumerate(sloped) for b in sloped[i + 1:])
    bins = {int(((p["aspect_deg"] + 45.0) % 360.0) // 90.0) for p in sloped}
    if len(bins) >= 3:
        return ("HIPPED" if opposing and len(sloped) <= 6 else "COMPLEX"), pitch
    if opposing:
        return "GABLE", pitch
    if len(bins) == 1:
        return "SHED", pitch
    return "COMPLEX", pitch


# =============================================================================
# footprint
# =============================================================================
def footprint_polygon(tm, mask):
    """Regularised outline [[x, z], ...] (counter-clockwise seen from above) and rectangularity."""
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return [], 0.0
    c = max(cs, key=cv2.contourArea)
    area_cells = float(mask.sum())
    rect = cv2.minAreaRect(c.astype(np.float32))
    (rw, rh) = rect[1]
    rect_area = max(1e-6, (rw + 1.0) * (rh + 1.0))
    rectangularity = min(1.0, area_cells / rect_area)
    if rectangularity >= 0.85:
        (cx, cy), (w, h), ang = rect
        box = cv2.boxPoints(((cx, cy), (w + 1.0, h + 1.0), ang))        # cell centres -> cell edges
        poly = box
    else:
        eps = max(1.0, 0.012 * cv2.arcLength(c, True))
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
    # pixel (col, row) -> model (x, z); contour points are cell centres
    x = tm.x0 + (poly[:, 0] + 0.5) * tm.res
    z = tm.z0 + (poly[:, 1] + 0.5) * tm.res
    xz = np.c_[x, z]
    # counter-clockwise seen from above (x East, North = -z): signed area in (x, -z)
    a = 0.5 * np.sum(xz[:, 0] * np.roll(-xz[:, 1], -1) - np.roll(xz[:, 0], -1) * (-xz[:, 1]))
    if a < 0:
        xz = xz[::-1]
    return xz.round(3).tolist(), round(rectangularity, 3)


def _perimeter(poly):
    p = np.asarray(poly)
    return float(np.sum(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1))) if len(p) >= 3 else 0.0


# =============================================================================
# facades
# =============================================================================
def facade_coverage(he, mask, base_y, eave_h, cell=0.5):
    """Share of the wall surface (perimeter x eaves) that carries reconstructed points."""
    tm = he.tm
    if eave_h is None or eave_h < 1.5:
        return None
    edge = mask & ~ndi.binary_erosion(mask)
    ei, ej = np.nonzero(edge)
    if len(ei) < 4:
        return None
    ex, ez = tm.centre(ei, ej)
    band = ndi.binary_dilation(mask, iterations=max(1, int(round(1.0 / tm.res)))) & \
        ~ndi.binary_erosion(mask, iterations=max(1, int(round(1.0 / tm.res))))
    idx = he.points_in(band)
    if len(idx) == 0:
        return 0.0
    p = he.xyz[idx]
    lo, hi = base_y + 0.4, base_y + eave_h - 0.3
    p = p[(p[:, 1] >= lo) & (p[:, 1] <= hi)]
    n_hbins = max(1, int((hi - lo) / cell))
    if len(p) == 0:
        return 0.0
    d, k = cKDTree(np.c_[ex, ez]).query(p[:, [0, 2]])
    p, k = p[d <= 1.0], k[d <= 1.0]
    hb = np.clip(((p[:, 1] - lo) / cell).astype(np.int64), 0, n_hbins - 1)
    covered = len(np.unique(k * n_hbins + hb))
    return round(100.0 * covered / float(len(ex) * n_hbins), 1)


# =============================================================================
# extraction
# =============================================================================
def extract_buildings(he, max_n=150, min_area=12.0, log=print):
    tm = he.tm
    nd = np.nan_to_num(tm.ndsm, nan=0.0)
    m = (tm.cls == T.BUILDING) & (nd >= 1.5) & tm.roi
    lab, n = ndi.label(m, np.ones((3, 3)))
    if not n:
        return []
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:] * tm.res ** 2
    comps = [k + 1 for k in np.argsort(-area) if area[k] >= min_area][:max_n]
    objs = ndi.find_objects(lab)
    out = []
    for lab_id in comps:
        sl = objs[lab_id - 1]
        sub = lab[sl] == lab_id
        ci, cj = np.nonzero(sub)
        ci, cj = ci + sl[0].start, cj + sl[1].start
        # measure from the cell nearest the component centre that belongs to it
        mi, mj = ci.mean(), cj.mean()
        k = int(np.argmin((ci - mi) ** 2 + (cj - mj) ** 2))
        x, z = tm.centre(ci[k], cj[k])
        try:
            r = he.measure_at(float(x), float(z))
        except Exception as e:                    # one odd object must not stop the inventory
            log(f"[Buildings] skipped object at ({x:.1f}, {z:.1f}): {e}")
            continue
        if r.get("status") != "ok" or r.get("kind") != "building":
            continue
        mask = np.zeros_like(m)
        mask[ci, cj] = True
        base_y = r["base"][1]
        height = r["height_m"]
        eave = r.get("eave_height_m")
        # roof form and eaves come from the height engine (the same roof planes the height tool shows)
        roof = r.get("roof") or {"type": "IRREGULAR", "pitch_deg": None, "planes": []}
        roof_type, pitch = roof["type"], roof["pitch_deg"]
        fp_area = float(mask.sum()) * tm.res ** 2
        poly, rect = footprint_polygon(tm, mask)
        perim = _perimeter(poly)
        roof_area = fp_area / max(0.2, math.cos(math.radians(pitch or 0.0)))
        eave_h = eave if (eave is not None and 1.0 <= eave <= height + 0.05) else (height if roof_type == "FLAT" else None)
        wall_area = perim * eave_h if eave_h else None
        cov = facade_coverage(he, mask, base_y, eave_h)
        volume = float(np.sum(np.clip(nd[mask], 0, None))) * tm.res ** 2
        storeys = max(1, int(round((eave_h or height) / 3.0))) if (eave_h or height) >= 2.2 else 1
        out.append({
            "id": None, "height_m": height, "uncertainty_m": r["uncertainty_m"], "method": r["method"],
            "eaves_m": round(eave_h, 2) if eave_h else None, "max_point_m": r.get("max_point_height_m"),
            "roof_type": roof_type, "roof_label": ROOF_LABEL[roof_type], "roof_pitch_deg": pitch,
            "roof_planes": roof["planes"],
            "footprint_m2": round(fp_area, 1), "roof_area_m2": round(roof_area, 1),
            "perimeter_m": round(perim, 1), "wall_area_m2": round(wall_area, 1) if wall_area else None,
            "facade_coverage_pct": cov, "volume_m3": round(volume, 0), "storeys_est": storeys,
            "length_m": r.get("length_m"), "width_m": r.get("width_m"), "orientation_deg": r.get("orientation_deg"),
            "rectangularity": rect, "footprint": poly, "base_y": round(base_y, 3),
            "centroid": r.get("centroid"), "top": r.get("top"), "gps_lat": r.get("gps_lat"), "gps_lon": r.get("gps_lon"),
        })
    out.sort(key=lambda b: -b["footprint_m2"])
    for i, b in enumerate(out):
        b["id"] = f"BLD-{i + 1:03d}"
    return out


def summary(buildings):
    if not buildings:
        return {"count": 0}
    types = {}
    for b in buildings:
        types[b["roof_type"]] = types.get(b["roof_type"], 0) + 1
    cov = [b["facade_coverage_pct"] for b in buildings if b.get("facade_coverage_pct") is not None]
    return {"count": len(buildings), "roof_types": types,
            "total_footprint_m2": round(sum(b["footprint_m2"] for b in buildings), 0),
            "total_volume_m3": round(sum(b["volume_m3"] for b in buildings), 0),
            "tallest_m": max(b["height_m"] for b in buildings),
            "median_facade_coverage_pct": round(float(np.median(cov)), 1) if cov else None}


# =============================================================================
# GIS / digital-twin encodings
# =============================================================================
def _props(b, gf):
    datum = (gf.origin[2] + gf.v_offset) if gf else 0.0
    return {"id": b["id"], "height_m": b["height_m"], "uncertainty_m": b["uncertainty_m"], "eaves_m": b["eaves_m"],
            "roof_type": b["roof_type"], "roof_pitch_deg": b["roof_pitch_deg"], "storeys_est": b["storeys_est"],
            "footprint_m2": b["footprint_m2"], "roof_area_m2": b["roof_area_m2"], "wall_area_m2": b["wall_area_m2"],
            "facade_coverage_pct": b["facade_coverage_pct"], "volume_m3": b["volume_m3"],
            "base_elevation_m": round(b["base_y"] + datum, 3), "measurement_method": b["method"]}


def to_geojson_features(buildings, gf):
    import geo_export as G
    feats = []
    for b in buildings:
        if len(b["footprint"]) < 3:
            continue
        ring = G.lonlat(gf, b["footprint"])
        ring.append(ring[0])
        props = _props(b, gf)
        props["layer"] = "building"
        feats.append(G.feature("Polygon", [ring], props))
    return feats


def to_cityjson(buildings, gf, title="PRISM survey"):
    """CityJSON 1.1, LoD 1.2 block solids (footprint extruded from the base to the roof mean height)."""
    verts, objs = [], {}
    all_u = []
    for b in buildings:
        fp = np.asarray(b["footprint"], float)
        if len(fp) < 3:
            continue
        base = b["base_y"]
        # LoD1 convention: mean roof height (eaves + half the roof rise for pitched roofs)
        top_rel = b["height_m"] if b["roof_type"] == "FLAT" or not b["eaves_m"] else 0.5 * (b["eaves_m"] + b["height_m"])
        bot = gf.model_to_utm(np.c_[fp[:, 0], np.full(len(fp), base), fp[:, 1]])
        top = gf.model_to_utm(np.c_[fp[:, 0], np.full(len(fp), base + top_rel), fp[:, 1]])
        i0 = len(verts)
        verts.extend(bot.tolist())
        verts.extend(top.tolist())
        all_u.append(bot)
        all_u.append(top)
        nb = len(fp)
        B = list(range(i0, i0 + nb))
        U = list(range(i0 + nb, i0 + 2 * nb))
        surfaces = [[B[::-1]], [U]]
        sem = [0, 1]
        for k in range(nb):
            a, c = k, (k + 1) % nb
            surfaces.append([[B[a], B[c], U[c], U[a]]])
            sem.append(2)
        objs[b["id"]] = {
            "type": "Building",
            "attributes": dict(_props(b, gf), roofType=ROOF_CODES[b["roof_type"]], measuredHeight=b["height_m"],
                               storeysAboveGround=b["storeys_est"]),
            "geometry": [{"type": "Solid", "lod": "1.2", "boundaries": [surfaces],
                          "semantics": {"surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}],
                                        "values": [sem]}}],
        }
    if not verts:
        return None
    v = np.asarray(verts)
    tr = np.floor(v.min(0))
    q = np.round((v - tr) / 0.001).astype(np.int64)
    ext = np.r_[v.min(0), v.max(0)]
    return {
        "type": "CityJSON", "version": "1.1",
        "transform": {"scale": [0.001, 0.001, 0.001], "translate": tr.tolist()},
        "metadata": {"referenceSystem": f"https://www.opengis.net/def/crs/EPSG/0/{gf.epsg}", "title": title,
                     "geographicalExtent": ext.round(3).tolist(),
                     "pointOfContact": {"contactName": "PRISM survey intelligence"}},
        "CityObjects": objs, "vertices": q.tolist(),
    }
