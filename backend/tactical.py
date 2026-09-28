"""
PRISM // Tactical terrain analysis (tactical.py)
================================================
Two planning tools built on the metric terrain model (terrain.py):

1. EXPOSURE  - probability that a person standing at a cell is seen by at least one observer.
   Observers default to where people are: around every building (doors/windows, 1.7 m eyes, plus a
   roof-level post on tall buildings) and along roads; operators can add their own threat posts.
   For each observer a line-of-sight viewshed is computed on the surface model (trees and buildings
   block sight). Detection falls off with range, p(d) = 0.95 * 2^-(d / 350 m)^2, and is reduced
   where the target is concealed (inside/against tree cover x0.35, in tall grass/reeds x0.6).
   Exposure = 1 - prod(1 - p_k)  (independent observers).

2. BASE-SITE FINDER - "a clearing hidden between trees, away from houses":
   hard constraints: open, flat (slope <= max), dry (away from water / no-return), fully inside
   the surveyed area, a free disk of the requested radius;
   scored on: enclosure by trees on all four sides (36 rays, quadrant coverage), distance from
   buildings, concealment (1 - exposure over the site), flatness, road access, drainage
   (not a local depression). Non-maximum suppression returns distinct sites.

3. COVERT ROUTE PLANNER - least-cost path on an 8-connected grid (Dijkstra, directed edges):
   edge cost = walking time (Tobler's hiking function, direction-aware uphill/downhill)
               x terrain factor (road 0.9, open 1.0, grass 1.2, under trees 1.5, steep 3.0)
               x (1 + w_exposure * exposure) + w_slope * steepness penalty.
   Buildings, water/no-return and cliffs are impassable; steep ground is allowed but expensive, so
   flat ground is preferred and high/steep ground is used only when nothing else connects.
   Three routes are returned: COVERT (exposure dominates), BALANCED and FASTEST.
"""

import math
import time

import numpy as np
from scipy import ndimage as ndi

try:
    import terrain as T
except ImportError:
    from backend import terrain as T

D50_M = 350.0
P0 = 0.95


# =============================================================================
# observers & exposure
# =============================================================================
def default_observers(tm, max_obs=40, include_roads=False, road_step_m=80.0):
    """
    Default watchers = occupied buildings. Each building gets one observer just above its roof at the
    footprint centre: it sees over its own roof in every direction, i.e. windows on all sides. That is
    deliberately pessimistic for us (a real window sees less), so concealment is never over-stated.
    Road observers (passing traffic) are optional; operators add their own threat posts on top.
    """
    obs = []
    bl, bn = ndi.label(tm.cls == T.BUILDING, np.ones((3, 3)))
    if bn:
        idx = np.arange(1, bn + 1)
        area = np.bincount(bl.ravel(), minlength=bn + 1)[1:] * tm.res ** 2
        hmax = np.asarray(ndi.maximum(np.nan_to_num(tm.ndsm), bl, idx))
        com = ndi.center_of_mass(np.ones_like(bl), bl, idx)
        for k in np.argsort(-area):
            if area[k] < 15.0:
                continue
            ci, cj = com[k]
            cx, cz = tm.centre(ci, cj)
            eye = float(tm.sample(tm.dtm, cx, cz) + max(hmax[k], 2.5) + 0.3)
            obs.append({"x": float(cx), "z": float(cz), "eye_abs": eye, "kind": "building",
                        "height_m": round(float(hmax[k]), 1)})
    road = getattr(tm, "road", None)
    if include_roads and road is not None and road.any():
        rr = ndi.binary_erosion(road, iterations=2)
        ii, jj = np.nonzero(rr if rr.any() else road)
        rx, rz = tm.centre(ii, jj)
        picked = _farthest_points(np.column_stack([rx, rz]), road_step_m)
        for x, z in picked:
            obs.append({"x": float(x), "z": float(z), "eye_h": 1.7, "kind": "road"})
    if len(obs) > max_obs:
        xy = np.array([[o["x"], o["z"]] for o in obs])
        keep = _farthest_indices(xy, max_obs)
        obs = [obs[i] for i in keep]
    return obs


def _farthest_points(xz, spacing):
    if len(xz) == 0:
        return []
    out = [xz[0]]
    d = np.linalg.norm(xz - xz[0], axis=1)
    while True:
        k = int(np.argmax(d))
        if d[k] < spacing:
            break
        out.append(xz[k])
        d = np.minimum(d, np.linalg.norm(xz - xz[k], axis=1))
    return out


def _farthest_indices(xz, n):
    sel = [0]
    d = np.linalg.norm(xz - xz[0], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, np.linalg.norm(xz - xz[k], axis=1))
    return sorted(set(sel))


def concealment_factor(tm):
    """Multiplier (<= 1) on detection probability from vegetation around the target."""
    tree = (tm.cls == T.TREE)
    near_tree = ndi.binary_dilation(tree, iterations=max(1, int(round(1.5 / tm.res))))
    tall_grass = (tm.cls == T.LOW_VEG) & (np.nan_to_num(tm.ndsm) >= 0.8)
    f = np.ones(tm.cls.shape, np.float32)
    f[tall_grass] = 0.6
    f[near_tree] = 0.35
    return f


_EXPO_CACHE = {}


def exposure_map(tm, observers, radius=500.0, target_h=1.7):
    """Probability of being seen by >= 1 observer, per cell (0 outside the survey). Cached per observer set."""
    key = (id(tm), tm.nx, tm.nz, round(tm.res, 3), radius, target_h,
           tuple((round(o["x"], 1), round(o["z"], 1), o.get("eye_h"), o.get("eye_abs")) for o in observers))
    if key in _EXPO_CACHE:
        return _EXPO_CACHE[key]
    not_seen = np.ones((tm.nz, tm.nx), np.float64)
    count = np.zeros((tm.nz, tm.nx), np.int16)
    for o in observers:
        vis, dist = tm.viewshed(o["x"], o["z"], eye_h=o.get("eye_h", 1.7), target_h=target_h,
                                radius=radius, eye_abs=o.get("eye_abs"))
        p = P0 * np.exp2(-(dist / D50_M) ** 2)
        not_seen *= np.where(vis, 1.0 - p, 1.0)
        count += vis
    expo = (1.0 - not_seen) * concealment_factor(tm)
    expo[~tm.roi] = 0.0
    out = (expo.astype(np.float32), count)
    if len(_EXPO_CACHE) > 6:
        _EXPO_CACHE.clear()
    _EXPO_CACHE[key] = out
    return out


# =============================================================================
# base-site finder
# =============================================================================
def find_base_sites(tm, telem=None, radius_m=15.0, min_building_dist_m=150.0, max_slope_deg=6.0,
                    max_road_dist_m=500.0, top_k=5, observers=None, log=print):
    t0 = time.time()
    f = max(1, int(round(1.0 / tm.res)))
    c = tm.coarsen(f)
    res = c.res
    obs_list = observers if observers is not None else default_observers(c)
    expo, _ = exposure_map(c, obs_list) if obs_list else (np.zeros((c.nz, c.nx), np.float32), None)
    cls = c.cls
    tree = cls == T.TREE
    bld = cls == T.BUILDING
    water = (cls == T.WATER) | ~c.roi
    d_bld = ndi.distance_transform_edt(~bld) * res if bld.any() else np.full(cls.shape, 9999.0)
    road = getattr(c, "road", None)
    d_road = ndi.distance_transform_edt(~road) * res if (road is not None and road.any()) else np.full(cls.shape, 9999.0)
    d_water = ndi.distance_transform_edt(~water) * res
    slope = c.slope
    open_ground = np.isin(cls, (T.GROUND, T.LOW_VEG)) & c.roi & c.observed & (slope <= max_slope_deg) & (d_water >= 8.0)
    free = ndi.distance_transform_edt(open_ground) * res
    cand = free >= radius_m
    relaxed = []
    if not cand.any():
        for r in (0.75 * radius_m, 0.5 * radius_m):
            cand = free >= r
            if cand.any():
                relaxed.append(f"no {2 * radius_m:.0f} m clearing: searched for {2 * r:.0f} m instead")
                radius_m = r
                break
    if not cand.any():
        return {"sites": [], "message": "No open, flat, dry clearing of the requested size inside the survey.",
                "observers": obs_list, "seconds": round(time.time() - t0, 2)}
    step = max(1, int(round(radius_m / 2 / res)))
    ii, jj = np.nonzero(cand)
    sel = ((ii % step) == 0) & ((jj % step) == 0)
    if sel.sum() < 20:
        sel = np.ones(len(ii), bool)
    ii, jj = ii[sel], jj[sel]
    # enclosure by trees: 36 rays from the site edge out to radius + 45 m
    n_rays = 36
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dists = np.arange(radius_m, radius_m + 45.0, res)
    ri = ii[:, None, None] + (np.sin(th)[None, :, None] * dists[None, None, :] / res)
    rj = jj[:, None, None] + (np.cos(th)[None, :, None] * dists[None, None, :] / res)
    ri = np.clip(np.rint(ri).astype(np.int32), 0, c.nz - 1)
    rj = np.clip(np.rint(rj).astype(np.int32), 0, c.nx - 1)
    hit = tree[ri, rj] | bld[ri, rj]
    covered = hit.any(axis=2)                                    # (cand, rays)
    first = np.where(covered, np.argmax(hit, axis=2) * res + radius_m, np.nan)
    enclosure = covered.mean(axis=1)
    # compass quadrants: rays are at angles from +x (east) towards +z (south)
    ang = (np.degrees(th) + 360) % 360
    quad = {"E": (ang >= 315) | (ang < 45), "S": (ang >= 45) & (ang < 135), "W": (ang >= 135) & (ang < 225),
            "N": (ang >= 225) & (ang < 315)}
    qcov = {k: covered[:, m].mean(axis=1) for k, m in quad.items()}
    min_q = np.min(np.stack(list(qcov.values()), 1), axis=1)
    # site-disk statistics (sampled)
    disk_off = [(0, 0)] + [(radius_m * 0.7 * math.sin(a), radius_m * 0.7 * math.cos(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    e_site = np.zeros(len(ii))
    s_site = np.zeros(len(ii))
    elev = np.zeros((len(ii), len(disk_off)))
    for k, (di, dj) in enumerate(disk_off):
        pi_ = np.clip(np.rint(ii + di / res).astype(int), 0, c.nz - 1)
        pj_ = np.clip(np.rint(jj + dj / res).astype(int), 0, c.nx - 1)
        e_site += expo[pi_, pj_]
        s_site = np.maximum(s_site, slope[pi_, pj_])
        elev[:, k] = c.dtm[pi_, pj_]
    e_site /= len(disk_off)
    relief = elev.max(1) - elev.min(1)
    tpi = c.dtm[ii, jj] - ndi.uniform_filter(c.dtm, size=max(3, int(60 / res)) | 1, mode="nearest")[ii, jj]
    db, dr = d_bld[ii, jj], d_road[ii, jj]
    # scores in [0, 1]
    s_enc = 0.6 * enclosure + 0.4 * min_q
    s_iso = np.clip(db / max(min_building_dist_m * 2, 1.0), 0, 1)
    s_con = 1.0 - np.clip(e_site / 0.6, 0, 1)
    s_flat = 1.0 - np.clip(s_site / max_slope_deg, 0, 1) * 0.6 - np.clip(relief / 2.0, 0, 1) * 0.4
    s_acc = np.where(dr <= max_road_dist_m, 1.0 - np.clip((dr - 60.0) / max(max_road_dist_m, 1), 0, 1) * 0.6, 0.2)
    s_acc = np.where(dr < 25.0, 0.4, s_acc)                    # right next to a road = visible traffic
    s_dry = np.clip(0.5 + tpi / 1.0, 0, 1)
    score = 100 * (0.30 * s_enc + 0.22 * s_iso + 0.25 * s_con + 0.10 * s_flat + 0.08 * s_acc + 0.05 * s_dry)
    hard_ok = db >= min_building_dist_m
    meets = hard_ok.copy()
    if not hard_ok.any():
        relaxed.append(f"no clearing is {min_building_dist_m:.0f} m from buildings: best available shown "
                       f"(max {float(db.max()):.0f} m)")
        hard_ok = db >= 0.8 * float(db.max())
    score = np.where(hard_ok, score, 0.5 * score)
    order = np.argsort(-score)
    chosen = []
    for k in order:
        if len(chosen) >= top_k:
            break
        x, z = c.centre(ii[k], jj[k])
        if any(math.hypot(x - s["center"][0], z - s["center"][2]) < 2.5 * radius_m for s in chosen):
            continue
        y = float(c.dtm[ii[k], jj[k]])
        lat, lon = T.to_latlon(telem, x, y, z)
        rays = [None if not np.isfinite(v) else round(float(v), 1) for v in first[k]]
        why = []
        why.append(f"trees on {int(round(100 * enclosure[k]))}% of the horizon "
                   f"(N {int(100 * qcov['N'][k])}%, E {int(100 * qcov['E'][k])}%, S {int(100 * qcov['S'][k])}%, W {int(100 * qcov['W'][k])}%)")
        why.append(f"nearest building {db[k]:.0f} m" if db[k] < 9000 else "no buildings in the survey")
        why.append(f"seen by observers with {100 * e_site[k]:.0f}% probability")
        why.append(f"slope <= {s_site[k]:.1f} deg, relief {relief[k]:.1f} m")
        why.append(f"road access {dr[k]:.0f} m" if dr[k] < 9000 else "no road detected")
        chosen.append({
            "center": [round(float(x), 2), round(y, 2), round(float(z), 2)], "radius_m": round(radius_m, 1),
            "area_m2": round(math.pi * radius_m ** 2, 0), "gps_lat": lat, "gps_lon": lon,
            "score": round(float(score[k]), 1), "meets_all_constraints": bool(meets[k] and not relaxed),
            "enclosure_pct": round(100 * float(enclosure[k]), 1),
            "quadrant_cover_pct": {q: round(100 * float(v[k]), 0) for q, v in qcov.items()},
            "four_sides_covered": bool(min_q[k] >= 0.5),
            "nearest_building_m": round(float(db[k]), 1) if db[k] < 9000 else None,
            "nearest_road_m": round(float(dr[k]), 1) if dr[k] < 9000 else None,
            "detection_probability_pct": round(100 * float(e_site[k]), 1),
            "max_slope_deg": round(float(s_site[k]), 1), "relief_m": round(float(relief[k]), 2),
            "drainage": "local low point - flood risk" if tpi[k] < -0.3 else ("raised / well drained" if tpi[k] > 0.2 else "level"),
            "cover_ray_distances_m": rays, "rationale": why,
        })
    for i, s in enumerate(chosen):
        s["id"] = f"SITE-{chr(65 + i)}"
    log(f"[Sites] {len(chosen)} site(s) from {len(ii)} candidates in {time.time() - t0:.1f}s")
    return {"sites": chosen, "relaxed": relaxed, "observers": obs_list, "radius_m": radius_m,
            "criteria": {"min_building_dist_m": min_building_dist_m, "max_slope_deg": max_slope_deg,
                         "max_road_dist_m": max_road_dist_m},
            "seconds": round(time.time() - t0, 2)}


# =============================================================================
# route planning
# =============================================================================
TERRAIN_FACTOR = {T.GROUND: 1.0, T.ROAD: 0.9, T.LOW_VEG: 1.2, T.TREE: 1.5, T.VEHICLE: np.inf,
                  T.BUILDING: np.inf, T.WATER: np.inf, T.NODATA: np.inf}
MODES = {"covert": dict(w_exp=14.0, w_slope=0.6, label="Covert"),
         "balanced": dict(w_exp=4.0, w_slope=0.35, label="Balanced"),
         "fastest": dict(w_exp=0.0, w_slope=0.1, label="Fastest")}


def tobler_kmh(grade):
    return 6.0 * np.exp(-3.5 * np.abs(grade + 0.05))


def _grid_for_routes(tm, max_cells=450_000):
    roi_cells = int(tm.roi.sum())
    f = max(1, int(math.ceil(math.sqrt(max(roi_cells, 1) / max_cells))))
    f = max(f, int(round(1.0 / tm.res)))
    return tm.coarsen(f)


def plan_routes(tm, telem, start, end, modes=("covert", "balanced", "fastest"), observers=None,
                max_slope_deg=35.0, log=print):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    t0 = time.time()
    c = _grid_for_routes(tm)
    res = c.res
    obs_list = observers if observers is not None else default_observers(c)
    expo, _ = exposure_map(c, obs_list) if obs_list else (np.zeros((c.nz, c.nx), np.float32), None)
    fac = np.full(c.cls.shape, np.inf)
    for k, v in TERRAIN_FACTOR.items():
        fac[c.cls == k] = v
    fac[~c.roi] = np.inf
    steep = c.slope > 25.0
    fac = np.where(steep & np.isfinite(fac), fac * 3.0, fac)      # "high ground only if nothing else"
    fac[c.slope > max_slope_deg] = np.inf
    passable = np.isfinite(fac)

    def snap(p):
        i, j = c.cell_of(p[0], p[2] if len(p) > 2 else p[1])
        i, j = int(np.clip(i, 0, c.nz - 1)), int(np.clip(j, 0, c.nx - 1))
        if passable[i, j]:
            return i, j, 0.0
        d, idx = ndi.distance_transform_edt(~passable, return_indices=True)
        return int(idx[0][i, j]), int(idx[1][i, j]), float(d[i, j] * res)

    si, sj, s_snap = snap(start)
    ei, ej, e_snap = snap(end)
    nid = -np.ones(c.cls.shape, np.int64)
    pi, pj = np.nonzero(passable)
    nid[pi, pj] = np.arange(len(pi))
    dtm = c.dtm.astype(np.float64)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    routes = []
    for mode in modes:
        prm = MODES[mode]
        rows, cols, data = [], [], []
        for di, dj in nbrs:
            qi, qj = pi + di, pj + dj
            ok = (qi >= 0) & (qi < c.nz) & (qj >= 0) & (qj < c.nx)
            a_i, a_j, b_i, b_j = pi[ok], pj[ok], qi[ok], qj[ok]
            okb = passable[b_i, b_j]
            a_i, a_j, b_i, b_j = a_i[okb], a_j[okb], b_i[okb], b_j[okb]
            dh = math.hypot(di, dj) * res
            dz = dtm[b_i, b_j] - dtm[a_i, a_j]
            grade = dz / dh
            t_s = np.sqrt(dh * dh + dz * dz) / (tobler_kmh(grade) / 3.6)
            steepness = (np.degrees(np.arctan(np.abs(grade))) / 10.0) ** 2
            cost = t_s * fac[b_i, b_j] * (1.0 + prm["w_exp"] * expo[b_i, b_j]) + prm["w_slope"] * t_s * steepness
            rows.append(nid[a_i, a_j])
            cols.append(nid[b_i, b_j])
            data.append(cost)
        g = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(len(pi), len(pi)))
        s_node, e_node = nid[si, sj], nid[ei, ej]
        dist, pred = dijkstra(g, directed=True, indices=s_node, return_predecessors=True)
        if not np.isfinite(dist[e_node]):
            moved = [f"{name} moved {d:.0f} m to walkable ground" for name, d in (("start", s_snap), ("end", e_snap)) if d > 1.0]
            return {"routes": [], "message": "No passable route: the two points are separated by water, buildings or cliffs"
                    + (f" ({'; '.join(moved)})" if moved else "") + ". Pick both points on the same side.",
                    "seconds": round(time.time() - t0, 2)}
        path = [e_node]
        while path[-1] != s_node:
            path.append(pred[path[-1]])
        path = path[::-1]
        ri, rj = pi[path], pj[path]
        routes.append(_route_metrics(c, telem, mode, prm, ri, rj, expo, fac))
    log(f"[Routes] {len(routes)} route(s) on {c.nx}x{c.nz} grid @ {res:.1f} m in {time.time() - t0:.1f}s")
    return {"routes": routes, "observers": obs_list, "grid_res_m": res,
            "start_snap_m": round(s_snap, 1), "end_snap_m": round(e_snap, 1), "seconds": round(time.time() - t0, 2)}


def _route_metrics(c, telem, mode, prm, ri, rj, expo, fac):
    x, z = c.centre(ri, rj)
    y = c.dtm[ri, rj].astype(np.float64)
    seg_h = np.hypot(np.diff(x), np.diff(z))
    dz = np.diff(y)
    seg3 = np.sqrt(seg_h ** 2 + dz ** 2)
    grade = np.divide(dz, seg_h, out=np.zeros_like(dz), where=seg_h > 0)
    t_s = seg3 / (tobler_kmh(grade) / 3.6) * np.minimum(fac[ri[1:], rj[1:]], 3.0)
    e = expo[ri, rj].astype(np.float64)
    e_seg = 0.5 * (e[1:] + e[:-1])
    cls = c.cls[ri, rj]
    length = float(seg3.sum())
    comp = {}
    for k, name in ((T.ROAD, "road"), (T.GROUND, "open ground"), (T.LOW_VEG, "grass / reeds"), (T.TREE, "tree cover")):
        comp[name] = round(100 * float(seg3[cls[1:] == k].sum()) / max(length, 1e-9), 1)
    # display polyline: simplify but keep the exposure profile
    keep = _simplify_idx(np.c_[x, z], tol=max(0.6, 0.6 * c.res))
    pts = np.c_[x[keep], y[keep] + 0.35, z[keep]].round(2).tolist()
    lat0, lon0 = T.to_latlon(telem, x[0], y[0], z[0])
    lat1, lon1 = T.to_latlon(telem, x[-1], y[-1], z[-1])
    exposed = float(seg3[e_seg > 0.3].sum())
    # time-weighted mean exposure: the share of the movement time spent where observers would see you
    dt = t_s
    risk = float(np.sum(e_seg * dt) / max(float(dt.sum()), 1e-9))
    exposed_time = float(dt[e_seg > 0.3].sum()) / 60.0
    return {
        "mode": mode, "label": prm["label"], "points": pts, "exposure": [round(float(v), 3) for v in e[keep]],
        "length_m": round(length, 1), "time_min": round(float(t_s.sum()) / 60.0, 1),
        "climb_m": round(float(dz[dz > 0].sum()), 1), "descent_m": round(float(-dz[dz < 0].sum()), 1),
        "max_slope_deg": round(float(np.degrees(np.arctan(np.abs(grade).max()))) if len(grade) else 0.0, 1),
        "mean_exposure_pct": round(100 * float(np.sum(e_seg * seg3) / max(length, 1e-9)), 1),
        "max_exposure_pct": round(100 * float(e.max()), 1),
        "exposed_distance_m": round(exposed, 1), "exposed_time_min": round(exposed_time, 1),
        "concealed_pct": round(100 * float(seg3[e_seg < 0.15].sum()) / max(length, 1e-9), 1),
        "detection_risk_pct": round(100 * risk, 1), "terrain_mix_pct": comp,
        "start_gps": [lat0, lon0], "end_gps": [lat1, lon1],
    }


def _simplify_idx(pts, tol):
    """Douglas-Peucker returning kept indices."""
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        v = q - p
        n = np.hypot(*v)
        seg = pts[a + 1:b]
        d = np.abs(v[0] * (seg[:, 1] - p[1]) - v[1] * (seg[:, 0] - p[0])) / n if n > 1e-12 else np.hypot(*(seg - p).T)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.flatnonzero(keep)
