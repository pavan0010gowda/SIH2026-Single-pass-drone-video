"""
PRISM // Road network & pothole engine (road_engine.py)
=======================================================
Roads are extracted from the metric 3D model, not from a fixed image window:

  1. Road likelihood per 0.5 m cell = product of fuzzy memberships
       at ground level (nDSM), gentle slope, smooth surface (local plane RMS), not vegetated
       (excess green), and road-like colour (asphalt/concrete: grey; earth/gravel: tan-brown).
  2. Hysteresis segmentation (strong seeds grown into weak cells), then shape filtering: a road is a
     long, narrow band. Per component the skeleton length and the distance-transform width give
     elongation = length / width; only bands 2-25 m wide with elongation >= 4 are kept.
  3. Road completion ("torn" roads): gaps left by moving/parked vehicles, shadows or missing stereo
     returns are closed along the road (closing + bounded hole filling). Vehicles standing on the
     road are reported separately.
  4. Centrelines are traced from the skeleton graph and simplified (Douglas-Peucker).

Potholes are measured on the road surface points at fine resolution (~0.15-0.2 m):
  * reference surface = iteratively re-weighted Gaussian smoothing of the road (depressions are
    down-weighted each round, so the reference follows the intact pavement / crown / camber);
  * residual = surface - reference; noise sigma = 1.4826 * MAD on the road;
  * limit of detection LoD = max(3 sigma, 2 cm); depressions deeper than LoD with area 0.1-25 m^2
    become potholes; depth = 95th percentile of the depression (robust maximum), plus mean depth,
    area, equivalent diameter and volume;
  * severity follows ASTM D6433 (depth x diameter matrix).
Nothing is invented: when the surface noise is above the depth of interest the report says so.
"""

import math

import numpy as np
from scipy import ndimage as ndi

try:
    import terrain as T
except ImportError:
    from backend import terrain as T


# =============================================================================
# colour helpers
# =============================================================================
def rgb_to_hsv(rgb):
    r, g, b = [rgb[..., k].astype(np.float64) / 255.0 for k in range(3)]
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    rm, gm, bm = m & (mx == r), m & (mx == g) & (mx != r), m & (mx == b) & (mx != r) & (mx != g)
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h *= 60.0
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0.0)
    return h, s, mx


def _ramp(x, a, b):
    """0 below a, 1 above b (a < b), linear in between; reversed when a > b."""
    if a < b:
        return np.clip((x - a) / (b - a), 0.0, 1.0)
    return np.clip((a - x) / (a - b), 0.0, 1.0)


def road_likelihood(tm):
    nd = np.nan_to_num(getattr(tm, "nd_med", tm.ndsm), nan=9.0)
    rough = np.nan_to_num(getattr(tm, "rough_med", tm.rough), nan=1.0)
    h, s, v = rgb_to_hsv(tm.rgb)
    s_level = _ramp(nd, 0.45, 0.15)
    s_slope = _ramp(tm.slope, 18.0, 7.0) ** 0.5
    s_smooth = _ramp(rough, 0.28, 0.07) ** 0.5
    s_bare = _ramp(tm.exg, 0.10, 0.03)
    dirt = _ramp(v, 0.30, 0.45) * _ramp(s, 0.06, 0.14) * _ramp(s, 0.60, 0.45) * \
        np.clip(np.minimum(_ramp(h, 5.0, 18.0), _ramp(h, 62.0, 48.0)), 0, 1)
    paved = _ramp(s, 0.22, 0.12) * _ramp(v, 0.12, 0.2) * _ramp(v, 0.97, 0.9)
    s_col = np.maximum(dirt, paved)
    L = s_level * s_slope * s_smooth * s_bare * s_col
    L[~tm.observed] = 0.0
    return ndi.gaussian_filter(L, 1.0 / tm.res * 0.6), dict(dirt=dirt, paved=paved, h=h, s=s, v=v)


# =============================================================================
# skeleton -> polylines
# =============================================================================
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skel):
    """Ordered pixel paths between endpoints/junctions of a skeleton."""
    sk = np.pad(skel.astype(bool), 1)
    deg = sum(np.roll(np.roll(sk, -di, 0), -dj, 1).astype(np.int8) for di, dj in _NB) * sk
    pts = set(zip(*np.nonzero(sk)))
    nodes = {p for p in pts if deg[p] != 2}
    visited_edges = set()
    paths = []

    def nbrs(p):
        return [(p[0] + di, p[1] + dj) for di, dj in _NB if (p[0] + di, p[1] + dj) in pts]

    starts = list(nodes) if nodes else ([next(iter(pts))] if pts else [])
    for s in starts:
        for n in nbrs(s):
            if (s, n) in visited_edges:
                continue
            path = [s, n]
            visited_edges.update({(s, n), (n, s)})
            prev, cur = s, n
            while cur not in nodes:
                nxt = [q for q in nbrs(cur) if q != prev and (cur, q) not in visited_edges]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                visited_edges.update({(prev, cur), (cur, prev)})
                path.append(cur)
                if cur == s:
                    break
            paths.append([(i - 1, j - 1) for i, j in path])
    return paths


def skeleton_paths(skel, min_len_px=6, max_paths=12):
    """
    Centrelines of a skeleton as geodesic paths: repeatedly take the longest shortest-path (graph
    diameter) through the remaining pixels, then remove it and its immediate neighbourhood. Robust to
    the staircase pixels a thinned diagonal band always has (they would look like junctions).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra, connected_components
    ij = np.argwhere(skel)
    if len(ij) < 2:
        return []
    index = -np.ones(skel.shape, np.int64)
    index[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
    rows, cols, wts = [], [], []
    h, w = skel.shape
    for di, dj in _NB:
        a = ij + np.array([di, dj])
        ok = (a[:, 0] >= 0) & (a[:, 0] < h) & (a[:, 1] >= 0) & (a[:, 1] < w)
        nb = np.full(len(ij), -1)
        nb[ok] = index[a[ok, 0], a[ok, 1]]
        m = nb >= 0
        rows.append(np.nonzero(m)[0])
        cols.append(nb[m])
        wts.append(np.full(m.sum(), math.hypot(di, dj)))
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ij), len(ij))).tocsr()
    alive = np.ones(len(ij), bool)
    paths = []
    for _ in range(max_paths):
        idx_alive = np.flatnonzero(alive)
        if len(idx_alive) < min_len_px:
            break
        sub = g[idx_alive][:, idx_alive]
        ncomp, lab = connected_components(sub, directed=False)
        sizes = np.bincount(lab)
        comp = int(np.argmax(sizes))
        if sizes[comp] < min_len_px:
            break
        members = np.flatnonzero(lab == comp)
        d0 = dijkstra(sub, indices=members[0], limit=np.inf)
        a = members[int(np.argmax(np.where(np.isfinite(d0[members]), d0[members], -1)))]
        da, pred = dijkstra(sub, indices=a, return_predecessors=True)
        b = members[int(np.argmax(np.where(np.isfinite(da[members]), da[members], -1)))]
        if da[b] < min_len_px:
            alive[idx_alive[members]] = False
            continue
        path, cur = [], b
        while cur != -9999 and cur >= 0:
            path.append(idx_alive[cur])
            if cur == a:
                break
            cur = pred[cur]
        path = path[::-1]
        paths.append([tuple(ij[p]) for p in path])
        # remove the path and the pixels right next to it (short side spurs)
        used = np.zeros(skel.shape, bool)
        used[ij[path, 0], ij[path, 1]] = True
        used = ndi.binary_dilation(used, iterations=2)
        alive &= ~used[ij[:, 0], ij[:, 1]]
    return paths


def douglas_peucker(pts, tol):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-12:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        d = np.abs(np.cross(ab, pts - a)) / n
    k = int(np.argmax(d))
    if d[k] > tol:
        left = douglas_peucker(pts[:k + 1], tol)
        right = douglas_peucker(pts[k:], tol)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


# =============================================================================
# road detection
# =============================================================================
def detect_roads(tm, log=print, min_len_m=20.0, min_w=2.0, max_w=25.0, min_elong=4.0):
    res = tm.res
    L, col = road_likelihood(tm)
    strong, weak = L > 0.45, L > 0.22
    seeds = ndi.binary_opening(strong, np.ones((3, 3)))
    grown = ndi.binary_propagation(seeds, mask=weak)
    grown = ndi.binary_opening(grown, np.ones((3, 3)))
    r_close = max(1, int(round(1.5 / res)))
    yy, xx = np.mgrid[-r_close:r_close + 1, -r_close:r_close + 1]
    disk = (xx * xx + yy * yy) <= r_close * r_close
    grown = ndi.binary_closing(grown, disk)
    lab, n = ndi.label(grown, np.ones((3, 3)))
    mask = np.zeros_like(grown)
    segments = []
    rejected = {"shape": 0, "width_varies": 0, "no_contrast": 0}
    if n:
        area_cells = np.bincount(lab.ravel(), minlength=n + 1)
        objs = ndi.find_objects(lab)
        pad = int(round(4.0 / res)) + 2
        for k in range(1, n + 1):
            if area_cells[k] * res * res < 30.0:
                continue
            sl = objs[k - 1]
            sl = (slice(max(0, sl[0].start - pad), min(lab.shape[0], sl[0].stop + pad)),
                  slice(max(0, sl[1].start - pad), min(lab.shape[1], sl[1].stop + pad)))
            comp = lab[sl] == k
            dt = ndi.distance_transform_edt(comp) * res
            skel = T.zhang_suen_thinning(comp)
            if skel.sum() < 3:
                continue
            widths = 2.0 * dt[skel]
            w_med = float(np.median(widths))
            length = float(skel.sum()) * res * 1.12        # 8-connected pixel steps average ~1.12 cells
            elong = length / max(w_med, 1e-6)
            if length < min_len_m or not (min_w <= w_med <= max_w) or elong < min_elong:
                rejected["shape"] += 1
                continue
            # judge the MAIN centreline (longest geodesic path): a road's is many times longer than the road
            # is wide and keeps a steady width; a blob of bare ground has a short main path whose width
            # swells in the middle (its many skeleton branches must not count as length)
            main = skeleton_paths(skel, min_len_px=3, max_paths=1)
            if main:
                mp = np.array(main[0])
                steps = np.hypot(*np.diff(mp, axis=0).T).sum() * res if len(mp) > 1 else 0.0
                wm = 2.0 * dt[mp[:, 0], mp[:, 1]]
                cut = max(1, len(wm) // 10)
                wm = wm[cut:-cut] if len(wm) > 2 * cut + 3 else wm
                w90 = float(np.percentile(wm, 90)) if len(wm) else w_med
                cv = float(np.std(wm) / max(np.mean(wm), 1e-6)) if len(wm) else 1.0
                ratio = steps / max(w90, 1e-6)
                # (Danube: main road 23x / cv 0.33, road into a farmyard 14x / 0.47; moorland blobs 4x with
                #  w90 18-34 m or cv >= 0.53) - very wide candidates must also be very long
                if ratio < 3.5 or cv > 0.5 or (w90 > 15.0 and ratio < 6.0):
                    rejected["width_varies"] += 1
                    continue
            # a road stands out from its verges (grass, crops, scrub); moorland or a bare field does not
            band = ndi.binary_dilation(comp, iterations=int(round(3.0 / res))) & ~ndi.binary_dilation(comp, iterations=1)
            band &= tm.observed[sl]
            if band.sum() >= 10 and float(L[sl][comp].mean() - L[sl][band].mean()) < 0.16:
                rejected["no_contrast"] += 1
                continue
            # prune side-branches wider than the road (fields touching the road): keep cells within
            # 1.6x the median half-width of the skeleton
            near = ndi.distance_transform_edt(~skel) * res <= 0.8 * w_med + res
            comp &= near
            m_glob = np.zeros_like(mask)
            m_glob[sl] = comp
            mask |= m_glob
            segments.append(dict(slice=sl, skel=skel & comp, dt=dt, width=w_med, length=length, elong=elong))
    # completion: fill bounded holes (vehicles, shadows, no-return) inside the road network
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    hl, hn = ndi.label(holes)
    if hn:
        ha = np.bincount(hl.ravel(), minlength=hn + 1) * res * res
        small = (hl > 0) & (ha[hl] <= 60.0)
        repaired = int(small.sum())
        mask |= small
    else:
        repaired = 0
    # vehicles standing on or at the edge of the road
    veh = (tm.cls == T.VEHICLE) & ndi.binary_dilation(mask, iterations=max(1, int(round(1.0 / res))))
    mask |= veh
    road_cells = int(mask.sum())
    # per-segment descriptors
    h, s, v = col["h"], col["s"], col["v"]
    out_segments = []
    lab2, n2 = ndi.label(mask, np.ones((3, 3)))
    for k in range(1, n2 + 1):
        m = lab2 == k
        area = float(m.sum()) * res * res
        if area < 30.0:
            continue
        sl = ndi.find_objects(m.astype(np.int8))[0]
        sub = m[sl]
        skel = T.zhang_suen_thinning(sub)
        dt = ndi.distance_transform_edt(sub) * res
        widths = 2.0 * dt[skel] if skel.any() else np.array([0.0])
        paths = skeleton_paths(skel, min_len_px=max(6, int(round(6.0 / res))))
        lines = []
        grades = []
        for pth in paths:
            if len(pth) < 4:
                continue
            ij = np.array(pth, float) + np.array([sl[0].start, sl[1].start])
            x, z = tm.centre(ij[:, 0], ij[:, 1])
            poly = douglas_peucker(np.c_[x, z], max(0.75, res))
            y = tm.sample(tm.dtm, poly[:, 0], poly[:, 1])
            lines.append(np.c_[poly[:, 0], y, poly[:, 1]].round(3).tolist())
            # grade along the centreline: DTM resampled every 1 m on the dense skeleton path and
            # differenced over a 10 m baseline (vertex-to-vertex differences over 2 m turned DTM noise
            # at the road edge into 20 % "grades" on flat land)
            arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
            if arc[-1] < 5.0:
                continue
            base = min(10.0, float(arc[-1]))
            su = np.arange(0.0, arc[-1] + 1e-6, 1.0)
            yu = tm.sample(tm.dtm, np.interp(su, arc, x), np.interp(su, arc, z))
            k = int(round(base))
            if len(yu) > k:
                grades.extend((100.0 * np.abs(yu[k:] - yu[:-k]) / base).tolist())
        length = sum(float(np.sum(np.linalg.norm(np.diff(np.asarray(l)[:, [0, 2]], axis=0), axis=1))) for l in lines)
        paved_share = float(np.mean(s[m] < 0.16))
        surface = "PAVED" if paved_share >= 0.6 else ("UNPAVED" if paved_share <= 0.35 else "MIXED")
        cx, cz = tm.centre(*[np.mean(a) for a in np.nonzero(m)])
        out_segments.append(dict(
            id=f"RD-{len(out_segments) + 1:02d}", area_m2=round(area, 1), length_m=round(length, 1),
            width_median_m=round(float(np.median(widths)), 2),
            width_min_m=round(float(np.percentile(widths, 5)), 2),
            width_max_m=round(float(np.percentile(widths, 95)), 2),
            max_grade_pct=round(float(np.percentile(grades, 95)), 1) if grades else 0.0,
            surface=surface, paved_share=round(paved_share, 2),
            centre=[round(float(cx), 2), round(float(tm.sample(tm.dtm, cx, cz)), 2), round(float(cz), 2)],
            centerlines=lines))
    out_segments.sort(key=lambda d: -d["length_m"])
    for i, sgm in enumerate(out_segments):
        sgm["id"] = f"RD-{i + 1:02d}"
    total_len = float(sum(sg["length_m"] for sg in out_segments))
    summary = dict(road_cells=road_cells, road_area_m2=round(road_cells * res * res, 1),
                   segments=len(out_segments), total_length_m=round(total_len, 1),
                   repaired_gap_area_m2=round(repaired * res * res, 1),
                   vehicles_on_road=int(ndi.label(veh)[1]), rejected_candidates=rejected)
    log(f"[Roads] {len(out_segments)} road segment(s), {total_len:.0f} m, "
        f"{summary['repaired_gap_area_m2']} m^2 of torn surface completed")
    return dict(mask=mask, segments=out_segments, summary=summary, likelihood=L)


# =============================================================================
# potholes
# =============================================================================
def astm_severity(depth_m, diameter_m):
    """ASTM D6433 pothole severity (LOW / MEDIUM / HIGH) from depth and mean diameter."""
    d_mm, dia_mm = depth_m * 1000.0, diameter_m * 1000.0
    if dia_mm > 750.0:                         # large: treated as equivalent area of several holes
        return "MEDIUM" if d_mm <= 25.0 else "HIGH"
    col = 0 if dia_mm <= 200.0 else (1 if dia_mm <= 450.0 else 2)
    row = 0 if d_mm <= 25.0 else (1 if d_mm <= 50.0 else 2)
    table = [["LOW", "LOW", "MEDIUM"], ["LOW", "MEDIUM", "HIGH"], ["MEDIUM", "MEDIUM", "HIGH"]]
    return table[row][col]


_ADVICE = {
    "HIGH": ("Wheeled convoys: slow to 10-15 km/h and steer around; risk of tyre, rim and suspension damage "
             "for heavy trucks. Mark for immediate repair."),
    "MEDIUM": "Reduce speed to 25-30 km/h; light vehicles should steer around. Schedule repair.",
    "LOW": "Passable at normal convoy speed; monitor for growth after rain.",
}
_REPAIR = {
    "HIGH": "Cut to sound pavement, clean, tack coat, fill with hot/cold mix in layers, compact.",
    "MEDIUM": "Clean and fill with cold-mix (or compacted granular fill on unpaved roads).",
    "LOW": "Surface seal or granular top-up during routine maintenance.",
}


def robust_reference(z, valid, sigma_cells, iters=4):
    w = valid.astype(np.float64)
    ref = None
    sig = np.nan
    for _ in range(iters):
        ref, den = T.masked_smooth(z, w, sigma_cells)
        r = z - ref
        rv = r[valid & np.isfinite(r)]
        if len(rv) < 20:
            break
        med = np.median(rv)
        sig = 1.4826 * np.median(np.abs(rv - med))
        w = (valid & np.isfinite(r) & (r > med - 2.5 * max(sig, 1e-3))).astype(np.float64)
    return ref, float(sig)


def detect_potholes(tm, xyz, road_mask, telem=None, fine_res=None, log=print):
    """Pothole measurement on the road surface points. Returns (potholes list, surface stats)."""
    res = tm.res
    i, j = tm.cell_of(xyz[:, 0], xyz[:, 2])
    ok = tm.inside(i, j)
    idx = np.nonzero(ok)[0]
    on_road = np.zeros(len(xyz), bool)
    road_core = ndi.binary_erosion(road_mask, iterations=1) & ~(tm.cls == T.VEHICLE)
    on_road[idx] = road_core[i[idx], j[idx]]
    pts = xyz[on_road]
    # keep only points close to the ground (vehicles, people, overhanging branches excluded)
    hag = pts[:, 1] - tm.sample(tm.dtm, pts[:, 0], pts[:, 2])
    pts = pts[np.abs(hag) < 0.6]
    if len(pts) < 200:
        return [], dict(road_points=int(len(pts)), noise_sigma_m=None, lod_m=None,
                        note="Too few 3D points on the road surface to measure potholes.")
    if fine_res is None:
        sub = pts[np.random.default_rng(0).choice(len(pts), min(20000, len(pts)), replace=False)]
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(pts).query(sub, k=2)
        spacing = float(np.median(dd[:, 1]))
        fine_res = float(np.clip(1.3 * spacing, 0.08, 0.30))
    x0, z0 = pts[:, 0].min() - 1.0, pts[:, 2].min() - 1.0
    fx = int(math.ceil((pts[:, 0].max() + 1.0 - x0) / fine_res))
    fz = int(math.ceil((pts[:, 2].max() + 1.0 - z0) / fine_res))
    fi = np.floor((pts[:, 2] - z0) / fine_res).astype(np.int64)
    fj = np.floor((pts[:, 0] - x0) / fine_res).astype(np.int64)
    key = fi * fx + fj
    order = np.lexsort((pts[:, 1], key))
    uk, st, cnt = np.unique(key[order], return_index=True, return_counts=True)
    zf = np.full(fx * fz, np.nan)
    zf[uk] = pts[order, 1][st + (cnt - 1) // 2]            # per-cell median height
    zf = zf.reshape(fz, fx)
    has = np.isfinite(zf)
    # road membership on the fine grid (from the coarse mask)
    cz_, cx_ = np.mgrid[0:fz, 0:fx]
    wx, wz = x0 + (cx_ + 0.5) * fine_res, z0 + (cz_ + 0.5) * fine_res
    ci, cj = tm.cell_of(wx, wz)
    inside = tm.inside(ci, cj)
    road_f = np.zeros((fz, fx), bool)
    road_f[inside] = road_core[ci[inside], cj[inside]]
    valid = has & road_f
    ref, sig = robust_reference(zf, valid, sigma_cells=1.0 / fine_res)
    resid = zf - ref
    # local noise: the reconstruction is noisier far from the flight path / in clutter, so the limit of
    # detection is computed per 4 m block (robust MAD), never below the global value
    sig_loc = local_sigma(resid, valid, max(4, int(round(4.0 / fine_res))), sig)
    lod_map = np.maximum(3.0 * sig_loc, 0.02)
    lod = max(3.0 * sig, 0.02)
    cand = valid & (resid < -lod_map)
    # depressions may contain empty cells (dark water pools, steep walls): close tiny gaps
    cand = ndi.binary_closing(cand, np.ones((3, 3))) & road_f
    lab, n = ndi.label(cand, np.ones((3, 3)))
    potholes, rejected = [], {"edge": 0, "shape": 0, "support": 0}
    cell_a = fine_res * fine_res
    min_cells = max(5, int(math.ceil(0.10 / cell_a)))
    edge_dist = ndi.distance_transform_edt(road_f) * fine_res        # distance to the road edge
    objs = ndi.find_objects(lab)
    for k in range(1, n + 1):
        sl0 = objs[k - 1]
        nc0 = int((lab[sl0] == k).sum())
        area0 = nc0 * cell_a
        if nc0 < min_cells or area0 > 25.0:
            continue
        # the support ring must clear the shallow rim of the bowl: it scales with the pothole size
        r_eq = math.sqrt(area0 / math.pi)
        ring_in = max(1, int(round(max(0.3, 0.6 * r_eq) / fine_res)))
        ring_out = max(ring_in + 2, int(round(max(0.9, 1.4 * r_eq) / fine_res)))
        pad = ring_out + 2
        sl = (slice(max(0, sl0[0].start - pad), min(fz, sl0[0].stop + pad)),
              slice(max(0, sl0[1].start - pad), min(fx, sl0[1].stop + pad)))
        m = lab[sl] == k
        nc = int(m.sum())
        area = nc * cell_a
        # a pothole sits inside the pavement: not cut by the road edge (ditch, verge, kerb drop)
        if float(np.min(edge_dist[sl][m])) < 0.35:
            rejected["edge"] += 1
            continue
        # compact, not a long rut / ditch: area vs oriented extent
        ii, jj = np.nonzero(m)
        cov = np.cov(np.c_[ii, jj].T) if nc > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))
        aspect = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
        if aspect > 4.0:
            rejected["shape"] += 1
            continue
        # intact support ring: at least 65 % of the surrounding pavement measured and near the reference
        dd = ndi.distance_transform_edt(~m)
        ring = (dd >= ring_in) & (dd <= ring_out) & road_f[sl]
        ring_valid = valid[sl] & ring
        if ring.sum() == 0 or ring_valid.sum() < 0.65 * ring.sum() or \
                np.median(np.abs(resid[sl][ring_valid])) > 2.0 * max(float(np.median(sig_loc[sl][m])), 0.01):
            rejected["support"] += 1
            continue
        r = resid[sl][m & has[sl]]
        if len(r) < max(4, 0.5 * nc):
            continue
        depth_max = float(-np.percentile(r, 5))
        depth_mean = float(-np.mean(r))
        sig_here = float(np.median(sig_loc[sl][m]))
        if depth_max < max(3.0 * sig_here, 0.02):
            continue
        snr = depth_max / max(sig_here, 1e-3)
        gi, gj = ii + sl[0].start, jj + sl[1].start
        px, pz = x0 + (gj.mean() + 0.5) * fine_res, z0 + (gi.mean() + 0.5) * fine_res
        py = float(np.nanmedian(ref[sl][m]))
        dia = 2.0 * math.sqrt(area / math.pi)
        ext = 4.0 * np.sqrt(np.maximum(ev[::-1], 0)) * fine_res          # ~ full length/width (2 sigma each side)
        vol = float(np.sum(np.clip(-r, 0, None)) * cell_a * (nc / max(1, len(r))))
        kind = "CRATER" if (depth_max > 0.35 and area >= 1.0) else "POTHOLE"
        sev = "HIGH" if kind == "CRATER" else astm_severity(depth_max, dia)
        lat, lon = T.to_latlon(telem, px, py, pz) if telem else (None, None)
        potholes.append(dict(
            kind=kind, severity=sev, depth_cm=round(100 * depth_max, 1), avg_depth_cm=round(100 * depth_mean, 1),
            depth_uncertainty_cm=round(100 * math.sqrt(sig_here ** 2 / max(1.0, len(r) / 8.0) + (0.25 * sig_here) ** 2), 1),
            diameter_cm=round(100 * dia, 0), length_cm=round(100 * float(max(ext)), 0),
            width_cm=round(100 * float(min(ext)), 0), area_m2=round(area, 3), volume_liters=round(1000 * vol, 1),
            position=[round(float(px), 3), round(py, 3), round(float(pz), 3)], gps_lat=lat, gps_lon=lon,
            snr=round(snr, 1), confidence=round(float(np.clip((snr - 2.0) / 6.0, 0.05, 0.99)), 2),
            local_lod_cm=round(100 * max(3.0 * sig_here, 0.02), 1),
            convoy_impact=_ADVICE[sev], recommended_action=_REPAIR[sev], points_in_cluster=int(len(r))))
    potholes.sort(key=lambda p: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[p["severity"]], -p["depth_cm"]))
    for i_, p in enumerate(potholes):
        p["id"] = f"{'CR' if p['kind'] == 'CRATER' else 'PH'}-{i_ + 1:02d}"
    stats = dict(road_points=int(len(pts)), fine_res_m=round(fine_res, 3),
                 noise_sigma_m=round(sig, 4), lod_m=round(lod, 3),
                 lod_range_m=[round(float(np.nanpercentile(lod_map[valid], 10)), 3),
                              round(float(np.nanpercentile(lod_map[valid], 90)), 3)] if valid.any() else None,
                 surface_rms_m=round(float(np.sqrt(np.nanmean(resid[valid] ** 2))), 4) if valid.any() else None,
                 measured_area_m2=round(float(valid.sum()) * cell_a, 1), rejected=rejected)
    log(f"[Potholes] {len(potholes)} depression(s) above the local LoD (global {100 * lod:.1f} cm, "
        f"surface noise sigma {100 * sig:.1f} cm, grid {100 * fine_res:.0f} cm); rejected {rejected}")
    return potholes, stats


def local_sigma(resid, valid, block, floor):
    """Robust (MAD) noise per block x block tile, smoothed and upsampled; never below `floor`."""
    nz, nx = resid.shape
    pz, px = (-nz) % block, (-nx) % block
    r = np.pad(np.where(valid, np.abs(resid), np.nan), ((0, pz), (0, px)), constant_values=np.nan)
    tiles = r.reshape(r.shape[0] // block, block, r.shape[1] // block, block).transpose(0, 2, 1, 3)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mad = 1.4826 * np.nanmedian(tiles.reshape(tiles.shape[0], tiles.shape[1], -1), axis=2)
    mad = T.nan_fill_nearest(mad)
    mad = ndi.uniform_filter(mad, 3, mode="nearest")
    up = np.repeat(np.repeat(mad, block, 0), block, 1)[:nz, :nx]
    return np.maximum(up, floor)


def condition_score(potholes, road_area_m2, noise_sigma):
    """Simplified PCI-style surface condition (0-100) from pothole density/severity and roughness."""
    if road_area_m2 <= 0:
        return None
    w = {"LOW": 4.0, "MEDIUM": 10.0, "HIGH": 22.0}
    deduct = sum(w[p["severity"]] for p in potholes) * (100.0 / max(road_area_m2, 100.0))
    deduct = 60.0 * (1.0 - math.exp(-deduct / 40.0))          # saturating, like PCI deduct curves
    if noise_sigma is not None:
        deduct += float(np.clip((noise_sigma - 0.02) / 0.06, 0, 1)) * 15.0
    return int(round(max(0.0, 100.0 - deduct)))
