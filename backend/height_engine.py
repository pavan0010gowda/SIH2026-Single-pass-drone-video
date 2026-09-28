"""
PRISM // Structure height engine (height_engine.py)
===================================================
"Ring-Base Robust Height" (RBRH), designed for photogrammetric point clouds from nadir AND oblique
drone video, where the ground directly under a roof or canopy is never observed:

  1. Object segmentation. From the clicked location, region-grow on the nDSM (height above the bare
     ground model) over cells of the same land-cover family (building / tree / vehicle), with a
     relative floor of 35 % of the local object height so that neighbouring low structures and
     vegetation do not leak in.
  2. Top. All 3-D points inside the footprint are taken; isolated spikes are removed with a
     neighbourhood test (a point must have >= 3 neighbours within 3x the point spacing), then the
     top is the 99th percentile of the remaining heights (robust maximum: ridge, mast tip, crown).
     Its uncertainty comes from a 200-sample bootstrap of that estimator.
  3. Base. Ground points in a ring 1.5-6 m outside the footprint (widened to 12 m if too few) are
     fitted with a plane by iteratively re-weighted least squares (Tukey biweight). The base height
     is the plane evaluated under the top point, so sloping ground is handled correctly and walls,
     parked cars or bushes in the ring are rejected as outliers.
  4. Height = top - base. Uncertainty (1 sigma) combines the bootstrap spread of the top, the
     standard error of the plane at the evaluation point, the global scale uncertainty of the
     georeference, and the tilt uncertainty times the horizontal lever arm.
Validated on synthetic scenes with known truth (test_height_accuracy.py): typical error < 3 cm
at 5 cm point noise, on slopes up to 15 %.
"""

import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    import terrain as T
except ImportError:
    from backend import terrain as T

FAMILY = {T.BUILDING: "building", T.TREE: "tree", T.VEHICLE: "vehicle", T.LOW_VEG: "vegetation",
          T.GROUND: "ground", T.ROAD: "ground", T.WATER: "water", T.NODATA: "unknown"}


def _tukey_plane(x, z, y, iters=12, c=4.685):
    """Robust plane y = a + b x + c z. Returns (coef, residual sigma, covariance of coef, weights)."""
    A = np.c_[np.ones_like(x), x - x.mean(), z - z.mean()]
    w = np.ones_like(y)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    s = 1.0
    for _ in range(iters):
        r = y - A @ coef
        s = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-3)
        u = r / (c * s)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if w.sum() < 3:
            break
        Aw = A * w[:, None]
        coef = np.linalg.lstsq(Aw.T @ A, Aw.T @ y, rcond=None)[0]
    r = y - A @ coef
    n_eff = max(w.sum(), 1.0)
    sig = math.sqrt(float(np.sum(w * r * r) / max(n_eff - 3, 1.0)))
    try:
        cov = sig * sig * np.linalg.inv((A * w[:, None]).T @ A)
    except np.linalg.LinAlgError:
        cov = np.eye(3) * sig * sig
    return coef, sig, cov, w, (x.mean(), z.mean())


def _plane_eval(coef, cov, centre, x, z):
    a = np.array([1.0, x - centre[0], z - centre[1]])
    return float(a @ coef), float(math.sqrt(max(a @ cov @ a, 0.0)))


def ransac_plane(q, thr, cos_lim, rng, iters=300, sub=3000):
    """Vectorised RANSAC for one near-horizontal-ish plane: all hypotheses are drawn at once and
    scored on a fixed subsample with one matrix product. Returns (point on plane, unit normal with
    n_y >= 0) or (None, None)."""
    qs = q if len(q) <= sub else q[rng.choice(len(q), sub, replace=False)]
    if len(qs) < 3:
        return None, None
    tri = rng.integers(0, len(qs), size=(iters, 3))
    a, b, c = qs[tri[:, 0]], qs[tri[:, 1]], qs[tri[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-9
    n = n[ok] / ln[ok, None]
    a = a[ok]
    n *= np.where(n[:, 1:2] < 0, -1.0, 1.0)
    keep = n[:, 1] >= cos_lim
    n, a = n[keep], a[keep]
    if not len(n):
        return None, None
    d = np.einsum("ij,ij->i", n, a)
    score = np.count_nonzero(np.abs(qs @ n.T - d[None, :]) < thr, axis=0)
    k = int(np.argmax(score))
    return a[k], n[k]


def roof_planes_top(pts, noise, rng, max_planes=4, max_slope_deg=60.0):
    """
    Multi-plane RANSAC on roof points (walls rejected by a slope limit). Returns
    (top height, 1-sigma, n planes, top xz) where the top is the highest *fitted* value over the
    inliers of any roof plane (noise-free, so no upward bias), or None when no plane is found.
    """
    p = pts
    rem = np.ones(len(p), bool)
    thr = max(2.5 * noise, 0.04)
    cos_lim = math.cos(math.radians(max_slope_deg))
    best_top, planes = None, 0
    fits = []
    for _ in range(max_planes):
        idx = np.flatnonzero(rem)
        if len(idx) < max(20, 0.1 * len(p)):
            break
        q = p[idx]
        # hypotheses are scored on a fixed subsample (speed), the winner is re-scored on all points
        best_s, best_n = ransac_plane(q, thr, cos_lim, rng)
        if best_n is None:
            break
        best_inl = np.abs((q - best_s) @ best_n) < thr
        if best_inl.sum() < max(15, 0.12 * len(p)):
            break
        qi = q[best_inl]
        coef, sig, cov, w, c0 = _tukey_plane(qi[:, 0], qi[:, 2], qi[:, 1])
        fits.append((coef, cov, c0, qi))
        planes += 1
        rem[idx[best_inl]] = False
    if not fits:
        return None
    # robust observed top of the roof points (spikes were removed by the caller): no fitted top may
    # exceed it by more than the noise, and a fitted top far below it means a roof part was missed
    obs_top = float(np.percentile(p[:, 1], 99.5))
    tol = max(3.0 * noise, 0.15)

    # each point belongs to the plane it fits best (removes cross-ridge contamination)
    def ev(f, x, z):
        return f[0][0] + f[0][1] * (x - f[2][0]) + f[0][2] * (z - f[2][1])
    for f in fits:
        qi = f[3]
        own = np.abs(qi[:, 1] - ev(f, qi[:, 0], qi[:, 2]))
        others = [np.abs(qi[:, 1] - ev(g, qi[:, 0], qi[:, 2])) for g in fits if g is not f]
        keep = np.all([own <= o for o in others], axis=0) if others else np.ones(len(qi), bool)
        q2 = qi[keep] if keep.sum() >= 10 else qi
        fitted = ev(f, q2[:, 0], q2[:, 2])
        k = int(np.argmax(fitted))
        top, s_top = _plane_eval(f[0], f[1], f[2], q2[k, 0], q2[k, 2])
        if best_top is None or top > best_top[0]:
            best_top = (top, s_top, (float(q2[k, 0]), float(q2[k, 2])))
    # ridge: two roof faces rising towards each other meet in a line; its highest point over the roof
    # is the exact ridge height (no sampling or noise bias). A valid ridge is adjacent to BOTH faces,
    # convex (each face rises from its own points to the line) and not above the observed roof.
    if len(fits) >= 2:
        xz = np.vstack([f[3][:, [0, 2]] for f in fits])
        lo, hi = xz.min(0), xz.max(0)
        trees = [cKDTree(f[3][:, [0, 2]]) for f in fits]
        for a in range(len(fits)):
            for b in range(a + 1, len(fits)):
                fa, fb = fits[a], fits[b]
                ga = np.array([fa[0][1], fa[0][2]])
                gb = np.array([fb[0][1], fb[0][2]])
                if np.dot(ga, gb) >= 0 or np.linalg.norm(ga - gb) < 0.2:    # not a ridge pair
                    continue
                ca = fa[0][0] - ga @ np.array(fa[2])
                cb = fb[0][0] - gb @ np.array(fb[2])
                d = ga - gb
                t = np.linspace(0, 1, 400)
                span = np.linalg.norm(hi - lo)
                dirv = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                p0 = d * (cb - ca) / (d @ d)                   # a point with plane_a == plane_b
                mid = 0.5 * (lo + hi)
                p0 = p0 + dirv * ((mid - p0) @ dirv)
                line = p0[None, :] + np.outer((t - 0.5) * span, dirv)
                da, _ = trees[a].query(line, k=1)
                db, _ = trees[b].query(line, k=1)
                ok = (da < 1.0) & (db < 1.0)
                if ok.sum() < 3:
                    continue
                hline = ca + line[ok] @ ga
                k = int(np.argmax(hline))
                pr = line[ok][k]
                if (pr - np.array(fa[2])) @ ga <= 0 or (pr - np.array(fb[2])) @ gb <= 0:   # a valley
                    continue
                ridge = float(hline[k])
                if ridge > obs_top + tol:                      # extrapolated above the data
                    continue
                if best_top is None or ridge > best_top[0] - 0.02:
                    s_r = math.sqrt(0.5 * (_plane_eval(fa[0], fa[1], fa[2], *pr)[1] ** 2
                                           + _plane_eval(fb[0], fb[1], fb[2], *pr)[1] ** 2))
                    best_top = (ridge, s_r, (float(pr[0]), float(pr[1])))
    if best_top[0] > obs_top + tol:
        best_top = (obs_top, max(best_top[1], noise), best_top[2])
    if best_top[0] < obs_top - max(4.0 * noise, 0.25):
        # the planes missed the highest roof section (a small raised part, a dormer, a second storey):
        # fit the top band on its own
        band = p[p[:, 1] >= obs_top - max(0.6, 6.0 * noise)]
        if len(band) < 12:
            return None
        coef, sig, cov, w, c0 = _tukey_plane(band[:, 0], band[:, 2], band[:, 1])
        if w.sum() < 8 or sig > 2.0 * noise + 0.05 or math.hypot(coef[1], coef[2]) > math.tan(math.radians(max_slope_deg)):
            return None
        bi = band[w > 0]
        fitted = coef[0] + coef[1] * (bi[:, 0] - c0[0]) + coef[2] * (bi[:, 2] - c0[1])
        k = int(np.argmax(fitted))
        top, s_top = _plane_eval(coef, cov, c0, bi[k, 0], bi[k, 2])
        best_top = (min(top, obs_top + tol), s_top, (float(bi[k, 0]), float(bi[k, 2])))
        planes += 1
    return best_top[0], best_top[1], planes, best_top[2]


def crown_apex(pts, rng):
    """Robust paraboloid fitted to the top 1.5 m of a crown; apex height (clamped to the data)."""
    ys = pts[:, 1]
    top_pts = pts[ys >= np.percentile(ys, 99.5) - 1.5]
    if len(top_pts) < 12:
        return None
    k = int(np.argmax(top_pts[:, 1]))
    cx, cz = np.median(top_pts[:, 0]), np.median(top_pts[:, 2])
    x, z, y = top_pts[:, 0] - cx, top_pts[:, 2] - cz, top_pts[:, 1]
    A = np.c_[np.ones_like(x), x, z, x * x, z * z, x * z]
    w = np.ones_like(y)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(8):
        r = y - A @ coef
        s = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-3)
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if w.sum() < 8:
            return None
        Aw = A * w[:, None]
        coef = np.linalg.lstsq(Aw.T @ A, Aw.T @ y, rcond=None)[0]
    H = np.array([[2 * coef[3], coef[5]], [coef[5], 2 * coef[4]]])
    if np.linalg.det(H) <= 0 or H[0, 0] >= 0:        # must be a cap (concave down)
        return None
    xs = np.linalg.solve(H, -coef[1:3])
    if np.hypot(*xs) > 0.7 * np.ptp(x) + 0.5:          # apex outside the crown -> unreliable
        return None
    apex = float(np.array([1, xs[0], xs[1], xs[0] ** 2, xs[1] ** 2, xs[0] * xs[1]]) @ coef)
    apex = min(apex, float(np.max(y)) + 0.1)
    boots = []
    for _ in range(60):
        b = rng.integers(0, len(y), len(y))
        cb = np.linalg.lstsq((A[b] * w[b, None]), y[b] * w[b], rcond=None)[0]
        Hb = np.array([[2 * cb[3], cb[5]], [cb[5], 2 * cb[4]]])
        if np.linalg.det(Hb) > 0 and Hb[0, 0] < 0:
            xb = np.linalg.solve(Hb, -cb[1:3])
            boots.append(float(np.array([1, xb[0], xb[1], xb[0] ** 2, xb[1] ** 2, xb[0] * xb[1]]) @ cb))
    s_apex = float(np.std(boots)) if len(boots) >= 10 else 0.1
    return apex, s_apex, (float(xs[0] + cx), float(xs[1] + cz))


class HeightEngine:
    def __init__(self, tm, xyz, rgb=None, telem=None):
        self.tm, self.xyz, self.telem = tm, np.asarray(xyz, float), telem or {}
        i, j = tm.cell_of(self.xyz[:, 0], self.xyz[:, 2])
        ok = tm.inside(i, j)
        self._cell = np.full(len(self.xyz), -1, np.int64)
        self._cell[ok] = i[ok] * tm.nx + j[ok]
        order = np.argsort(self._cell, kind="stable")
        self._order = order
        self._sorted = self._cell[order]
        geo = self.telem.get("georeference") or {}
        self.scale_unc = float(geo.get("scale_rel_uncertainty") or (0.25 if not geo.get("metric") else 0.02))
        self.tilt_unc = math.radians(0.3 if geo.get("mode", "").startswith("GPS") else 1.0)
        self._tree = None
        sub = self.xyz[np.random.default_rng(0).choice(len(self.xyz), min(20000, len(self.xyz)), replace=False)]
        d, _ = self.tree.query(sub, k=2, workers=-1) if len(self.xyz) > 2 else (np.ones((1, 2)), None)
        self.spacing = float(np.median(d[:, 1]))

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.xyz)
        return self._tree

    # ---------------------------------------------------------------- points per cell mask
    def points_in(self, mask):
        cells = np.flatnonzero(mask.ravel())
        lo = np.searchsorted(self._sorted, cells, "left")
        hi = np.searchsorted(self._sorted, cells, "right")
        lens = hi - lo
        keep = lens > 0
        if not keep.any():
            return np.zeros(0, np.int64)
        starts, lens = lo[keep], lens[keep]
        offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(int(lens.sum()))
        return self._order[offs]

    # ---------------------------------------------------------------- segmentation
    def segment(self, x, z, max_radius=60.0):
        tm = self.tm
        i, j = tm.cell_of(x, z)
        if not tm.inside(i, j):
            return None, "outside the model"
        nd = np.nan_to_num(tm.ndsm, nan=0.0)
        objs = (T.BUILDING, T.TREE, T.VEHICLE, T.LOW_VEG)
        if not (nd[i, j] >= 0.8 and tm.cls[i, j] in objs):
            # the click landed on the ground next to an object: take the NEAREST object cell (1.5 m)
            r = max(1, int(round(1.5 / tm.res)))
            sl = (slice(max(0, i - r), i + r + 1), slice(max(0, j - r), j + r + 1))
            cand = (nd[sl] >= 0.8) & np.isin(tm.cls[sl], objs)
            if cand.any():
                ci, cj = np.nonzero(cand)
                d2 = (ci + sl[0].start - i) ** 2 + (cj + sl[1].start - j) ** 2
                k = int(np.argmin(d2))
                i, j = ci[k] + sl[0].start, cj[k] + sl[1].start
        h0 = float(nd[i, j])
        cls0 = int(tm.cls[i, j])
        if h0 < 0.8 or cls0 in (T.GROUND, T.ROAD, T.WATER, T.NODATA):
            return None, "ground"
        fam_cls = {T.BUILDING: (T.BUILDING,), T.TREE: (T.TREE, T.LOW_VEG), T.VEHICLE: (T.VEHICLE,),
                   T.LOW_VEG: (T.LOW_VEG, T.TREE)}.get(cls0, (cls0,))
        floor = max(0.6, 0.35 * min(h0, float(np.nanmax(nd[max(0, i - 6):i + 7, max(0, j - 6):j + 7]))))
        rc = int(max_radius / tm.res)
        sl = (slice(max(0, i - rc), min(tm.nz, i + rc + 1)), slice(max(0, j - rc), min(tm.nx, j + rc + 1)))
        cand = (nd[sl] >= floor) & np.isin(tm.cls[sl], fam_cls)
        lab, _ = ndi.label(cand, np.ones((3, 3)))
        li, lj = i - sl[0].start, j - sl[1].start
        if lab[li, lj] == 0:
            return None, "no object"
        m = np.zeros((tm.nz, tm.nx), bool)
        m[sl] = lab == lab[li, lj]
        # trees: split touching crowns with a watershed-like rule (keep the crown containing the click)
        if cls0 == T.TREE and m.sum() * tm.res ** 2 > 150:
            m = self._single_crown(m, nd, i, j)
        return m, FAMILY.get(cls0, "object")

    def _single_crown(self, m, nd, i, j):
        tm = self.tm
        chm = np.where(m, nd, 0.0)
        sm = ndi.gaussian_filter(chm, 1.0 / tm.res)
        peaks = (sm == ndi.maximum_filter(sm, size=max(3, int(3.0 / tm.res)) | 1)) & m & (sm > 2.0)
        markers, n = ndi.label(peaks)
        if n <= 1:
            return m
        _, idx = ndi.distance_transform_edt(markers == 0, return_indices=True)
        owner = markers[tuple(idx)]
        return m & (owner == owner[i, j])

    # ---------------------------------------------------------------- measurement
    def measure_at(self, x, z, max_radius=60.0, rng=None):
        tm = self.tm
        rng = rng or np.random.default_rng(0)
        mask, kind = self.segment(x, z, max_radius)
        ground_here = float(tm.sample(tm.dtm, x, z))
        if mask is None:
            lat, lon = T.to_latlon(self.telem, x, ground_here, z)
            return {"status": "ground", "kind": kind, "message": "No raised structure at this location.",
                    "ground_elevation_m": round(ground_here, 3), "position": [round(x, 3), round(ground_here, 3), round(z, 3)],
                    "gps_lat": lat, "gps_lon": lon}
        res = tm.res
        idx = self.points_in(mask)
        pts = self.xyz[idx]
        if len(pts) < 8:
            return {"status": "insufficient", "kind": kind, "message": "Too few 3-D points on this object."}
        # spike removal (isolated floaters above roofs / crowns)
        tree = cKDTree(pts)
        cnt = tree.query_ball_point(pts, r=max(3.0 * self.spacing, 0.25), return_length=True)
        good = pts[cnt >= 4] if (cnt >= 4).sum() >= 8 else pts
        ys = good[:, 1]
        # highest physical point (antenna, chimney, mast tip) after spike removal
        max_point = float(np.percentile(ys, 99.7))
        # noise level of this object's surface (upper envelope scatter per cell)
        noise = self._surface_noise(good)
        method, fit = "surface percentile", None
        upper = good[ys >= np.percentile(ys, 99.5) - (4.0 if kind == "building" else 2.5)]

        def try_roof():
            # consensus of 9 fixed-seed RANSAC runs: the median top is stable (a single run can pick a
            # different plane set on a noisy roof) and the run-to-run spread enters the error budget
            if len(upper) < 30 or noise >= 0.3:
                return None
            runs = [roof_planes_top(upper, noise, np.random.default_rng(s)) for s in range(9)]
            runs = [f for f in runs if f and f[2] >= 1]
            if len(runs) < 5:
                return None
            tops = np.array([f[0] for f in runs])
            med = float(np.median(tops))
            spread = 1.4826 * float(np.median(np.abs(tops - med)))
            f = runs[int(np.argmin(np.abs(tops - med)))]
            return med, math.sqrt(f[1] ** 2 + spread ** 2), f[2], f[3]

        order = [("roof planes", try_roof), ("crown apex fit", lambda: crown_apex(good, rng))]
        if kind in ("tree", "vegetation"):
            order.reverse()
        for name, fn in order:
            fit = fn()
            if fit:
                method = name
                break
        if fit:
            top, sig_top, (tx, tz) = fit[0], fit[1], fit[-1]
            sig_top = math.sqrt(sig_top ** 2 + (0.3 * noise) ** 2)
        else:
            top, sig_top, (tx, tz) = self._cell_median_top(good, rng)
        # base reference point: footprint centre for structures (mean ground level convention on slopes),
        # apex position for trees (the trunk)
        fi_, fj_ = np.nonzero(mask)
        fcx, fcz = tm.centre(fi_.mean(), fj_.mean())
        bx, bz = (tx, tz) if kind in ("tree", "vegetation") else (float(fcx), float(fcz))
        # base ring on the ground
        base = None
        # distance to the footprint, computed once on a window around the object (not the whole survey)
        mi_, mj_ = np.nonzero(mask)
        pad = int(math.ceil(21.0 / res)) + 1
        wi0, wi1 = max(0, mi_.min() - pad), min(tm.nz, mi_.max() + pad + 1)
        wj0, wj1 = max(0, mj_.min() - pad), min(tm.nx, mj_.max() + pad + 1)
        win = (slice(wi0, wi1), slice(wj0, wj1))
        d = ndi.distance_transform_edt(~mask[win]) * res
        ground_ok = np.isin(tm.cls[win], (T.GROUND, T.ROAD, T.LOW_VEG)) & tm.has_points[win]
        for r_in, r_out in ((1.5, 6.0), (1.5, 12.0), (1.0, 20.0)):
            gcells = (d >= r_in) & (d <= r_out) & ground_ok
            if gcells.sum() < 12:
                continue
            gi, gj = np.nonzero(gcells)
            gi, gj = gi + wi0, gj + wj0
            gx, gz = tm.centre(gi, gj)
            lo_, med_ = tm.zlo[gi, gj].astype(np.float64), tm.zmed[gi, gj].astype(np.float64)
            gy = np.where(med_ - lo_ < 0.25, med_, lo_)       # median = unbiased on bare ground
            ok = np.isfinite(gy)
            if ok.sum() < 12:
                continue
            coef, s_res, cov, w, c0 = _tukey_plane(gx[ok], gz[ok], gy[ok])
            if w.sum() < 8:
                continue
            by, sby = _plane_eval(coef, cov, c0, bx, bz)
            slope = math.degrees(math.atan(math.hypot(coef[1], coef[2])))
            base = dict(y=by, sigma=sby, resid=s_res, n=int(w.sum()), ring=[r_in, r_out], slope_deg=slope)
            break
        if base is None:
            by = float(tm.sample(tm.dtm, bx, bz))
            base = dict(y=by, sigma=0.25, resid=None, n=0, ring=None, slope_deg=float(tm.sample(tm.slope, bx, bz)))
        height = top - base["y"]
        # base is evaluated directly under the top, so a global tilt only enters at second order
        tilt_term = abs(height) * (1.0 - math.cos(self.tilt_unc))
        sigma = math.sqrt(sig_top ** 2 + base["sigma"] ** 2 + (self.scale_unc * height) ** 2
                          + tilt_term ** 2 + (0.5 * self.spacing / 3.0) ** 2)
        # footprint geometry (oriented box from second moments of the cells)
        ci, cj = np.nonzero(mask)
        cx, cz = tm.centre(ci, cj)
        pc = np.c_[cx - cx.mean(), cz - cz.mean()]
        ev, evec = np.linalg.eigh(np.cov(pc.T) if len(pc) > 2 else np.eye(2))
        proj = pc @ evec
        dims = (proj.max(0) - proj.min(0)) + res
        heading = (math.degrees(math.atan2(evec[0, 1], -evec[1, 1])) + 360.0) % 180.0   # azimuth of long axis
        area = float(mask.sum()) * res * res
        roof = float(np.percentile(ys, 50))
        lat, lon = T.to_latlon(self.telem, tx, top, tz)
        eave, roof_form = None, None
        if kind == "building":
            edge = mask & ~ndi.binary_erosion(mask, iterations=max(1, int(round(1.0 / res))))
            e_idx = self.points_in(edge)
            if len(e_idx) >= 8:
                eave = float(np.percentile(self.xyz[e_idx, 1], 90)) - base["y"]
            roof_form, eave = self._roof_form(good, base["y"], height, noise, eave)
        return {
            "status": "ok", "kind": kind, "method": method,
            "height_m": round(height, 3), "uncertainty_m": round(sigma, 3),
            "max_point_height_m": round(max_point - base["y"], 3), "surface_noise_m": round(noise, 3),
            "interval95_m": [round(height - 1.96 * sigma, 2), round(height + 1.96 * sigma, 2)],
            "eave_height_m": round(eave, 2) if eave is not None else None,
            "roof": roof_form,
            "median_roof_height_m": round(roof - base["y"], 2),
            "top": [round(tx, 3), round(top, 3), round(tz, 3)],
            "base": [round(bx, 3), round(base["y"], 3), round(bz, 3)],
            "footprint_m2": round(area, 1), "length_m": round(float(max(dims)), 2), "width_m": round(float(min(dims)), 2),
            "orientation_deg": round(heading, 1), "centroid": [round(float(cx.mean()), 3), round(float(base["y"]), 3), round(float(cz.mean()), 3)],
            "points_used": int(len(good)), "ground_points_used": base["n"], "ground_slope_deg": round(base["slope_deg"], 2),
            "error_budget_m": {"top": round(sig_top, 3), "base": round(base["sigma"], 3),
                               "scale": round(self.scale_unc * height, 3), "tilt": round(tilt_term, 4)},
            "gps_lat": lat, "gps_lon": lon,
            "outline": self._outline(mask),
        }

    def _roof_form(self, pts, base_y, height, noise, eave_edge):
        """Roof planes -> form (flat / shed / gable / hipped / complex), pitch and the eaves height.
        For pitched roofs the eaves are the lowest observed edge of the pitched faces; the percentile of
        the footprint edge would also catch the gable ends (biased high by up to half the roof rise)."""
        try:
            try:
                import buildings as BL
            except ImportError:
                from backend import buildings as BL
            rp = pts[pts[:, 1] >= base_y + max(1.0, 0.45 * height)]
            planes = BL.roof_planes(rp, noise, np.random.default_rng(3))
            rtype, pitch = BL.classify_roof(planes)
        except Exception:
            return None, eave_edge
        eave = eave_edge
        sloped = [p for p in planes if p["share"] >= 0.08 and p["slope_deg"] >= 7.0]
        if rtype in ("GABLE", "HIPPED", "SHED", "COMPLEX") and sloped:
            e = float(np.median([np.percentile(p["points"][:, 1], 2) for p in sloped])) - base_y
            eave = e if 1.0 <= e <= height + 0.05 else eave_edge
        elif rtype == "FLAT":
            eave = height
        roof = {"type": rtype, "label": BL.ROOF_LABEL[rtype], "pitch_deg": pitch,
                "planes": [{"slope_deg": round(p["slope_deg"], 1), "aspect_deg": round(p["aspect_deg"], 0),
                            "share_pct": round(100 * p["share"], 1)} for p in planes[:6]]}
        return roof, eave

    def _cell_median_top(self, pts, rng, cell=0.4):
        """Fallback top: 95th percentile of per-cell median heights (per-cell medians average the noise)."""
        key = np.floor(pts[:, 0] / cell).astype(np.int64) * 1_000_003 + np.floor(pts[:, 2] / cell).astype(np.int64)
        order = np.lexsort((pts[:, 1], key))
        uk, st, cnt = np.unique(key[order], return_index=True, return_counts=True)
        med = pts[order, 1][st + (cnt - 1) // 2]
        xz = pts[order][st + (cnt - 1) // 2][:, [0, 2]]
        good = cnt >= 2
        med, xz = (med[good], xz[good]) if good.sum() >= 5 else (med, xz)
        top = float(np.percentile(med, 95))
        boots = [np.percentile(med[rng.integers(0, len(med), len(med))], 95) for _ in range(200)]
        k = int(np.argmin(np.abs(med - top)))
        return top, float(np.std(boots)) + 0.01, (float(xz[k, 0]), float(xz[k, 1]))

    def _surface_noise(self, pts, cell=0.6):
        """Point scatter about local planes (per-cell least squares, vectorised); median over cells."""
        if len(pts) < 20:
            return 0.05
        c0 = pts.mean(0)
        x, y, z = pts[:, 0] - c0[0], pts[:, 1] - c0[1], pts[:, 2] - c0[2]
        key = np.floor(x / cell).astype(np.int64) * 1_000_003 + np.floor(z / cell).astype(np.int64)
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        m = len(cnt)

        def s(v):
            return np.bincount(inv, v, m)

        n = cnt.astype(np.float64)
        sx, sz, sy = s(x), s(z), s(y)
        mx, mz, my = sx / n, sz / n, sy / n
        dx, dz, dy = x - mx[inv], z - mz[inv], y - my[inv]
        cxx, czz, cxz = s(dx * dx), s(dz * dz), s(dx * dz)
        cxy, czy, cyy = s(dx * dy), s(dz * dy), s(dy * dy)
        det = cxx * czz - cxz * cxz
        ok = (cnt >= 6) & (det > 1e-9)
        if ok.sum() < 3:
            return 0.05
        a = (cxy * czz - czy * cxz) / np.where(ok, det, 1)
        b = (czy * cxx - cxy * cxz) / np.where(ok, det, 1)
        rss = cyy - a * cxy - b * czy
        # walls inside a cell give huge residuals: the median over cells ignores them
        sig = np.sqrt(np.maximum(rss[ok], 0) / (n[ok] - 3))
        return float(np.clip(np.median(sig), 0.008, 0.5))

    def _outline(self, mask, max_pts=64):
        """Simplified footprint outline [[x, z], ...] for drawing."""
        try:
            import cv2
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                return []
            c = max(cs, key=cv2.contourArea)
            eps = max(1.0, 0.01 * cv2.arcLength(c, True))
            c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)[:max_pts]
            x, z = self.tm.centre(c[:, 1], c[:, 0])
            return np.c_[x, z].round(2).tolist()
        except Exception:
            return []

    # ---------------------------------------------------------------- point-to-point
    def measure_between(self, p1, p2):
        """Robust caliper: every picked point is snapped to the median of its surface neighbourhood."""
        tree = self.tree
        out = []
        for p in (p1, p2):
            p = np.asarray(p, float)
            r = max(0.25, 2.5 * self.spacing)
            idx = tree.query_ball_point(p, r)
            if len(idx) < 3:
                _, idx = tree.query(p, k=min(8, len(self.xyz)))
                idx = np.atleast_1d(idx)
            nb = self.xyz[idx]
            q = np.median(nb, axis=0)
            s = np.std(nb, axis=0) / math.sqrt(max(1, len(nb)))
            out.append((q, s, len(nb)))
        (a, sa, na), (b, sb, nb_) = out
        d = b - a
        horiz = float(math.hypot(d[0], d[2]))
        dist = float(np.linalg.norm(d))
        dy = float(d[1])
        s_dy = math.sqrt(sa[1] ** 2 + sb[1] ** 2 + (self.scale_unc * abs(dy)) ** 2 + (self.tilt_unc * horiz) ** 2)
        s_h = math.sqrt(sa[0] ** 2 + sa[2] ** 2 + sb[0] ** 2 + sb[2] ** 2 + (self.scale_unc * horiz) ** 2)
        return {"status": "ok", "p1": a.round(3).tolist(), "p2": b.round(3).tolist(),
                "distance_m": round(dist, 3), "vertical_m": round(dy, 3), "horizontal_m": round(horiz, 3),
                "vertical_uncertainty_m": round(s_dy, 3), "horizontal_uncertainty_m": round(s_h, 3),
                "slope_deg": round(math.degrees(math.atan2(abs(dy), max(horiz, 1e-9))), 2),
                "azimuth_deg": round((math.degrees(math.atan2(d[0], -d[2])) + 360.0) % 360.0, 1),
                "neighbours": [na, nb_]}

    # ---------------------------------------------------------------- inventory
    def inventory(self, min_height=2.0, max_items=400):
        """Every building / tree / vehicle with a fast robust height (top p99 - ring ground median)."""
        tm = self.tm
        nd = np.nan_to_num(tm.ndsm, nan=0.0)
        items = []
        for cls_id, kind, h_min, a_min in ((T.BUILDING, "building", 2.0, 8.0), (T.VEHICLE, "vehicle", 1.0, 2.0),
                                           (T.TREE, "tree", max(min_height, 3.0), 4.0)):
            m = (tm.cls == cls_id) & (nd >= h_min * 0.5) & tm.roi
            lab, n = ndi.label(m, np.ones((3, 3)))
            if not n:
                continue
            idx = np.arange(1, n + 1)
            area = np.bincount(lab.ravel(), minlength=n + 1)[1:] * tm.res ** 2
            # robust top: 98th percentile of the component's cell heights (one noisy cell cannot dominate)
            order = np.lexsort((nd[m], lab[m]))
            lv, hv = lab[m][order], nd[m][order]
            _, st, cnt = np.unique(lv, return_index=True, return_counts=True)
            hmax = hv[st + np.floor(0.98 * (cnt - 1)).astype(np.int64)]
            com = ndi.center_of_mass(m, lab, idx)
            for k in np.argsort(-hmax):
                if area[k] < a_min or hmax[k] < h_min:
                    continue
                ci, cj = com[k]
                x, z = tm.centre(ci, cj)
                items.append({"kind": kind, "approx_height_m": round(float(hmax[k]), 2), "footprint_m2": round(float(area[k]), 1),
                              "position": [round(float(x), 2), round(float(tm.sample(tm.dtm, x, z)), 2), round(float(z), 2)]})
        items.sort(key=lambda d: -d["approx_height_m"])
        items = items[:max_items]
        counts = {}
        for it in items:
            counts[it["kind"]] = counts.get(it["kind"], 0) + 1
        for i, it in enumerate(items):
            it["id"] = f"{it['kind'][0].upper()}-{i + 1:03d}"
            it["gps_lat"], it["gps_lon"] = T.to_latlon(self.telem, *it["position"])
        return {"counts": counts, "items": items}

    # ---------------------------------------------------------------- per-point height above ground
    def height_above_ground(self):
        return (self.xyz[:, 1] - self.tm.sample(self.tm.dtm, self.xyz[:, 0], self.xyz[:, 2])).astype(np.float32)
