"""
PRISM // survey-to-survey change detection (change_engine.py)

Day-1 vs day-2 (or any two surveys of the same place): finds what appeared, disappeared or was dug
between the flights - a tent, a camouflage net, a vehicle, a mast, a trench - with an honest noise floor.

  1. location check   both flight logs / surveyed areas must overlap (otherwise the comparison is refused
                      and the distance between them is reported)
  2. georeference     day-1 points are carried into day-2's metric frame through both GPS georeferences
                      (ECEF, exact)
  3. co-registration  the GNSS error between two days (metres) is removed on the data itself:
                        - horizontal: correlation of the high-passed surface models (relief edges) -
                          coarse shift, then per-tile shifts -> rigid 2-D fit (rotation + translation)
                        - vertical: robust plane fitted to height differences on unchanged ground
  4. differencing     0.5 m surface grids (upper envelope); a new height is compared with the highest
                      day-1 surface in the 3 x 3 neighbourhood, so a 0.5 m misregistration at a wall
                      cannot create a fake change
  5. noise floor      per cell: LoD95 = 1.96 * sqrt(roughness of the reference surface^2 + noise of the
                      other survey^2 + registration residual^2); nothing below it is reported
  6. objects          connected changes -> tent / shelter, new structure, mast, vehicle, low object,
                      excavation / trench, removed object; clusters of new objects -> possible camp.
                      Changes inside existing tree canopy (wind, growth) are not alerts; a green net on
                      previously open ground is.
"""
import math
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

try:
    import terrain as T
    from georeference import enu_to_enu_transform
except ImportError:                                    # pragma: no cover
    from backend import terrain as T
    from backend.georeference import enu_to_enu_transform

M2E = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])      # model (x E, y U, z S) -> ENU
E2M = M2E.T

SEVERITY_RANK = {"HIGH": 0, "ELEVATED": 1, "ADVISORY": 2}
_DEBUG = {}                     # last comparison's grids (diagnostics / tests)


# =============================================================================
# helpers
# =============================================================================
def _origin(telem):
    o = ((telem or {}).get("georeference") or {}).get("origin") or {}
    if o.get("lat") is None:
        return None
    return float(o["lat"]), float(o["lon"]), float(o.get("alt") or 0.0)


def model_to_model(origin_a, origin_b):
    """4 x 4: survey-A model coordinates -> survey-B model coordinates (through ECEF)."""
    te = enu_to_enu_transform(origin_a, origin_b)
    t = np.eye(4)
    t[:3, :3] = E2M @ te[:3, :3] @ M2E
    t[:3, 3] = E2M @ te[:3, 3]
    return t


def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - np.asarray(lon1)) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rasterize(xyz, x0, z0, res, nx, nz, q=0.9):
    """Per-cell q-quantile(s) of height and the point count. q may be a tuple: one sort for all."""
    qs = q if isinstance(q, (tuple, list)) else (q,)
    j = np.floor((xyz[:, 0] - x0) / res).astype(np.int64)
    i = np.floor((xyz[:, 2] - z0) / res).astype(np.int64)
    ok = (i >= 0) & (i < nz) & (j >= 0) & (j < nx)
    key = i[ok] * nx + j[ok]
    y = xyz[ok, 1]
    order = np.lexsort((y, key))
    ys = y[order]
    uk, st, cnt = np.unique(key[order], return_index=True, return_counts=True)
    outs = []
    for qq in qs:
        dsm = np.full(nz * nx, np.nan, np.float32)
        dsm[uk] = ys[st + np.floor(qq * (cnt - 1)).astype(np.int64)]
        outs.append(dsm.reshape(nz, nx))
    count = np.zeros(nz * nx, np.int32)
    count[uk] = cnt
    if isinstance(q, (tuple, list)):
        return outs, count.reshape(nz, nx)
    return outs[0], count.reshape(nz, nx)


def _highpass(dsm, sigma_cells):
    f = T.nan_fill_nearest(dsm.astype(np.float64))
    hp = f - ndi.gaussian_filter(f, sigma_cells)
    hp[~np.isfinite(dsm)] = 0.0
    return np.clip(hp, -6.0, 6.0)


def _xcorr_shift(hp_b, hp_a, max_shift_cells):
    """Integer + parabolic sub-cell shift k (rows, cols) with hp_b(m) ~ hp_a(m - k); peak sharpness."""
    h, w = hp_b.shape
    c = fftconvolve(hp_b, hp_a[::-1, ::-1], mode="full")
    ci, cj = h - 1, w - 1
    r = max_shift_cells
    win = c[ci - r:ci + r + 1, cj - r:cj + r + 1]
    pi, pj = np.unravel_index(int(np.argmax(win)), win.shape)

    def sub(v, k):
        if 0 < k < len(v) - 1:
            d = v[k - 1] - 2 * v[k] + v[k + 1]
            return 0.5 * (v[k - 1] - v[k + 1]) / d if d < 0 else 0.0
        return 0.0
    di = pi - r + sub(win[:, pj], pi)
    dj = pj - r + sub(win[pi, :], pj)
    peak = float(win[pi, pj])
    sharp = peak / (float(np.median(np.abs(win))) + 1e-9)
    return float(di), float(dj), sharp


def _rigid2d(src, dst):
    """Least-squares rotation + translation (2-D) taking src -> dst."""
    ms, md = src.mean(0), dst.mean(0)
    h = (src - ms).T @ (dst - md)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[1] *= -1
        r = vt.T @ u.T
    return r, md - r @ ms


def _tukey_plane(x, z, d, iters=8):
    a = np.c_[x, z, np.ones_like(x)]
    w = np.ones_like(d)
    coef = np.linalg.lstsq(a, d, rcond=None)[0]
    for _ in range(iters):
        r = d - a @ coef
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.01)
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if w.sum() < 20:
            break
        aw = a * w[:, None]
        coef = np.linalg.lstsq(aw.T @ a, aw.T @ d, rcond=None)[0]
    r = d - a @ coef
    good = w > 0
    sig = 1.4826 * float(np.median(np.abs(r[good] - np.median(r[good])))) if good.sum() > 10 else float(np.std(r))
    return coef, sig, int(good.sum())


# =============================================================================
# the comparison
# =============================================================================
class Registration:
    """A-model -> B-model: georeference transform, then 2-D rigid correction, then a vertical plane."""

    def __init__(self, t_geo):
        self.t_geo = t_geo
        self.r2 = np.eye(2)
        self.t2 = np.zeros(2)
        self.plane = np.zeros(3)

    def apply(self, xyz):
        p = xyz @ self.t_geo[:3, :3].T + self.t_geo[:3, 3]
        xz = p[:, [0, 2]] @ self.r2.T + self.t2
        p[:, 0], p[:, 2] = xz[:, 0], xz[:, 1]
        p[:, 1] += self.plane[0] * p[:, 0] + self.plane[1] * p[:, 2] + self.plane[2]
        return p

    def inverse_xz(self, x, z):
        xz = np.c_[np.ravel(x), np.ravel(z)]
        xz = (xz - self.t2) @ self.r2              # inverse rotation
        p = np.c_[xz[:, 0], np.zeros(len(xz)), xz[:, 1]]
        a = (p - self.t_geo[:3, 3]) @ self.t_geo[:3, :3]
        return a[:, 0].reshape(np.shape(x)), a[:, 2].reshape(np.shape(z))

    def describe(self):
        ang = math.degrees(math.atan2(self.r2[1, 0], self.r2[0, 0]))
        return {"horizontal_shift_m": [round(float(self.t2[0]), 2), round(float(self.t2[1]), 2)],
                "horizontal_shift_total_m": round(float(np.hypot(*self.t2)), 2),
                "rotation_deg": round(ang, 3), "vertical_offset_m": round(float(self.plane[2]), 2),
                "tilt_mm_per_100m": [round(1e5 * float(self.plane[0]), 1), round(1e5 * float(self.plane[1]), 1)]}


def location_check(telem_a, telem_b, tm_a, tm_b, t_geo, min_overlap_m2=400.0):
    out = {"georeferenced": bool(_origin(telem_a) and _origin(telem_b))}
    wa = [(w["latitude"], w["longitude"]) for w in (telem_a.get("waypoints") or []) if w.get("latitude") is not None]
    wb = [(w["latitude"], w["longitude"]) for w in (telem_b.get("waypoints") or []) if w.get("latitude") is not None]
    if wa and wb:
        a = np.asarray(wa)[:: max(1, len(wa) // 400)]
        b = np.asarray(wb)[:: max(1, len(wb) // 400)]
        d = _hav(a[:, None, 0], a[:, None, 1], b[None, :, 0], b[None, :, 1])
        out["track_min_distance_m"] = round(float(d.min()), 1)
        out["track_centre_distance_m"] = round(float(_hav(np.median(a[:, 0]), np.median(a[:, 1]),
                                                          np.median(b[:, 0]), np.median(b[:, 1]))), 1)
    if not out["georeferenced"]:
        out.update(ok=False, message="Both surveys must be GPS-georeferenced to be compared.")
        return out
    # the verified georeference origins are the reliable measure of how far apart the two surveys are
    oa, ob = _origin(telem_a), _origin(telem_b)
    out["track_centre_distance_m"] = round(float(_hav(oa[0], oa[1], ob[0], ob[1])), 1)
    # surveyed-area overlap: day-1 analysed cells carried into day 2
    ii, jj = np.nonzero(tm_a.roi)
    if len(ii) > 200000:
        s = np.random.default_rng(0).choice(len(ii), 200000, replace=False)
        ii, jj = ii[s], jj[s]
    scale = float(tm_a.roi.sum()) / max(1, len(ii))
    x, z = tm_a.centre(ii, jj)
    p = np.c_[x, np.zeros(len(x)), z] @ t_geo[:3, :3].T + t_geo[:3, 3]
    bi, bj = tm_b.cell_of(p[:, 0], p[:, 2])
    ok = tm_b.inside(bi, bj)
    inside = np.zeros(len(p), bool)
    inside[ok] = tm_b.roi[bi[ok], bj[ok]]
    area = float(inside.sum()) * scale * tm_a.res ** 2
    area_a = float(tm_a.roi.sum()) * tm_a.res ** 2
    area_b = float(tm_b.roi.sum()) * tm_b.res ** 2
    out["overlap_m2"] = round(area, 0)
    out["overlap_pct_of_smaller"] = round(100.0 * area / max(1.0, min(area_a, area_b)), 1)
    ok = area >= min_overlap_m2 and out["overlap_pct_of_smaller"] >= 5.0
    out["ok"] = bool(ok)
    if ok:
        out["message"] = (f"Same place: the surveyed areas overlap by {area / 1e4:.2f} ha "
                          f"({out['overlap_pct_of_smaller']:.0f} % of the smaller survey)"
                          + (f", flight tracks {out['track_min_distance_m']:.0f} m apart at the closest." if "track_min_distance_m" in out else "."))
    else:
        dist = out.get("track_centre_distance_m")
        out["message"] = ("Different places: the surveyed areas do not overlap"
                          + (f" (flights {dist / 1000:.2f} km apart)" if dist is not None and dist >= 1000 else
                             f" (flights {dist:.0f} m apart)" if dist is not None else "") + ". Nothing can be compared.")
    return out


def register(pa, pb, tm_b, reg, log=print):
    """Removes the GNSS error between the two days (horizontal from relief, vertical from ground)."""
    # ---- coarse horizontal shift, 1 m grid over day-2's analysed area
    ii, jj = np.nonzero(tm_b.roi)
    x0, z0 = tm_b.centre(ii.min(), jj.min())
    x1, z1 = tm_b.centre(ii.max(), jj.max())
    res1, pad = 1.0, 25.0
    gx0, gz0 = x0 - pad, z0 - pad
    nx1, nz1 = int((x1 - x0 + 2 * pad) / res1) + 1, int((z1 - z0 + 2 * pad) / res1) + 1
    a1 = reg.apply(pa)
    da, _ = rasterize(a1, gx0, gz0, res1, nx1, nz1)
    db, _ = rasterize(pb, gx0, gz0, res1, nx1, nz1)
    hpa, hpb = _highpass(da, 4.0), _highpass(db, 4.0)
    di, dj, sharp = _xcorr_shift(hpb, hpa, int(20 / res1))
    reg.t2 = np.array([dj * res1, di * res1])
    log(f"[Change] coarse shift {reg.t2[0]:+.2f} E / {-reg.t2[1]:+.2f} N m (peak sharpness {sharp:.1f})")
    # ---- per-tile refinement (0.5 m) -> rotation + translation
    res2, tile = 0.5, 100.0
    a2 = reg.apply(pa)
    nx2, nz2 = int(nx1 * res1 / res2), int(nz1 * res1 / res2)
    da2, _ = rasterize(a2, gx0, gz0, res2, nx2, nz2)
    db2, _ = rasterize(pb, gx0, gz0, res2, nx2, nz2)
    hpa2, hpb2 = _highpass(da2, 6.0), _highpass(db2, 6.0)
    ts = int(tile / res2)
    src, dst = [], []
    for r0 in range(0, nz2 - ts // 2, ts // 2):
        for c0 in range(0, nx2 - ts // 2, ts // 2):
            sb = hpb2[r0:r0 + ts, c0:c0 + ts]
            sa = hpa2[r0:r0 + ts, c0:c0 + ts]
            va = np.isfinite(da2[r0:r0 + ts, c0:c0 + ts])
            vb = np.isfinite(db2[r0:r0 + ts, c0:c0 + ts])
            if (va & vb).mean() < 0.3 or sb[vb].std() < 0.15 or sa[va].std() < 0.15:
                continue
            ti, tj, sh = _xcorr_shift(sb, sa, int(3.0 / res2))
            if sh < 3.0 or max(abs(ti), abs(tj)) >= int(3.0 / res2) - 0.5:
                continue
            cx = gx0 + (c0 + sb.shape[1] / 2) * res2
            cz = gz0 + (r0 + sb.shape[0] / 2) * res2
            src.append([cx, cz])
            dst.append([cx + tj * res2, cz + ti * res2])
    src, dst = np.asarray(src), np.asarray(dst)
    n_tiles = len(src)
    if n_tiles >= 4:
        keep = np.ones(n_tiles, bool)
        for _ in range(4):
            r2, t2 = _rigid2d(src[keep], dst[keep])
            resid = np.linalg.norm(src @ r2.T + t2 - dst, axis=1)
            new = resid < max(0.4, 2.5 * float(np.median(resid[keep])))
            if new.sum() < 4 or (new == keep).all():
                break
            keep = new
        # compose: corrected = r2 @ (r_old @ p + t_old) + t2
        reg.r2, reg.t2 = r2 @ reg.r2, r2 @ reg.t2 + t2
        tiles_used = int(keep.sum())
    elif n_tiles:
        reg.t2 = reg.t2 + np.median(dst - src, axis=0)
        tiles_used = n_tiles
    else:
        tiles_used = 0
    return {"coarse_peak_sharpness": round(sharp, 1), "tiles_used": tiles_used}


def vertical_fit(reg, da, ca, db, cb, tm_b, cls_a):
    """da / db: median surfaces on day-2's grid (day 1 already horizontally registered)."""
    stable = np.isin(tm_b.cls, (T.GROUND, T.ROAD)) & np.isin(cls_a, (T.GROUND, T.ROAD)) & (ca >= 2) & (cb >= 2)
    stable &= np.isfinite(da) & np.isfinite(db)
    d = (db - da)[stable].astype(np.float64)
    if len(d) < 200:
        return None
    si, sj = np.nonzero(stable)
    x, z = tm_b.centre(si, sj)
    med = float(np.median(d))
    near = np.abs(d - med) < 2.0
    coef, sig, n = _tukey_plane(x[near], z[near], d[near])
    reg.plane = reg.plane + coef
    return {"ground_cells": n, "registration_sigma_m": round(sig, 3)}


def compare(sa, sb, *, min_height_m=0.5, min_area_m2=2.0, log=print):
    """
    sa / sb: dicts with tm (TerrainModel), xyz (points), telem, name, id - earlier (day 1) / later (day 2).
    Everything is returned in day-2's model frame.
    """
    t0 = time.time()
    tm_a, tm_b = sa["tm"], sb["tm"]
    oa, ob = _origin(sa["telem"]), _origin(sb["telem"])
    base = {"earlier": {"id": sa["id"], "name": sa["name"]}, "later": {"id": sb["id"], "name": sb["name"]},
            "frame": sb["id"], "min_height_m": min_height_m, "min_area_m2": min_area_m2}
    if not (oa and ob):
        return dict(base, status="not_georeferenced", location={"ok": False, "message": "Both surveys must be GPS-georeferenced."},
                    alerts=[], alert_count=0)
    reg = Registration(model_to_model(oa, ob))
    loc = location_check(sa["telem"], sb["telem"], tm_a, tm_b, reg.t_geo)
    base["location"] = loc
    if not loc["ok"]:
        return dict(base, status="disjoint", alerts=[], alert_count=0, message=loc["message"])

    pa, pb = np.asarray(sa["xyz"], float), np.asarray(sb["xyz"], float)
    reg_info = register(pa, pb, tm_b, reg, log=log)
    grid = (tm_b.x0, tm_b.z0, tm_b.res, tm_b.nx, tm_b.nz)
    res = tm_b.res
    # day-1 terrain layers on the day-2 grid
    ii, jj = np.mgrid[0:tm_b.nz, 0:tm_b.nx]
    cx, cz = tm_b.centre(ii, jj)
    ax, az = reg.inverse_xz(cx, cz)
    ai, aj = tm_a.cell_of(ax, az)
    a_in = tm_a.inside(ai, aj)
    ai, aj = np.clip(ai, 0, tm_a.nz - 1), np.clip(aj, 0, tm_a.nx - 1)
    cls_a = np.where(a_in, tm_a.cls[ai, aj], T.NODATA)
    roi_a = a_in & tm_a.roi[ai, aj]
    nd_a = np.where(a_in, np.nan_to_num(tm_a.ndsm, nan=0.0)[ai, aj], 0.0)
    # ---- surfaces: upper envelope (objects) and median (a narrow trench barely lowers the envelope)
    (da, da50), ca = rasterize(reg.apply(pa), *grid, q=(0.9, 0.5))
    (db, db50), cb = rasterize(pb, *grid, q=(0.9, 0.5))
    plane0 = reg.plane.copy()
    vinfo = vertical_fit(reg, da50, ca, db50, cb, tm_b, cls_a) or {"ground_cells": 0, "registration_sigma_m": 0.15}
    sig_reg = float(vinfo["registration_sigma_m"])
    dp = reg.plane - plane0                     # the vertical correction is smooth: applied to the grids
    corr = (dp[0] * cx + dp[1] * cz + dp[2]).astype(np.float32)
    da, da50 = da + corr, da50 + corr
    # water on both days = reflections / no stereo -> not compared; water on one day only is a change
    # one point per cell is enough here: every object below must also carry >= 6 points on each day
    valid = (ca >= 1) & (cb >= 1) & roi_a & tm_b.roi & ~((tm_b.cls == T.WATER) & (cls_a == T.WATER))
    fa, fb = T.nan_fill_nearest(da.astype(np.float64)), T.nan_fill_nearest(db.astype(np.float64))
    # at a height step that already existed on day 1 (wall, roof edge) a residual misregistration of one
    # cell would look like a change: there the other surface's highest cell around is used; elsewhere the
    # direct difference is kept, so new narrow features (a 1 m trench, a pole) are not erased by their own edges
    step_a = (ndi.maximum_filter(fa, 3) - ndi.minimum_filter(fa, 3)) > 0.5
    gain = fb - np.where(step_a, ndi.maximum_filter(fa, 3), fa)
    loss = fa - np.where(step_a, ndi.maximum_filter(fb, 3), fb)
    loss_med = T.nan_fill_nearest(da50.astype(np.float64)) - T.nan_fill_nearest(db50.astype(np.float64))
    loss = np.where(step_a, loss, np.maximum(loss, loss_med))
    ra, _ = T.local_plane_rms(fa, np.isfinite(da), size=5)
    rb, _ = T.local_plane_rms(fb, np.isfinite(db), size=5)
    ra = np.clip(np.nan_to_num(ra, nan=0.5), 0.02, 1.0)
    rb = np.clip(np.nan_to_num(rb, nan=0.5), 0.02, 1.0)
    ground_b = float(np.median(rb[valid & np.isin(tm_b.cls, (T.GROUND, T.ROAD))])) if (valid & np.isin(tm_b.cls, (T.GROUND, T.ROAD))).any() else 0.05
    ground_a = float(np.median(ra[valid & np.isin(cls_a, (T.GROUND, T.ROAD))])) if (valid & np.isin(cls_a, (T.GROUND, T.ROAD))).any() else 0.05
    # surface texture from the smoother of the two days: real texture (vegetation, walls that exist on both
    # days) is rough on both, while a new tent or trench makes only one day rough and must not raise its
    # own detection limit
    tex = np.minimum(ra, rb)
    lod_gain = 1.96 * np.sqrt(tex ** 2 + max(ground_a, ground_b) ** 2 + sig_reg ** 2)
    lod_loss = lod_gain
    lod_ground = float(1.96 * math.sqrt(ground_a ** 2 + ground_b ** 2 + sig_reg ** 2))
    new = valid & (gain > np.maximum(min_height_m, lod_gain))
    gone = valid & (loss > np.maximum(min_height_m, lod_loss))
    _DEBUG.update(gain=gain, loss=loss, valid=valid, lod_gain=lod_gain, lod_loss=lod_loss, ca=ca, cb=cb, roi_a=roi_a,
                  cls_a=cls_a, new=new.copy(), gone=gone.copy(), da=da, db=db)
    new = ndi.binary_opening(new, np.ones((2, 2), bool))
    gone = ndi.binary_opening(gone, np.ones((2, 2), bool))

    # ---- context for classification
    bld_a = (cls_a == T.BUILDING) & roi_a
    dist_bld = ndi.distance_transform_edt(~bld_a) * res if bld_a.any() else np.full(bld_a.shape, 1e4)
    tree_a = cls_a == T.TREE
    road = (tm_b.cls == T.ROAD) | (cls_a == T.ROAD)

    alerts, suppressed = [], {"tree_canopy": 0, "crop_growth": 0, "too_small": 0}
    telem_b = sb["telem"]

    def geometry(mask):
        pi, pj = np.nonzero(mask)
        px, pz = tm_b.centre(pi, pj)
        pc = np.c_[px - px.mean(), pz - pz.mean()]
        ev, evec = np.linalg.eigh(np.cov(pc.T) if len(pc) > 2 else np.eye(2))
        proj = pc @ evec
        dims = proj.max(0) - proj.min(0) + res
        corners = np.array([[proj[:, 0].min() - res / 2, proj[:, 1].min() - res / 2], [proj[:, 0].max() + res / 2, proj[:, 1].min() - res / 2],
                            [proj[:, 0].max() + res / 2, proj[:, 1].max() + res / 2], [proj[:, 0].min() - res / 2, proj[:, 1].max() + res / 2]])
        box = corners @ evec.T + [px.mean(), pz.mean()]
        return float(px.mean()), float(pz.mean()), float(max(dims)), float(min(dims)), box.round(2).tolist()

    def components(mask):
        """Fragments of one object (a trench broken by a missing cell) closer than ~1 m are one object."""
        lab, n = ndi.label(ndi.binary_dilation(mask, iterations=2), np.ones((3, 3), bool))
        lab = np.where(mask, lab, 0)
        return lab, ndi.find_objects(lab)

    lab, objs = components(new)
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == k
        area = float(m.sum()) * res * res
        if area < min_area_m2:
            suppressed["too_small"] += 1
            continue
        mask = np.zeros(new.shape, bool)
        mask[sl] = m
        if int(ca[mask].sum()) < 6 or int(cb[mask].sum()) < 6:
            suppressed["too_few_points"] = suppressed.get("too_few_points", 0) + 1
            continue
        g = gain[mask]
        h95, hmean = float(np.percentile(g, 95)), float(np.mean(g))
        if float(np.mean(tree_a[mask])) >= 0.5 and h95 < 6.0:
            suppressed["tree_canopy"] += 1              # wind / growth inside canopy that was already there
            continue
        low_veg = float(np.mean(cls_a[mask] == T.LOW_VEG)) >= 0.5
        if low_veg and h95 < 1.0 and area > 150:
            suppressed["crop_growth"] += 1
            continue
        x, z, length, width, box = geometry(mask)
        lod = float(np.median(lod_gain[mask]))
        on_road = float(np.mean(road[mask])) >= 0.5
        isolated = float(np.min(dist_bld[mask])) > 30.0
        near_b = float(np.min(dist_bld[mask]))
        vol = float(np.sum(np.clip(g, 0, None))) * res * res
        top = float(np.nanpercentile(db[mask], 95))
        base_y = float(np.nanmedian(da[mask]))
        reasons = [f"height gain +{h95:.2f} m ({h95 / max(lod, 1e-3):.0f} x the {lod:.2f} m noise floor)"]
        if h95 >= 4.0 and area <= 25.0:
            cat, sev = "MAST / TOWER", "HIGH"
        elif area >= 60.0 and h95 >= 2.0:
            cat, sev = "NEW STRUCTURE", "HIGH"
        elif 1.0 <= h95 <= 4.2 and width <= 3.6 and length >= 1.3 * width and length <= 14.0:
            cat, sev = ("VEHICLE", "ADVISORY") if on_road else ("VEHICLE (off road)", "ELEVATED")
        elif 1.0 <= h95 <= 4.5 and area <= 150.0:
            cat, sev = "TENT / SHELTER", "HIGH" if isolated else "ELEVATED"
        elif h95 < 1.0:
            cat, sev = "LOW OBJECT", "ELEVATED" if (isolated and area >= 6.0) else "ADVISORY"
            reasons.append("stockpile, crates, sandbag wall or a low tent")
        else:
            cat, sev = "NEW OBJECT", "ELEVATED"
        if isolated:
            reasons.append(f"isolated: {near_b:.0f} m from the nearest day-1 building")
        if float(np.mean(tm_b.exg[mask] > 0.04)) > 0.5 and float(np.mean(tree_a[mask])) < 0.2:
            reasons.append("green surface on previously open ground (possible camouflage net)")
            if sev == "ADVISORY":
                sev = "ELEVATED"
        lat, lon = T.to_latlon(telem_b, x, top, z)
        alerts.append({"kind": "gain", "category": cat, "severity": sev, "height_change_m": round(h95, 2),
                       "mean_change_m": round(hmean, 2), "uncertainty_m": round(lod / 1.96, 3), "noise_floor_m": round(lod, 2),
                       "area_m2": round(area, 1), "length_m": round(length, 1), "width_m": round(width, 1),
                       "volume_m3": round(vol, 1), "position": [round(x, 2), round(top, 2), round(z, 2)], "base_y": round(base_y, 2),
                       "footprint": box, "on_road": on_road, "isolated": isolated, "nearest_building_m": round(near_b, 1),
                       "gps_lat": lat, "gps_lon": lon, "reasons": reasons})

    lab, objs = components(gone)
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == k
        area = float(m.sum()) * res * res
        if area < min_area_m2:
            continue
        mask = np.zeros(gone.shape, bool)
        mask[sl] = m
        if int(ca[mask].sum()) < 6 or int(cb[mask].sum()) < 6:
            suppressed["too_few_points"] = suppressed.get("too_few_points", 0) + 1
            continue
        l95 = float(np.percentile(loss[mask], 95))
        if float(np.mean(tree_a[mask])) >= 0.5 and l95 < 6.0:
            suppressed["tree_canopy"] += 1
            continue
        x, z, length, width, box = geometry(mask)
        lod = float(np.median(lod_loss[mask]))
        was_object = float(np.median(nd_a[mask])) > 0.8
        if was_object:
            cat, sev = ("REMOVED STRUCTURE", "HIGH") if area >= 60 else ("REMOVED OBJECT", "ADVISORY")
            reason = "present on day 1, gone on day 2 (moved vehicle, dismantled tent, demolition)"
        else:
            cat, sev = "EXCAVATION / TRENCH", "ELEVATED" if l95 >= 0.4 else "ADVISORY"
            reason = "ground lowered: trench, pit, crater or dug fighting position"
        top = float(np.nanmedian(db[mask]))
        lat, lon = T.to_latlon(telem_b, x, top, z)
        alerts.append({"kind": "loss", "category": cat, "severity": sev, "height_change_m": round(-l95, 2),
                       "uncertainty_m": round(lod / 1.96, 3), "noise_floor_m": round(lod, 2), "area_m2": round(area, 1),
                       "length_m": round(length, 1), "width_m": round(width, 1),
                       "volume_m3": round(float(np.sum(np.clip(loss[mask], 0, None))) * res * res, 1),
                       "position": [round(x, 2), round(top, 2), round(z, 2)], "base_y": round(top, 2), "footprint": box,
                       "gps_lat": lat, "gps_lon": lon,
                       "reasons": [f"height loss -{l95:.2f} m ({l95 / max(lod, 1e-3):.0f} x the noise floor)", reason]})

    # ---- clusters of new objects off the road network -> possible camp
    camp = [a for a in alerts if a["kind"] == "gain" and a["category"] not in ("VEHICLE",)]
    groups = []
    if len(camp) >= 3:
        pts = np.array([[a["position"][0], a["position"][2]] for a in camp])
        parent = list(range(len(pts)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if d[i, j] <= 40.0:
                    parent[find(i)] = find(j)
        clusters = {}
        for i in range(len(pts)):
            clusters.setdefault(find(i), []).append(i)
        for members in clusters.values():
            if len(members) >= 3:
                c = pts[members].mean(0)
                rad = float(np.max(np.hypot(*(pts[members] - c).T))) + 10.0
                groups.append((members, c, rad))
    alerts.sort(key=lambda a: (SEVERITY_RANK[a["severity"]], -abs(a["height_change_m"]) * a["area_m2"]))
    for i, a in enumerate(alerts):
        a["id"] = f"CHG-{i + 1:02d}"
    for gi, (members, c, rad) in reversed(list(enumerate(groups))):
        ids = [camp[m]["id"] for m in members]
        y = float(np.median([camp[m]["position"][1] for m in members]))
        lat, lon = T.to_latlon(telem_b, float(c[0]), y, float(c[1]))
        alerts.insert(0, {"id": f"CAMP-{gi + 1}", "kind": "group",
                          "category": "POSSIBLE CAMP", "severity": "HIGH", "members": ids,
                          "position": [round(float(c[0]), 2), round(y, 2), round(float(c[1]), 2)], "radius_m": round(rad, 1),
                          "area_m2": round(sum(camp[m]["area_m2"] for m in members), 1), "height_change_m": None,
                          "gps_lat": lat, "gps_lon": lon,
                          "reasons": [f"{len(members)} new objects within {rad:.0f} m of each other ({', '.join(ids)})"]})

    # ---- change map (day-2 grid, north-up PNG handled by the API)
    sig_gain = np.where(valid & (gain > 0), gain / np.maximum(lod_gain, 1e-3), 0.0)
    sig_loss = np.where(valid & (loss > 0), loss / np.maximum(lod_loss, 1e-3), 0.0)
    img = np.zeros((tm_b.nz, tm_b.nx, 4), np.uint8)
    show_g = new
    show_l = gone
    img[show_g] = [226, 70, 52, 0]
    img[show_l] = [70, 140, 230, 0]
    alpha = np.clip(np.maximum(sig_gain, sig_loss) * 60, 0, 230).astype(np.uint8)
    img[..., 3] = np.where(show_g | show_l, np.maximum(alpha, 120), 0)
    counts = {}
    for a in alerts:
        counts[a["severity"]] = counts.get(a["severity"], 0) + 1
    compared = float(valid.sum()) * res * res
    log(f"[Change] {len(alerts)} alert(s) in {time.time() - t0:.1f}s; compared {compared / 1e4:.2f} ha, "
        f"ground noise floor {lod_ground:.2f} m, registration {reg.describe()}")
    return dict(base, status="completed", location=loc,
                registration=dict(reg.describe(), **reg_info, **vinfo),
                compared_area_m2=round(compared, 0), noise_floor_ground_m=round(lod_ground, 3),
                alerts=alerts, alert_count=len(alerts), severity_counts=counts, suppressed=suppressed,
                change_image=img, grid={"x0": tm_b.x0, "z0": tm_b.z0, "res": tm_b.res, "nx": tm_b.nx, "nz": tm_b.nz},
                seconds=round(time.time() - t0, 1),
                message=(f"{counts.get('HIGH', 0)} high, {counts.get('ELEVATED', 0)} elevated, {counts.get('ADVISORY', 0)} advisory "
                         f"change(s); nothing below the {lod_ground:.2f} m noise floor is reported."))
