"""
PRISM // Terrain analysis layer (terrain.py)
============================================
Turns the metric point cloud (x = East, y = Up, z = South, metres) into analysis rasters that every
downstream feature shares: height measurement, roads & potholes, base-site finding, covert routes
and change detection.

Rasters (row i runs along +z = south, column j along +x = east, cell centre = origin + (idx+0.5)*res)
  count      points per cell
  zlo/zmed/zhi   10th / 50th / 95th percentile heights (zhi = DSM, top surface)
  dtm        bare-earth elevation (progressive morphological filter + harmonic in-fill)
  ndsm       dsm - dtm (height of objects above the local ground)
  slope      terrain slope in degrees (from the DTM)
  rough      roof/canopy roughness: RMS residual of a local plane fitted to the DSM (5x5 cells)
  rgb        top-surface colour, exg = excess-green index
  cls        land cover: NODATA GROUND LOW_VEG TREE BUILDING WATER ROAD VEHICLE
  roi        cells that were seen at a usable grazing angle (> MIN_GRAZING_DEG) from the flight path

Ground filter: SMRF-style progressive opening (Pingel et al. 2013) on the 10th-percentile surface,
with low-outlier (water reflection) rejection first. Classification uses geometry first (nDSM,
planarity, component size/shape) and colour only to split vegetation from man-made surfaces.
"""

import json
import math
import os
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    import plyio
except ImportError:
    from backend import plyio

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "data")

NODATA, GROUND, LOW_VEG, TREE, BUILDING, WATER, ROAD, VEHICLE = range(8)
CLASS_NAMES = ["nodata", "ground", "low_vegetation", "tree", "building", "water", "road", "vehicle"]
MIN_GRAZING_DEG = 8.0
CACHE_VERSION = 5


# =============================================================================
# helpers
# =============================================================================
def nan_fill_nearest(a):
    """Fills NaNs with the value of the nearest valid cell."""
    a = np.asarray(a, np.float64)
    bad = ~np.isfinite(a)
    if not bad.any():
        return a.copy()
    if bad.all():
        return np.zeros_like(a)
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def harmonic_fill(a, known, iters=300):
    """Laplace (harmonic) interpolation of unknown cells, initialised with nearest values."""
    out = nan_fill_nearest(np.where(known, a, np.nan))
    unk = ~known
    if not unk.any():
        return out
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    for _ in range(iters):
        sm = ndi.convolve(out, k, mode="nearest")
        out[unk] = sm[unk]
    return out


def masked_smooth(a, w, sigma):
    """Normalised Gaussian smoothing of `a` with weights `w` (NaN-safe)."""
    a0 = np.where(np.isfinite(a) & (w > 0), a, 0.0)
    num = ndi.gaussian_filter(a0 * w, sigma, mode="nearest")
    den = ndi.gaussian_filter(w.astype(np.float64), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan), den


def local_plane_rms(z, valid, size=5):
    """RMS residual of a least-squares plane fitted in every size x size window (box-filter moments)."""
    w = valid.astype(np.float64)
    zz = np.where(valid, z, 0.0)
    r = size // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    box = np.ones((size, size))

    def conv(img, ker):
        return ndi.correlate(img, ker, mode="constant", cval=0.0)

    n = conv(w, box)
    sz = conv(zz, box)
    szz = conv(zz * zz, box)
    sxz = conv(zz, xx)       # sum over window of x_offset * z (kernel positions are offsets)
    syz = conv(zz, yy)
    sx = conv(w, xx)
    sy = conv(w, yy)
    sxx = conv(w, xx * xx)
    syy = conv(w, yy * yy)
    sxy = conv(w, xx * yy)
    with np.errstate(invalid="ignore", divide="ignore"):
        mz = sz / n
        # centred moments (windows may be partially empty -> do the full 3x3 normal equations)
        cxx = sxx - sx * sx / n
        cyy = syy - sy * sy / n
        cxy = sxy - sx * sy / n
        cxz = sxz - sx * sz / n
        cyz = syz - sy * sz / n
        czz = szz - sz * sz / n
        det = cxx * cyy - cxy * cxy
        b = (cxz * cyy - cyz * cxy) / det
        c = (cyz * cxx - cxz * cxy) / det
        rss = czz - b * cxz - c * cyz
        rms = np.sqrt(np.maximum(rss, 0.0) / np.maximum(n - 3, 1))
    rms[(n < 6) | ~np.isfinite(rms)] = np.nan
    return rms, mz


def zhang_suen_thinning(mask, max_iter=200):
    """Morphological skeleton (Zhang-Suen), vectorised."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = img
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            c = p[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(x.astype(np.int16) for x in nb[:8])
            a = sum(((nb[k] == 0) & (nb[k + 1] == 1)).astype(np.int16) for k in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def edt_dilate(mask, r_cells):
    return ndi.distance_transform_edt(~mask) <= r_cells


def edt_close(mask, r_cells):
    """Morphological closing with a disk of radius r (cells) via two distance transforms (fast)."""
    pad = int(math.ceil(r_cells)) + 2
    m = np.pad(mask, pad)
    dil = ndi.distance_transform_edt(~m) <= r_cells
    closed = ndi.distance_transform_edt(dil) > r_cells
    return closed[pad:-pad, pad:-pad]


def block_median_surface(z, block, smooth_size):
    """Median of every block x block tile (NaN-aware), median-smoothed, upsampled back to z's shape."""
    nz, nx = z.shape
    pz, px = (-nz) % block, (-nx) % block
    zp = np.pad(z, ((0, pz), (0, px)), constant_values=np.nan)
    tiles = zp.reshape(zp.shape[0] // block, block, zp.shape[1] // block, block).transpose(0, 2, 1, 3)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            coarse = np.nanmedian(tiles.reshape(tiles.shape[0], tiles.shape[1], -1), axis=2)
    coarse = ndi.median_filter(nan_fill_nearest(coarse), size=smooth_size)
    up = np.repeat(np.repeat(coarse, block, 0), block, 1)
    return up[:nz, :nx]


# =============================================================================
# the terrain model
# =============================================================================
class TerrainModel:
    def __init__(self, res, x0, z0, nx, nz):
        self.res, self.x0, self.z0, self.nx, self.nz = float(res), float(x0), float(z0), int(nx), int(nz)
        self.meta = {}

    # ---------------- coordinates ----------------
    def cell_of(self, x, z):
        j = np.floor((np.asarray(x) - self.x0) / self.res).astype(np.int64)
        i = np.floor((np.asarray(z) - self.z0) / self.res).astype(np.int64)
        return i, j

    def inside(self, i, j):
        return (i >= 0) & (i < self.nz) & (j >= 0) & (j < self.nx)

    def centre(self, i, j):
        return self.x0 + (np.asarray(j) + 0.5) * self.res, self.z0 + (np.asarray(i) + 0.5) * self.res

    def sample(self, raster, x, z):
        """Bilinear sample of a raster at world (x, z); scalars in -> scalar out."""
        scalar = np.ndim(x) == 0
        fx = (np.atleast_1d(np.asarray(x, float)) - self.x0) / self.res - 0.5
        fz = (np.atleast_1d(np.asarray(z, float)) - self.z0) / self.res - 0.5
        src = raster if not np.isnan(raster).any() else np.nan_to_num(raster, nan=float(np.nanmedian(raster)))
        out = ndi.map_coordinates(src.astype(np.float64), [fz, fx], order=1, mode="nearest")
        return float(out[0]) if scalar else out

    def height_above_ground(self, xyz):
        return np.asarray(xyz)[:, 1] - self.sample(self.dtm, xyz[:, 0], xyz[:, 2])

    # ---------------- construction ----------------
    @classmethod
    def from_points(cls, xyz, rgb=None, res=0.5, cams=None, min_grazing_deg=MIN_GRAZING_DEG, log=print):
        t0 = time.time()
        xyz = np.asarray(xyz, np.float64)
        keep = np.isfinite(xyz).all(1)
        roi_pts = None
        if cams is not None and len(cams) >= 3:
            cams = np.asarray(cams, float)
            d, k = cKDTree(cams[:, [0, 2]]).query(xyz[:, [0, 2]], workers=-1)
            ang = np.degrees(np.arctan2(cams[k, 1] - xyz[:, 1], np.maximum(d, 1e-6)))
            roi_pts = ang >= min_grazing_deg
        if rgb is None:
            rgb = np.full((len(xyz), 3), 150, np.uint8)
        ext_pts = xyz[keep & roi_pts] if (roi_pts is not None and (keep & roi_pts).sum() > 1000) else xyz[keep]
        lo = np.percentile(ext_pts[:, [0, 2]], 0.05, axis=0) - 10 * res
        hi = np.percentile(ext_pts[:, [0, 2]], 99.95, axis=0) + 10 * res
        keep &= (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) & (xyz[:, 2] >= lo[1]) & (xyz[:, 2] <= hi[1])
        nx = int(math.ceil((hi[0] - lo[0]) / res))
        nz = int(math.ceil((hi[1] - lo[1]) / res))
        tm = cls(res, lo[0], lo[1], nx, nz)
        p, c = xyz[keep], np.asarray(rgb)[keep]
        roi_p = roi_pts[keep] if roi_pts is not None else None
        i, j = tm.cell_of(p[:, 0], p[:, 2])
        key = i * nx + j
        order = np.lexsort((p[:, 1], key))
        ks, ys = key[order], p[order, 1]
        uk, st, cnt = np.unique(ks, return_index=True, return_counts=True)

        def pct(q):
            out = np.full(nx * nz, np.nan, np.float32)
            out[uk] = ys[st + np.floor(q * (cnt - 1)).astype(np.int64)]
            return out.reshape(nz, nx)

        tm.count = np.zeros(nx * nz, np.int32)
        tm.count[uk] = cnt
        tm.count = tm.count.reshape(nz, nx)
        tm.zlo, tm.zmed, tm.zhi = pct(0.10), pct(0.50), pct(0.95)
        tm.zmax = pct(1.0)
        # top-surface colour: mean colour of the upper half of every cell's points
        grp = np.repeat(np.arange(len(uk)), cnt)
        rank = np.arange(len(ks)) - np.repeat(st, cnt)
        upper = rank >= (np.repeat(cnt, cnt) // 2)
        cs = c[order].astype(np.float64)
        wsum = np.bincount(grp[upper], minlength=len(uk)).astype(np.float64)
        rgbm = np.zeros((nx * nz, 3), np.float32)
        for ch in range(3):
            rgbm[uk, ch] = np.bincount(grp[upper], cs[upper, ch], minlength=len(uk)) / np.maximum(wsum, 1)
        tm.rgb = rgbm.reshape(nz, nx, 3)
        # observed region: cells with points, with sparse speckle gaps (1-2 cells) closed
        observed = tm.count > 0
        region = ndi.binary_closing(observed, np.ones((3, 3)), iterations=2) | observed
        # geometric ROI: ground seen from the flight path at a usable grazing angle (independent of
        # whether the surface returned points -> water / occlusions inside it are detectable)
        if cams is not None and len(cams) >= 3:
            ci, cj = tm.cell_of(cams[:, 0], cams[:, 2])
            track = np.zeros((nz, nx), bool)
            ok = tm.inside(ci, cj)
            track[ci[ok], cj[ok]] = True
            if track.any():
                dist = ndi.distance_transform_edt(~track) * res
                agl = float(np.median(cams[:, 1]) - np.percentile(p[:, 1], 20))
                reach = max(40.0, agl / math.tan(math.radians(min_grazing_deg)))
                roi = dist <= reach
            else:
                roi = np.ones((nz, nx), bool)
        else:
            roi = np.ones((nz, nx), bool)
        # analysis region = densely observed ground (>= 45 % of cells within 5 m hold points); gaps up
        # to ~30 m wide between dense data (canals, ponds, occlusions) belong to the scene, the sparse
        # far-field fringe does not (reported as "insufficient data" instead of guessed)
        fill = ndi.uniform_filter(observed.astype(np.float32), size=max(3, int(round(5.0 / res))) | 1)
        dense = fill >= 0.45
        enclosed = edt_close(dense, 15.0 / res)
        near_data = enclosed | edt_dilate(dense, 3.0 / res)
        tm.roi = roi & near_data
        tm.enclosed = enclosed
        tm.fill_ratio = fill
        tm.observed = region
        tm.has_points = observed
        # fill the speckle cells of the region from their nearest observed neighbour
        fill = region & ~observed
        if fill.any():
            idx = ndi.distance_transform_edt(~observed, return_distances=False, return_indices=True)
            for name in ("zlo", "zmed", "zhi", "zmax"):
                a = getattr(tm, name)
                a[fill] = a[tuple(i[fill] for i in idx)]
            tm.rgb[fill] = tm.rgb[tuple(i[fill] for i in idx)]
        tm._classify(log)
        tm.meta.update(points=int(len(p)), res=res, seconds=round(time.time() - t0, 2),
                       roi_cells=int(roi.sum()), observed_cells=int(observed.sum()))
        log(f"[Terrain] {nx}x{nz} cells @ {res} m from {len(p):,} points in {time.time() - t0:.1f}s")
        return tm

    # ---------------- ground + classes ----------------
    def _classify(self, log=print):
        res = self.res
        obs = self.observed
        zlo = self.zlo.astype(np.float64)
        # 1. low-outlier rejection (reflections in water, MVS blunders below the surface):
        #    compare with a 2 m block-median surface smoothed over ~10 m
        ref = block_median_surface(zlo, max(1, int(round(2.0 / res))), 5)
        low_outlier = obs & (zlo < ref - 1.0)
        self.reflection = low_outlier
        zlo[low_outlier] = np.nan
        # 2. SMRF progressive opening on the minimum surface
        zmin = nan_fill_nearest(zlo)
        nonground = ~np.isfinite(zlo)
        slope_thr = 0.15                       # rise per metre tolerated before a cell is "object"
        prev = zmin
        rmax = max(2, int(round(18.0 / res)))
        radii = sorted({int(round(r)) for r in np.unique(np.geomspace(1, rmax, 9))})
        for r in radii:
            opened = ndi.grey_opening(prev, size=(2 * r + 1, 2 * r + 1))
            dh = prev - opened
            nonground |= dh > max(0.25, slope_thr * r * res)
            prev = opened
        ground_cells = obs & ~nonground
        dtm0 = harmonic_fill(zlo, ground_cells & np.isfinite(zlo), iters=200)
        # 3. refine: cells whose low surface is within the elevation threshold of the provisional DTM
        slope0 = self._slope(dtm0)
        thr = 0.35 + 1.25 * np.tan(np.radians(np.clip(slope0, 0, 60))) * res
        ground_cells = obs & np.isfinite(zlo) & (np.abs(zlo - dtm0) <= thr)
        # the 10th percentile sits ~1.3 sigma below a noisy surface; on confirmed ground cells the
        # median is the unbiased surface estimate (grass tops excepted: kept at the low percentile)
        zmed = self.zmed.astype(np.float64)
        zg = np.where(ground_cells & (zmed - zlo < 0.25), zmed, zlo)
        self.dtm = harmonic_fill(zg, ground_cells, iters=250)
        self.dtm = 0.5 * self.dtm + 0.5 * ndi.uniform_filter(self.dtm, 3, mode="nearest")
        self.dsm = np.where(obs, self.zhi, np.nan)
        self.ndsm = self.dsm - self.dtm
        self.slope = self._slope(self.dtm)
        self.rough, _ = local_plane_rms(nan_fill_nearest(self.dsm), obs, size=5)
        # the median surface is far less noisy than the 95th percentile: used for smooth surfaces (roads)
        self.rough_med, _ = local_plane_rms(nan_fill_nearest(self.zmed.astype(np.float64)), obs, size=5)
        self.nd_med = np.where(obs, self.zmed - self.dtm, np.nan)
        r, g, b = [self.rgb[..., k].astype(np.float64) for k in range(3)]
        s = r + g + b + 1e-6
        self.exg = (2 * g - r - b) / s           # excess green (chromatic coordinates)
        mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
        self.sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        self.val = mx / 255.0
        self.ground_mask = ground_cells

        cls = np.full(obs.shape, NODATA, np.uint8)
        nd = np.nan_to_num(self.ndsm, nan=0.0)
        green = self.exg > 0.04
        rough = np.nan_to_num(self.rough, nan=0.0)
        cls[obs] = GROUND
        cls[obs & (nd >= 0.35) & green] = LOW_VEG
        cls[obs & (nd >= 0.35) & ~green & (nd < 1.2)] = GROUND     # kerbs, low walls, noise
        tall = obs & (nd >= 1.2)
        lab, n = ndi.label(tall, np.ones((3, 3)))
        if n:
            idx = np.arange(1, n + 1)
            area = np.bincount(lab.ravel(), minlength=n + 1)[1:] * res * res
            g_med = np.asarray(ndi.median(self.exg, lab, idx))
            # roughness on the interior only: wall edges of small buildings look rough in any window
            inner = lab * ndi.binary_erosion(tall, np.ones((3, 3)))
            r_all = np.asarray(ndi.median(rough, lab, idx))
            r_in = np.asarray(ndi.median(rough, inner, idx))
            n_in = np.bincount(inner.ravel(), minlength=n + 1)[1:]
            r_med = np.where((n_in >= 4) & np.isfinite(r_in), r_in, r_all)
            # small tall structures (towers, water tanks): every window near the edge straddles a wall;
            # roughness from windows that do not cross a height step > 1.5 m sees the real roof
            dsm_f = nan_fill_nearest(self.dsm)
            step = ndi.maximum_filter(dsm_f, 5) - ndi.minimum_filter(dsm_f, 5)
            clean = lab * (step < 1.5)
            r_cl = np.asarray(ndi.median(rough, clean, idx))
            n_cl = np.bincount(clean.ravel(), minlength=n + 1)[1:]
            n_all = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            ok_cl = (n_cl >= np.maximum(4, 0.2 * n_all)) & np.isfinite(r_cl)
            r_med = np.where(ok_cl, np.minimum(r_med, r_cl), r_med)
            h_max = np.asarray(ndi.maximum(nd, lab, idx))
            # footprint shape from second moments (uniform rectangle: variance = L^2 / 12)
            ii, jj = np.mgrid[0:lab.shape[0], 0:lab.shape[1]]
            cnt = np.maximum(n_all, 1)
            mi = np.bincount(lab.ravel(), ii.ravel(), n + 1)[1:] / cnt
            mj = np.bincount(lab.ravel(), jj.ravel(), n + 1)[1:] / cnt
            vii = np.bincount(lab.ravel(), (ii * ii).ravel().astype(np.float64), n + 1)[1:] / cnt - mi ** 2
            vjj = np.bincount(lab.ravel(), (jj * jj).ravel().astype(np.float64), n + 1)[1:] / cnt - mj ** 2
            vij = np.bincount(lab.ravel(), (ii * jj).ravel().astype(np.float64), n + 1)[1:] / cnt - mi * mj
            tr_, det_ = vii + vjj, vii * vjj - vij ** 2
            disc = np.sqrt(np.maximum(tr_ ** 2 / 4 - det_, 0))
            length = np.sqrt(12 * np.maximum(tr_ / 2 + disc, 0)) * res + res
            width = np.sqrt(12 * np.maximum(tr_ / 2 - disc, 0)) * res + res
            # component-level decision (geometry first, colour only to split vegetation from man-made);
            # vehicles are narrow and elongated (cars 1.8 m, trucks 2.6 m, tanks up to 3.7 m wide)
            vehicle = (area >= 2.0) & (area <= 45.0) & (h_max <= 4.2) & (g_med < 0.03) & (r_med < 0.25) & \
                (width <= 4.0) & (length >= 1.35 * width)
            building = ~vehicle & (g_med < 0.03) & (r_med < 0.18) & (area >= 12.0)
            treeish = ~vehicle & ~building & ((g_med >= 0.03) | (r_med >= 0.18))
            other = ~vehicle & ~building & ~treeish
            comp = np.zeros(n + 1, np.uint8)
            comp[1:][vehicle] = VEHICLE
            comp[1:][building] = BUILDING
            comp[1:][treeish] = TREE
            comp[1:][other] = np.where(area[other] >= 4.0, BUILDING, TREE)
            area_map = np.concatenate([[0.0], area])[lab]
            ccls = comp[lab]
            # per-cell split inside mixed components (a tree touching a roof, a roof under a canopy)
            veg_cell = green | (rough > 0.22)
            ccls[(ccls == BUILDING) & veg_cell & (self.exg > 0.08)] = TREE
            ccls[(ccls == TREE) & ~veg_cell & (nd >= 2.0) & (area_map >= 40.0)] = BUILDING
            cls[tall] = ccls[tall]
            # a building fragment is only kept when it forms a sizeable blob
            bl, bn = ndi.label(cls == BUILDING, np.ones((3, 3)))
            if bn:
                barea = np.bincount(bl.ravel(), minlength=bn + 1) * res * res
                cls[(bl > 0) & (barea[bl] < 8.0)] = TREE
        # water / no-return: no data inside the geometric ROI (water gives no stereo matches), mirror
        # reflections below the ground, and dark, flat, non-green surfaces at ground level
        dark_flat = obs & (self.val < 0.25) & (nd < 0.3) & (self.slope < 4) & (self.exg < 0.0)
        # genuine gaps are wide (can hold a 4 m disk); sparse far-field coverage never is
        rr = max(1, int(round(2.0 / res)))
        yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        disk = (xx * xx + yy * yy) <= rr * rr
        has = getattr(self, "has_points", obs)
        enclosed = getattr(self, "enclosed", self.roi)
        holes = self.roi & enclosed & ndi.binary_opening(~has, disk)
        # a real gap is bounded by dense data: mean point coverage in a 3 m band around each gap
        hl, hn = ndi.label(holes, np.ones((3, 3)))
        if hn:
            band = edt_dilate(holes, 3.0 / res) & ~holes
            idx = ndi.distance_transform_edt(~holes, return_distances=False, return_indices=True)
            owner = hl[tuple(i[band] for i in idx)]
            cov = np.bincount(owner, weights=has[band].astype(np.float64), minlength=hn + 1) / \
                np.maximum(np.bincount(owner, minlength=hn + 1), 1)
            holes = (hl > 0) & (cov[hl] >= 0.55)
        water = holes | ndi.binary_opening(self.reflection | dark_flat, np.ones((3, 3)))
        wl, wn = ndi.label(water, np.ones((3, 3)))
        if wn:
            warea = np.bincount(wl.ravel(), minlength=wn + 1) * res * res
            big = (wl > 0) & (warea[wl] >= 25.0)
            cls[big & (cls != BUILDING) & (cls != TREE)] = WATER
        self.cls = cls
        for name in ("dtm", "dsm", "ndsm", "slope", "rough", "rough_med", "nd_med", "exg", "sat", "val"):
            setattr(self, name, getattr(self, name).astype(np.float32))
        log(f"[Terrain] classes: " + ", ".join(f"{CLASS_NAMES[c]} {100 * np.mean(cls[self.roi] == c):.1f}%"
                                              for c in range(1, 8) if np.any(cls == c)))

    def _slope(self, dtm):
        gz, gx = np.gradient(dtm, self.res)
        return np.degrees(np.arctan(np.hypot(gx, gz)))

    # ---------------- roads (set by road engine) ----------------
    def apply_road_mask(self, road):
        self.cls[road & (self.cls != BUILDING) & (self.cls != WATER)] = ROAD
        self.road = road

    # ---------------- visibility ----------------
    def viewshed(self, ox, oz, eye_h=1.7, target_h=1.7, radius=600.0, n_rays=None, surface=None, eye_abs=None):
        """
        Cells where a target standing target_h above the terrain can be seen by an observer at (ox, oz)
        with eyes eye_h above the terrain (or at absolute height eye_abs). Radial sweep: along every ray
        the running maximum elevation angle of the obstructing surface (DSM) is compared with the
        target's elevation angle. Returns (visible bool raster, distance float32 raster).
        """
        surf = self.dsm_filled if surface is None else surface
        res = self.res
        rc = int(radius / res)
        if n_rays is None:
            n_rays = int(np.clip(2 * math.pi * rc, 360, 2048))
        oi, oj = (oz - self.z0) / res - 0.5, (ox - self.x0) / res - 0.5
        h0 = float(eye_abs) if eye_abs is not None else float(self.sample(self.dtm, ox, oz)) + eye_h
        th = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
        steps = np.arange(1, rc + 1, dtype=np.float32)
        ii = (oi + np.outer(np.sin(th), steps)).astype(np.float32)
        jj = (oj + np.outer(np.cos(th), steps)).astype(np.float32)
        inside = (ii >= -0.5) & (ii <= self.nz - 0.5) & (jj >= -0.5) & (jj <= self.nx - 0.5)
        iic = np.clip(np.rint(ii), 0, self.nz - 1).astype(np.int32)
        jjc = np.clip(np.rint(jj), 0, self.nx - 1).astype(np.int32)
        del ii, jj
        dist = (steps * res)[None, :]
        ang_surf = np.where(inside, (surf[iic, jjc] - h0) / dist, -np.inf).astype(np.float32)
        ang_tgt = ((self.dtm[iic, jjc] + target_h - h0) / dist).astype(np.float32)
        run = np.maximum.accumulate(ang_surf, axis=1)
        vis = np.empty_like(inside)
        vis[:, 0] = inside[:, 0]
        vis[:, 1:] = (ang_tgt[:, 1:] >= run[:, :-1] - 1e-4) & inside[:, 1:]
        out = np.zeros((self.nz, self.nx), bool)
        out[iic[vis], jjc[vis]] = True
        out[int(np.clip(round(oi), 0, self.nz - 1)), int(np.clip(round(oj), 0, self.nx - 1))] = True
        # rays spread apart far away: close 1-cell gaps, but never inside obstructions
        out = ndi.binary_closing(out, np.ones((3, 3)))
        zi = ((np.arange(self.nz) - oi) * res).astype(np.float32)[:, None]
        xj = ((np.arange(self.nx) - oj) * res).astype(np.float32)[None, :]
        dmap = np.sqrt(zi * zi + xj * xj)
        return out & (dmap <= radius), dmap

    def coarsen(self, factor):
        """Aggregated copy at res*factor (dtm mean, dsm max, class by priority) for fast global analyses."""
        f = int(factor)
        if f <= 1:
            return self
        memo = self.__dict__.setdefault("_coarse", {})
        if f in memo:
            return memo[f]
        nz, nx = self.nz // f, self.nx // f

        def blocks(a):
            return a[:nz * f, :nx * f].reshape(nz, f, nx, f).transpose(0, 2, 1, 3).reshape(nz, nx, f * f)

        c = TerrainModel(self.res * f, self.x0, self.z0, nx, nz)
        c.meta = dict(self.meta, coarsened_from=self.res)
        c.dtm = blocks(self.dtm).mean(2)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c.dsm = np.nanmax(blocks(self.dsm), axis=2)
                c.ndsm = c.dsm - c.dtm
                c.rough = np.nanmean(blocks(self.rough), axis=2)
        c.count = blocks(self.count).sum(2)
        c.observed = blocks(self.observed).any(2)
        c.roi = blocks(self.roi).mean(2) >= 0.5
        cb = blocks(self.cls)
        # priority: building > water > vehicle > tree > road > low veg > ground > nodata  (by majority share)
        cls = np.full((nz, nx), NODATA, np.uint8)
        share = {k: (cb == k).mean(2) for k in range(8)}
        cls[share[GROUND] + share[LOW_VEG] + share[ROAD] + share[TREE] + share[BUILDING] + share[WATER] + share[VEHICLE] > 0] = GROUND
        for k, thr in ((LOW_VEG, 0.5), (ROAD, 0.35), (TREE, 0.4), (VEHICLE, 0.3), (WATER, 0.5), (BUILDING, 0.3)):
            cls[share[k] >= thr] = k
        c.cls = cls
        c.slope = c._slope(c.dtm)
        for k in ("rgb", "exg"):
            if hasattr(self, k):
                a = getattr(self, k)
                setattr(c, k, a[:nz * f, :nx * f].reshape(nz, f, nx, f, *a.shape[2:]).mean((1, 3)))
        if hasattr(self, "road"):
            c.road = blocks(self.road).mean(2) >= 0.35
        memo[f] = c
        return c

    @property
    def dsm_filled(self):
        if not hasattr(self, "_dsm_filled"):
            d = np.where(np.isfinite(self.dsm), self.dsm, self.dtm)
            # water / no-return cells are open (no obstruction)
            d = np.where(self.cls == WATER, self.dtm, d)
            self._dsm_filled = d
        return self._dsm_filled

    # ---------------- serialisation ----------------
    _ARRAYS = ("count", "zlo", "zmed", "zhi", "zmax", "rgb", "roi", "observed", "has_points", "dtm", "dsm", "ndsm",
               "slope", "rough", "rough_med", "nd_med", "exg", "sat", "val", "cls", "reflection", "ground_mask")

    def save(self, path, key):
        arrs = {k: getattr(self, k) for k in self._ARRAYS if hasattr(self, k)}
        if hasattr(self, "road"):
            arrs["road"] = self.road
        np.savez_compressed(path, _grid=np.array([self.res, self.x0, self.z0, self.nx, self.nz]),
                            _key=np.array(json.dumps(key)), _meta=np.array(json.dumps(self.meta)), **arrs)

    @classmethod
    def load(cls, path, key):
        z = np.load(path, allow_pickle=False)
        if json.loads(str(z["_key"])) != key:
            return None
        res, x0, z0, nx, nz = z["_grid"]
        tm = cls(res, x0, z0, int(nx), int(nz))
        tm.meta = json.loads(str(z["_meta"]))
        for k in z.files:
            if not k.startswith("_"):
                setattr(tm, k, z[k])
        return tm

    # ---------------- outputs ----------------
    def class_image(self):
        """RGBA uint8 top-down class map (north up) for the dashboard."""
        pal = np.array([[0, 0, 0, 0], [160, 150, 120, 150], [120, 170, 90, 150], [40, 110, 50, 170],
                        [200, 90, 60, 190], [60, 120, 200, 170], [230, 200, 120, 200], [230, 60, 60, 220]], np.uint8)
        return pal[self.cls]

    def summary(self):
        roi = self.roi
        area = float(roi.sum()) * self.res ** 2
        frac = {CLASS_NAMES[c]: round(float(np.mean(self.cls[roi] == c)) * 100, 1) for c in range(1, 8)}
        return {"res_m": self.res, "extent_m": [round(self.nx * self.res, 1), round(self.nz * self.res, 1)],
                "origin": [round(self.x0, 3), round(self.z0, 3)], "analysed_area_m2": round(area, 0),
                "class_percent": frac, "meta": self.meta}


# =============================================================================
# cache-aware accessors
# =============================================================================
def _active_paths(data_dir, baseline_id=None):
    if baseline_id:
        bdir = os.path.join(data_dir, "baselines", baseline_id)
        pts = next((p for p in (os.path.join(bdir, "model_cloud.ply"), os.path.join(bdir, "model.ply")) if os.path.exists(p)), None)
        return pts, os.path.join(bdir, "telemetry.json"), os.path.join(bdir, "terrain_cache.npz")
    models = os.path.join(data_dir, "models")
    pts = next((p for p in (os.path.join(models, "actionable_threat_map_points.ply"),
                            os.path.join(models, "actionable_threat_map_cloud.ply"),
                            os.path.join(models, "actionable_threat_map.ply")) if os.path.exists(p)), None)
    return pts, os.path.join(data_dir, "flight_telemetry.json"), os.path.join(models, "terrain_cache.npz")


def load_telemetry(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def camera_positions(telem, data_dir=DATA_DIR):
    traj = telem.get("camera_trajectory")
    if traj and len(traj) >= 3:
        return np.asarray(traj, float)
    fi = os.path.join(data_dir, "workspace", "frame_index.json")
    if os.path.exists(fi):
        try:
            with open(fi, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if isinstance(entries, list):
                pos = [e["position"] for e in entries if e.get("position") is not None]
                if len(pos) >= 3:
                    return np.asarray(pos, float)
        except Exception:
            pass
    return None


_MEM = {}


def get_terrain(data_dir=DATA_DIR, baseline_id=None, res=0.5, force=False, log=print):
    """Cached TerrainModel of the active model (or a baseline). Returns (terrain, points dict, telemetry)."""
    pts_path, telem_path, cache = _active_paths(data_dir, baseline_id)
    if not pts_path:
        raise FileNotFoundError("No 3D model is deployed yet.")
    st = os.stat(pts_path)
    key = {"v": CACHE_VERSION, "pts": os.path.basename(pts_path), "size": st.st_size, "mtime": int(st.st_mtime), "res": res}
    mk = (pts_path, json.dumps(key))
    telem = load_telemetry(telem_path)
    if not force and mk in _MEM:
        return _MEM[mk]
    d = plyio.read_ply(pts_path)
    pts = {"xyz": d.xyz, "rgb": d.rgb, "normals": d.normals, "path": pts_path}
    tm = None
    if not force and os.path.exists(cache):
        try:
            tm = TerrainModel.load(cache, key)
        except Exception:
            tm = None
    if tm is None:
        tm = TerrainModel.from_points(d.xyz, d.rgb, res=res, cams=camera_positions(telem, data_dir), log=log)
        try:
            from road_engine import detect_roads
        except ImportError:
            from backend.road_engine import detect_roads
        try:
            roads = detect_roads(tm, log=log)
            tm.apply_road_mask(roads["mask"])
            tm.meta["roads"] = roads["summary"]
        except Exception as e:     # road layer is an enhancement; the terrain stays usable
            log(f"[Terrain] road layer skipped: {e}")
        try:
            tm.save(cache, key)
        except Exception as e:
            log(f"[Terrain] cache not written: {e}")
    _MEM.clear()
    _MEM[mk] = (tm, pts, telem)
    return tm, pts, telem


def geo_origin(telem):
    geo = (telem or {}).get("georeference") or {}
    o = geo.get("origin")
    if o and o.get("lat") is not None:
        return (float(o["lat"]), float(o["lon"]), float(o.get("alt") or 0.0))
    return None


def to_latlon(telem, x, y, z):
    """Model (x East, y Up, z South) -> (lat, lon) or (None, None) when the model is not georeferenced."""
    o = geo_origin(telem)
    if o is None:
        return None, None
    try:
        from georeference import enu_to_geodetic
    except ImportError:
        from backend.georeference import enu_to_geodetic
    la, lo, _ = enu_to_geodetic(np.array([x, -z, y], float), o)
    return round(float(la), 7), round(float(lo), 7)
