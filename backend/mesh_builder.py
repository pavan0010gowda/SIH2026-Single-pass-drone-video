"""
PRISM // Gap-free, geometry-faithful surface meshing (mesh_builder.py)
======================================================================
Why the old meshes had gaps or blobs:
  * Ball pivoting only connects points that are close; wherever stereo produced no points (water,
    shadows behind trees, road sections under moving cars, sky-facing walls) it leaves holes.
  * Plain Poisson closes everything, including the empty air around the scene, and melts edges.

This builder combines both worlds:
  1. Scene gating: only points inside the surveyed area (terrain ROI) are meshed; statistical outlier
     removal drops floaters (sky, reflections, moving objects).
  2. Terrain-aware gap filling: holes of the observed ground that are enclosed by data are filled
     with synthetic samples lying on the bare-earth model (roads torn by traffic, shadows, missing
     patches). Water / no-return holes are closed at the lowest bank level as a flat, dark water
     surface. Colours for filled ground come from an in-painted orthophoto, so patches blend in.
  3. Normals are oriented towards the flight path (the side the cameras saw), which is what makes
     Poisson reconstruct walls of oblique views correctly.
  4. Screened Poisson (Kazhdan & Hoppe) with a point weight that keeps edges crisp, adaptive octree depth.
  5. Trimming by support: vertices far from any (real or filled) sample and low-density vertices
     are removed, so no surface balloons into empty air; tiny disconnected islands are dropped.
  6. Colour = inverse-distance weighted average of the 4 nearest samples; optional QEM decimation.
Outputs PLY + OBJ + GLB in the same metric frame as the point cloud.
"""

import math
import os
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    import plyio
    import terrain as T
except ImportError:
    from backend import plyio
    from backend import terrain as T

WATER_RGB = np.array([52, 66, 72], np.uint8)


def _spacing(xyz, n=20000):
    sub = xyz[np.random.default_rng(0).choice(len(xyz), min(n, len(xyz)), replace=False)]
    d, _ = cKDTree(xyz).query(sub, k=2, workers=-1)
    return float(max(np.median(d[:, 1]), 1e-4))


def gap_fill_samples(tm, spacing, max_hole_m2=400.0, log=print):
    """Synthetic surface samples for enclosed holes: ground holes on the DTM, water at bank level."""
    import cv2
    has = getattr(tm, "has_points", tm.observed)
    region = tm.roi & getattr(tm, "enclosed", tm.roi)
    holes = region & ~has
    lab, n = ndi.label(holes, np.ones((3, 3)))
    if not n:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), {}
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    area = cnt * tm.res ** 2
    water_share = np.bincount(lab.ravel(), weights=(tm.cls == T.WATER).ravel().astype(float), minlength=n + 1) / np.maximum(cnt, 1)
    refl = getattr(tm, "reflection", np.zeros_like(holes))
    near_refl = np.bincount(lab.ravel(), weights=ndi.binary_dilation(refl, iterations=2).ravel().astype(float),
                            minlength=n + 1) > 3
    # water = large elongated bodies (canals, rivers) or holes with mirror reflections; everything else
    # (tree / building occlusion shadows, torn road, missing patches) is ground
    idx = np.arange(n + 1)
    ii_all, jj_all = np.nonzero(lab)
    lab_nz = lab[ii_all, jj_all]
    s_i = np.bincount(lab_nz, ii_all, n + 1)
    s_j = np.bincount(lab_nz, jj_all, n + 1)
    s_ii = np.bincount(lab_nz, ii_all.astype(float) ** 2, n + 1)
    s_jj = np.bincount(lab_nz, jj_all.astype(float) ** 2, n + 1)
    s_ij = np.bincount(lab_nz, ii_all.astype(float) * jj_all, n + 1)
    c_ = np.maximum(cnt, 1)
    vii = s_ii / c_ - (s_i / c_) ** 2
    vjj = s_jj / c_ - (s_j / c_) ** 2
    vij = s_ij / c_ - (s_i / c_) * (s_j / c_)
    tr, det = vii + vjj, vii * vjj - vij ** 2
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    elong = np.sqrt(np.maximum(tr / 2 + disc, 1e-9) / np.maximum(tr / 2 - disc, 1e-9))
    is_water = (water_share >= 0.5) & (((area >= 150.0) & (elong >= 4.0)) | (area >= 800.0) | near_refl)
    is_water[0] = False
    fill_water = (lab > 0) & is_water[lab]
    fill_ground = (lab > 0) & ~is_water[lab] & (area[lab] <= max_hole_m2)
    del idx
    # orthophoto in-painting for ground colours
    rgb = np.clip(tm.rgb, 0, 255).astype(np.uint8)
    inpaint_mask = (~has).astype(np.uint8)
    ortho = cv2.inpaint(rgb, inpaint_mask, 3, cv2.INPAINT_TELEA) if inpaint_mask.any() else rgb
    step = max(1, int(round(1.5 * spacing / tm.res))) if spacing > tm.res else 1
    sub = max(1, int(round(tm.res / (1.5 * spacing)))) if spacing < tm.res else 1
    pts, cols = [], []
    rng = np.random.default_rng(1)
    # ground holes: `sub` x `sub` samples per cell on the DTM
    ii, jj = np.nonzero(fill_ground)
    if len(ii):
        off = (np.arange(sub) + 0.5) / sub - 0.5
        oi, oj = [a.ravel() for a in np.meshgrid(off, off, indexing="ij")]
        gi = (ii[:, None] + 0.5 + oi[None, :]).ravel()
        gj = (jj[:, None] + 0.5 + oj[None, :]).ravel()
        x, z = tm.x0 + gj * tm.res, tm.z0 + gi * tm.res
        y = tm.sample(tm.dtm, x, z)
        pts.append(np.c_[x, y, z])
        cols.append(ortho[np.repeat(ii, sub * sub), np.repeat(jj, sub * sub)])
    # water: flat surface at the lowest bank level of each water body
    wl, wn = ndi.label(fill_water, np.ones((3, 3)))
    water_levels = []
    for k in range(1, wn + 1):
        m = wl == k
        ring = ndi.binary_dilation(m, iterations=3) & ~m & has
        if ring.sum() < 5:
            continue
        level = float(np.percentile(tm.zlo[ring], 5)) - 0.15
        water_levels.append(level)
        wi, wj = np.nonzero(m)
        wi, wj = wi[::step], wj[::step]
        x, z = tm.centre(wi, wj)
        pts.append(np.c_[x, np.full(len(x), level), z])
        c = np.tile(WATER_RGB, (len(x), 1)).astype(np.int16) + rng.integers(-4, 5, (len(x), 3))
        cols.append(np.clip(c, 0, 255).astype(np.uint8))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), {}
    p, c = np.vstack(pts), np.vstack(cols)
    info = {"ground_hole_area_m2": round(float(fill_ground.sum()) * tm.res ** 2, 1),
            "water_area_m2": round(float(fill_water.sum()) * tm.res ** 2, 1), "samples": int(len(p)),
            "water_bodies": len(water_levels)}
    log(f"[Mesh] gap filling: {info['ground_hole_area_m2']} m^2 ground holes, {info['water_area_m2']} m^2 water, "
        f"{len(p):,} samples")
    return p, c, info


def orient_normals(xyz, nrm, cams):
    """Flip normals so they face the nearest camera of the flight path (or up without cameras)."""
    if nrm is None:
        return None
    nrm = np.asarray(nrm, np.float64)
    if cams is not None and len(cams) >= 3:
        _, k = cKDTree(cams).query(xyz, workers=-1)
        view = cams[k] - xyz
    else:
        view = np.tile([0.0, 1.0, 0.0], (len(xyz), 1))
    flip = np.einsum("ij,ij->i", nrm, view) < 0
    nrm[flip] *= -1
    return nrm


def _poisson_tile(o3d, xyz, nrm, depth, spacing):
    """Screened Poisson on one tile, trimmed to the sampled surface, islands removed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1,
                                                                              linear_fit=True, n_threads=-1)
    v, f, dens = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(dens)
    if len(f) == 0:
        return v, f
    ext = float(np.max(np.ptp(xyz, axis=0)))
    cell = 1.1 * ext / (2 ** depth)
    tol = 2.5 * max(spacing, cell)
    dist, _ = cKDTree(xyz).query(v, k=1, workers=-1, distance_upper_bound=4 * tol)
    bad = ~(dist < tol) | (dens < np.quantile(dens, 0.02))
    f = f[~bad[f].any(1)]
    if len(f):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(len(v), len(v)))
        _, lab = connected_components(g, directed=False)
        fl = lab[f[:, 0]]
        cnt = np.bincount(fl)
        f = f[cnt[fl] >= max(150, int(0.002 * len(f)))]
    return v, f


def build_mesh(points_path, out_ply, out_obj=None, out_glb=None, tm=None, telem=None, depth=None,
               max_points=1_600_000, target_faces=4_000_000, progress=None, log=print):
    import open3d as o3d
    t0 = time.time()

    def step(pct, msg):
        if progress:
            progress(pct, msg)
        log(f"[Mesh] {msg}")

    step(3, "Loading point cloud")
    d = plyio.read_ply(points_path)
    xyz, rgb, nrm = d.xyz, d.rgb, d.normals
    if rgb is None:
        rgb = np.full((len(xyz), 3), 170, np.uint8)
    cams = T.camera_positions(telem or {}) if telem is not None else None
    # 1. scene gating
    if tm is not None:
        i, j = tm.cell_of(xyz[:, 0], xyz[:, 2])
        ok = tm.inside(i, j)
        keep = np.zeros(len(xyz), bool)
        keep[ok] = tm.roi[i[ok], j[ok]]
        xyz, rgb = xyz[keep], rgb[keep]
        nrm = nrm[keep] if nrm is not None else None
    step(8, f"Cleaning {len(xyz):,} points")
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
        nrm = nrm[sel] if nrm is not None else None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)
    ind = np.asarray(ind)
    xyz, rgb = xyz[ind], rgb[ind]
    nrm = nrm[ind] if nrm is not None else None
    spacing = _spacing(xyz)
    # 2. gap filling
    fill_info = {}
    if tm is not None:
        step(14, "Filling enclosed gaps from the terrain model")
        fp, fc, fill_info = gap_fill_samples(tm, spacing, log=log)
        if len(fp):
            fn = np.tile([0.0, 1.0, 0.0], (len(fp), 1))
            if nrm is None:
                nrm_all = None
            else:
                nrm_all = np.vstack([nrm, fn])
            xyz = np.vstack([xyz, fp])
            rgb = np.vstack([rgb, fc])
            nrm = nrm_all
    # 3. normals
    step(20, "Estimating and orienting normals")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if nrm is None or len(nrm) != len(xyz):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(4 * spacing, 0.3), max_nn=24))
        nrm = np.asarray(pcd.normals)
    nrm = orient_normals(xyz, nrm, cams)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    # 4-5. tiled screened Poisson + support trimming: every tile is solved at full octree depth, so
    # the surface resolution follows the point spacing instead of the (huge) scene extent
    depth = int(depth or 10)
    # Poisson cell ~ 2x the sample spacing (finer than the stereo noise floor, and no decimation needed)
    tile = float(np.clip(2.0 * max(spacing, 0.1) * (2 ** depth) / 1.1, 80.0, 500.0))
    margin = max(8.0, 0.06 * tile)
    lo, hi = xyz[:, [0, 2]].min(0), xyz[:, [0, 2]].max(0)
    nxt, nzt = max(1, int(math.ceil((hi[0] - lo[0]) / tile))), max(1, int(math.ceil((hi[1] - lo[1]) / tile)))
    tree = cKDTree(xyz)
    vs, fs, off = [], [], 0
    n_tiles = nxt * nzt
    for ti in range(nxt):
        for tj in range(nzt):
            k = ti * nzt + tj
            x0, z0 = lo[0] + ti * tile, lo[1] + tj * tile
            x1, z1 = x0 + tile, z0 + tile
            sel = (xyz[:, 0] >= x0 - margin) & (xyz[:, 0] < x1 + margin) & (xyz[:, 2] >= z0 - margin) & (xyz[:, 2] < z1 + margin)
            if sel.sum() < 2000:
                continue
            step(28 + int(50 * k / n_tiles), f"Poisson tile {k + 1}/{n_tiles} ({int(sel.sum()):,} points, depth {depth})")
            v, f = _poisson_tile(o3d, xyz[sel], nrm[sel], depth, spacing)
            if len(f) == 0:
                continue
            cen = v[f].mean(1)
            core = (cen[:, 0] >= x0) & (cen[:, 0] < x1) & (cen[:, 2] >= z0) & (cen[:, 2] < z1)
            f = f[core]
            used = np.zeros(len(v), bool)
            used[f.ravel()] = True
            remap = -np.ones(len(v), np.int64)
            remap[used] = np.arange(used.sum())
            vs.append(v[used])
            fs.append(remap[f] + off)
            off += int(used.sum())
    if not vs:
        raise RuntimeError("Poisson produced no surface (point cloud too sparse).")
    v, f = np.vstack(vs), np.vstack(fs)
    # 6. decimation for the web viewer
    if target_faces and len(f) > target_faces * 1.05:
        step(80, f"Simplifying {len(f):,} -> {target_faces:,} faces (QEM)")
        m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        m2 = m2.simplify_quadric_decimation(int(target_faces))
        m2.remove_unreferenced_vertices()
        v, f = np.asarray(m2.vertices), np.asarray(m2.triangles)
    step(88, "Transferring photographic colours")
    dist, nn = tree.query(v, k=4, workers=-1)
    w = 1.0 / np.maximum(dist, 1e-4) ** 2
    col = (rgb[nn].astype(np.float64) * w[..., None]).sum(1) / w.sum(1)[:, None]
    col = np.clip(np.round(col), 0, 255).astype(np.uint8)
    m3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    m3.compute_vertex_normals()
    vn = np.asarray(m3.vertex_normals).astype(np.float32)
    step(93, "Writing PLY / OBJ / GLB")
    plyio.write_ply(out_ply, v, col, vn, f)
    if out_obj:
        plyio.write_obj(out_obj, v, col, f)
    if out_glb:
        plyio.write_glb(out_glb, v, col, f, vn)
    stats = {"vertices": int(len(v)), "triangles": int(len(f)), "depth": depth, "spacing_m": round(spacing, 3),
             "gap_fill": fill_info, "seconds": round(time.time() - t0, 1), "ply_path": out_ply, "obj_path": out_obj}
    step(100, f"Done: {len(v):,} vertices, {len(f):,} faces in {stats['seconds']} s")
    return stats
