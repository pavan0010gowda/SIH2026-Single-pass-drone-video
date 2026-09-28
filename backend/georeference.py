"""
PRISM // Metric Georeferencing Engine  (georeference.py)
=========================================================
COLMAP reconstructions have NO physical scale and an ARBITRARY orientation
(the world frame is the camera frame of one image, whose +Y axis points DOWN).
Every height computed directly in that frame is meaningless.

This module converts a reconstruction into a METRIC, GRAVITY-ALIGNED frame:

  1. Reads the registered camera poses (images.bin / images.txt).
  2. Time-synchronises every registered frame with the flight log (SRT timestamps,
     or a uniform-time fallback) and converts GPS to local ENU metres (exact WGS-84).
  3. Robust similarity fit (RANSAC + Umeyama) camera centres -> ENU
     => scale, rotation, translation, with uncertainty estimates.
  4. Independent gravity cues from the reconstruction itself:
       a) dominant ground plane (RANSAC), sign fixed by the cameras being above it,
       b) horizon constraint: gimbal-stabilised cameras keep their image x-axis level,
       c) viewing-direction sign (drone cameras look down / level, never up),
     fused with the GPS cue by anisotropic weighted least squares on the unit sphere.
     This also solves straight-line (collinear) flights, where GPS alone cannot fix roll.
  5. Fallbacks without usable GPS: scale from flight altitude above ground (AGL).
  6. Ground levelled to height 0 and exported in a Y-up metric frame
     (x = East, y = Up, z = South)  -> any Y-up viewer shows the model upright.

Shared utilities (WGS-84 geodesy, PLY I/O that works with or without Open3D,
COLMAP readers) are also used by change_detector.py and pipeline_manager.py.
"""

import os
import re
import math
import struct

import numpy as np
from scipy.spatial import cKDTree

# =============================================================================
# CONFIGURATION
# =============================================================================
# True  -> the deployed model (actionable_threat_map.ply) is written in metres, Y-up.
#          metric_scale_factor becomes 1.0 (calipers read metres directly).
# False -> the deployed model stays in raw COLMAP coordinates (legacy display);
#          metric_scale_factor is set to the true scale. Height maths is identical.
EXPORT_DISPLAY_MODEL_METRIC = True

GPS_NOISE_FLOOR_M = 1.5        # typical consumer GNSS horizontal noise (1 sigma)
MIN_GPS_TRAVEL_M = 20.0        # GPS travel must dwarf GNSS noise before it can give scale
MIN_REGISTERED_CAMERAS = 3
DEFAULT_ASSUMED_ALTITUDE_M = 25.0

# ENU (x=E, y=N, z=U)  <->  Y-up display frame (x=E, y=U, z=-N = South). Proper rotations.
YUP_FROM_ENU = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])
ENU_FROM_YUP = YUP_FROM_ENU.T


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _kd_query(tree, x, **kw):
    try:
        return tree.query(x, workers=-1, **kw)
    except TypeError:  # SciPy < 1.6
        return tree.query(x, **kw)


# =============================================================================
# 1. WGS-84 GEODESY (exact, no flat-earth approximations)
# =============================================================================
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(h_m, dtype=float)
    sl, cl = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.stack([(n + h) * cl * np.cos(lon),
                     (n + h) * cl * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + h) * sl], axis=-1)


def ecef_to_geodetic(xyz):
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sl = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        lat = np.arctan2(z + WGS84_E2 * n * sl, p)
    sl, cl = np.sin(lat), np.cos(lat)
    h = p * cl + z * sl - WGS84_A * np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.degrees(lat), np.degrees(lon), h


def ecef_to_enu_rotation(lat0_deg, lon0_deg):
    la, lo = math.radians(lat0_deg), math.radians(lon0_deg)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def geodetic_to_enu(lat, lon, h, origin):
    lat0, lon0, h0 = origin
    r = ecef_to_enu_rotation(lat0, lon0)
    d = geodetic_to_ecef(lat, lon, h) - geodetic_to_ecef(lat0, lon0, h0)
    return d @ r.T


def enu_to_geodetic(enu, origin):
    lat0, lon0, h0 = origin
    r = ecef_to_enu_rotation(lat0, lon0)
    ecef = np.asarray(enu, dtype=float) @ r + geodetic_to_ecef(lat0, lon0, h0)
    return ecef_to_geodetic(ecef)


def enu_to_enu_transform(origin_from, origin_to):
    """4x4 rigid transform taking ENU coordinates of origin_from into ENU of origin_to."""
    ra = ecef_to_enu_rotation(origin_from[0], origin_from[1])
    rb = ecef_to_enu_rotation(origin_to[0], origin_to[1])
    ea = geodetic_to_ecef(*origin_from)
    eb = geodetic_to_ecef(*origin_to)
    t = np.eye(4)
    t[:3, :3] = rb @ ra.T
    t[:3, 3] = rb @ (ea - eb)
    return t


def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


# =============================================================================
# 2. POINT CLOUD I/O  (Open3D when available, pure-NumPy PLY fallback otherwise)
# =============================================================================
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
              "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _read_ply_numpy(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (no end_header): {path}")
            s = line.decode("ascii", errors="ignore").strip()
            header.append(s)
            if s == "end_header":
                break
        fmt, elements = None, []
        for s in header:
            parts = s.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append([parts[1], int(parts[2]), []])
            elif parts[0] == "property" and elements:
                if parts[1] == "list":
                    elements[-1][2].append((parts[-1], "list"))
                else:
                    elements[-1][2].append((parts[2], parts[1]))
        if not elements or elements[0][0] != "vertex":
            raise ValueError("PLY vertex element must come first")
        _, count, props = elements[0]
        if any(t == "list" for _, t in props):
            raise ValueError("List properties inside the vertex element are not supported")
        if fmt == "ascii":
            rows = []
            for _ in range(count):
                rows.append(f.readline().decode("ascii", errors="ignore").split()[:len(props)])
            data = np.asarray(rows, dtype=float).reshape(count, len(props))
            cols = {p: data[:, i] for i, (p, _) in enumerate(props)}
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(p, endian + _PLY_TYPES[t]) for p, t in props])
            arr = np.frombuffer(f.read(dt.itemsize * count), dtype=dt, count=count)
            cols = {p: arr[p] for p, _ in props}
    pts = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    normals = None
    if all(k in cols for k in ("nx", "ny", "nz")):
        normals = np.stack([cols["nx"], cols["ny"], cols["nz"]], axis=1).astype(np.float64)
    colors = None
    for keys in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if all(k in cols for k in keys):
            c = np.stack([cols[k] for k in keys], axis=1).astype(np.float64)
            if np.issubdtype(cols[keys[0]].dtype, np.integer) or c.max() > 1.0:
                c = c / 255.0
            colors = np.clip(c, 0.0, 1.0)
            break
    return pts, normals, colors


def _write_ply_numpy(path, pts, normals=None, colors=None):
    n = len(pts)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if normals is not None:
        arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if colors is not None:
        c = np.clip(np.round(np.asarray(colors) * 255.0), 0, 255).astype(np.uint8)
        arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    head += [f"property {'float' if t == '<f4' else 'uchar'} {name}" for name, t in fields]
    head += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(arr.tobytes())


def load_point_cloud(path):
    """Returns (points Nx3, normals Nx3 | None, colors Nx3 in [0,1] | None)."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float64)
        if len(pts) > 0:
            normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
            colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
            return pts, normals, colors
    except ImportError:
        pass
    except Exception:
        pass
    return _read_ply_numpy(path)


def save_point_cloud(path, points, normals=None, colors=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, dtype=np.float64), 0, 1))
        if o3d.io.write_point_cloud(path, pcd):
            return path
    except ImportError:
        pass
    _write_ply_numpy(path, np.asarray(points), normals, colors)
    return path


# =============================================================================
# 3. COLMAP MODEL READERS
# =============================================================================
def qvec_to_rotmat(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def _pose_record(qvec, tvec, camera_id=None):
    r = qvec_to_rotmat(qvec)          # world -> camera
    t = np.asarray(tvec, dtype=float)
    return {"R": r, "t": t, "C": -r.T @ t, "camera_id": camera_id}


_CAM_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
               4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
               8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


def read_colmap_cameras(model_dir):
    """dict camera_id -> {model, width, height, params, fy}  (cameras.bin or cameras.txt)."""
    out = {}
    bin_path, txt_path = os.path.join(model_dir, "cameras.bin"), os.path.join(model_dir, "cameras.txt")
    try:
        if os.path.exists(bin_path):
            with open(bin_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
                    name, npar = _CAM_MODELS.get(mid, ("UNKNOWN", 0))
                    params = struct.unpack("<" + "d" * npar, f.read(8 * npar)) if npar else ()
                    out[cid] = {"model": name, "width": w, "height": h, "params": list(params)}
        elif os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if ln.startswith("#") or not ln.strip():
                        continue
                    p = ln.split()
                    out[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                                      "params": [float(v) for v in p[4:]]}
    except Exception:
        return {}
    for c in out.values():
        pr = c["params"]
        two_focal = c["model"] in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE")
        c["fy"] = float(pr[1] if two_focal and len(pr) > 1 else (pr[0] if pr else 0.0))
    return out


def _read_images_bin(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            props = struct.unpack("<idddddddi", f.read(64))
            name = bytearray()
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2d, 1)
            q = np.array(props[1:5])
            if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.0) > 0.05:
                raise ValueError("Unexpected images.bin layout")
            out[name.decode("utf-8", errors="ignore")] = _pose_record(q, props[5:8], int(props[8]))
    return out


def _read_images_txt(path):
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            try:
                q = [float(v) for v in parts[1:5]]
                t = [float(v) for v in parts[5:8]]
                name = " ".join(parts[9:])
                out[name] = _pose_record(q, t, int(parts[8]))
                i += 2          # skip the POINTS2D line
                continue
            except ValueError:
                pass
        i += 1
    return out


def read_colmap_images(model_dir):
    """dict: image name -> {'R': world->cam 3x3, 't': 3, 'C': camera centre 3}."""
    bin_path = os.path.join(model_dir, "images.bin")
    txt_path = os.path.join(model_dir, "images.txt")
    if os.path.exists(bin_path):
        try:
            return _read_images_bin(bin_path)
        except Exception:
            pass
    if os.path.exists(txt_path):
        return _read_images_txt(txt_path)
    return {}


def read_colmap_points3d(model_dir):
    """Sparse points (xyz, rgb in [0,1]) from points3D.bin / points3D.txt; used if dense fails."""
    bin_path = os.path.join(model_dir, "points3D.bin")
    txt_path = os.path.join(model_dir, "points3D.txt")
    xyz, rgb = [], []
    if os.path.exists(bin_path):
        with open(bin_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                vals = struct.unpack("<QdddBBBd", f.read(43))
                track_len = struct.unpack("<Q", f.read(8))[0]
                f.seek(8 * track_len, 1)
                xyz.append(vals[1:4])
                rgb.append(vals[4:7])
    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("#"):
                    continue
                p = ln.split()
                if len(p) >= 7:
                    xyz.append([float(v) for v in p[1:4]])
                    rgb.append([float(v) for v in p[4:7]])
    if not xyz:
        return np.zeros((0, 3)), None
    return np.asarray(xyz, dtype=float), np.asarray(rgb, dtype=float) / 255.0


# =============================================================================
# 4. ROBUST GEOMETRY
# =============================================================================
def _normalize(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-15)


def rotation_between(a, b):
    """Minimal rotation matrix taking direction a onto direction b."""
    a, b = _normalize(a), _normalize(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:
        axis = _normalize(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_normalize(a), _normalize(b)), -1.0, 1.0))))


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity: dst ~= s * R @ src + t (Umeyama 1991)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    n = len(src)
    u, d, vt = np.linalg.svd(xd.T @ xs / n)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    r = u @ s_mat @ vt
    var_s = (xs ** 2).sum() / n
    s = float((d * np.diag(s_mat)).sum() / var_s) if (with_scale and var_s > 0) else 1.0
    return s, r, mu_d - s * r @ mu_s


def ransac_similarity(src, dst, rng, iters=500, init_thresh=6.0):
    """RANSAC + iterative re-weighting Umeyama. Returns s, R, t, inlier mask, inlier RMSE."""
    n = len(src)
    best, best_cnt, best_med = None, -1, np.inf
    for _ in range(min(iters, max(20, n * (n - 1) * (n - 2) // 6))):
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.norm(np.ptp(src[idx], axis=0)) < 1e-9:
            continue
        s, r, t = umeyama(src[idx], dst[idx])
        if not np.isfinite(s) or s <= 0:
            continue
        res = np.linalg.norm(dst - (s * src @ r.T + t), axis=1)
        inl = res < init_thresh
        cnt = int(inl.sum())
        med = float(np.median(res[inl])) if cnt else np.inf
        if cnt > best_cnt or (cnt == best_cnt and med < best_med):
            best, best_cnt, best_med = inl, cnt, med
    inl = best if (best is not None and best.sum() >= 3) else np.ones(n, bool)
    for _ in range(6):
        s, r, t = umeyama(src[inl], dst[inl])
        res = np.linalg.norm(dst - (s * src @ r.T + t), axis=1)
        thr = max(2.5 * GPS_NOISE_FLOOR_M, 2.5 * float(np.median(res[inl])))
        new = res < thr
        if new.sum() < 3 or np.array_equal(new, inl):
            break
        inl = new
    rmse = float(np.sqrt(np.mean(res[inl] ** 2)))
    return s, r, t, inl, rmse


def fit_plane_lsq(p):
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    return n, -float(n @ c)


def ransac_ground_plane(pts, cams, thresh, rng, iters=500):
    """Dominant plane that has (almost) all cameras on one side -> ground. Normal points to cameras."""
    sample = pts if len(pts) <= 30000 else pts[rng.choice(len(pts), 30000, replace=False)]
    m = len(sample)
    tri = rng.integers(0, m, size=(iters, 3))
    p0, p1, p2 = sample[tri[:, 0]], sample[tri[:, 1]], sample[tri[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    nrm, p0 = nrm[ok] / ln[ok, None], p0[ok]
    d = -(nrm * p0).sum(1)
    side = cams @ nrm.T + d
    frac = np.maximum((side > 0).mean(0), (side < 0).mean(0))
    nrm, d = nrm[frac >= 0.9], d[frac >= 0.9]
    if len(nrm) == 0:
        return None
    best_cnt, best = -1, None
    for s in range(0, len(nrm), 64):
        cnt = (np.abs(sample @ nrm[s:s + 64].T + d[s:s + 64]) < thresh).sum(0)
        j = int(cnt.argmax())
        if cnt[j] > best_cnt:
            best_cnt, best = cnt[j], (nrm[s + j], d[s + j])
    n, d0 = best
    for th in (thresh, 0.6 * thresh):
        inl = np.abs(pts @ n + d0) < th
        if inl.sum() < 10:
            break
        n, d0 = fit_plane_lsq(pts[inl])
    if np.median(cams @ n + d0) < 0:
        n, d0 = -n, -d0
    return n, float(d0), float((np.abs(pts @ n + d0) < thresh).mean())


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def _tangent_basis(u):
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _normalize(np.cross(u, a))
    return e1, np.cross(u, e1)


class UpVectorCost:
    """Sum of squared angular residuals (radians^2 / sigma^2) of independent gravity cues."""

    def __init__(self):
        self.terms, self.names = [], []

    def add(self, name, fn):
        self.names.append(name)
        self.terms.append(fn)

    def __call__(self, u):
        u = np.atleast_2d(u)
        total = np.zeros(len(u))
        for fn in self.terms:
            total += fn(u)
        return total


def minimize_on_sphere(cost, seeds=(), n_grid=12000):
    from scipy.optimize import minimize
    grid = _fibonacci_sphere(n_grid)
    vals = np.concatenate([cost(grid[i:i + 1500]) for i in range(0, n_grid, 1500)])
    cands = [grid[int(np.argmin(vals))]] + [_normalize(s) for s in seeds]
    best_val, best_u = np.inf, cands[0]
    for u0 in cands:
        e1, e2 = _tangent_basis(u0)

        def f(ab, u0=u0, e1=e1, e2=e2):
            return float(cost(_normalize(u0 + ab[0] * e1 + ab[1] * e2))[0])

        res = minimize(f, [0.0, 0.0], method="Nelder-Mead",
                       options={"initial_simplex": [[0, 0], [0.02, 0], [0, 0.02]],
                                "xatol": 1e-7, "fatol": 1e-12, "maxiter": 600})
        u = _normalize(u0 + res.x[0] * e1 + res.x[1] * e2)
        if res.fun < best_val:
            best_val, best_u = res.fun, u
    return best_u, float(best_val)


def camera_heights_above_ground(pz, cz):
    """Per-camera height above the local ground (same units as inputs, Z-up frame)."""
    g0 = float(np.percentile(pz[:, 2], 20))
    h0 = float(np.median(cz[:, 2] - g0))
    if not h0 > 0:
        return None
    tree = cKDTree(pz[:, :2])
    out = np.full(len(cz), np.nan)
    for i, idx in enumerate(tree.query_ball_point(cz[:, :2], 0.25 * h0)):
        if len(idx) >= 30:
            out[i] = cz[i, 2] - np.percentile(pz[idx, 2], 15)
    if np.isfinite(out).sum() < max(3, 0.2 * len(cz)):   # oblique views: ground below not visible
        out = np.where(np.isfinite(out), out, cz[:, 2] - g0)
    return out


def apply_transform(t4, pts):
    return np.asarray(pts, float) @ t4[:3, :3].T + t4[:3, 3]


def transform_normals(t4, nrm):
    a = t4[:3, :3]
    r = a / np.cbrt(np.linalg.det(a))
    n = np.asarray(nrm, float) @ r.T
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# =============================================================================
# 5. FLIGHT-LOG <-> FRAME SYNCHRONISATION
# =============================================================================
def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def has_real_gps(telemetry):
    if not isinstance(telemetry, dict) or not telemetry.get("waypoints"):
        return False
    if telemetry.get("is_synthetic"):
        return False
    return not str(telemetry.get("source", "")).upper().startswith("SYNTHETIC")


def _telemetry_arrays(telemetry, swap_latlon=False):
    wps = telemetry.get("waypoints") or []
    lat = np.array([_f(w.get("latitude")) for w in wps])
    lon = np.array([_f(w.get("longitude")) for w in wps])
    if swap_latlon:
        lat, lon = lon, lat
    alt_ref = str(telemetry.get("altitude_reference", "relative")).lower()
    key = "absolute_altitude_m" if alt_ref == "absolute" else "relative_altitude_m"
    alt = np.array([_f(w.get(key, w.get("relative_altitude_m"))) for w in wps])
    t = np.array([_f(w.get("time_s")) for w in wps])
    ok = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
          & ~((np.abs(lat) < 1e-9) & (np.abs(lon) < 1e-9)))
    lat, lon, alt, t = lat[ok], lon[ok], alt[ok], t[ok]
    if len(lat) < 3:
        return None
    if telemetry.get("altitude_is_estimated") or not np.isfinite(alt).any():
        alt, alt_ref = np.zeros_like(lat), "estimated"
    elif not np.isfinite(alt).all():
        idx = np.arange(len(alt))
        good = np.isfinite(alt)
        alt = np.interp(idx, idx[good], alt[good])
    has_t = bool(np.isfinite(t).all() and t[-1] > t[0] and np.all(np.diff(t) >= 0))
    return {"lat": lat, "lon": lon, "alt": alt, "t": t if has_t else None, "alt_ref": alt_ref}


def _fresh(v):
    """Keep the first sample of every constant run (removes GPS 'staircase' repeats)."""
    v = np.asarray(v).reshape(len(v), -1)
    m = np.ones(len(v), bool)
    m[1:] = np.any(np.abs(np.diff(v, axis=0)) > 1e-12, axis=1)
    return m


def _interp_fresh(q, x, *series):
    out = []
    for ser in series:
        m = _fresh(ser if ser.ndim > 1 else ser[:, None])
        if m.sum() < 2:
            out.append(np.full((len(q),) + ser.shape[1:], ser[0]))
        elif ser.ndim > 1:
            out.append(np.stack([np.interp(q, x[m], ser[m][:, k]) for k in range(ser.shape[1])], axis=1))
        else:
            out.append(np.interp(q, x[m], ser[m]))
    return out


def _trailing_int(name):
    m = re.findall(r"(\d+)", os.path.basename(str(name)))
    return int(m[-1]) if m else None


def associate_gps(names, telemetry, frame_info, swap_latlon=False):
    """GPS/altitude at the capture instant of each registered image, as local ENU metres."""
    if not has_real_gps(telemetry):
        return None
    arr = _telemetry_arrays(telemetry, swap_latlon)
    if arr is None:
        return None
    if isinstance(frame_info, list):          # pose-list format (bundle / post-georeference frame_index.json)
        frame_info = {"frames": {f.get("filename"): {"time_s": f.get("timestamp_sec")}
                                 for f in frame_info if isinstance(f, dict) and f.get("filename")}}
    frames = (frame_info or {}).get("frames") or {}
    tq = np.array([_f((frames.get(n) or {}).get("time_s")) for n in names])
    latlon = np.c_[arr["lat"], arr["lon"]]
    if arr["t"] is not None and len(tq) and np.isfinite(tq).all():
        t = arr["t"]
        sync = "timestamp"
        valid = (tq >= t[0] - 0.5) & (tq <= t[-1] + 0.5)
        ll, alt = _interp_fresh(np.clip(tq, t[0], t[-1]), t, latlon, arr["alt"])
    else:
        fq = np.array([_f((frames.get(n) or {}).get("fraction")) for n in names])
        if not np.isfinite(fq).all():
            dur = _f((frame_info or {}).get("duration_s"))
            if np.isfinite(tq).all() and np.isfinite(dur) and dur > 0:
                fq = tq / dur
            else:
                k = np.array([_f(_trailing_int(n)) for n in names])
                total = _f((frame_info or {}).get("total_saved_frames"))
                kmax = total - 1 if np.isfinite(total) and total > 1 else np.nanmax(k)
                fq = k / kmax if (np.isfinite(k).all() and kmax > 0) else np.linspace(0, 1, len(names))
        sync = "uniform_fraction"
        valid = np.isfinite(fq)
        param = np.linspace(0.0, 1.0, len(arr["lat"]))
        ll, alt = _interp_fresh(np.clip(np.nan_to_num(fq), 0, 1), param, latlon, arr["alt"])
    origin = (float(np.median(ll[:, 0])), float(np.median(ll[:, 1])), float(np.median(alt)))
    enu = geodetic_to_enu(ll[:, 0], ll[:, 1], alt, origin)
    return {"enu": enu, "valid": valid & np.isfinite(enu).all(1), "origin": origin,
            "sync": sync, "alt_ref": arr["alt_ref"], "alt": alt}


def _alt_is_agl_like(gps):
    """True when the logged altitude can stand in for height above the take-off ground."""
    if gps is None:
        return False
    if gps["alt_ref"] == "relative":
        return True
    if gps["alt_ref"] == "gps_altitude":
        med = float(np.nanmedian(gps["alt"])) if len(gps["alt"]) else np.nan
        return bool(np.isfinite(med) and 2.0 < med < 500.0)
    return False


def _quick_sim3_rmse(c_src, gps, rng):
    """RMSE (m) of a robust similarity camera centres -> GPS ENU, and the implied up vector."""
    m = gps["valid"]
    if m.sum() < 5:
        return np.inf, None
    s, r, t, inl, rmse = ransac_similarity(c_src[m], gps["enu"][m], rng, iters=300)
    if not (np.isfinite(s) and s > 0):
        return np.inf, None
    res = np.linalg.norm(gps["enu"][m] - (s * c_src[m] @ r.T + t), axis=1)
    return float(np.sqrt(np.mean(res ** 2))), r.T @ np.array([0.0, 0.0, 1.0])


def resolve_latlon_order(names, cam_centres, telemetry, frame_info, rng, diag=None):
    """
    Decides the latitude/longitude order of an ambiguous flight log with the camera trajectory.

    Exchanging latitude and longitude mirrors the GPS track (and stretches it by cos(lat)), so only
    the true order can be matched by a proper similarity transform. Both orders are fitted and the
    one with the lower residual wins; a clear winner needs <= 60 % of the other RMSE.
    Returns (gps association for the chosen order, swapped?).
    """
    gps = associate_gps(names, telemetry, frame_info)
    if gps is None or not (telemetry or {}).get("latlon_ambiguous"):
        return gps, False, True
    alt = associate_gps(names, telemetry, frame_info, swap_latlon=True)
    if alt is None:
        return gps, False, False
    r0, _ = _quick_sim3_rmse(cam_centres, gps, rng)
    r1, _ = _quick_sim3_rmse(cam_centres, alt, rng)
    if diag is not None:
        diag["latlon_test_rmse_m"] = {"as_logged": round(r0, 3) if np.isfinite(r0) else None,
                                      "swapped": round(r1, 3) if np.isfinite(r1) else None}
    if np.isfinite(r1) and r1 < 0.6 * r0:
        if diag is not None:
            diag["latlon_order_resolved"] = "swapped"
        return alt, True, True
    decided = bool(np.isfinite(r0) and r0 < 0.6 * r1)
    if diag is not None:
        diag["latlon_order_resolved"] = "as_logged" if decided else "undecided"
    return gps, False, decided


def levelled_gps_fit(cam, gps, r_lvl):
    """
    4-DOF similarity (scale, heading, translation) between gravity-levelled camera centres and GPS ENU,
    with iterative outlier rejection. Robust for straight flights and immune to the mirrored solutions a
    free 3-D fit can produce on nearly planar / linear tracks. Returns (s, R, t, inliers, rmse_h).
    """
    m = gps["valid"]
    p = cam[m] @ r_lvl.T
    e = gps["enu"][m]
    inl = np.ones(len(p), bool)
    for _ in range(8):
        s, r2, t2 = umeyama_2d(p[inl, :2], e[inl, :2])
        res = np.linalg.norm(e[:, :2] - (s * p[:, :2] @ r2.T + t2), axis=1)
        new = res < max(2.5 * GPS_NOISE_FLOOR_M, 3.0 * float(np.median(res[inl])))
        if new.sum() < 4 or np.array_equal(new, inl):
            break
        inl = new
    tz = float(np.median(e[inl, 2] - s * p[inl, 2]))
    r3 = np.eye(3)
    r3[:2, :2] = r2
    full_inl = np.zeros(len(m), bool)
    full_inl[np.flatnonzero(m)[inl]] = True
    return s, r3 @ r_lvl, np.array([t2[0], t2[1], tz]), full_inl, float(np.sqrt(np.mean(res[inl] ** 2)))


def umeyama_2d(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[1, 1] = -1
    r = u @ sgn @ vt
    var = (a ** 2).sum() / len(src)
    s = float((sv * np.diag(sgn)).sum() / var) if var > 0 else 1.0
    return s, r, md - s * r @ ms


def estimate_ground_height(h):
    """Lowest strong mode of the height histogram: a robust 'ground level' for nadir AND oblique views."""
    h = np.asarray(h, float)
    h = h[np.isfinite(h)]
    if len(h) < 20:
        return float(np.percentile(h, 5)) if len(h) else 0.0
    lo, hi = np.percentile(h, [0.5, 75])
    if hi - lo < 1e-9:
        return float(lo)
    hist, edges = np.histogram(h[(h >= lo) & (h <= hi)], bins=240, range=(lo, hi))
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    sm = np.convolve(hist.astype(float), k / k.sum(), mode="same")
    i = int(np.argmax(sm >= 0.3 * sm.max()))
    while i + 1 < len(sm) and sm[i + 1] >= sm[i]:
        i += 1
    return float(0.5 * (edges[i] + edges[i + 1]))


# =============================================================================
# 6. MAIN: METRIC, GRAVITY-ALIGNED GEOREFERENCING
# =============================================================================
def georeference_reconstruction(model_dir, raw_points, telemetry=None, frame_info=None,
                                assumed_altitude_m=None, log=print, seed=0):
    """
    Returns a dict with status 'ok' and 'transform_raw_to_metric_yup' (4x4), or status 'failed'.
    Frame produced: metres, x = East, y = Up, z = South, ground ~ y = 0.
    """
    rng = np.random.default_rng(seed)
    warnings, diag = [], {}
    cams = read_colmap_images(model_dir) if model_dir else {}
    if len(cams) < MIN_REGISTERED_CAMERAS:
        return {"status": "failed", "reason": f"Only {len(cams)} registered camera poses found in '{model_dir}'."}
    names = sorted(cams, key=natural_key)
    cc = np.array([cams[n]["C"] for n in names])
    rw = np.array([cams[n]["R"] for n in names])
    pts = np.asarray(raw_points, dtype=float)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 200:
        return {"status": "failed", "reason": "Point cloud too small for georeferencing."}
    if len(pts) > 300000:
        pts = pts[rng.choice(len(pts), 300000, replace=False)]
    dcs, _ = _kd_query(cKDTree(cc), pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)], k=1)
    view_dist = float(np.median(dcs))
    diag.update(registered_cameras=len(names), view_distance_units=round(view_dist, 5))

    # ---------------- GPS similarity (scale + rotation + translation) ----------------
    gps, swap, order_decided = resolve_latlon_order(names, cc, telemetry, frame_info, rng, diag)
    if swap:
        warnings.append("Flight log writes GPS(lon, lat): latitude/longitude order corrected using the "
                        "camera trajectory.")
    fit = None
    if gps is not None:
        m = gps["valid"]
        diag.update(gps_matched_cameras=int(m.sum()), time_sync=gps["sync"], altitude_reference=gps["alt_ref"])
        if m.sum() >= 5:
            pg, cg = gps["enu"][m], cc[m]
            _, sv, vt = np.linalg.svd(pg - pg.mean(0), full_matrices=False)
            spread = sv / np.sqrt(len(pg))
            travel = float(np.linalg.norm(np.ptp(pg, axis=0)))
            diag.update(gps_travel_m=round(travel, 2), gps_spread_m=[round(float(v), 2) for v in spread])
            s0 = None
            enough = travel >= MIN_GPS_TRAVEL_M and spread[0] >= 3.5 * GPS_NOISE_FLOOR_M
            if enough:
                s0, r0, t0, inl, rmse = ransac_similarity(cg, pg, rng)
                if not (s0 > 0) or rmse > max(15.0, 0.5 * travel) or inl.sum() < 5:
                    warnings.append(f"GPS/camera fit rejected (RMSE {rmse:.1f} m): check time sync of the log.")
                    s0 = None
            if enough and s0 is not None:
                fit = {"s": s0, "R": r0, "inl": inl, "rmse": rmse, "spread": spread, "dirs": vt,
                       "n_eff": int(min(inl.sum(), 10)), "pg": pg, "cg": cg}
                diag.update(gps_inliers=int(inl.sum()), gps_fit_rmse_m=round(rmse, 3))
            elif not enough:
                warnings.append(f"GPS track too short ({travel:.1f} m) compared with GNSS noise; "
                                "using altitude for scale instead.")
        else:
            warnings.append("Too few frames could be time-matched to the flight log.")

    # ---------------- Gravity cues from the reconstruction ----------------
    cost = UpVectorCost()
    seeds = []
    plane = ransac_ground_plane(pts, cc, 0.03 * view_dist, rng)
    if plane is not None:
        n_pl, _, frac_pl = plane
        sig_pl = math.radians(5.0 if frac_pl >= 0.35 else (10.0 if frac_pl >= 0.15 else 25.0))
        cost.add("ground_plane", lambda u, n=n_pl, s=sig_pl: (np.arccos(np.clip(u @ n, -1, 1)) / s) ** 2)
        seeds.append(n_pl)
        diag["ground_plane_inlier_fraction"] = round(frac_pl, 3)
    x_ax, y_ax, z_ax = rw[:, 0, :], rw[:, 1, :], rw[:, 2, :]
    sig_h = math.radians(2.0)
    keep_h = max(1, int(0.8 * len(x_ax)))

    def horizon(u, x=x_ax, k=keep_h, s=sig_h):
        a = np.arcsin(np.clip(np.abs(x @ u.T), 0, 1)) ** 2
        return np.partition(a, k - 1, axis=0)[:k].mean(0) / s ** 2

    cost.add("horizon", horizon)
    look = y_ax + z_ax      # drone cameras look down/level: (y+z).up = -(sin p + cos p) <= -1
    cost.add("view_sign", lambda u, l=look: 100.0 * np.maximum(0.0, (u @ l.T).mean(1) + 0.5) ** 2)
    if fit is not None:
        floor = math.radians(5.0 if gps["alt_ref"] == "estimated" else 0.5)
        sp = fit["spread"]
        noise = max(fit["rmse"], GPS_NOISE_FLOOR_M)
        fit["active_axes"] = 0
        for k in range(2):
            if sp[k] < 2.5 * noise:      # this track direction is GNSS noise, not motion -> no information
                continue
            fit["active_axes"] += 1
            d_k = fit["dirs"][k]
            e_k = fit["R"].T @ d_k
            c_k = float(np.clip(d_k[2], -1, 1))
            # tilt towards d_k is fixed by the track spread along d_k (and vertically)
            perp = math.sqrt(sp[k] ** 2 + sp[2] ** 2)
            sig = max(floor, math.atan2(fit["rmse"], max(perp, 1e-6) * math.sqrt(fit["n_eff"])))
            cost.add(f"gps_axis_{k}", lambda u, e=e_k, c=c_k, s=sig:
                     ((np.arccos(np.clip(u @ e, -1, 1)) - math.acos(c)) / s) ** 2)
        seeds.append(fit["R"].T @ np.array([0.0, 0.0, 1.0]))
    up, _ = minimize_on_sphere(cost, seeds=seeds)
    if (look @ up).mean() > 0:      # safety: never accept an upside-down solution
        up = -up
    diag["up_vector_raw"] = [round(float(v), 6) for v in up]
    if plane is not None:
        diag["angle_up_vs_ground_plane_deg"] = round(angle_deg(up, plane[0]), 2)
    if fit is not None:
        a = angle_deg(up, fit["R"].T @ np.array([0.0, 0.0, 1.0]))
        diag["angle_up_vs_gps_deg"] = round(a, 2)
        if a > 60.0 and fit["active_axes"] == 2:
            warnings.append("Flight-log vertical disagrees with the scene geometry (check telemetry axes).")

    # ---------------- Rotation, scale, translation ----------------
    ez = np.array([0.0, 0.0, 1.0])
    origin, heading_known, scale_rel_unc = None, False, None
    if fit is not None:
        # gravity is fixed by `up`; scale, heading and offset come from a levelled 4-DOF fit (robust for
        # straight flights, and it cannot inherit a mirrored heading from the free 3-D fit)
        r_lvl = rotation_between(up, ez)
        s, r_final, t, inl_all, rmse = levelled_gps_fit(cc, gps, r_lvl)
        alt_order = associate_gps(names, telemetry, frame_info, swap_latlon=not swap) \
            if (telemetry or {}).get("latlon_ambiguous") and not order_decided else None
        if alt_order is not None and _alt_is_agl_like(gps):
            # straight flight: geometry cannot tell the lat/lon order, but swapping them rescales east-west
            # distances by cos(latitude); only the true order makes GPS scale agree with the altitude scale
            agl_u = camera_heights_above_ground(pts @ r_lvl.T, cc @ r_lvl.T)
            if agl_u is not None:
                okm = gps["valid"] & np.isfinite(agl_u) & (agl_u > 0) & (gps["alt"] > 5.0)
                if okm.sum() >= 3:
                    s_alt = float(np.median(gps["alt"][okm] / agl_u[okm]))
                    s2, r2_, t2_, inl2, rmse2 = levelled_gps_fit(cc, alt_order, r_lvl)
                    d1, d2 = abs(math.log(s / s_alt)), abs(math.log(s2 / s_alt))
                    diag["latlon_altitude_test"] = {"scale_as_used": round(s, 5), "scale_swapped": round(s2, 5),
                                                    "scale_from_altitude": round(s_alt, 5)}
                    if d2 < 0.5 * d1 and d2 < math.log(1.25):
                        gps, swap = alt_order, not swap
                        s, r_final, t, inl_all, rmse = s2, r2_, t2_, inl2, rmse2
                        diag["latlon_order_resolved"] = "swapped (altitude test)"
                        warnings.append("Latitude/longitude order decided from the altitude-based scale "
                                        "(straight flight path).")
                    elif d1 < 0.5 * d2 and d1 < math.log(1.25):
                        diag["latlon_order_resolved"] = "as_logged (altitude test)"
        pg, cg = gps["enu"][gps["valid"]], cc[gps["valid"]]
        inl = inl_all[gps["valid"]]
        res = np.linalg.norm(pg - (s * cg @ r_final.T + t), axis=1)
        spread_tot = float(np.sqrt((fit["spread"] ** 2).sum()))
        scale_rel_unc = rmse / (spread_tot * math.sqrt(fit["n_eff"]))
        i, j = rng.integers(0, len(pg), (2, 4000))
        dp = np.linalg.norm(pg[i] - pg[j], axis=1)
        dc = np.linalg.norm(cg[i] - cg[j], axis=1)
        far = (dp > max(8.0, 0.3 * float(np.max(dp)))) & (dc > 0)
        if far.sum() >= 10:
            s_pair = float(np.median(dp[far] / dc[far]))
            diag["scale_pairwise_crosscheck"] = round(s_pair, 6)
            if abs(s_pair / s - 1.0) > 0.05:
                warnings.append(f"Scale cross-check differs by {100 * abs(s_pair / s - 1):.1f}%.")
        if _alt_is_agl_like(gps):
            agl_u = camera_heights_above_ground(pts @ r_final.T, cc @ r_final.T)
            if agl_u is not None:
                ok = gps["valid"] & np.isfinite(agl_u) & (gps["alt"] > 5.0) & (agl_u > 0)
                if ok.sum() >= 3:
                    s_alt = float(np.median(gps["alt"][ok] / agl_u[ok]))
                    diag["scale_altitude_crosscheck"] = round(s_alt, 6)
                    dev = abs(s_alt / s - 1.0)
                    if dev > 0.15:
                        warnings.append(f"GPS scale and altitude-based scale differ by {100 * dev:.0f}% "
                                        "(take-off point not at scene ground level, or GPS problem).")
                    if dev > 0.30:
                        scale_rel_unc = max(scale_rel_unc, 0.10)
        mode = "GPS_SIM3" if fit.get("active_axes", 0) == 2 else "GPS_LINE"
        origin, heading_known = gps["origin"], True
        diag.update(final_gps_rmse_m=round(rmse, 3), final_gps_inliers=int(inl.sum()))
        if gps["sync"] == "timestamp" and scale_rel_unc <= 0.02 and rmse <= 4.0:
            confidence = "HIGH"
        elif scale_rel_unc <= 0.08 and rmse <= 8.0:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    else:
        r_final = rotation_between(up, ez)
        agl_u = camera_heights_above_ground(pts @ r_final.T, cc @ r_final.T)
        if agl_u is None or not np.isfinite(agl_u).any():
            return {"status": "failed", "reason": "Cannot find the ground below the cameras.", "diagnostics": diag}
        alt_m = None
        if gps is not None and _alt_is_agl_like(gps):
            ok = gps["valid"] & np.isfinite(agl_u) & (gps["alt"] > 2.0) & (agl_u > 0)
            if ok.sum() >= 3:
                s = float(np.median(gps["alt"][ok] / agl_u[ok]))
                alt_m = float(np.median(gps["alt"][ok]))
                mode, confidence = "ALTITUDE_AGL", "MEDIUM"
                scale_rel_unc = 0.05
        if alt_m is None:
            alt_m = float(assumed_altitude_m or DEFAULT_ASSUMED_ALTITUDE_M)
            s = alt_m / float(np.nanmedian(agl_u))
            mode, confidence = "ASSUMED_ALTITUDE", "LOW"
            scale_rel_unc = 0.25
            warnings.append(f"No usable GPS: scale assumes the drone flew {alt_m:.1f} m above ground. "
                            "Pass the real flight altitude for correct heights.")
        xy = np.median(cc @ r_final.T, axis=0)[:2] * s
        t = np.array([-xy[0], -xy[1], 0.0])
        if gps is not None:
            origin = gps["origin"]
        diag["assumed_or_logged_altitude_m"] = round(alt_m, 2)

    # ---------------- Level the ground to height 0 ----------------
    pm = s * pts @ r_final.T + t
    cm = s * cc @ r_final.T + t
    agl_m = camera_heights_above_ground(pm, cm)
    ground = estimate_ground_height(pm[:, 2])
    t = t - np.array([0.0, 0.0, ground])
    if origin is not None:
        la, lo, h = enu_to_geodetic(np.array([0.0, 0.0, ground]), origin)
        origin = (float(la), float(lo), float(h))
    if agl_m is not None:
        diag["median_camera_height_above_ground_m"] = round(float(np.nanmedian(agl_m)), 2)
    diag["median_camera_height_above_datum_m"] = round(float(np.median(cm[:, 2] - ground)), 2)

    t4 = np.eye(4)
    t4[:3, :3] = YUP_FROM_ENU @ (s * r_final)
    t4[:3, 3] = YUP_FROM_ENU @ t
    log(f"[Georef] mode={mode} confidence={confidence} scale={s:.6f} m/unit "
        f"(+/-{100 * (scale_rel_unc or 0):.2f}%)")
    return {
        "status": "ok", "version": 3, "mode": mode, "confidence": confidence, "metric": True,
        "heading_known": heading_known, "latlon_swapped": bool(swap),
        "origin": {"lat": origin[0], "lon": origin[1], "alt": origin[2]} if origin else None,
        "scale_m_per_unit": s,
        "scale_rel_uncertainty": float(scale_rel_unc) if scale_rel_unc is not None else None,
        "transform_raw_to_metric_yup": t4.tolist(),
        "display_frame": "METRIC_YUP" if EXPORT_DISPLAY_MODEL_METRIC else "RAW",
        "diagnostics": diag, "warnings": warnings,
    }


def georef_transform(georef):
    return np.asarray(georef["transform_raw_to_metric_yup"], dtype=float)


def _rotmat_to_quat_xyzw(r):
    r = np.asarray(r, float)
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def metric_camera_frames(model_dir, t4, frame_info=None):
    """
    Registered keyframes in the metric Y-up frame, in the bundle's frame_index format:
    [{index, filename, timestamp_sec, registered, position, quaternion_xyzw, fov_y_deg}, ...]
    (three.js camera convention: looks down its local -Z, +Y up).
    """
    cams = read_colmap_images(model_dir)
    intr = read_colmap_cameras(model_dir)
    frames = ((frame_info or {}).get("frames") or {}) if isinstance(frame_info, dict) else {}
    t4 = np.asarray(t4, float)
    s = np.cbrt(np.linalg.det(t4[:3, :3]))
    rs = t4[:3, :3] / s
    flip = np.diag([1.0, -1.0, -1.0])
    out = []
    for k, name in enumerate(sorted(cams, key=natural_key)):
        c = cams[name]
        pos = t4[:3, :3] @ c["C"] + t4[:3, 3]
        r3 = rs @ c["R"].T @ flip
        e = {"index": k, "filename": name, "registered": True,
             "position": [round(float(v), 4) for v in pos],
             "quaternion_xyzw": [round(float(v), 6) for v in _rotmat_to_quat_xyzw(r3)]}
        ci = intr.get(c.get("camera_id"))
        if ci and ci.get("fy"):
            e["fov_y_deg"] = round(float(2 * math.degrees(math.atan(ci["height"] / (2 * ci["fy"])))), 3)
        fr = frames.get(name) or {}
        if fr.get("time_s") is not None:
            e["timestamp_sec"] = float(fr["time_s"])
        if fr.get("source_frame") is not None:
            e["frame_number"] = int(fr["source_frame"])
        out.append(e)
    return out


def export_georeferenced_cloud(raw_ply, georef, out_paths):
    """Transforms the raw COLMAP cloud (points + normals) into the metric Y-up frame."""
    pts, nrm, col = load_point_cloud(raw_ply)
    t4 = georef_transform(georef)
    pm = apply_transform(t4, pts)
    nm = transform_normals(t4, nrm) if nrm is not None and len(nrm) == len(pts) else None
    for p in out_paths:
        save_point_cloud(p, pm, nm, col)
    return len(pm)
