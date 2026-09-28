"""
PRISM // georeferenced exports (geo_export.py)

Turns the metric model (x East, y Up, z South, metres, origin = GPS origin on the ground datum) into
standard GIS / 3-D deliverables without GDAL, PDAL or pyproj:

  point cloud   LAS 1.2 (point format 3: RGB + ASPRS class), UTM / WGS 84, GeoKeys VLR
  rasters       GeoTIFF (deflate) in UTM: DSM, DTM, nDSM (height above ground), orthophoto (RGBA),
                land-cover classes - every pixel is resampled from an exact UTM grid, not from the
                local tangent-plane grid, so meridian convergence and the UTM scale factor are honoured
  meshes        glTF 2.0 (.gltf + .bin), FBX 7.4 binary (vertex colours), OBJ / PLY / GLB (existing)
  vectors       GeoJSON (RFC 7946, WGS 84 lon/lat)

Coordinates: model -> ECEF -> geodetic is exact (no flat-earth approximation); geodetic -> UTM uses
the Krueger n-series to 6th order (sub-millimetre inside a zone).
Heights: the model datum (y = 0) sits at `origin.alt` in the flight log's altitude reference. With a
relative log (DJI SRT) that is height above the take-off point; pass `vertical_offset_m` (the
take-off elevation) to get elevations above sea level.
"""
import io
import json
import math
import os
import struct
import time
import zipfile
import zlib

import numpy as np

try:
    from georeference import enu_to_geodetic, geodetic_to_enu
except ImportError:                                    # pragma: no cover
    from backend.georeference import enu_to_geodetic, geodetic_to_enu

# =============================================================================
# UTM (WGS 84), Krueger series
# =============================================================================
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2 - _F)
_E = math.sqrt(_E2)
_N = _F / (2 - _F)
_K0 = 0.9996
_AA = _A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64 + _N ** 6 / 256)
_n = _N
_ALPHA = [
    _n / 2 - 2 * _n ** 2 / 3 + 5 * _n ** 3 / 16 + 41 * _n ** 4 / 180 - 127 * _n ** 5 / 288 + 7891 * _n ** 6 / 37800,
    13 * _n ** 2 / 48 - 3 * _n ** 3 / 5 + 557 * _n ** 4 / 1440 + 281 * _n ** 5 / 630 - 1983433 * _n ** 6 / 1935360,
    61 * _n ** 3 / 240 - 103 * _n ** 4 / 140 + 15061 * _n ** 5 / 26880 + 167603 * _n ** 6 / 181440,
    49561 * _n ** 4 / 161280 - 179 * _n ** 5 / 168 + 6601661 * _n ** 6 / 7257600,
    34729 * _n ** 5 / 80640 - 3418889 * _n ** 6 / 1995840,
    212378941 * _n ** 6 / 319334400,
]
_BETA = [
    _n / 2 - 2 * _n ** 2 / 3 + 37 * _n ** 3 / 96 - _n ** 4 / 360 - 81 * _n ** 5 / 512 + 96199 * _n ** 6 / 604800,
    _n ** 2 / 48 + _n ** 3 / 15 - 437 * _n ** 4 / 1440 + 46 * _n ** 5 / 105 - 1118711 * _n ** 6 / 3870720,
    17 * _n ** 3 / 480 - 37 * _n ** 4 / 840 - 209 * _n ** 5 / 4480 + 5569 * _n ** 6 / 90720,
    4397 * _n ** 4 / 161280 - 11 * _n ** 5 / 504 - 830251 * _n ** 6 / 7257600,
    4583 * _n ** 5 / 161280 - 108847 * _n ** 6 / 3991680,
    20648693 * _n ** 6 / 638668800,
]


def utm_zone(lat, lon):
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:          # Norway
        zone = 32
    if 72.0 <= lat < 84.0:                                 # Svalbard
        if 0.0 <= lon < 9.0:
            zone = 31
        elif 9.0 <= lon < 21.0:
            zone = 33
        elif 21.0 <= lon < 33.0:
            zone = 35
        elif 33.0 <= lon < 42.0:
            zone = 37
    return max(1, min(60, zone))


def utm_epsg(zone, north):
    return (32600 if north else 32700) + int(zone)


def _lon0(zone):
    return math.radians((zone - 1) * 6 - 180 + 3)


def utm_forward(lat, lon, zone, north=True):
    """Geodetic degrees -> UTM easting / northing (metres)."""
    phi = np.radians(np.asarray(lat, float))
    lam = np.radians(np.asarray(lon, float)) - _lon0(zone)
    sphi = np.sin(phi)
    t = np.sinh(np.arctanh(sphi) - _E * np.arctanh(_E * sphi))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, 1):
        xi = xi + a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    e = 500000.0 + _K0 * _AA * eta
    n = _K0 * _AA * xi + (0.0 if north else 10000000.0)
    return e, n


def utm_inverse(e, n, zone, north=True):
    """UTM easting / northing -> geodetic degrees."""
    xi = (np.asarray(n, float) - (0.0 if north else 10000000.0)) / (_K0 * _AA)
    eta = (np.asarray(e, float) - 500000.0) / (_K0 * _AA)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, 1):
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    e2 = _E2
    phi = (chi + (e2 / 2 + 5 * e2 ** 2 / 24 + e2 ** 3 / 12 + 13 * e2 ** 4 / 360) * np.sin(2 * chi)
           + (7 * e2 ** 2 / 48 + 29 * e2 ** 3 / 240 + 811 * e2 ** 4 / 11520) * np.sin(4 * chi)
           + (7 * e2 ** 3 / 120 + 81 * e2 ** 4 / 1120) * np.sin(6 * chi)
           + (4279 * e2 ** 4 / 161280) * np.sin(8 * chi))
    return np.degrees(phi), np.degrees(lam + _lon0(zone))


# =============================================================================
# the georeferencing frame of a survey
# =============================================================================
class GeoFrame:
    """Model frame <-> WGS 84 / UTM for one georeferenced survey."""

    def __init__(self, telem, vertical_offset_m=None):
        geo = (telem or {}).get("georeference") or {}
        o = geo.get("origin") or {}
        if not geo.get("metric") or o.get("lat") is None:
            raise ValueError("This model is not georeferenced (no GPS origin); geographic exports need a metric, "
                             "GPS-referenced model.")
        self.origin = (float(o["lat"]), float(o["lon"]), float(o.get("alt") or 0.0))
        self.zone = utm_zone(self.origin[0], self.origin[1])
        self.north = self.origin[0] >= 0
        self.epsg = utm_epsg(self.zone, self.north)
        ref = str((telem or {}).get("altitude_reference") or "relative").lower()
        auto = _takeoff_elevation(telem)
        if vertical_offset_m is not None:
            self.v_offset, self.v_datum = float(vertical_offset_m), "elevation (take-off elevation supplied by the user)"
        elif ref in ("absolute", "gps_altitude", "msl"):
            self.v_offset, self.v_datum = 0.0, "elevation as logged by the aircraft GNSS (absolute altitude)"
        elif auto is not None:
            self.v_offset, self.v_datum = auto, "elevation (take-off elevation from the flight log's absolute altitude)"
        else:
            self.v_offset, self.v_datum = 0.0, "height above the take-off point (relative flight log)"
        self.confidence = geo.get("confidence")
        self.scale_unc = geo.get("scale_rel_uncertainty")

    @property
    def crs_name(self):
        return f"WGS 84 / UTM zone {self.zone}{'N' if self.north else 'S'}"

    def model_to_geodetic(self, xyz):
        xyz = np.asarray(xyz, float)
        enu = np.c_[xyz[:, 0], -xyz[:, 2], xyz[:, 1]]
        la, lo, h = enu_to_geodetic(enu, self.origin)
        return la, lo, h + self.v_offset

    def model_to_utm(self, xyz, chunk=500000):
        xyz = np.asarray(xyz, float)
        out = np.empty((len(xyz), 3), np.float64)
        for s in range(0, len(xyz), chunk):
            la, lo, h = self.model_to_geodetic(xyz[s:s + chunk])
            e, n = utm_forward(la, lo, self.zone, self.north)
            out[s:s + chunk] = np.c_[e, n, h]
        return out

    def utm_to_model_xz(self, e, n):
        """UTM grid -> model (x, z) plus the tangent-plane drop u0 (for height conversion)."""
        la, lo = utm_inverse(e, n, self.zone, self.north)
        enu = geodetic_to_enu(la, lo, np.full(np.shape(la), self.origin[2]), self.origin)
        return enu[..., 0], -enu[..., 1], enu[..., 2]

    def describe(self):
        return {"crs": self.crs_name, "epsg": self.epsg, "utm_zone": self.zone, "hemisphere": "N" if self.north else "S",
                "origin_wgs84": {"lat": self.origin[0], "lon": self.origin[1]},
                "model_axes": "x = East, y = Up, z = South (metres); model origin = origin_wgs84 on the ground datum",
                "vertical": self.v_datum, "vertical_offset_m": round(self.v_offset, 3),
                "datum_height_m": round(self.origin[2] + self.v_offset, 3),
                "georeference_confidence": self.confidence, "scale_rel_uncertainty": self.scale_unc}


def _takeoff_elevation(telem):
    """Take-off elevation from a log that carries both absolute and relative altitude."""
    wps = (telem or {}).get("waypoints") or []
    d = [w["absolute_altitude_m"] - w["relative_altitude_m"] for w in wps
         if isinstance(w.get("absolute_altitude_m"), (int, float)) and isinstance(w.get("relative_altitude_m"), (int, float))]
    if len(d) >= 5:
        v = float(np.median(d))
        if abs(v) > 0.5 and float(np.percentile(d, 90) - np.percentile(d, 10)) < 5.0:
            return v
    return None


# =============================================================================
# LAS 1.2 (point data format 3)
# =============================================================================
# terrain class -> ASPRS LAS class
_ASPRS = {"unclassified": 1, "ground": 2, "low_veg": 3, "medium_veg": 4, "high_veg": 5, "building": 6,
          "low_noise": 7, "water": 9, "road": 11}


def geokeys(epsg, vertical_units=True):
    keys = [(1024, 0, 1, 1), (1025, 0, 1, 1), (3072, 0, 1, int(epsg)), (3076, 0, 1, 9001)]
    if vertical_units:
        keys.append((4099, 0, 1, 9001))
    out = [1, 1, 0, len(keys)]
    for k in keys:
        out.extend(k)
    return out


def classify_points(xyz, tm):
    """Per-point ASPRS class from the terrain model (class of the cell + height above ground)."""
    import terrain as T
    n = len(xyz)
    cls = np.full(n, _ASPRS["unclassified"], np.uint8)
    if tm is None:
        return cls
    i, j = tm.cell_of(xyz[:, 0], xyz[:, 2])
    ok = tm.inside(i, j)
    ii, jj = np.clip(i, 0, tm.nz - 1), np.clip(j, 0, tm.nx - 1)
    c = tm.cls[ii, jj]
    hag = xyz[:, 1] - tm.sample(tm.dtm, xyz[:, 0], xyz[:, 2])
    low = hag < 0.3
    cls[ok & low] = _ASPRS["ground"]
    cls[ok & low & (c == T.ROAD)] = _ASPRS["road"]
    cls[ok & low & (c == T.WATER)] = _ASPRS["water"]
    up = ok & ~low
    cls[up & (c == T.BUILDING)] = _ASPRS["building"]
    veg = up & np.isin(c, (T.TREE, T.LOW_VEG))
    cls[veg & (hag < 0.5)] = _ASPRS["low_veg"]
    cls[veg & (hag >= 0.5) & (hag < 2.0)] = _ASPRS["medium_veg"]
    cls[veg & (hag >= 2.0)] = _ASPRS["high_veg"]
    cls[ok & (hag < -1.0)] = _ASPRS["low_noise"]
    return cls


def write_las(path, xyz, rgb=None, classes=None, epsg=None, scale=0.001, software="PRISM geo_export"):
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)
    mins, maxs = xyz.min(0), xyz.max(0)
    offs = np.floor(mins)
    vlrs = b""
    nvlr = 0
    if epsg:
        gk = geokeys(epsg)
        body = struct.pack(f"<{len(gk)}H", *gk)
        vlrs += struct.pack("<H16sHH32s", 0, b"LASF_Projection", 34735, len(body), b"GeoKeyDirectoryTag") + body
        nvlr += 1
    header_size = 227
    offset_to_points = header_size + len(vlrs)
    today = time.gmtime()
    hdr = struct.pack(
        "<4sHHIHH8sBB32s32sHHHIIBHI5I3d3d6d",
        b"LASF", 0, 0, 0, 0, 0, b"\x00" * 8, 1, 2,
        b"PRISM".ljust(32, b"\x00"), software.encode()[:31].ljust(32, b"\x00"),
        today.tm_yday, today.tm_year, header_size, offset_to_points, nvlr, 3, 34, n,
        n, 0, 0, 0, 0,
        scale, scale, scale, offs[0], offs[1], offs[2],
        maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
    assert len(hdr) == header_size, len(hdr)
    dt = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"), ("flags", "u1"),
                   ("cls", "u1"), ("scan", "i1"), ("user", "u1"), ("src", "<u2"), ("gps", "<f8"),
                   ("r", "<u2"), ("g", "<u2"), ("b", "<u2")])
    assert dt.itemsize == 34
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(hdr)
        fh.write(vlrs)
        step = 1_000_000
        for s in range(0, n, step):
            p = xyz[s:s + step]
            rec = np.zeros(len(p), dt)
            q = np.round((p - offs) / scale).astype(np.int64)
            rec["x"], rec["y"], rec["z"] = q[:, 0], q[:, 1], q[:, 2]
            rec["flags"] = 0b00001001                  # return 1 of 1
            rec["cls"] = classes[s:s + step] if classes is not None else 1
            if rgb is not None:
                c = np.asarray(rgb[s:s + step], np.uint16) * 257
                rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
                rec["intensity"] = (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]).astype(np.uint16)
            fh.write(rec.tobytes())
    os.replace(tmp, path)
    return path


# =============================================================================
# GeoTIFF (classic TIFF, strips, Adobe deflate)
# =============================================================================
def write_geotiff(path, data, x_ul, y_ul, res, epsg, nodata=None, rgba=False, description=""):
    """data: (H, W) float32/uint8 or (H, W, 4) uint8 RGBA. (x_ul, y_ul) = outer corner of the top-left pixel."""
    a = np.ascontiguousarray(data)
    h, w = a.shape[:2]
    spp = a.shape[2] if a.ndim == 3 else 1
    bps = a.dtype.itemsize * 8
    fmt = 3 if a.dtype.kind == "f" else (2 if a.dtype.kind == "i" else 1)
    row_bytes = w * spp * a.dtype.itemsize
    rps = max(1, min(h, (256 * 1024) // max(1, row_bytes)))
    strips = []
    for r in range(0, h, rps):
        strips.append(zlib.compress(a[r:r + rps].tobytes(), 6))
    gk = geokeys(epsg, vertical_units=False)
    tags = []            # (tag, type, count, value-bytes)

    def tag(t, typ, vals):
        code = {"H": 3, "I": 4, "d": 12, "s": 2}[typ]
        if typ == "s":
            b = vals.encode("ascii") + b"\x00"
            tags.append((t, code, len(b), b))
        else:
            vals = list(vals) if isinstance(vals, (list, tuple, np.ndarray)) else [vals]
            tags.append((t, code, len(vals), struct.pack(f"<{len(vals)}{typ}", *vals)))

    tag(256, "I", w)
    tag(257, "I", h)
    tag(258, "H", [bps] * spp)
    tag(259, "H", 8)
    tag(262, "H", 2 if spp >= 3 else 1)
    if description:
        tag(270, "s", description)
    tag(273, "I", [0] * len(strips))                # patched below
    tag(277, "H", spp)
    tag(278, "I", rps)
    tag(279, "I", [len(s) for s in strips])
    tag(284, "H", 1)
    tag(305, "s", "PRISM geo_export")
    if rgba and spp == 4:
        tag(338, "H", 2)
    tag(339, "H", [fmt] * spp)
    tag(33550, "d", [res, res, 0.0])
    tag(33922, "d", [0.0, 0.0, 0.0, x_ul, y_ul, 0.0])
    tag(34735, "H", gk)
    if nodata is not None:
        tag(42113, "s", str(nodata))
    tags.sort(key=lambda t: t[0])
    # layout: header | IFD | overflow values | strips
    n = len(tags)
    ifd_off = 8
    ifd_size = 2 + 12 * n + 4
    over_off = ifd_off + ifd_size
    over = bytearray()
    entries = []
    strip_tag_idx = None
    for k, (t, code, cnt, b) in enumerate(tags):
        if t == 273:
            strip_tag_idx = k
        if len(b) <= 4:
            entries.append((t, code, cnt, b.ljust(4, b"\x00"), None))
        else:
            if (over_off + len(over)) % 2:
                over += b"\x00"
            entries.append((t, code, cnt, None, over_off + len(over)))
            over += b
    data_off = over_off + len(over)
    data_off += data_off % 2
    offsets, pos = [], data_off
    for s in strips:
        offsets.append(pos)
        pos += len(s)
    # patch strip offsets
    t, code, cnt, inline, ext = entries[strip_tag_idx]
    ob = struct.pack(f"<{len(offsets)}I", *offsets)
    if inline is not None:
        entries[strip_tag_idx] = (t, code, cnt, ob.ljust(4, b"\x00"), None)
    else:
        rel = ext - over_off
        over[rel:rel + len(ob)] = ob
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(b"II*\x00" + struct.pack("<I", ifd_off))
        fh.write(struct.pack("<H", n))
        for t, code, cnt, inline, ext in entries:
            fh.write(struct.pack("<HHI", t, code, cnt))
            fh.write(inline if inline is not None else struct.pack("<I", ext))
        fh.write(struct.pack("<I", 0))
        fh.write(bytes(over))
        fh.write(b"\x00" * (data_off - over_off - len(over)))
        for s in strips:
            fh.write(s)
    os.replace(tmp, path)
    return path


# =============================================================================
# terrain rasters resampled onto a true UTM grid
# =============================================================================
def utm_grid(gf, tm, res=None, max_pixels=40e6):
    """Output UTM grid covering the terrain model's analysed area."""
    roi = tm.roi if hasattr(tm, "roi") else np.ones((tm.nz, tm.nx), bool)
    ii, jj = np.nonzero(roi)
    if not len(ii):
        ii, jj = np.array([0, tm.nz - 1]), np.array([0, tm.nx - 1])
    x0, z0 = tm.centre(ii.min(), jj.min())
    x1, z1 = tm.centre(ii.max(), jj.max())
    corners = np.array([[x0, 0, z0], [x1, 0, z0], [x0, 0, z1], [x1, 0, z1]], float)
    u = gf.model_to_utm(corners)
    res = float(res or tm.res)
    area = (u[:, 0].max() - u[:, 0].min()) * (u[:, 1].max() - u[:, 1].min())
    if area / res ** 2 > max_pixels:
        res = math.ceil(math.sqrt(area / max_pixels) * 100) / 100.0
    e0 = math.floor((u[:, 0].min() - res) / res) * res
    n1 = math.ceil((u[:, 1].max() + res) / res) * res
    w = int(math.ceil((u[:, 0].max() + res - e0) / res))
    h = int(math.ceil((n1 - (u[:, 1].min() - res)) / res))
    return e0, n1, res, w, h


def _grid_model_coords(gf, e0, n1, res, w, r0, r1):
    ee = e0 + (np.arange(w) + 0.5) * res
    nn = n1 - (np.arange(r0, r1) + 0.5) * res
    E, N = np.meshgrid(ee, nn)
    return gf.utm_to_model_xz(E, N)


class _Sampler:
    """Bilinear / nearest sampling of terrain rasters at model (x, z)."""

    def __init__(self, tm):
        self.tm = tm
        self._src = {}

    def lin(self, name, raster, x, z):
        from scipy import ndimage as ndi
        tm = self.tm
        if name not in self._src:
            src = raster.astype(np.float64)
            bad = ~np.isfinite(src)
            if bad.any():
                src[bad] = np.nanmedian(src) if (~bad).any() else 0.0
            self._src[name] = src
        fx = (x - tm.x0) / tm.res - 0.5
        fz = (z - tm.z0) / tm.res - 0.5
        return ndi.map_coordinates(self._src[name], [fz, fx], order=1, mode="nearest")

    def near(self, raster, x, z):
        tm = self.tm
        i = np.clip(np.floor((z - tm.z0) / tm.res).astype(np.int64), 0, tm.nz - 1)
        j = np.clip(np.floor((x - tm.x0) / tm.res).astype(np.int64), 0, tm.nx - 1)
        return raster[i, j]


def terrain_rasters(gf, tm, out_dir, prefix="prism", products=("dsm", "dtm", "ndsm", "classes"), rows_per_chunk=256):
    """Writes the requested GeoTIFFs on an exact UTM grid; returns {product: path}."""
    import terrain as T
    e0, n1, res, w, h = utm_grid(gf, tm)
    nod = -9999.0
    datum = gf.origin[2] + gf.v_offset
    arrs = {}
    for p in products:
        arrs[p] = np.zeros((h, w), np.uint8) if p == "classes" else np.full((h, w), nod, np.float32)
    smp = _Sampler(tm)
    nd = np.maximum(np.nan_to_num(tm.ndsm, nan=0.0), 0.0)
    xmax, zmax = tm.x0 + tm.nx * tm.res, tm.z0 + tm.nz * tm.res
    for r0 in range(0, h, rows_per_chunk):
        r1 = min(h, r0 + rows_per_chunk)
        x, z, u0 = _grid_model_coords(gf, e0, n1, res, w, r0, r1)
        roi = (x >= tm.x0) & (x < xmax) & (z >= tm.z0) & (z < zmax)
        roi &= smp.near(tm.roi, x, z).astype(bool)
        if not roi.any():
            continue
        if "dsm" in arrs:
            v = smp.lin("dsm", tm.dsm_filled, x, z) + datum - u0
            arrs["dsm"][r0:r1] = np.where(roi, v, nod)
        if "dtm" in arrs:
            v = smp.lin("dtm", tm.dtm, x, z) + datum - u0
            arrs["dtm"][r0:r1] = np.where(roi, v, nod)
        if "ndsm" in arrs:
            v = smp.lin("ndsm", nd, x, z)
            arrs["ndsm"][r0:r1] = np.where(roi, v, nod)
        if "classes" in arrs:
            arrs["classes"][r0:r1] = np.where(roi, smp.near(tm.cls, x, z), 0)
    out = {}
    names = {"dsm": ("DSM", f"PRISM digital surface model, metres, {gf.v_datum}"),
             "dtm": ("DTM", f"PRISM bare-earth terrain model, metres, {gf.v_datum}"),
             "ndsm": ("heights_above_ground", "PRISM object heights above ground (nDSM), metres"),
             "classes": ("land_cover", "PRISM land cover classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(T.CLASS_NAMES)))}
    for p, a in arrs.items():
        stem, desc = names[p]
        out[p] = write_geotiff(os.path.join(out_dir, f"{prefix}_{stem}.tif"), a, e0, n1, res, gf.epsg,
                               nodata=0 if p == "classes" else nod, description=desc)
    return out, dict(res_m=res, width=w, height=h, ul=[e0, n1])


def orthophoto(gf, xyz, rgb, out_path, res=None, max_pixels=36e6):
    """True orthophoto: top-most point colour per UTM cell, pinholes closed; RGBA (alpha = coverage)."""
    from scipy import ndimage as ndi
    u = gf.model_to_utm(xyz)
    if res is None:
        from scipy.spatial import cKDTree
        sub = xyz[np.random.default_rng(0).choice(len(xyz), min(len(xyz), 20000), replace=False)]
        d, _ = cKDTree(sub[:, [0, 2]]).query(sub[:, [0, 2]], k=2)
        spacing = float(np.median(d[:, 1])) * math.sqrt(len(sub) / len(xyz))
        res = max(0.05, round(1.6 * spacing, 2))
    lo, hi = np.percentile(u[:, :2], 0.01, 0), np.percentile(u[:, :2], 99.99, 0)
    area = (hi[0] - lo[0]) * (hi[1] - lo[1])
    if area / res ** 2 > max_pixels:
        res = math.ceil(math.sqrt(area / max_pixels) * 100) / 100.0
    e0 = math.floor(lo[0] / res) * res
    n1 = math.ceil(hi[1] / res) * res
    w = int(math.ceil((hi[0] - e0) / res)) + 1
    h = int(math.ceil((n1 - lo[1]) / res)) + 1
    col = np.floor((u[:, 0] - e0) / res).astype(np.int64)
    row = np.floor((n1 - u[:, 1]) / res).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    key = row[ok] * w + col[ok]
    order = np.lexsort((u[ok, 2], key))                 # highest point of each cell last
    ks = key[order]
    last = np.r_[ks[1:] != ks[:-1], True]
    cells, src = ks[last], np.flatnonzero(ok)[order[last]]
    img = np.zeros((h * w, 4), np.uint8)
    img[cells, :3] = np.asarray(rgb)[src]
    img[cells, 3] = 255
    img = img.reshape(h, w, 4)
    # close 1-2 pixel pinholes left by the point spacing (normalised 5x5 colour average)
    have = img[..., 3] > 0
    hole = ndi.binary_closing(have, iterations=2) & ~have
    if hole.any():
        wgt = ndi.uniform_filter(have.astype(np.float32), 5)
        for c in range(3):
            s = ndi.uniform_filter(np.where(have, img[..., c], 0).astype(np.float32), 5)
            img[..., c][hole] = np.clip(s[hole] / np.maximum(wgt[hole], 1e-6), 0, 255).astype(np.uint8)
        img[..., 3][hole] = 255
    write_geotiff(out_path, img, e0, n1, res, gf.epsg, rgba=True,
                  description="PRISM true orthophoto (top-most surface colour), RGBA, alpha = coverage")
    return out_path, dict(res_m=res, width=w, height=h, ul=[e0, n1], coverage_pct=round(100.0 * float(np.mean(img[..., 3] > 0)), 1))


# =============================================================================
# glTF (.gltf + .bin) from GLB
# =============================================================================
def glb_to_gltf_zip(glb_path, zip_path, extra_files=None, stem="prism_model"):
    with open(glb_path, "rb") as fh:
        b = fh.read()
    magic, ver, total = struct.unpack_from("<III", b, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    off = 12
    js, binb = None, b""
    while off < len(b):
        ln, typ = struct.unpack_from("<II", b, off)
        chunk = b[off + 8: off + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif typ == 0x004E4942:
            binb = chunk
        off += 8 + ln
    js["buffers"] = [dict(byteLength=len(binb), uri=f"{stem}.bin")]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}.gltf", json.dumps(js, indent=1))
        z.writestr(f"{stem}.bin", binb)
        for name, data in (extra_files or {}).items():
            z.writestr(name, data)
    return zip_path


# =============================================================================
# FBX 7.4 binary
# =============================================================================
_FBX_HEAD = b"Kaydara FBX Binary\x20\x20\x00\x1a\x00"
_FBX_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FBX_FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
_FBX_FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
_FBX_TIME = "1970-01-01 10:00:00:000"


class _Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, *props, children=None):
        self.name = name.encode() if isinstance(name, str) else name
        self.props = list(props)
        self.children = children or []

    def add(self, name, *props, children=None):
        n = _Node(name, *props, children=children)
        self.children.append(n)
        return n


class _P:
    """Typed FBX property value."""
    __slots__ = ("t", "v")

    def __init__(self, t, v):
        self.t, self.v = t, v


def _enc_prop(p):
    if isinstance(p, _P):
        t, v = p.t, p.v
    elif isinstance(p, bool):
        t, v = "C", p
    elif isinstance(p, int):
        t, v = "I", p
    elif isinstance(p, float):
        t, v = "D", p
    elif isinstance(p, (bytes, bytearray)):
        t, v = "S", bytes(p)
    elif isinstance(p, str):
        t, v = "S", p.encode("utf-8")
    elif isinstance(p, np.ndarray):
        t, v = ("d" if p.dtype.kind == "f" else "i"), p
    else:
        raise TypeError(type(p))
    tb = t.encode()
    if t == "C":
        return tb + struct.pack("<?", bool(v))
    if t == "Y":
        return tb + struct.pack("<h", v)
    if t == "I":
        return tb + struct.pack("<i", v)
    if t == "L":
        return tb + struct.pack("<q", v)
    if t == "F":
        return tb + struct.pack("<f", v)
    if t == "D":
        return tb + struct.pack("<d", v)
    if t in ("S", "R"):
        return tb + struct.pack("<I", len(v)) + v
    if t in ("d", "i", "l", "f"):
        dt = {"d": "<f8", "i": "<i4", "l": "<i8", "f": "<f4"}[t]
        raw = np.ascontiguousarray(v, dtype=dt).tobytes()
        n = int(np.asarray(v).size)
        if len(raw) > 128:
            comp = zlib.compress(raw, 6)
            return tb + struct.pack("<III", n, 1, len(comp)) + comp
        return tb + struct.pack("<III", n, 0, len(raw)) + raw
    raise TypeError(t)


def _write_node(fh, node, is_last):
    props = b"".join(_enc_prop(p) for p in node.props)
    start = fh.tell()
    fh.write(b"\x00" * 12)
    fh.write(bytes((len(node.name),)) + node.name)
    fh.write(props)
    if node.children:
        for k, c in enumerate(node.children):
            _write_node(fh, c, k == len(node.children) - 1)
        fh.write(b"\x00" * 13)
    elif not node.props and not is_last:
        fh.write(b"\x00" * 13)
    end = fh.tell()
    fh.seek(start)
    fh.write(struct.pack("<III", end, len(node.props), len(props)))
    fh.seek(end)


def _p70(parent, rows):
    p = parent.add("Properties70")
    for r in rows:
        p.add("P", *r)
    return p


def write_fbx(path, xyz, faces, rgb=None, name="PRISM_survey", unit_scale_cm=100.0):
    """Binary FBX 7.4 with one mesh (triangles) and per-vertex colours; y-up, metres."""
    xyz = np.asarray(xyz, np.float64)
    faces = np.asarray(faces, np.int64)
    pvi = faces.astype(np.int32).copy()
    pvi[:, 2] = ~pvi[:, 2]                             # polygon end marker
    root = []
    lt = time.localtime()
    hx = _Node("FBXHeaderExtension")
    hx.add("FBXHeaderVersion", 1003)
    hx.add("FBXVersion", 7400)
    hx.add("EncryptionType", 0)
    ts = hx.add("CreationTimeStamp")
    for k, v in (("Version", 1000), ("Year", lt.tm_year), ("Month", lt.tm_mon), ("Day", lt.tm_mday),
                 ("Hour", lt.tm_hour), ("Minute", lt.tm_min), ("Second", lt.tm_sec), ("Millisecond", 0)):
        ts.add(k, v)
    hx.add("Creator", "PRISM geo_export")
    root.append(hx)
    root.append(_Node("FileId", _P("R", _FBX_FILE_ID)))
    root.append(_Node("CreationTime", _FBX_TIME))
    root.append(_Node("Creator", "PRISM geo_export"))
    gs = _Node("GlobalSettings")
    gs.add("Version", 1000)
    _p70(gs, [
        ("UpAxis", "int", "Integer", "", 1), ("UpAxisSign", "int", "Integer", "", 1),
        ("FrontAxis", "int", "Integer", "", 2), ("FrontAxisSign", "int", "Integer", "", 1),
        ("CoordAxis", "int", "Integer", "", 0), ("CoordAxisSign", "int", "Integer", "", 1),
        ("OriginalUpAxis", "int", "Integer", "", 1), ("OriginalUpAxisSign", "int", "Integer", "", 1),
        ("UnitScaleFactor", "double", "Number", "", float(unit_scale_cm)),
        ("OriginalUnitScaleFactor", "double", "Number", "", float(unit_scale_cm)),
        ("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0),
        ("DefaultCamera", "KString", "", "", "Producer Perspective"),
        ("TimeMode", "enum", "", "", 11), ("TimeSpanStart", "KTime", "Time", "", _P("L", 0)),
        ("TimeSpanStop", "KTime", "Time", "", _P("L", 46186158000)),
    ])
    root.append(gs)
    docs = _Node("Documents")
    docs.add("Count", 1)
    doc = docs.add("Document", _P("L", 1000000), "", "Scene")
    _p70(doc, [("SourceObject", "object", "", ""), ("ActiveAnimStackName", "KString", "", "", "")])
    doc.add("RootNode", _P("L", 0))
    root.append(docs)
    root.append(_Node("References"))
    defs = _Node("Definitions")
    defs.add("Version", 100)
    defs.add("Count", 4)
    for t, c in (("GlobalSettings", 1), ("Model", 1), ("Geometry", 1), ("Material", 1)):
        ot = defs.add("ObjectType", t)
        ot.add("Count", c)
    root.append(defs)
    GEO, MOD, MAT = 2000001, 2000002, 2000003
    objs = _Node("Objects")
    g = objs.add("Geometry", _P("L", GEO), f"{name}\x00\x01Geometry".encode(), "Mesh")
    g.add("Properties70")
    g.add("GeometryVersion", 124)
    g.add("Vertices", _P("d", xyz.reshape(-1)))
    g.add("PolygonVertexIndex", _P("i", pvi.reshape(-1)))
    layer_elems = []
    if rgb is not None:
        cols = np.ones((len(xyz), 4), np.float64)
        cols[:, :3] = np.asarray(rgb, np.float64) / 255.0
        lc = g.add("LayerElementColor", 0)
        lc.add("Version", 101)
        lc.add("Name", "Col")
        lc.add("MappingInformationType", "ByPolygonVertex")
        lc.add("ReferenceInformationType", "IndexToDirect")
        lc.add("Colors", _P("d", cols.reshape(-1)))
        lc.add("ColorIndex", _P("i", faces.astype(np.int32).reshape(-1)))
        layer_elems.append("LayerElementColor")
    lm = g.add("LayerElementMaterial", 0)
    lm.add("Version", 101)
    lm.add("Name", "")
    lm.add("MappingInformationType", "AllSame")
    lm.add("ReferenceInformationType", "IndexToDirect")
    lm.add("Materials", _P("i", np.zeros(1, np.int32)))
    layer_elems.append("LayerElementMaterial")
    lay = g.add("Layer", 0)
    lay.add("Version", 100)
    for le in layer_elems:
        e = lay.add("LayerElement")
        e.add("Type", le)
        e.add("TypedIndex", 0)
    m = objs.add("Model", _P("L", MOD), f"{name}\x00\x01Model".encode(), "Mesh")
    m.add("Version", 232)
    _p70(m, [("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0),
             ("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0),
             ("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0),
             ("DefaultAttributeIndex", "int", "Integer", "", 0)])
    m.add("Shading", True)
    m.add("Culling", "CullingOff")
    mt = objs.add("Material", _P("L", MAT), "photogrammetry\x00\x01Material".encode(), "")
    mt.add("Version", 102)
    mt.add("ShadingModel", "lambert")
    mt.add("MultiLayer", 0)
    _p70(mt, [("DiffuseColor", "Color", "", "A", 1.0, 1.0, 1.0),
              ("AmbientColor", "Color", "", "A", 0.2, 0.2, 0.2),
              ("Emissive", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)])
    root.append(objs)
    con = _Node("Connections")
    con.add("C", "OO", _P("L", MOD), _P("L", 0))
    con.add("C", "OO", _P("L", GEO), _P("L", MOD))
    con.add("C", "OO", _P("L", MAT), _P("L", MOD))
    root.append(con)
    takes = _Node("Takes")
    takes.add("Current", "")
    root.append(takes)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(_FBX_HEAD + struct.pack("<I", 7400))
        for k, n in enumerate(root):
            _write_node(fh, n, k == len(root) - 1)
        fh.write(b"\x00" * 13)
        fh.write(_FBX_FOOT_ID)
        fh.write(b"\x00" * 4)
        ofs = fh.tell()
        pad = ((ofs + 15) & ~15) - ofs
        fh.write(b"\x00" * (pad or 16))
        fh.write(struct.pack("<I", 7400))
        fh.write(b"\x00" * 120)
        fh.write(_FBX_FOOT_MAGIC)
    os.replace(tmp, path)
    return path


# =============================================================================
# GeoJSON helpers
# =============================================================================
def lonlat(gf, pts_xz_or_xyz):
    """Model points (N x 2 as x,z or N x 3) -> [[lon, lat], ...] rounded to ~1 cm."""
    a = np.asarray(pts_xz_or_xyz, float)
    if a.ndim == 1:
        a = a[None]
    xyz = np.c_[a[:, 0], np.zeros(len(a)), a[:, 1]] if a.shape[1] == 2 else a
    la, lo, _ = gf.model_to_geodetic(xyz)
    return [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lo, la)]


def feature(geom_type, coords, props):
    return {"type": "Feature", "geometry": {"type": geom_type, "coordinates": coords}, "properties": props}


def feature_collection(features, name, gf=None):
    fc = {"type": "FeatureCollection", "name": name, "features": features}
    if gf is not None:
        fc["prism"] = {"georeference": gf.describe(), "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return fc


def zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()
