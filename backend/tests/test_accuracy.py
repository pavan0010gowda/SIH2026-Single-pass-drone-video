"""
PRISM ground-truth accuracy tests (no drone data or GPU needed).

Run from the project root:   python backend/tests/test_accuracy.py
Every test builds a synthetic survey with known answers (tests/synthetic_scene.py), runs the real
engines and checks the errors. Exit code 0 = all passed.
"""

import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import terrain as T                       # noqa: E402
import height_engine as H                 # noqa: E402
import road_engine as R                   # noqa: E402
import synthetic_scene as S               # noqa: E402

_CACHE = {}


def scene(noise=0.03):
    if noise not in _CACHE:
        pts, rgb, truth = S.build_scene(noise=noise)
        tm = T.TerrainModel.from_points(pts, rgb, res=0.5, log=lambda *a: None)
        _CACHE[noise] = (pts, rgb, truth, tm)
    return _CACHE[noise]


# --------------------------------------------------------------------------- telemetry parsing
def test_srt_gps_triplet_order_and_altitude():
    import tempfile
    from telemetry_parser import parse_srt_records
    txt = "".join(f"{i + 1}\n00:00:{i // 30:02d},{(i % 30) * 33:03d} --> 00:00:{(i + 1) // 30:02d},{((i + 1) % 30) * 33:03d}\n"
                  f"{'HOME(45.31130865,29.29587291)' + chr(10) if i == 0 else ''}"
                  f"GPS({45.3113 + i * 1e-6:.8f},{29.2958 - i * 1e-6:.8f},{29.9 + 0.001 * i:.3f})\n\n" for i in range(60))
    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
        f.write(txt)
    recs, meta = parse_srt_records(f.name)
    os.remove(f.name)
    assert len(recs) == 60
    assert meta["latlon_ambiguous"] is True and meta["latlon_order"] == "lat_lon"
    assert abs(recs[0]["lat"] - 45.3113) < 1e-9 and abs(recs[0]["lon"] - 29.2958) < 1e-9
    assert recs[0]["alt"] is not None and abs(recs[0]["alt"] - 29.9) < 1e-9, "altitude inside GPS() must be read"
    return "GPS(lat,lon,alt) parsed; altitude kept; order flagged ambiguous"


def test_srt_dji_legacy_satellites_not_altitude():
    import tempfile
    from telemetry_parser import parse_srt_records
    txt = "".join(f"{i + 1}\n00:00:0{i},000 --> 00:00:0{i + 1},000\nHOME(149.0251,-35.2170) 2016.10.03 13:28:42\n"
                  f"GPS(149.02{50 + i},-35.2168,17) BAROMETER:3{i}.9\n\n" for i in range(5))
    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
        f.write(txt)
    recs, meta = parse_srt_records(f.name)
    os.remove(f.name)
    assert meta["latlon_ambiguous"] is False and meta["latlon_order"] == "lon_lat"
    assert abs(recs[0]["lat"] + 35.2168) < 1e-9 and recs[0]["alt"] == 30.9, "barometer is the altitude, 17 = satellites"
    return "legacy DJI GPS(lon,lat,sats)+BAROMETER handled"


def test_latlon_order_resolved_from_camera_track():
    import recalibrate as RC
    rng = np.random.default_rng(3)
    t = np.linspace(0, 120, 90)
    # a curved flight: 250 m x 180 m, 30 m altitude, GPS noise 0.8 m
    e = 120 * np.sin(t / 40.0)
    n = 90 * np.cos(t / 25.0) + 0.8 * t
    lat0, lon0 = 45.3106, 29.2953
    lat = lat0 + n / 111_132.0
    lon = lon0 + e / (111_320.0 * math.cos(math.radians(lat0)))
    recs = [{"time_s": float(a), "lat": float(b), "lon": float(c), "alt": 30.0 + 0.2 * math.sin(a)} for a, b, c in zip(t, lat, lon)]
    # model frame: scaled by 0.7, rotated 40 deg about Y, shifted; Y-up (x=E, y=U, z=S)
    s, th = 0.7, math.radians(40)
    enu = np.c_[e, n, np.full_like(e, 30.0)] + rng.normal(0, 0.8, (len(e), 3))
    yup = np.c_[enu[:, 0], enu[:, 2], -enu[:, 1]]
    Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    cams = s * yup @ Ry.T + np.array([5.0, -3.0, 7.0])
    swapped = [{**r, "lat": r["lon"], "lon": r["lat"]} for r in recs]       # log written as (lon, lat)
    fit = RC.fit_model_to_gps(cams, t, swapped, ambiguous=True)
    assert fit["swap"] is True, "swapped log must be detected"
    assert abs(fit["s"] - 1 / s) / (1 / s) < 0.01, f"scale {fit['s']:.4f} vs {1 / s:.4f}"
    return f"swap detected, scale error {100 * abs(fit['s'] * s - 1):.2f}%, RMSE {fit['rmse']:.2f} m"


# --------------------------------------------------------------------------- terrain / heights
def test_terrain_ground_model():
    pts, rgb, truth, tm = scene()
    ii, jj = np.nonzero(tm.ground_mask & tm.roi)
    x, z = tm.centre(ii, jj)
    err = tm.dtm[ii, jj] - S.terrain_y(x, z)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # under buildings/trees the ground was never seen: interpolation must still be sane
    b = truth["boxes"][1]
    under = float(tm.sample(tm.dtm, b.cx, b.cz)) - S.terrain_y(b.cx, b.cz)
    assert rmse < 0.06, f"DTM RMSE {rmse:.3f} m"
    assert abs(under) < 0.3, f"DTM under roof off by {under:.2f} m"
    return f"DTM RMSE {100 * rmse:.1f} cm on open ground, {100 * abs(under):.0f} cm under a 20x10 m roof"


def test_structure_heights():
    pts, rgb, truth, tm = scene()
    he = H.HeightEngine(tm, pts, rgb, telem={"georeference": {"metric": True, "mode": "GPS_SIM3", "scale_rel_uncertainty": 0.002}})
    rows, worst = [], 0.0
    for obj in truth["boxes"] + [t for t in truth["trees"] if not t.name.startswith("belt")]:
        r = he.measure_at(obj.cx + 0.4, obj.cz - 0.3)
        assert r["status"] == "ok", f"{obj.name}: {r}"
        err = r["height_m"] - obj.truth_height()
        worst = max(worst, abs(err))
        z = abs(err) / max(r["uncertainty_m"], 1e-3)
        rows.append(f"{obj.name:>11}: {r['height_m']:6.2f} m (truth {obj.truth_height():5.2f}, err {100 * err:+5.1f} cm, "
                    f"+/-{100 * r['uncertainty_m']:.1f} cm, {r['method']})")
        tol = 0.10 if isinstance(obj, S.Box) else 0.30
        assert abs(err) <= tol, rows[-1]
        assert z < 4.0, f"uncertainty not honest: {rows[-1]}"
    return "\n      " + "\n      ".join(rows)


def test_caliper_snaps_to_surface():
    pts, rgb, truth, tm = scene()
    he = H.HeightEngine(tm, pts, rgb, telem={})
    b = truth["boxes"][0]
    base = S.terrain_y(b.cx, b.cz)
    p1 = [b.cx + 7.5, S.terrain_y(b.cx + 7.5, b.cz) + 0.05, b.cz]          # ground next to the building
    p2 = [b.cx, base + b.H + 0.05, b.cz]                                    # roof
    r = he.measure_between(p1, p2)
    truth_dy = base + b.H - S.terrain_y(b.cx + 7.5, b.cz)
    assert abs(r["vertical_m"] - truth_dy) < 0.06, r
    return f"vertical {r['vertical_m']:.3f} m vs truth {truth_dy:.3f} m (+/-{r['vertical_uncertainty_m']:.3f})"


# --------------------------------------------------------------------------- roads / potholes
def test_road_extraction():
    pts, rgb, truth, tm = scene()
    roads = R.detect_roads(tm, log=lambda *a: None)
    assert roads["segments"], "road not found"
    main = roads["segments"][0]
    ra, rb, w = truth["road"]
    true_len = float(np.linalg.norm(rb - ra))
    assert main["length_m"] > 0.8 * true_len, main
    assert abs(main["width_median_m"] - w) < 1.5, main
    return f"length {main['length_m']:.0f} m (truth {true_len:.0f}), width {main['width_median_m']:.1f} m (truth {w})"


def test_pothole_depths():
    pts, rgb, truth, tm = scene()
    roads = R.detect_roads(tm, log=lambda *a: None)
    ph, st = R.detect_potholes(tm, pts, roads["mask"], log=lambda *a: None)
    lod = st["lod_m"]
    rows = []
    for (hx, hz, depth, dia) in truth["holes"]:
        near = [p for p in ph if math.hypot(p["position"][0] - hx, p["position"][2] - hz) < max(1.0, dia)]
        if depth < lod + 0.02:
            rows.append(f"hole {100 * depth:.0f} cm (below LoD {100 * lod:.1f} cm): {'found' if near else 'not reported'}")
            continue
        assert near, f"missed {depth} m pothole at ({hx:.1f},{hz:.1f}); found {[p['position'] for p in ph]}"
        p = near[0]
        err = p["depth_cm"] / 100 - depth
        rows.append(f"hole {100 * depth:.0f} cm / {100 * dia:.0f} cm: measured {p['depth_cm']:.1f} cm deep, "
                    f"{p['diameter_cm']:.0f} cm wide ({p['severity']})")
        assert abs(err) < max(0.04, 0.25 * depth), rows[-1]
    false_pos = [p for p in ph if not any(math.hypot(p["position"][0] - hx, p["position"][2] - hz) < 1.5
                                          for hx, hz, _, _ in truth["holes"])]
    assert len(false_pos) == 0, f"false potholes: {[p['position'] for p in false_pos]}"
    return f"LoD {100 * lod:.1f} cm; " + "; ".join(rows)


# --------------------------------------------------------------------------- buildings / rooftops
def test_building_roofs_and_eaves():
    import buildings as B
    pts, rgb, truth, tm = scene()
    he = H.HeightEngine(tm, pts, rgb, telem={})
    found = B.extract_buildings(he, log=lambda *a: None)
    assert len(found) == len(truth["boxes"]), f"{len(found)} buildings found, {len(truth['boxes'])} in the scene"
    rows = []
    for box in truth["boxes"]:
        b = min(found, key=lambda q: math.hypot(q["centroid"][0] - box.cx, q["centroid"][2] - box.cz))
        if box.eave is None:
            assert b["roof_type"] == "FLAT", f"{box.name}: {b['roof_type']}"
            rows.append(f"{box.name}: FLAT")
        else:
            true_pitch = math.degrees(math.atan((box.H - box.eave) / (box.W / 2.0)))
            assert b["roof_type"] == "GABLE", f"{box.name}: {b['roof_type']}"
            assert abs(b["roof_pitch_deg"] - true_pitch) < 1.5, f"pitch {b['roof_pitch_deg']} vs {true_pitch:.1f}"
            assert abs(b["eaves_m"] - box.eave) < 0.25, f"eaves {b['eaves_m']} vs {box.eave}"
            rows.append(f"{box.name}: GABLE pitch {b['roof_pitch_deg']:.1f} deg (truth {true_pitch:.1f}), "
                        f"eaves {b['eaves_m']:.2f} m (truth {box.eave:.2f})")
    return "; ".join(rows)


def test_height_estimator_seed_independent():
    pts, rgb, truth, tm = scene()
    he = H.HeightEngine(tm, pts, rgb, telem={})
    worst = 0.0
    for box in truth["boxes"]:
        hs = [he.measure_at(box.cx + 0.3, box.cz - 0.2, rng=np.random.default_rng(s))["height_m"] for s in range(5)]
        worst = max(worst, max(hs) - min(hs))
    assert worst < 0.03, f"height changes by {100 * worst:.1f} cm with the random seed"
    return f"max spread over 5 seeds: {100 * worst:.1f} cm"


# --------------------------------------------------------------------------- georeferenced exports
def test_utm_against_independent_formula():
    import geo_export as G

    def snyder(lat, lon, zone):
        a, f = 6378137.0, 1 / 298.257223563
        e2 = f * (2 - f)
        ep2, k0 = e2 / (1 - e2), 0.9996
        phi, lam, lam0 = math.radians(lat), math.radians(lon), math.radians((zone - 1) * 6 - 180 + 3)
        N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
        Tt, C, A = math.tan(phi) ** 2, ep2 * math.cos(phi) ** 2, math.cos(phi) * (lam - lam0)
        M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
                 + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi) - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
        x = k0 * N * (A + (1 - Tt + C) * A ** 3 / 6 + (5 - 18 * Tt + Tt * Tt + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
        y = k0 * (M + N * math.tan(phi) * (A * A / 2 + (5 - Tt + 9 * C + 4 * C * C) * A ** 4 / 24
                                           + (61 - 58 * Tt + Tt * Tt + 600 * C - 330 * ep2) * A ** 6 / 720))
        return x, y + (0 if lat >= 0 else 1e7)

    worst = 0.0
    for lat, lon in ((45.31, 29.29), (57.98, -4.0), (28.6, 77.2), (-33.9, 151.2), (10.0, 5.9)):
        z = G.utm_zone(lat, lon)
        e, n = G.utm_forward(lat, lon, z, lat >= 0)
        x, y = snyder(lat, lon, z)
        worst = max(worst, abs(float(e) - x), abs(float(n) - y))
        la, lo = G.utm_inverse(e, n, z, lat >= 0)
        assert abs(float(la) - lat) < 1e-9 and abs(float(lo) - lon) < 1e-9
    assert worst < 0.005, f"UTM differs by {worst:.4f} m"
    return f"UTM forward within {1000 * worst:.2f} mm of the USGS series; inverse round trip < 1e-9 deg"


def test_las_geotiff_fbx_writers():
    import struct
    import tempfile
    import zlib
    import geo_export as G
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    xyz = np.c_[500000 + rng.uniform(0, 50, 1000), 5000000 + rng.uniform(0, 50, 1000), rng.uniform(0, 10, 1000)]
    rgb = rng.integers(0, 255, (1000, 3)).astype(np.uint8)
    G.write_las(os.path.join(d, "t.las"), xyz, rgb, np.full(1000, 2, np.uint8), 32635)
    with open(os.path.join(d, "t.las"), "rb") as fh:
        b = fh.read()
    assert b[:4] == b"LASF" and b[24:26] == bytes((1, 2))
    off, nvlr, fmt, rl, npts = struct.unpack_from("<IIBHI", b, 96)
    assert (fmt, rl, npts, nvlr) == (3, 34, 1000, 1)
    sx, sy, sz, ox, oy, oz = struct.unpack_from("<6d", b, 131)
    x0 = struct.unpack_from("<i", b, off)[0] * sx + ox
    assert abs(x0 - xyz[0, 0]) < 0.001
    img = rng.normal(0, 1, (37, 53)).astype(np.float32)
    G.write_geotiff(os.path.join(d, "t.tif"), img, 500000.0, 5000100.0, 0.5, 32635, nodata=-9999.0)
    try:
        import tifffile
        with tifffile.TiffFile(os.path.join(d, "t.tif")) as t:
            assert np.array_equal(t.pages[0].asarray(), img)
            assert t.geotiff_metadata["ProjectedCSTypeGeoKey"] == 32635
    except ImportError:
        pass
    v = rng.uniform(-1, 1, (30, 3))
    f = np.array([[0, 1, 2], [2, 3, 4]])
    G.write_fbx(os.path.join(d, "t.fbx"), v, f, rgb[:30])
    with open(os.path.join(d, "t.fbx"), "rb") as fh:
        fb = fh.read()
    assert fb[:21] == b"Kaydara FBX Binary  \x00" and struct.unpack_from("<I", fb, 23)[0] == 7400
    k = fb.find(b"PolygonVertexIndex")
    n_arr, enc, clen = struct.unpack_from("<III", fb, k + len("PolygonVertexIndex") + 1)
    raw = fb[k + len("PolygonVertexIndex") + 13: k + len("PolygonVertexIndex") + 13 + clen]
    arr = np.frombuffer(zlib.decompress(raw) if enc else raw, "<i4")
    assert arr.tolist() == [0, 1, -3, 2, 3, -5], arr
    return "LAS 1.2 / GeoTIFF (EPSG:32635) / FBX 7.4 structure verified"


# --------------------------------------------------------------------------- change detection (day 1 vs day 2)
def test_change_detection_day1_day2():
    import change_engine as CE
    from georeference import enu_to_geodetic
    pts, rgb, truth, tm_a = scene()
    rng = np.random.default_rng(11)
    oa = (45.30, 29.30, 0.0)
    la, lo, al = enu_to_geodetic(np.array([30.0, -18.0, 2.0]), oa)
    ob = (float(la), float(lo), float(al))
    # day 2: another sampling, a new tent (4 x 3 x 2.4 m) and a 0.6 m deep trench on open ground
    p2 = pts[rng.random(len(pts)) < 0.85].copy()
    tent_c = (10.0, -10.0)
    from scipy import ndimage as _ndi
    open_ground = _ndi.binary_erosion((tm_a.cls == T.GROUND) & tm_a.roi, iterations=12)
    cand = [tm_a.centre(i, j) for i, j in np.argwhere(open_ground)[::97]]
    trench_c = next((float(x), float(z)) for x, z in cand if math.hypot(x - tent_c[0], z - tent_c[1]) > 30)
    inside = (np.abs(p2[:, 0] - tent_c[0]) <= 2.0) & (np.abs(p2[:, 2] - tent_c[1]) <= 1.5)
    p2 = p2[~inside]
    g = S.terrain_y(*tent_c)
    xs, zs = np.meshgrid(np.arange(-2.0, 2.0, 0.15), np.arange(-1.5, 1.5, 0.15))
    tent = np.c_[tent_c[0] + xs.ravel(), np.full(xs.size, g + 2.4), tent_c[1] + zs.ravel()]
    ins = (np.abs(p2[:, 0] - trench_c[0]) <= 4.0) & (np.abs(p2[:, 2] - trench_c[1]) <= 0.6)
    p2[ins, 1] -= 0.6
    p2 = np.vstack([p2, tent])
    c2 = np.full((len(p2), 3), 150, np.uint8)
    # day-2 georeference with a GNSS error of (1.1 E, 0.7 S, +3.5 up) and 0.2 deg heading
    t_true = CE.model_to_model(oa, ob)

    def to_b(p):
        q = p @ t_true[:3, :3].T + t_true[:3, 3]
        th = math.radians(0.2)
        r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        xz = q[:, [0, 2]] @ r.T + np.array([1.1, 0.7])
        q[:, 0], q[:, 2] = xz[:, 0], xz[:, 1]
        q[:, 1] += 3.5
        return q
    pb = to_b(p2)
    tm_b = T.TerrainModel.from_points(pb, c2, res=0.5, log=lambda *a: None)
    geo = lambda o: {"georeference": {"metric": True, "origin": {"lat": o[0], "lon": o[1], "alt": o[2]}}}
    r = CE.compare(dict(tm=tm_a, xyz=pts, telem=geo(oa), id="d1", name="Day 1"),
                   dict(tm=tm_b, xyz=pb, telem=geo(ob), id="d2", name="Day 2"), min_height_m=0.4, log=lambda *a: None)
    assert r["status"] == "completed", r.get("location")
    reg = r["registration"]
    assert abs(reg["horizontal_shift_m"][0] - 1.1) < 0.2 and abs(reg["horizontal_shift_m"][1] - 0.7) < 0.2, reg
    assert abs(reg["vertical_offset_m"] - 3.5) < 0.1, reg
    found = {}
    for name, (x, z), want in (("tent", tent_c, "gain"), ("trench", trench_c, "loss")):
        tb = to_b(np.array([[x, 0.0, z]]))[0]
        hits = [a for a in r["alerts"] if a["kind"] == want and math.hypot(a["position"][0] - tb[0], a["position"][2] - tb[2]) < 3.0]
        assert hits, f"{name} not detected: {[(a['category'], a['position']) for a in r['alerts']]}"
        found[name] = hits[0]
    extra = [a for a in r["alerts"] if a["kind"] != "group" and a not in found.values()]
    assert not extra, f"false alarms: {[(a['category'], a['height_change_m'], a['position']) for a in extra]}"
    assert abs(found["tent"]["height_change_m"] - 2.4) < 0.25, found["tent"]
    assert abs(found["trench"]["height_change_m"] + 0.6) < 0.2, found["trench"]
    return (f"GNSS error recovered ({reg['horizontal_shift_m']} m, {reg['rotation_deg']} deg, {reg['vertical_offset_m']} m); "
            f"tent +{found['tent']['height_change_m']} m, trench {found['trench']['height_change_m']} m, 0 false alarms")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    t0 = time.time()
    failed = 0
    for fn in TESTS:
        t1 = time.time()
        try:
            msg = fn()
            print(f"PASS  {fn.__name__} ({time.time() - t1:.1f}s): {msg}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__} ({time.time() - t1:.1f}s): {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed in {time.time() - t0:.1f}s")
    sys.exit(1 if failed else 0)
