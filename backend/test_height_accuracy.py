"""
PRISM // Height-accuracy self-test (ground truth, no COLMAP / drone needed)

Builds a synthetic site with structures of KNOWN height, simulates two drone sorties exactly the
way COLMAP outputs them (arbitrary rotation, arbitrary scale, camera poses in images.bin), adds
realistic GNSS errors (1.2 m drifting noise, per-flight bias, different take-off height), then runs
the real georeference.py + change_detector.py and compares the reported heights with the truth.

Usage (from the backend folder):   python test_height_accuracy.py
"""
import os
import sys
import json
import time
import struct
import shutil
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import georeference as G
import change_detector as CD

BASE_LAT, BASE_LON = 12.9716, 77.5946


# ----------------------------------------------------------------------------- synthetic site
def terrain_z(x, y):
    return 0.02 * x + 0.01 * y + 0.3 * np.sin(x / 15.0) * np.cos(y / 20.0)


class Struct:
    def __init__(self, name, cx, cy, L, W, H, yaw_deg=0.0, eave=None):
        self.name, self.cx, self.cy, self.L, self.W, self.H = name, cx, cy, L, W, H
        self.yaw, self.eave, self.base = np.radians(yaw_deg), eave, terrain_z(cx, cy)

    def local(self, x, y):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        dx, dy = x - self.cx, y - self.cy
        return c * dx + s * dy, -s * dx + c * dy

    def inside(self, x, y):
        u, v = self.local(x, y)
        return (np.abs(u) <= self.L / 2) & (np.abs(v) <= self.W / 2)

    def top(self, x, y):
        u, v = self.local(x, y)
        if self.eave is None:
            return np.full_like(x, self.base + self.H)
        return self.base + self.eave + (self.H - self.eave) * (1.0 - np.abs(v) / (self.W / 2))

    def sample(self, sp, rng):
        U, V = np.meshgrid(np.arange(-self.L / 2, self.L / 2 + 1e-9, sp), np.arange(-self.W / 2, self.W / 2 + 1e-9, sp))
        U, V = U.ravel(), V.ravel()
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        X, Y = self.cx + c * U - s * V, self.cy + s * U + c * V
        parts = [np.c_[X, Y, self.top(X, Y)]]
        for (u0, v0, u1, v1) in [(-1, -1, 1, -1), (1, -1, 1, 1), (1, 1, -1, 1), (-1, 1, -1, -1)]:
            n = max(2, int(np.hypot((u1 - u0) * self.L / 2, (v1 - v0) * self.W / 2) / sp))
            t = np.linspace(0, 1, n)
            uu, vv = (u0 + (u1 - u0) * t) * self.L / 2, (v0 + (v1 - v0) * t) * self.W / 2
            xx, yy = self.cx + c * uu - s * vv, self.cy + s * uu + c * vv
            zt = self.top(xx, yy)
            for k in range(len(xx)):
                zs = np.arange(self.base, zt[k], sp)
                zs = zs[rng.random(len(zs)) < 0.35]          # walls are poorly seen from the air
                parts.append(np.c_[np.full(len(zs), xx[k]), np.full(len(zs), yy[k]), zs])
        return np.vstack(parts)


EXISTING = [Struct("building (unchanged)", -30, 10, 15, 10, 6.0, yaw_deg=10)]
NEW = [Struct("watchtower", 20, -10, 2.0, 2.0, 8.0),
       Struct("tent (ridge)", 0, 15, 6.0, 4.0, 3.0, yaw_deg=25, eave=2.0),
       Struct("vehicle", 35, 20, 4.5, 2.0, 1.8, yaw_deg=-40),
       Struct("barricade", -10, -25, 12.0, 0.6, 1.2, yaw_deg=5)]
REMOVED = [Struct("container", 40, -25, 6.0, 2.5, 2.6, yaw_deg=15)]


def make_cloud(epoch, seed, sp=0.15):
    rng = np.random.default_rng(seed)
    structs = EXISTING + (NEW if epoch == "recon" else REMOVED)
    X, Y = np.meshgrid(np.arange(-60, 60, sp), np.arange(-40, 40, sp))
    X = X.ravel() + rng.uniform(-sp / 2, sp / 2, X.size)
    Y = Y.ravel() + rng.uniform(-sp / 2, sp / 2, Y.size)
    occ = np.zeros(X.shape, bool)
    for s in structs:
        occ |= s.inside(X, Y)
    parts = [np.c_[X[~occ], Y[~occ], terrain_z(X[~occ], Y[~occ])]] + [s.sample(sp, rng) for s in structs]
    th, r = rng.uniform(0, 2 * np.pi, 4000), 3.0 * np.sqrt(rng.uniform(0, 1, 4000))       # a tree
    tx, ty = -45 + r * np.cos(th), -20 + r * np.sin(th)
    parts.append(np.c_[tx, ty, terrain_z(tx, ty) + 7.0 - (r / 3.0) ** 2 * 2.5 + rng.normal(0, 0.3, 4000)])
    P = np.vstack(parts) + rng.normal(0, 0.04, (sum(len(p) for p in parts), 3))            # MVS noise
    k = int(0.003 * len(P))
    P[rng.choice(len(P), k, replace=False)] += rng.normal(0, 1.2, (k, 3))                   # floaters
    return P


# ----------------------------------------------------------------------------- simulated flights
def cam_rotation(yaw_deg, pitch_deg):
    psi, th = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([np.sin(psi) * np.cos(th), np.cos(psi) * np.cos(th), -np.sin(th)])
    r = np.array([np.cos(psi), -np.sin(psi), 0.0])
    return np.stack([r, np.cross(f, r), f], axis=0)


def flight(kind, alt):
    cams = []
    if kind == "EW":
        for i, yl in enumerate(np.arange(-30, 31, 15)):
            xs = np.arange(-50, 51, 2.5)[::(-1 if i % 2 else 1)]
            cams += [((x, yl, alt), 90 if i % 2 == 0 else 270, 90) for x in xs]
    else:
        for i, xl in enumerate(np.arange(-45, 46, 15)):
            ys = np.arange(-35, 36, 2.5)[::(-1 if i % 2 else 1)]
            cams += [((xl, y, alt), 0 if i % 2 == 0 else 180, 90) for y in ys]
    return np.array([c[0] for c in cams], float), np.array([cam_rotation(c[1], c[2]) for c in cams])


def rotmat_to_qvec(R):
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return q if q[0] >= 0 else -q


def simulate_sortie(workdir, epoch, kind, alt, bias, seed, with_gps=True, assumed_alt=None):
    rng = np.random.default_rng(seed)
    P, (C, R) = make_cloud(epoch, seed), flight(kind, alt)
    # COLMAP-like arbitrary frame: rotation of a random camera, scale normalised, origin arbitrary
    Rc0, s_c, Cbar = R[rng.integers(0, len(C))], 10.0 / np.ptp(C, axis=0).max(), C.mean(0)
    to_c = lambda X: s_c * (Rc0 @ (X - Cbar).T).T
    P_c, C_c = to_c(P), to_c(C)
    R_c = np.einsum("nij,kj->nik", R, Rc0)
    t_c = -np.einsum("nij,nj->ni", R_c, C_c)
    model_dir = os.path.join(workdir, f"colmap_{epoch}")
    os.makedirs(model_dir, exist_ok=True)
    names = [f"frame_{i:04d}.jpg" for i in range(len(C))]
    with open(os.path.join(model_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, (n, r, t) in enumerate(zip(names, R_c, t_c)):
            f.write(struct.pack("<idddddddi", i + 1, *rotmat_to_qvec(r), *t, 1) + n.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    dt = 0.5
    frame_info = {"frames": {n: {"time_s": i * dt} for i, n in enumerate(names)}}
    if with_gps:           # 30 Hz log, GNSS drift + bias + different take-off height
        ts = np.arange(0, (len(C) - 1) * dt + 1e-9, 1 / 30.0)
        true = np.stack([np.interp(ts, np.arange(len(C)) * dt, C[:, k]) for k in range(3)], 1)
        walk = np.cumsum(rng.normal(0, 0.3, (len(ts), 2)), axis=0)
        walk = (walk - walk.mean(0)) * (1.2 / max(walk.std(), 1e-9)) + rng.normal(0, 0.3, (len(ts), 2))
        enu = np.c_[true[:, :2] + walk + bias[:2], true[:, 2] + rng.normal(0, 0.25, len(ts)) + bias[2]]
        lat, lon, _ = G.enu_to_geodetic(enu, (BASE_LAT, BASE_LON, 0.0))
        telem = {"source": "DJI_SRT", "is_synthetic": False, "altitude_reference": "relative",
                 "waypoints": [{"frame_id": i, "time_s": float(ts[i]), "latitude": round(float(lat[i]), 6),
                                "longitude": round(float(lon[i]), 6), "relative_altitude_m": float(enu[i, 2])}
                               for i in range(len(ts))]}
    else:
        telem = {"source": "SYNTHETIC_ESTIMATE", "is_synthetic": True, "waypoints": []}
    geo = G.georeference_reconstruction(model_dir, P_c, telem, frame_info, assumed_altitude_m=assumed_alt,
                                        log=lambda *a: None)
    if geo["status"] != "ok":
        raise RuntimeError(f"Georeferencing failed: {geo}")
    telem["georeference"], telem["metric_scale_factor"] = geo, 1.0
    world_to_display = lambda W: G.apply_transform(G.georef_transform(geo), to_c(np.atleast_2d(W)))
    return telem, G.apply_transform(G.georef_transform(geo), P_c), geo, world_to_display


def run_scenario(title, base_kw, recon_kw, threshold=1.0):
    work = tempfile.mkdtemp(prefix="prism_selftest_")
    try:
        data = os.path.join(work, "data")
        bdir = os.path.join(data, "baselines", "b1")
        os.makedirs(bdir)
        os.makedirs(os.path.join(data, "models"))
        bt, bc, bg, _ = simulate_sortie(work, "baseline", **base_kw)
        rt, rc, rg, to_disp = simulate_sortie(work, "recon", **recon_kw)
        G.save_point_cloud(os.path.join(bdir, "model_cloud.ply"), bc)
        shutil.copy(os.path.join(bdir, "model_cloud.ply"), os.path.join(bdir, "model.ply"))
        json.dump(bt, open(os.path.join(bdir, "telemetry.json"), "w"))
        G.save_point_cloud(os.path.join(data, "models", "actionable_threat_map_cloud.ply"), rc)
        shutil.copy(os.path.join(data, "models", "actionable_threat_map_cloud.ply"),
                    os.path.join(data, "models", "actionable_threat_map.ply"))
        json.dump(rt, open(os.path.join(data, "flight_telemetry.json"), "w"))
        t0 = time.time()
        res = CD.TemporalChangeDetector(data).compare_active_against_baseline("b1", threshold)
        print(f"\n=== {title}  ({time.time() - t0:.1f}s)")
        print(f"    calibration: baseline {bg['mode']}/{bg['confidence']}, recon {rg['mode']}/{rg['confidence']}; "
              f"LoD95 {res.get('level_of_detection_m')} m")
        ok = True
        found = {a["id"]: a for a in res.get("alerts", [])}
        for s in NEW:
            expect = s.H >= threshold
            # true structure position expressed in the displayed recon model (Y-up: compare x and z)
            tp = to_disp([s.cx, s.cy, s.base])[0]
            dist = lambda a: np.hypot(a["position"][0] - tp[0], a["position"][2] - tp[2])
            best = min(found.values(), key=dist, default=None)
            hit = best is not None and dist(best) < 5.0
            if hit:
                err = best["height_above_ground_m"] - s.H
                good = abs(err) <= max(0.3, 0.06 * s.H)
                ok &= good
                print(f"    {s.name:16s} true {s.H:4.1f} m -> measured {best['height_above_ground_m']:5.2f} m "
                      f"(±{best['height_uncertainty_m']:.2f}) err {err:+.2f} m  {'PASS' if good else 'FAIL'}")
            else:
                ok &= not expect
                print(f"    {s.name:16s} true {s.H:4.1f} m -> not detected  {'FAIL' if expect else 'ok (below threshold)'}")
        for r in res.get("removed_structures", []):
            print(f"    removed          true 2.6 m -> measured loss {r['max_height_loss_m']:.2f} m")
        false_alarms = len(found) - sum(1 for s in NEW if s.H >= threshold)
        print(f"    false alarms: {max(false_alarms, 0)}   ->  {'ALL PASS' if ok and false_alarms <= 0 else 'CHECK'}")
        return ok
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    a = run_scenario("Two GPS sorties (different pattern/altitude, GNSS bias, 3 m take-off offset)",
                     dict(kind="EW", alt=35.0, bias=(0, 0, 0), seed=21),
                     dict(kind="NS", alt=40.0, bias=(2.5, -1.8, 3.0), seed=22))
    b = run_scenario("No GPS log in either sortie (scale from known flight altitude)",
                     dict(kind="EW", alt=35.0, bias=(0, 0, 0), seed=31, with_gps=False, assumed_alt=35.0),
                     dict(kind="NS", alt=40.0, bias=(0, 0, 0), seed=32, with_gps=False, assumed_alt=40.0))
    print("\nRESULT:", "PASS" if (a and b) else "SOME CHECKS FAILED")
