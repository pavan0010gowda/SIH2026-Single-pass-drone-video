"""
Synthetic drone-survey scene with KNOWN ground truth, used by the accuracy tests.

Terrain: 6 % / 3 % slope plus 0.4 m undulation. Objects are sampled the way oblique photogrammetry
sees them: roofs and crowns densely, walls only on the sides facing the flight path, and NO ground
under roofs or canopies (occluded). Every coordinate gets Gaussian noise (MVS-like).
Frame: x = East, y = Up, z = South (metres), like the deployed PRISM models.
"""

import math

import numpy as np


def terrain_y(x, z):
    return 0.06 * x + 0.03 * z + 0.4 * np.sin(x / 20.0) * np.cos(z / 25.0)


GRASS = (95, 140, 70)
ROAD = (188, 170, 132)
ROOF_GREY = (150, 152, 158)
ROOF_RED = (160, 72, 60)
WALL = (205, 200, 190)
LEAF = (48, 108, 50)


class Box:
    def __init__(self, name, cx, cz, L, W, H, yaw_deg=0.0, eave=None, colour=ROOF_GREY):
        self.name, self.cx, self.cz, self.L, self.W, self.H = name, cx, cz, L, W, H
        self.yaw, self.eave, self.colour = math.radians(yaw_deg), eave, colour

    def local(self, x, z):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dz = x - self.cx, z - self.cz
        return c * dx + s * dz, -s * dx + c * dz

    def inside(self, x, z, pad=0.0):
        u, v = self.local(x, z)
        return (np.abs(u) <= self.L / 2 + pad) & (np.abs(v) <= self.W / 2 + pad)

    def base_at(self, x, z):
        return terrain_y(x, z)

    def roof_rel(self, u, v):
        if self.eave is None:
            return np.full_like(u, self.H)
        return self.eave + (self.H - self.eave) * (1.0 - np.abs(v) / (self.W / 2))

    @property
    def truth_top_xz(self):
        return self.cx, self.cz

    def truth_height(self):
        # height = roof top above the ground directly below it (flat roof: constant; gable: ridge)
        return self.H

    def sample(self, sp, rng):
        us = np.arange(-self.L / 2, self.L / 2 + 1e-9, sp)
        vs = np.arange(-self.W / 2, self.W / 2 + 1e-9, sp)
        U, V = [a.ravel() for a in np.meshgrid(us, vs)]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        X, Z = self.cx + c * U - s * V, self.cz + s * U + c * V
        base = terrain_y(self.cx, self.cz)          # building sits on a levelled pad at its centre height
        Y = base + self.roof_rel(U, V)
        pts = [np.c_[X, Y, Z]]
        cols = [np.tile(self.colour, (len(X), 1))]
        # two visible walls (south and east faces)
        for (u0, v0, u1, v1) in [(-1, 1, 1, 1), (1, -1, 1, 1)]:
            n = max(2, int(math.hypot((u1 - u0) * self.L / 2, (v1 - v0) * self.W / 2) / sp))
            t = np.linspace(0, 1, n)
            uu = (u0 + (u1 - u0) * t) * self.L / 2
            vv = (v0 + (v1 - v0) * t) * self.W / 2
            top = self.roof_rel(uu, vv)
            for f in np.arange(0.2, 1.0, 0.25):
                h = top * f
                pts.append(np.c_[self.cx + c * uu - s * vv, base + h, self.cz + s * uu + c * vv])
                cols.append(np.tile(WALL, (n, 1)))
        return np.vstack(pts), np.vstack(cols)


class Tree:
    def __init__(self, name, cx, cz, H, R):
        self.name, self.cx, self.cz, self.H, self.R = name, cx, cz, H, R

    def inside(self, x, z, pad=0.0):
        return (x - self.cx) ** 2 + (z - self.cz) ** 2 <= (self.R + pad) ** 2

    def truth_height(self):
        return self.H

    def sample(self, sp, rng):
        n = int(2 * math.pi * self.R ** 2 / sp ** 2 * 0.6)
        th = rng.uniform(0, 2 * math.pi, n)
        ph = np.arccos(rng.uniform(0.0, 1.0, n))       # upper hemisphere
        rv = 0.45 * self.H                             # crown vertical semi-axis
        base = terrain_y(self.cx, self.cz)
        X = self.cx + self.R * np.sin(ph) * np.cos(th)
        Z = self.cz + self.R * np.sin(ph) * np.sin(th)
        Y = base + (self.H - rv) + rv * np.cos(ph)
        rough = rng.normal(0, 0.12, n) * np.sin(ph)    # leafy surface
        cols = np.clip(np.array(LEAF) + rng.normal(0, 12, (n, 3)), 0, 255)
        return np.c_[X, Y + rough, Z], cols


def pothole_profile(x, z, holes):
    d = np.zeros_like(x)
    for (hx, hz, depth, dia) in holes:
        r = np.hypot(x - hx, z - hz) / (dia / 2)
        d = np.minimum(d, np.where(r < 1, -depth * (1 - r ** 2), 0.0))
    return d


def build_scene(seed=0, sp=0.18, noise=0.03, size=150.0):
    rng = np.random.default_rng(seed)
    boxes = [Box("flat_6m", -40, 40, 12, 8, 6.0, 0, None, ROOF_GREY),
             Box("gable_7.5m", 30, 45, 20, 10, 7.5, 30, 4.0, ROOF_RED),
             Box("shed_3.2m", 50, -40, 6, 6, 3.2, 10, None, ROOF_GREY),
             Box("tower_11m", -55, -20, 5, 5, 11.0, 0, None, ROOF_GREY)]
    trees = [Tree("tree_12m", -20, -45, 12.0, 4.0), Tree("tree_8m", 5, -55, 8.0, 3.0), Tree("tree_15m", 62, 12, 15.0, 5.0)]
    # a tree belt (for concealment tests) along z = 20..24, x = -10..30
    for k, xb in enumerate(np.arange(-8, 30, 5.0)):
        trees.append(Tree(f"belt_{k}", xb, 58, 9.0 + (k % 3), 3.0))
    # road: band from (-70,-25) to (70,25), 6 m wide, with potholes of known size
    ra, rb = np.array([-70.0, -25.0]), np.array([70.0, 25.0])
    rdir = (rb - ra) / np.linalg.norm(rb - ra)
    rnrm = np.array([-rdir[1], rdir[0]])
    holes = []
    for t, depth, dia in ((0.2, 0.10, 1.0), (0.4, 0.18, 1.4), (0.6, 0.30, 1.8), (0.8, 0.07, 0.9)):
        p = ra + t * (rb - ra)
        holes.append((p[0], p[1], depth, dia))

    g = np.arange(-size / 2, size / 2, sp)
    X, Z = [a.ravel() for a in np.meshgrid(g, g)]
    X = X + rng.uniform(-sp / 3, sp / 3, len(X))
    Z = Z + rng.uniform(-sp / 3, sp / 3, len(Z))
    occ = np.zeros(len(X), bool)
    for b in boxes:
        occ |= b.inside(X, Z, pad=0.3)
    for t in trees:
        occ |= t.inside(X, Z, pad=0.0)
    X, Z = X[~occ], Z[~occ]
    rel = np.c_[X, Z] - ra
    along, across = rel @ rdir, rel @ rnrm
    on_road = (along >= 0) & (along <= np.linalg.norm(rb - ra)) & (np.abs(across) <= 3.0)
    Y = terrain_y(X, Z) + np.where(on_road, pothole_profile(X, Z, holes), 0.0)
    cols = np.where(on_road[:, None], np.array(ROAD), np.array(GRASS)).astype(float)
    cols += rng.normal(0, 8, cols.shape)
    parts, pcols = [np.c_[X, Y, Z]], [cols]
    for obj in boxes + trees:
        p, c = obj.sample(sp, rng)
        parts.append(p)
        pcols.append(c)
    pts = np.vstack(parts)
    rgb = np.clip(np.vstack(pcols), 0, 255).astype(np.uint8)
    pts = pts + rng.normal(0, noise, pts.shape)
    truth = {"boxes": boxes, "trees": trees, "holes": holes, "road": (ra, rb, 6.0)}
    return pts, rgb, truth
