"""
PRISM // image masks for reconstruction (masks.py)

Single-pass video has two kinds of pixels that must not become geometry:
  sky / horizon        oblique video: clouds give false matches and floaters -> sky mask (straight
                       horizon fitted per frame; nothing is masked in nadir views)
  dynamic objects      vehicles, people, animals (problem-statement challenge iv). Two steps:
    1. detection       instance segmentation (YOLO-seg, COCO classes person / vehicles / animals)
                       -> every instance is removed from feature matching (poses stay clean)
    2. motion test     after SfM the relative pose of two keyframes is known, so the fundamental
                       matrix F is exact. Features inside an instance are tracked (pyramidal
                       Lucas-Kanade, forward-backward checked) into the neighbouring keyframe; a
                       static object obeys x2' F x1 = 0, a moving one does not (Sampson distance).
                       Only MOVING instances are removed from depth fusion, so parked vehicles stay
                       in the model (they are real obstacles) while traffic leaves no ghosts or
                       torn roads.
Detection needs the optional `ultralytics` package; without it only sky masks are written.
COLMAP convention: the mask of 'frame_0001.jpg' is 'frame_0001.jpg.png', 0 = ignore, 255 = use.
"""
import json
import os

import numpy as np

# COCO ids: people, vehicles, animals
DYNAMIC_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck", 8: "boat",
                   14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
                   22: "zebra", 23: "giraffe"}
YOLO_WEIGHTS = ("yolo11n-seg.pt", "yolov8n-seg.pt")


# =============================================================================
# sky
# =============================================================================
def sky_mask(bgr):
    """255 = use, 0 = sky; None when the frame shows no horizon (nadir / steep views)."""
    import cv2
    h, w = bgr.shape[:2]
    s = 480.0 / max(h, w)
    small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.GaussianBlur(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)), (7, 7), 0)
    skyish = (V > 0.45) & (grad < 18.0) & ((S < 0.30) | ((H >= 88) & (H <= 132) & (S < 0.75)))
    sh, sw = skyish.shape
    if skyish[: max(2, sh // 30)].mean() < 0.7:
        return None
    _, lab = cv2.connectedComponents(skyish.astype(np.uint8))
    top_labels = set(np.unique(lab[0][skyish[0]])) - {0}
    region = np.isin(lab, list(top_labels))
    profile = np.argmin(np.vstack([region, np.zeros((1, sw), bool)]), axis=0)
    xs = np.arange(sw, dtype=np.float64)
    ok = profile > 0.02 * sh
    if ok.sum() < 0.3 * sw:
        return None
    a, b = np.polyfit(xs[ok], profile[ok], 1)
    for _ in range(4):
        res = profile - (a * xs + b)
        inl = ok & (np.abs(res) <= max(3.0, 2.5 * 1.4826 * np.median(np.abs(res[ok]))))
        if inl.sum() < 0.2 * sw:
            break
        a, b = np.polyfit(xs[inl], profile[inl], 1)
    line = a * xs + b
    if line.min() < 0 or line.max() > 0.7 * sh or abs(a) > 0.35:
        return None
    horizon = np.maximum(0, line - 3)
    rows = np.arange(sh)[:, None]
    keep_small = (rows >= horizon[None, :]).astype(np.uint8) * 255
    return cv2.resize(keep_small, (w, h), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# dynamic objects: detection
# =============================================================================
def load_detector(device=None, log=print):
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    for w in YOLO_WEIGHTS:
        try:
            m = YOLO(w)
            m.prism_device = device
            return m
        except Exception as e:              # weights not downloadable here: try the next one
            log(f"[Masks] {w} unavailable ({e})")
    return None


def detect_instances(model, paths, imgsz=1280, conf=0.25, batch=16):
    """-> {name: [{"cls": id, "label": str, "conf": c, "poly": [[x, y], ...]}, ...]}"""
    out = {}
    classes = sorted(DYNAMIC_CLASSES)
    dev = getattr(model, "prism_device", None)
    for s in range(0, len(paths), batch):
        chunk = paths[s:s + batch]
        kw = dict(imgsz=imgsz, conf=conf, classes=classes, verbose=False, retina_masks=False)
        if dev is not None:
            kw["device"] = dev
        res = model.predict(chunk, **kw)
        for p, r in zip(chunk, res):
            inst = []
            if r.masks is not None and r.boxes is not None:
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for poly, c, f in zip(r.masks.xy, cls, cf):
                    if poly is None or len(poly) < 3:
                        continue
                    inst.append({"cls": int(c), "label": DYNAMIC_CLASSES.get(int(c), str(c)), "conf": round(float(f), 3),
                                 "poly": np.asarray(poly, float).round(1).tolist()})
            out[os.path.basename(p)] = inst
    return out


def instance_mask(shape, poly, dilate_px=0):
    import cv2
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32)], 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m > 0


# =============================================================================
# dynamic objects: motion test with the known relative pose
# =============================================================================
def fundamental_from_poses(K1, R1, T1, K2, R2, T2):
    """x2^T F x1 = 0 for world->camera poses (x_cam = R X + T)."""
    R = R2 @ R1.T
    t = T2 - R @ T1
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K2).T @ tx @ R @ np.linalg.inv(K1)
    return F / (np.linalg.norm(F) + 1e-12)


def sampson(F, p1, p2):
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.einsum("ij,ij->i", x2, Fx1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def _track(g1, g2, mask, max_pts=60):
    import cv2
    p = cv2.goodFeaturesToTrack(g1, maxCorners=max_pts, qualityLevel=0.01, minDistance=3,
                                mask=mask.astype(np.uint8) * 255 if mask is not None else None, blockSize=5)
    if p is None or len(p) < 3:
        return None, None
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    q, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p, None, **lk)
    b, st2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, q, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((b - p).reshape(-1, 2), axis=1) < 0.7)
    if ok.sum() < 3:
        return None, None
    return p.reshape(-1, 2)[ok], q.reshape(-1, 2)[ok]


def moving_flags(img1, img2, F, instances, min_px=2.5):
    """
    For the instances of img1: True = moving, False = static, None = undecided (too little texture).
    Returns (flags, background median Sampson distance in px). A bad F (background not consistent)
    makes every instance 'undecided'.
    """
    import cv2
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    union = np.zeros(g1.shape, bool)
    masks = []
    for inst in instances:
        m = instance_mask(g1.shape, inst["poly"])
        masks.append(m)
        union |= m
    bg_mask = ~cv2.dilate(union.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    p, q = _track(g1, g2, bg_mask, max_pts=400)
    if p is None:
        return [None] * len(instances), None
    bg = float(np.median(sampson(F, p, q)))
    if bg > 1.5:                                       # the pose does not explain the background
        return [None] * len(instances), bg
    thr = max(min_px, 4.0 * bg)
    flags = []
    for m in masks:
        ero = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        pi, qi = _track(g1, g2, ero if ero.sum() > 30 else m)
        if pi is None or len(pi) < 4:
            flags.append(None)
            continue
        flags.append(bool(np.median(sampson(F, pi, qi)) > thr))
    return flags, bg


# =============================================================================
# writing masks
# =============================================================================
def write_masks(image_dir, names, out_dir, instances=None, moving=None, dilate_frac=0.006, sky=True):
    """
    Writes COLMAP masks. instances: {name: [inst...]} to remove; moving: {name: [flag...]} restricts the
    removal to moving (True) or undecided (None) instances. Returns stats.
    """
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    n_sky, n_inst = 0, 0
    for n in names:
        img = cv2.imread(os.path.join(image_dir, n))
        if img is None:
            continue
        keep = np.full(img.shape[:2], 255, np.uint8)
        if sky:
            sm = sky_mask(img)
            if sm is not None:
                keep = np.minimum(keep, sm)
                n_sky += 1
        dil = max(2, int(round(dilate_frac * max(img.shape[:2]))))
        flags = (moving or {}).get(n) or []
        for k, inst in enumerate((instances or {}).get(n, [])):
            if moving is not None and k < len(flags) and flags[k] is False:
                continue                           # static (parked) object: keep it in the model
            keep[instance_mask(img.shape, inst["poly"], dil)] = 0
            n_inst += 1
        cv2.imwrite(os.path.join(out_dir, n + ".png"), keep)
    return {"frames": len(names), "sky_frames": n_sky, "masked_instances": n_inst}


def summarise(instances, moving=None):
    by = {}
    mov = stat = und = 0
    for n, lst in (instances or {}).items():
        for k, inst in enumerate(lst):
            by[inst["label"]] = by.get(inst["label"], 0) + 1
            if moving is not None:
                f = (moving.get(n) or [])[k] if k < len(moving.get(n) or []) else None
                mov += f is True
                stat += f is False
                und += f is None
    out = {"detections": sum(by.values()), "by_class": by}
    if moving is not None:
        out.update(moving=mov, static=stat, undecided=und)
    return out


def save_instances(path, instances):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instances, f)
