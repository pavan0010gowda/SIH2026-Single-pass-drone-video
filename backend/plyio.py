"""
PRISM // Fast PLY and GLB geometry I/O (plyio.py)

NumPy-only readers/writers for point clouds and triangle meshes:
  * read_ply(path)  -> PlyData(xyz, rgb uint8 | None, normals | None, faces (F,3) | None)
  * write_ply(path, xyz, rgb=None, normals=None, faces=None)   binary little-endian
  * write_obj(path, xyz, rgb=None, faces=None)                 vertex colours as "v x y z r g b"
  * write_glb(path, xyz, rgb, faces, normals=None)             glTF 2.0 binary, unlit vertex colours
  * transform_glb(path_in, path_out, T4)                       applies a similarity to POSITION/NORMAL

Binary files are memory-mapped, so 100+ MB meshes load in about a second.
"""

import json
import os
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

_T = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
      "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
      "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


@dataclass
class PlyData:
    xyz: np.ndarray
    rgb: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None

    @property
    def n(self):
        return len(self.xyz)


def _header(fh):
    lines = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("PLY header has no end_header")
        s = line.decode("ascii", "ignore").strip()
        lines.append(s)
        if s == "end_header":
            return lines, fh.tell()


def _elements(lines):
    fmt, els = "ascii", []
    for h in lines:
        p = h.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element":
            els.append({"name": p[1], "count": int(p[2]), "props": []})
        elif p[0] == "property" and els:
            if p[1] == "list":
                els[-1]["props"].append(("list", p[2], p[3], p[4]))
            else:
                els[-1]["props"].append(("scalar", p[1], p[2]))
    return fmt, els


def _vertex_arrays(cols):
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = None
    if all(k in cols for k in ("nx", "ny", "nz")):
        nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float32)
    rgb = None
    for keys in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if all(k in cols for k in keys):
            c = np.stack([np.asarray(cols[k]) for k in keys], 1)
            if c.dtype.kind == "f":
                c = c * (255.0 if float(np.nanmax(c)) <= 1.0 + 1e-6 else 1.0)
            rgb = np.clip(np.round(c), 0, 255).astype(np.uint8)
            break
    return xyz, rgb, nrm


def read_ply(path) -> PlyData:
    with open(path, "rb") as fh:
        lines, off = _header(fh)
    fmt, els = _elements(lines)
    if fmt == "ascii":
        return _read_ascii(path, off, els)
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY format {fmt}")
    end = "<" if fmt == "binary_little_endian" else ">"
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=off)
    pos = 0
    out = PlyData(xyz=np.zeros((0, 3)))
    for el in els:
        props = el["props"]
        if all(p[0] == "scalar" for p in props):
            dt = np.dtype([(p[2], end + _T[p[1]]) for p in props])
            nbytes = el["count"] * dt.itemsize
            arr = np.frombuffer(data[pos:pos + nbytes].tobytes(), dtype=dt, count=el["count"])
            pos += nbytes
            if el["name"] == "vertex":
                out.xyz, out.rgb, out.normals = _vertex_arrays({k: arr[k] for k in arr.dtype.names})
            continue
        li = next(i for i, p in enumerate(props) if p[0] == "list")
        if any(p[0] == "list" for p in props[li + 1:]):
            raise ValueError("PLY element with several list properties is not supported")
        ct, it = _T[props[li][1]], _T[props[li][2]]
        pre = [(p[2], end + _T[p[1]]) for p in props[:li]]
        post = [(p[2], end + _T[p[1]]) for p in props[li + 1:]]
        # fast path: every face is a triangle
        dt = np.dtype(pre + [("n", end + ct), ("v", end + it, (3,))] + post)
        nbytes = el["count"] * dt.itemsize
        arr = np.frombuffer(data[pos:pos + nbytes].tobytes(), dtype=dt, count=el["count"])
        if el["count"] and not np.all(arr["n"] == 3):
            raise ValueError("PLY faces must be triangles")
        pos += nbytes
        if el["name"] == "face":
            out.faces = np.ascontiguousarray(arr["v"]).astype(np.int64)
    del data
    return out


def _read_ascii(path, off, els):
    out = PlyData(xyz=np.zeros((0, 3)))
    with open(path, "rb") as fh:
        fh.seek(off)
        for el in els:
            if el["name"] == "vertex":
                names = [p[2] for p in el["props"] if p[0] == "scalar"]
                rows = [fh.readline().split()[:len(names)] for _ in range(el["count"])]
                arr = np.asarray(rows, dtype=np.float64).reshape(el["count"], len(names))
                cols = {nm: arr[:, i] for i, nm in enumerate(names)}
                for k in ("red", "green", "blue", "r", "g", "b"):
                    if k in cols:
                        cols[k] = cols[k].astype(np.uint8) if cols[k].max() > 1.0 else cols[k]
                out.xyz, out.rgb, out.normals = _vertex_arrays(cols)
            elif el["name"] == "face":
                f = []
                for _ in range(el["count"]):
                    v = fh.readline().split()
                    k = int(v[0])
                    idx = [int(x) for x in v[1:1 + k]]
                    for j in range(1, k - 1):
                        f.append((idx[0], idx[j], idx[j + 1]))
                out.faces = np.asarray(f, dtype=np.int64).reshape(-1, 3)
            else:
                for _ in range(el["count"]):
                    fh.readline()
    return out


def write_ply(path, xyz, rgb=None, normals=None, faces=None, comment="PRISM"):
    xyz = np.asarray(xyz)
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    v = np.empty(n, dtype=np.dtype(fields))
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if normals is not None:
        normals = np.asarray(normals)
        v["nx"], v["ny"], v["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.dtype.kind == "f":
            rgb = np.clip(np.round(rgb * (255.0 if rgb.max() <= 1.0 + 1e-6 else 1.0)), 0, 255)
        rgb = rgb.astype(np.uint8)
        v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hdr = ["ply", "format binary_little_endian 1.0", f"comment {comment}", f"element vertex {n}"]
    hdr += [f"property {'float' if t == '<f4' else 'uchar'} {nm}" for nm, t in fields]
    if faces is not None:
        hdr += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    hdr += ["end_header"]
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as fh:
        fh.write(("\n".join(hdr) + "\n").encode("ascii"))
        fh.write(v.tobytes())
        if faces is not None:
            f = np.empty(len(faces), dtype=np.dtype([("n", "u1"), ("v", "<i4", (3,))]))
            f["n"] = 3
            f["v"] = faces
            fh.write(f.tobytes())
    os.replace(tmp, path)
    return path


def write_obj(path, xyz, rgb=None, faces=None, chunk=200000):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(f"# PRISM mesh\n# vertices {len(xyz)} faces {0 if faces is None else len(faces)}\n")
        if rgb is not None:
            data = np.hstack([np.asarray(xyz, np.float64), np.asarray(rgb, np.float64) / 255.0])
            fmt = "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
        else:
            data = np.asarray(xyz, np.float64)
            fmt = "v %.4f %.4f %.4f\n"
        for i in range(0, len(data), chunk):
            blk = data[i:i + chunk]
            fh.write((fmt * len(blk)) % tuple(blk.ravel()))
        if faces is not None:
            f1 = np.asarray(faces, np.int64) + 1
            for i in range(0, len(f1), chunk):
                blk = f1[i:i + chunk]
                fh.write(("f %d %d %d\n" * len(blk)) % tuple(blk.ravel()))
    os.replace(tmp, path)
    return path


def write_glb(path, xyz, rgb, faces, normals=None, unlit=True, generator="PRISM"):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    parts, views, accessors = [], [], []
    off = 0

    def add(buf, target):
        nonlocal off
        b = buf.tobytes()
        pad = (-len(b)) % 4
        parts.append(b + b"\x00" * pad)
        views.append(dict(buffer=0, byteOffset=off, byteLength=len(b), target=target))
        off += len(b) + pad
        return len(views) - 1

    attrs = {}
    v = add(xyz, 34962)
    accessors.append(dict(bufferView=v, componentType=5126, count=len(xyz), type="VEC3",
                          min=[float(x) for x in xyz.min(0)], max=[float(x) for x in xyz.max(0)]))
    attrs["POSITION"] = len(accessors) - 1
    if normals is not None:
        v = add(np.ascontiguousarray(normals, dtype=np.float32), 34962)
        accessors.append(dict(bufferView=v, componentType=5126, count=len(xyz), type="VEC3"))
        attrs["NORMAL"] = len(accessors) - 1
    if rgb is not None:
        c = np.empty((len(xyz), 4), np.uint8)
        c[:, :3] = rgb
        c[:, 3] = 255
        v = add(c, 34962)
        accessors.append(dict(bufferView=v, componentType=5121, normalized=True, count=len(xyz), type="VEC4"))
        attrs["COLOR_0"] = len(accessors) - 1
    v = add(faces.reshape(-1), 34963)
    accessors.append(dict(bufferView=v, componentType=5125, count=int(faces.size), type="SCALAR"))
    mat = dict(name="photogrammetry", doubleSided=True,
               pbrMetallicRoughness=dict(baseColorFactor=[1, 1, 1, 1], metallicFactor=0.0, roughnessFactor=1.0))
    gltf = dict(asset=dict(version="2.0", generator=generator), scene=0, scenes=[dict(nodes=[0])],
                nodes=[dict(mesh=0, name="actionable_threat_mesh")],
                meshes=[dict(primitives=[dict(attributes=attrs, indices=len(accessors) - 1, material=0, mode=4)])],
                materials=[mat], accessors=accessors, bufferViews=views, buffers=[dict(byteLength=off)])
    if unlit:
        mat["extensions"] = {"KHR_materials_unlit": {}}
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
    _write_glb_chunks(path, gltf, b"".join(parts))
    return path


def _write_glb_chunks(path, gltf, binb):
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    binb = binb + b"\x00" * ((-len(binb)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binb)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(binb), 0x004E4942))
        fh.write(binb)
    os.replace(tmp, path)


def transform_glb(path_in, path_out, t4):
    """Applies a 4x4 similarity to every POSITION (and rotates NORMAL) accessor of a GLB file."""
    with open(path_in, "rb") as fh:
        raw = fh.read()
    magic, _, _ = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    jlen, _ = struct.unpack_from("<II", raw, 12)
    gltf = json.loads(raw[20:20 + jlen].decode("utf-8"))
    boff = 20 + jlen
    blen, _ = struct.unpack_from("<II", raw, boff)
    binb = bytearray(raw[boff + 8:boff + 8 + blen])
    t4 = np.asarray(t4, float)
    a = t4[:3, :3]
    rn = a / np.cbrt(np.linalg.det(a))
    done = set()
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            for key in ("POSITION", "NORMAL"):
                ai = prim.get("attributes", {}).get(key)
                if ai is None or (key, ai) in done:
                    continue
                done.add((key, ai))
                acc = gltf["accessors"][ai]
                view = gltf["bufferViews"][acc["bufferView"]]
                if acc.get("componentType") != 5126 or acc.get("type") != "VEC3":
                    continue
                start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
                stride = view.get("byteStride", 12)
                cnt = acc["count"]
                if stride == 12:
                    arr = np.frombuffer(binb, dtype="<f4", count=cnt * 3, offset=start).reshape(cnt, 3).astype(np.float64)
                else:
                    arr = np.stack([np.frombuffer(binb, dtype="<f4", count=3, offset=start + i * stride)
                                    for i in range(cnt)]).astype(np.float64)
                if key == "POSITION":
                    arr = arr @ a.T + t4[:3, 3]
                    acc["min"] = [float(x) for x in arr.min(0)]
                    acc["max"] = [float(x) for x in arr.max(0)]
                else:
                    arr = arr @ rn.T
                    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
                out = arr.astype("<f4")
                if stride == 12:
                    binb[start:start + cnt * 12] = out.tobytes()
                else:
                    for i in range(cnt):
                        binb[start + i * stride:start + i * stride + 12] = out[i].tobytes()
    _write_glb_chunks(path_out, gltf, bytes(binb))
    return path_out


def ply_counts(path):
    """(vertex_count, face_count) from the header only."""
    v = f = 0
    try:
        with open(path, "rb") as fh:
            for _ in range(60):
                s = fh.readline().decode("latin1", "ignore").strip()
                if s.startswith("element vertex"):
                    v = int(s.split()[-1])
                elif s.startswith("element face"):
                    f = int(s.split()[-1])
                elif s == "end_header":
                    break
    except OSError:
        pass
    return v, f
