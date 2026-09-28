// PRISM 3-D viewer (three.js r128, global THREE).
// World coordinates are the model's metric frame: x = East, y = Up, z = South (metres). Nothing is
// re-centred or re-rotated, so every coordinate from the API can be used directly.
import { turbo } from "./geo.js";

const COLORS = {
  accent: 0xc9a45c, danger: 0xd4654f, caution: 0xd9a441, ok: 0x78a86f, info: 0x7fa6c9, text: 0xe7e4dd, muted: 0x75726b,
};
export { COLORS };

export class Viewer {
  constructor({ canvas, labelsEl, compassNeedle, scaleLabel, scaleBar }) {
    this.canvas = canvas;
    this.labelsEl = labelsEl;
    this.compassNeedle = compassNeedle;
    this.scaleLabel = scaleLabel;
    this.scaleBar = scaleBar;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1113);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 20000);
    this.camera.position.set(0, 120, 160);
    this.controls = new THREE.OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.screenSpacePanning = true;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this.scene.add(new THREE.HemisphereLight(0xf2efe8, 0x3a3630, 0.95));
    const sun = new THREE.DirectionalLight(0xffffff, 0.75);
    sun.position.set(-120, 260, 180);
    this.scene.add(sun);

    this.model = new THREE.Group();
    this.scene.add(this.model);
    this.layers = {};
    this.labels = [];
    this.points = null;
    this.pickCloud = null;
    this.mesh = null;
    this.renderMode = "points";
    this.colorMode = "photo";
    this.hag = null;
    this.classImage = null;
    this.ceiling = null;
    this.bounds = null;
    this.onHover = null;
    this.onPick = null;
    this.picking = false;
    this.materials = {
      points: new THREE.PointsMaterial({ size: 0.25, vertexColors: true, sizeAttenuation: true }),
      surface: new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }),
      clay: new THREE.MeshStandardMaterial({ color: 0xcbc6bb, roughness: 0.85, metalness: 0.0, side: THREE.DoubleSide }),
      wire: new THREE.MeshBasicMaterial({ color: 0x6f7780, wireframe: true }),
    };
    this.raycaster = new THREE.Raycaster();
    this._bindPointer();
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this.renderer.setAnimationLoop(() => this._frame());
  }

  // ------------------------------------------------------------------ loading
  loadPLY(url, onProgress) {
    return new Promise((resolve, reject) => {
      new THREE.PLYLoader().load(url, resolve, (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); }, reject);
    });
  }

  async loadPoints(url, onProgress) {
    const geo = await this.loadPLY(url, onProgress);
    this.clearModel();
    if (!geo.hasAttribute("color")) {
      const n = geo.attributes.position.count;
      geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3).fill(0.7), 3));
    }
    geo.userData.photo = geo.attributes.color.array.slice();
    geo.computeBoundingBox();
    this.points = new THREE.Points(geo, this.materials.points);
    this.points.name = "points";
    this.model.add(this.points);
    this._buildPickCloud(geo);
    this.bounds = this._robustBounds(geo);
    const spacing = Math.sqrt((this.bounds.size.x * this.bounds.size.z) / Math.max(1, geo.attributes.position.count));
    this.materials.points.size = Math.min(1.2, Math.max(0.12, spacing * 1.6));
    this.pointSpacing = spacing;
    this.hag = null;
    this.applyRenderMode();
    return geo.attributes.position.count;
  }

  async loadMesh(url, onProgress) {
    const geo = await this.loadPLY(url, onProgress);
    if (this.mesh) { this.model.remove(this.mesh); this.mesh.geometry.dispose(); }
    if (!geo.hasAttribute("normal")) geo.computeVertexNormals();
    if (geo.hasAttribute("color")) geo.userData.photo = geo.attributes.color.array.slice();
    this.mesh = new THREE.Mesh(geo, this.materials.surface);
    this.mesh.name = "mesh";
    this.model.add(this.mesh);
    this.applyRenderMode();
    this.applyColorMode();
    return geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3;
  }

  clearModel() {
    for (const obj of [this.points, this.mesh, this.pickCloud]) {
      if (obj) { this.model.remove(obj); obj.geometry.dispose(); }
    }
    this.points = this.mesh = this.pickCloud = null;
  }

  _buildPickCloud(geo) {
    // every k-th point: fast hover picking on multi-million point clouds
    const pos = geo.attributes.position.array;
    const n = pos.length / 3;
    const k = Math.max(1, Math.floor(n / 250000));
    const sub = new Float32Array(Math.ceil(n / k) * 3);
    let o = 0;
    for (let i = 0; i < n; i += k) { sub[o++] = pos[3 * i]; sub[o++] = pos[3 * i + 1]; sub[o++] = pos[3 * i + 2]; }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(sub.subarray(0, o), 3));
    this.pickCloud = new THREE.Points(g, new THREE.PointsMaterial({ visible: false }));
    this.pickCloud.visible = false;
    this.model.add(this.pickCloud);
  }

  _robustBounds(geo) {
    const pos = geo.attributes.position.array;
    const n = pos.length / 3;
    const step = Math.max(1, Math.floor(n / 40000));
    const xs = [], ys = [], zs = [];
    for (let i = 0; i < n; i += step) { xs.push(pos[3 * i]); ys.push(pos[3 * i + 1]); zs.push(pos[3 * i + 2]); }
    const q = (a, p) => { const s = a.slice().sort((u, v) => u - v); return s[Math.floor(p * (s.length - 1))]; };
    const min = new THREE.Vector3(q(xs, 0.02), q(ys, 0.02), q(zs, 0.02));
    const max = new THREE.Vector3(q(xs, 0.98), q(ys, 0.995), q(zs, 0.98));
    return { min, max, size: max.clone().sub(min), center: min.clone().add(max).multiplyScalar(0.5), ground: q(ys, 0.3) };
  }

  // ------------------------------------------------------------------ appearance
  setRenderMode(mode) {
    this.renderMode = mode;
    this.applyRenderMode();
  }

  applyRenderMode() {
    const wantMesh = this.renderMode !== "points" && this.mesh;
    if (this.points) this.points.visible = !wantMesh;
    if (this.mesh) {
      this.mesh.visible = !!wantMesh;
      this.mesh.material = this.materials[this.renderMode] || this.materials.surface;
    }
  }

  setColorMode(mode) {
    this.colorMode = mode;
    this.applyColorMode();
  }

  setHeightAboveGround(hag) {
    this.hag = hag;
    if (this.colorMode === "height" || this.ceiling !== null) this.applyColorMode();
  }

  setClassImage(img) {
    this.classImage = img;
    if (this.colorMode === "cover") this.applyColorMode();
  }

  setCeiling(h) {
    this.ceiling = h;
    this.applyColorMode();
  }

  setTerrainGrid(grid) { this.grid = grid; }

  groundAt(x, z) {
    const g = this.grid;
    if (!g) return this.bounds ? this.bounds.ground : 0;
    const fx = (x - g.x0) / g.dx, fz = (z - g.z0) / g.dx;
    const i0 = Math.max(0, Math.min(g.nz - 2, Math.floor(fz))), j0 = Math.max(0, Math.min(g.nx - 2, Math.floor(fx)));
    const tx = Math.min(1, Math.max(0, fx - j0)), tz = Math.min(1, Math.max(0, fz - i0));
    const h = (i, j) => g.heights[i * g.nx + j];
    return (1 - tz) * ((1 - tx) * h(i0, j0) + tx * h(i0, j0 + 1)) + tz * ((1 - tx) * h(i0 + 1, j0) + tx * h(i0 + 1, j0 + 1));
  }

  applyColorMode() {
    for (const obj of [this.points, this.mesh]) {
      if (!obj || !obj.geometry.attributes.color || !obj.geometry.userData.photo) continue;
      const geo = obj.geometry;
      const col = geo.attributes.color.array;
      const photo = geo.userData.photo;
      const pos = geo.attributes.position.array;
      const n = pos.length / 3;
      const hagArr = obj === this.points && this.hag && this.hag.length === n ? this.hag : null;
      const hagOf = (i) => (hagArr ? hagArr[i] : pos[3 * i + 1] - this.groundAt(pos[3 * i], pos[3 * i + 2]));
      const mode = this.colorMode;
      if (mode === "photo" && this.ceiling === null) {
        col.set(photo);
      } else {
        const ci = this.classImage;
        for (let i = 0; i < n; i++) {
          let r = photo[3 * i], g = photo[3 * i + 1], b = photo[3 * i + 2];
          if (mode === "height") {
            const h = hagOf(i);
            if (h > 0.4) {
              [r, g, b] = turbo(Math.min(1, h / 20));
            } else {
              const l = 0.25 + 0.5 * (0.3 * r + 0.59 * g + 0.11 * b);
              r = g = b = l;
            }
          } else if (mode === "cover" && ci) {
            const jx = Math.floor((pos[3 * i] - ci.x0) / ci.res), iz = Math.floor((pos[3 * i + 2] - ci.z0) / ci.res);
            if (jx >= 0 && iz >= 0 && jx < ci.nx && iz < ci.nz) {
              const o = 4 * (iz * ci.nx + jx);
              if (ci.data[o + 3] > 0) {
                const a = 0.75;
                r = (1 - a) * r + a * ci.data[o] / 255; g = (1 - a) * g + a * ci.data[o + 1] / 255; b = (1 - a) * b + a * ci.data[o + 2] / 255;
              }
            }
          }
          if (this.ceiling !== null && hagOf(i) > this.ceiling) { r = 0.9; g = 0.28; b = 0.2; }
          col[3 * i] = r; col[3 * i + 1] = g; col[3 * i + 2] = b;
        }
      }
      geo.attributes.color.needsUpdate = true;
    }
  }

  // ------------------------------------------------------------------ camera
  frameAll() {
    if (!this.bounds) return;
    const b = this.bounds;
    const span = Math.max(b.size.x, b.size.z, 40);
    const target = new THREE.Vector3(b.center.x, b.ground, b.center.z);
    this.controls.target.copy(target);
    this.camera.position.set(target.x + span * 0.05, target.y + span * 0.55, target.z + span * 0.75);
    this.camera.near = Math.max(0.05, span / 5000);
    this.camera.far = span * 30;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  topView() {
    const t = this.controls.target;
    const span = this.bounds ? Math.max(this.bounds.size.x, this.bounds.size.z, 40) : 200;
    this.flyTo(t.clone(), new THREE.Vector3(t.x, t.y + span * 0.95, t.z + 0.01));
  }

  focus(point, distance = null) {
    const p = point.clone ? point.clone() : new THREE.Vector3(point[0], point[1], point[2]);
    const dir = this.camera.position.clone().sub(this.controls.target);
    const d = distance || Math.max(15, Math.min(dir.length() * 0.5, 120));
    dir.normalize();
    if (dir.y < 0.35) { dir.y = 0.35; dir.normalize(); }
    this.flyTo(p, p.clone().add(dir.multiplyScalar(d)));
  }

  flyTo(target, position, ms = 650) {
    const t0 = performance.now();
    const sT = this.controls.target.clone(), sP = this.camera.position.clone();
    const step = (now) => {
      let k = Math.min(1, (now - t0) / ms);
      k = 1 - Math.pow(1 - k, 3);
      this.controls.target.lerpVectors(sT, target, k);
      this.camera.position.lerpVectors(sP, position, k);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ------------------------------------------------------------------ picking
  _ndc(ev) {
    const r = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  }

  pickAt(ev, precise = true) {
    const ndc = this._ndc(ev);
    this.raycaster.setFromCamera(ndc, this.camera);
    const meshShown = this.mesh && this.mesh.visible;
    if (meshShown && precise) {
      const hit = this.raycaster.intersectObject(this.mesh, false)[0];
      if (hit) return hit.point;
    }
    const cloud = precise && this.points ? this.points : this.pickCloud;
    if (!cloud) return null;
    const dist = this.camera.position.distanceTo(this.controls.target);
    this.raycaster.params.Points.threshold = Math.max(0.12, dist * 0.004, (this.pointSpacing || 0.2) * (precise ? 1.2 : 3));
    const was = cloud.visible;
    cloud.visible = true;
    const hits = this.raycaster.intersectObject(cloud, false);
    cloud.visible = was;
    if (!hits.length) return null;
    // the first hit can be a floating speck: among hits within 2 m of the first, take the one closest to the ray
    const first = hits[0].distance;
    let best = hits[0];
    for (const h of hits) {
      if (h.distance > first + 2) break;
      if (h.distanceToRay < best.distanceToRay) best = h;
    }
    return best.point.clone();
  }

  _bindPointer() {
    let down = null, lastHover = 0;
    this.canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; });
    this.canvas.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4 || e.button !== 0) return;
      if (this.picking && this.onPick) {
        const p = this.pickAt(e, true);
        if (p) this.onPick(p, e);
      }
    });
    this.canvas.addEventListener("dblclick", (e) => {
      if (this.picking) return;
      const p = this.pickAt(e, true);
      if (p) this.focus(p);
    });
    this.canvas.addEventListener("pointermove", (e) => {
      const now = performance.now();
      if (now - lastHover < 90 || !this.onHover) return;
      lastHover = now;
      this.onHover(this.pickAt(e, false));
    });
  }

  setPicking(on) {
    this.picking = on;
    this.canvas.parentElement.classList.toggle("picking", on);
  }

  // ------------------------------------------------------------------ annotation layers
  layer(name) {
    if (!this.layers[name]) {
      const g = new THREE.Group();
      g.name = name;
      this.scene.add(g);
      this.layers[name] = g;
    }
    return this.layers[name];
  }

  clearLayer(name) {
    const g = this.layers[name];
    if (g) {
      g.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) { (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); }); } });
      g.clear();
    }
    this.labels = this.labels.filter((l) => { if (l.layer === name) { l.el.remove(); return false; } return true; });
  }

  setLayerVisible(name, on) {
    if (this.layers[name]) this.layers[name].visible = on;
    for (const l of this.labels) if (l.layer === name) l.hidden = !on;
  }

  label(layerName, pos, text, cls = "") {
    const el = document.createElement("div");
    el.className = `label3d ${cls}`;
    el.textContent = text;
    this.labelsEl.appendChild(el);
    const rec = { el, pos: new THREE.Vector3(pos[0], pos[1], pos[2]), layer: layerName, hidden: false };
    this.labels.push(rec);
    return rec;
  }

  sphere(layerName, pos, radius, color) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), new THREE.MeshBasicMaterial({ color }));
    m.position.set(pos[0], pos[1], pos[2]);
    this.layer(layerName).add(m);
    return m;
  }

  line(layerName, pts, color, { dashed = false, depthTest = true, opacity = 1 } = {}) {
    const g = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
    const mat = dashed
      ? new THREE.LineDashedMaterial({ color, dashSize: 0.6, gapSize: 0.4, depthTest, transparent: opacity < 1, opacity })
      : new THREE.LineBasicMaterial({ color, depthTest, transparent: opacity < 1, opacity });
    const l = new THREE.Line(g, mat);
    if (dashed) l.computeLineDistances();
    l.renderOrder = depthTest ? 0 : 10;
    this.layer(layerName).add(l);
    return l;
  }

  tube(layerName, pts, radius, colors) {
    // polyline as a tube; `colors` = one [r,g,b] per input point or a single hex
    const v = pts.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (v.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(v, false, "centripetal", 0.2);
    const seg = Math.min(2000, v.length * 4);
    const geo = new THREE.TubeGeometry(curve, seg, radius, 6, false);
    let mat;
    if (Array.isArray(colors)) {
      const n = geo.attributes.position.count;
      const col = new Float32Array(n * 3);
      const ring = 7;
      for (let i = 0; i < n; i++) {
        const t = Math.floor(i / ring) / seg;
        const k = Math.min(colors.length - 1, Math.round(t * (colors.length - 1)));
        col.set(colors[k], 3 * i);
      }
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      mat = new THREE.MeshBasicMaterial({ vertexColors: true });
    } else {
      mat = new THREE.MeshBasicMaterial({ color: colors });
    }
    const m = new THREE.Mesh(geo, mat);
    this.layer(layerName).add(m);
    return m;
  }

  disc(layerName, center, radius, color, opacity = 0.22) {
    const grp = new THREE.Group();
    const segs = 64;
    const ring = [];
    for (let k = 0; k <= segs; k++) {
      const a = (k / segs) * Math.PI * 2;
      const x = center[0] + radius * Math.cos(a), z = center[2] + radius * Math.sin(a);
      ring.push([x, this.groundAt(x, z) + 0.25, z]);
    }
    const lg = new THREE.BufferGeometry().setFromPoints(ring.map((p) => new THREE.Vector3(...p)));
    grp.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color })));
    const fill = new THREE.Mesh(new THREE.CircleGeometry(radius, segs),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide }));
    fill.rotation.x = -Math.PI / 2;
    fill.position.set(center[0], center[1] + 0.2, center[2]);
    grp.add(fill);
    this.layer(layerName).add(grp);
    return grp;
  }

  outline(layerName, xz, color) {
    if (!xz || xz.length < 3) return null;
    const pts = xz.concat([xz[0]]).map(([x, z]) => [x, this.groundAt(x, z) + 0.3, z]);
    return this.line(layerName, pts, color);
  }

  // extruded footprint (LoD1 building block): translucent walls + roof, crisp edges
  block(layerName, xz, baseY, topY, color, opacity = 0.28) {
    if (!xz || xz.length < 3 || !(topY > baseY)) return null;
    const grp = new THREE.Group();
    const shape = new THREE.Shape(xz.map(([x, z]) => new THREE.Vector2(x, -z)));
    const geo = new THREE.ExtrudeGeometry(shape, { depth: topY - baseY, bevelEnabled: false });
    geo.rotateX(-Math.PI / 2);                      // shape plane (x, -z) -> ground plane, extrusion -> +y
    geo.translate(0, baseY, 0);
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide });
    grp.add(new THREE.Mesh(geo, mat));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 20), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
    grp.add(edges);
    this.layer(layerName).add(grp);
    return grp;
  }

  dimension(layerName, base, top, color, text) {
    const b = [base[0], base[1], base[2]], t = [top[0], top[1], top[2]];
    this.line(layerName, [b, [b[0], t[1], b[2]]], color, { depthTest: false });
    if (Math.hypot(t[0] - b[0], t[2] - b[2]) > 0.3) this.line(layerName, [[b[0], t[1], b[2]], t], color, { dashed: true, depthTest: false });
    const tick = 0.8;
    this.line(layerName, [[b[0] - tick, b[1], b[2]], [b[0] + tick, b[1], b[2]]], color, { depthTest: false });
    this.line(layerName, [[b[0] - tick, t[1], b[2]], [b[0] + tick, t[1], b[2]]], color, { depthTest: false });
    if (text) this.label(layerName, [b[0], t[1] + 0.8, b[2]], text, "accent");
  }

  // draped raster (land cover, exposure...) on the terrain grid
  drape(layerName, info, grid, { opacity = 0.7, lift = 0.25 } = {}) {
    this.clearLayer(layerName);
    const img = new Image();
    img.src = info.image;
    const tex = new THREE.Texture(img);
    img.onload = () => { tex.needsUpdate = true; };
    tex.flipY = false;
    tex.minFilter = THREE.LinearFilter;
    const nx = grid.nx, nz = grid.nz;
    const pos = new Float32Array(nx * nz * 3), uv = new Float32Array(nx * nz * 2);
    const W = info.nx * info.res, H = info.nz * info.res;
    for (let i = 0; i < nz; i++) {
      for (let j = 0; j < nx; j++) {
        const k = i * nx + j, x = grid.x0 + j * grid.dx, z = grid.z0 + i * grid.dx;
        pos[3 * k] = x; pos[3 * k + 1] = grid.heights[k] + lift; pos[3 * k + 2] = z;
        uv[2 * k] = (x - info.x0) / W; uv[2 * k + 1] = (z - info.z0) / H;
      }
    }
    const idx = [];
    for (let i = 0; i < nz - 1; i++) for (let j = 0; j < nx - 1; j++) {
      const a = i * nx + j, b = a + 1, c = a + nx, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geo.setIndex(idx);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide }));
    mesh.renderOrder = 2;
    this.layer(layerName).add(mesh);
    return mesh;
  }

  // ------------------------------------------------------------------ flight path & drone camera
  setFlightPath(points) {
    this.clearLayer("flightpath");
    if (!points || points.length < 2) return;
    this.line("flightpath", points, COLORS.info, { opacity: 0.8 });
    this.droneFrustum = null;
  }

  setDronePose(pose) {
    const layer = this.layer("drone");
    if (!pose) { layer.visible = false; return; }
    layer.visible = true;
    if (!this.droneFrustum) {
      const s = 3.0, a = 1.6, h = 0.9;
      const pts = [[0, 0, 0], [-a, h, -s], [0, 0, 0], [a, h, -s], [0, 0, 0], [a, -h, -s], [0, 0, 0], [-a, -h, -s],
        [-a, h, -s], [a, h, -s], [a, h, -s], [a, -h, -s], [a, -h, -s], [-a, -h, -s], [-a, -h, -s], [-a, h, -s]];
      const g = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
      this.droneFrustum = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: COLORS.accent, depthTest: false }));
      this.droneFrustum.renderOrder = 12;
      layer.add(this.droneFrustum);
    }
    this.droneFrustum.position.set(...pose.position);
    this.droneFrustum.quaternion.set(...pose.q);
  }

  // ------------------------------------------------------------------ frame loop
  _resize() {
    const p = this.canvas.parentElement;
    const w = p.clientWidth, h = p.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _frame() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._updateLabels();
    this._updateCompass();
  }

  _updateLabels() {
    if (!this.labels.length) return;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const v = new THREE.Vector3();
    for (const l of this.labels) {
      if (l.hidden) { l.el.style.display = "none"; continue; }
      v.copy(l.pos).project(this.camera);
      if (v.z > 1 || v.x < -1.1 || v.x > 1.1 || v.y < -1.1 || v.y > 1.1) { l.el.style.display = "none"; continue; }
      l.el.style.display = "";
      l.el.style.left = `${((v.x + 1) / 2) * w}px`;
      l.el.style.top = `${((1 - v.y) / 2) * h - 4}px`;
    }
  }

  _updateCompass() {
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    // heading of the view direction measured from north (-z), clockwise
    const heading = Math.atan2(dir.x, -dir.z);
    if (this.compassNeedle) this.compassNeedle.setAttribute("transform", `rotate(${(-heading * 180) / Math.PI})`);
    if (this.scaleLabel && this.scaleBar) {
      const d = this.camera.position.distanceTo(this.controls.target);
      const mPerPx = (2 * d * Math.tan((this.camera.fov * Math.PI) / 360)) / Math.max(1, this.canvas.clientHeight);
      const target = mPerPx * 110;
      const pow = Math.pow(10, Math.floor(Math.log10(target)));
      const nice = [1, 2, 5, 10].map((k) => k * pow).reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
      const px = nice / mPerPx;
      if (this._lastScale !== nice || Math.abs((this._lastPx || 0) - px) > 1) {
        this._lastScale = nice; this._lastPx = px;
        this.scaleBar.style.width = `${px.toFixed(0)}px`;
        this.scaleLabel.textContent = nice >= 1000 ? `${nice / 1000} km` : `${nice} m`;
      }
    }
  }
}
