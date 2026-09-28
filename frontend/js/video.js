// Drone video, flight-data readouts and the augmented overlay.
// The overlay uses the camera pose of every reconstructed keyframe (position + orientation + field of
// view in the model frame); between keyframes the pose is interpolated, so 3-D results (potholes,
// sites, routes, measurements) are drawn exactly where they are in the footage.
import { $, fmt } from "./ui.js";
import { haversine, bearing } from "./geo.js";

const KIND_COLOR = { danger: "#d4654f", caution: "#d9a441", ok: "#78a86f", info: "#7fa6c9", accent: "#c9a45c", text: "#e7e4dd" };

export class VideoPanel {
  constructor({ video, canvas, empty, onPose }) {
    this.video = video;
    this.canvas = canvas;
    this.empty = empty;
    this.onPose = onPose;
    this.frames = [];
    this.wps = [];
    this.items = () => [];
    this.enabled = true;
    this.cam = new THREE.PerspectiveCamera(40, 16 / 9, 0.5, 5000);
    this.lastT = -1;
    const loop = () => { this._tick(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    video.addEventListener("error", () => this.showEmpty(true));
    video.addEventListener("loadeddata", () => this.showEmpty(false));
  }

  setSource(src) {
    if (!src) { this.video.removeAttribute("src"); this.video.load(); this.showEmpty(true); return; }
    this.video.src = src;
    this.video.load();
  }

  showEmpty(on) { this.empty.classList.toggle("hidden", !on); }

  setCameras(frames) {
    this.frames = (frames || []).filter((f) => f.position && f.q && isFinite(f.t)).sort((a, b) => a.t - b.t);
  }

  setTelemetry(telem) {
    const wps = (telem && telem.waypoints) || [];
    this.wps = wps.filter((w) => isFinite(w.latitude) && isFinite(w.longitude));
    this.hasTime = this.wps.length > 1 && this.wps.every((w) => isFinite(w.time_s));
    // cumulative distance over fresh fixes
    let d = 0;
    this.cum = new Float64Array(this.wps.length);
    for (let i = 1; i < this.wps.length; i++) {
      const a = this.wps[i - 1], b = this.wps[i];
      if (a.latitude !== b.latitude || a.longitude !== b.longitude) d += haversine(a.latitude, a.longitude, b.latitude, b.longitude);
      this.cum[i] = d;
    }
    $("#telem-source").textContent = telem && telem.source ? String(telem.source).replace(/_/g, " ") : "";
    if (!this.wps.length) for (const id of ["r-lat", "r-lon", "r-alt", "r-spd", "r-hdg", "r-dist"]) $(`#${id}`).textContent = "—";
    this.redraw();
  }

  _wpIndex(t) {
    const w = this.wps;
    if (!w.length) return -1;
    if (!this.hasTime) {
      const dur = this.video.duration || 1;
      return Math.min(w.length - 1, Math.max(0, Math.round((t / dur) * (w.length - 1))));
    }
    let lo = 0, hi = w.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (w[mid].time_s <= t) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  _readouts(t) {
    const i = this._wpIndex(t);
    if (i < 0) return;
    const w = this.wps[i];
    $("#r-lat").textContent = w.latitude.toFixed(6);
    $("#r-lon").textContent = w.longitude.toFixed(6);
    $("#r-alt").innerHTML = isFinite(w.relative_altitude_m) ? `${w.relative_altitude_m.toFixed(1)} <small>m</small>` : "—";
    // speed / heading over a 3 s window: consumer logs round GPS to ~1e-4 deg (~10 m), so a short window
    // turns rounding jumps into fake speed
    const j0 = this._wpIndex(Math.max(0, t - 1.5)), j1 = this._wpIndex(t + 1.5);
    const a = this.wps[j0], b = this.wps[j1];
    const dt = this.hasTime ? (b.time_s - a.time_s) : 1;
    const d = haversine(a.latitude, a.longitude, b.latitude, b.longitude);
    if (dt > 1.0 && d > 0.05) {
      $("#r-spd").innerHTML = `${((d / dt) * 3.6).toFixed(1)} <small>km/h</small>`;
      $("#r-hdg").innerHTML = `${bearing(a.latitude, a.longitude, b.latitude, b.longitude).toFixed(0)}<small>°</small>`;
    } else {
      $("#r-spd").innerHTML = "0.0 <small>km/h</small>";
    }
    $("#r-dist").textContent = fmt.len(this.cum[i]);
  }

  poseAt(t) {
    const f = this.frames;
    if (f.length < 2 || t < f[0].t - 2 || t > f[f.length - 1].t + 2) return null;
    let k = 0;
    while (k < f.length - 2 && f[k + 1].t <= t) k++;
    const a = f[k], b = f[k + 1];
    const u = Math.min(1, Math.max(0, (t - a.t) / Math.max(1e-6, b.t - a.t)));
    const qa = new THREE.Quaternion(...a.q), qb = new THREE.Quaternion(...b.q);
    const q = qa.clone().slerp(qb, u);
    const p = a.position.map((v, i) => v + (b.position[i] - v) * u);
    return { position: p, q: [q.x, q.y, q.z, q.w], fov: a.fov_y || 40 };
  }

  _tick() {
    const t = this.video.currentTime || 0;
    const moved = Math.abs(t - this.lastT) > 1e-3;
    if (moved) {
      this.lastT = t;
      this._readouts(t);
    }
    const pose = this.poseAt(t);
    if (moved && this.onPose) this.onPose(pose);
    this._draw(pose);
  }

  redraw() { this.lastT = -1; }

  _draw(pose) {
    const c = this.canvas;
    const W = c.clientWidth, H = c.clientHeight;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    if (!this.enabled || !pose || !this.video.videoWidth) return;
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    const s = Math.min(W / vw, H / vh);
    const dw = vw * s, dh = vh * s, ox = (W - dw) / 2, oy = (H - dh) / 2;
    const cam = this.cam;
    cam.fov = pose.fov;
    cam.aspect = vw / vh;
    cam.updateProjectionMatrix();
    cam.position.set(...pose.position);
    cam.quaternion.set(...pose.q);
    cam.updateMatrixWorld(true);
    const v = new THREE.Vector3();
    const proj = (p) => {
      v.set(p[0], p[1], p[2]).project(cam);
      const view = v.z;
      if (view > 1 || view < -1) return null;
      return [ox + ((v.x + 1) / 2) * dw, oy + ((1 - v.y) / 2) * dh, cam.position.distanceTo(new THREE.Vector3(p[0], p[1], p[2]))];
    };
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, dw, dh);
    ctx.clip();
    ctx.font = "11px 'IBM Plex Mono', monospace";
    for (const it of this.items()) {
      const col = KIND_COLOR[it.kind] || it.color || KIND_COLOR.accent;
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.lineWidth = it.width || 2;
      if (it.type === "polyline" || it.type === "ring") {
        const pts = it.points;
        ctx.beginPath();
        let pen = false;
        for (const p of pts) {
          const q = proj(p);
          if (!q) { pen = false; continue; }
          if (!pen) { ctx.moveTo(q[0], q[1]); pen = true; } else ctx.lineTo(q[0], q[1]);
        }
        ctx.globalAlpha = 0.9;
        ctx.stroke();
        ctx.globalAlpha = 1;
      } else if (it.type === "point") {
        const q = proj(it.pos);
        if (!q || q[2] > 600) continue;
        const r = Math.max(3, Math.min(10, 600 / q[2]));
        ctx.beginPath();
        ctx.arc(q[0], q[1], r, 0, Math.PI * 2);
        ctx.stroke();
        if (it.label) {
          const tw = ctx.measureText(it.label).width + 8;
          ctx.fillStyle = "rgba(20,22,24,0.85)";
          ctx.fillRect(q[0] + r + 3, q[1] - 17, tw, 16);
          ctx.fillStyle = col;
          ctx.fillText(it.label, q[0] + r + 7, q[1] - 5);
        }
      }
    }
    ctx.restore();
  }
}
