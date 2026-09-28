// North-up overview map: survey raster, flight path, live drone position, 3-D view target and
// analysis results. Click to move the 3-D view there.
const KIND_COLOR = { danger: "#d4654f", caution: "#d9a441", ok: "#78a86f", info: "#7fa6c9", accent: "#c9a45c", text: "#e7e4dd" };

export class MiniMap {
  constructor({ canvas, onClick }) {
    this.canvas = canvas;
    this.onClick = onClick;
    this.img = null;
    this.info = null;
    this.path = [];
    this.drone = null;
    this.target = null;
    this.items = () => [];
    canvas.addEventListener("click", (e) => {
      const w = this.toWorld(e.offsetX, e.offsetY);
      if (w && this.onClick) this.onClick(w[0], w[1]);
    });
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
  }

  setLayer(info) {
    this.info = info;
    if (!info) { this.img = null; this.draw(); return; }
    const img = new Image();
    img.onload = () => { this.img = img; this._fit(); this.draw(); };
    img.src = info.image;
  }

  setPath(pts) { this.path = pts || []; this.draw(); }
  setDrone(pose) { this.drone = pose; this.draw(); }
  setTarget(x, z) { this.target = [x, z]; this.draw(); }

  _fit() {
    const c = this.canvas, info = this.info;
    if (!info) return;
    const W = info.nx * info.res, H = info.nz * info.res;
    const s = Math.min(c.clientWidth / W, c.clientHeight / H) * 0.96;
    this.tf = { s, ox: (c.clientWidth - W * s) / 2 - info.x0 * s, oz: (c.clientHeight - H * s) / 2 - info.z0 * s };
  }

  toScreen(x, z) { const t = this.tf; return t ? [x * t.s + t.ox, z * t.s + t.oz] : null; }
  toWorld(px, py) { const t = this.tf; return t ? [(px - t.ox) / t.s, (py - t.oz) / t.s] : null; }

  draw() {
    const c = this.canvas;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; this._fit(); }
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    if (!this.info || !this.tf) return;
    const i = this.info, t = this.tf;
    if (this.img) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.img, i.x0 * t.s + t.ox, i.z0 * t.s + t.oz, i.nx * i.res * t.s, i.nz * i.res * t.s);
    }
    const poly = (pts, color, width = 1.5, alpha = 1) => {
      if (pts.length < 2) return;
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha;
      ctx.beginPath();
      pts.forEach((p, k) => { const s = this.toScreen(p[0], p[2]); if (k) ctx.lineTo(s[0], s[1]); else ctx.moveTo(s[0], s[1]); });
      ctx.stroke();
      ctx.globalAlpha = 1;
    };
    poly(this.path, "#7fa6c9", 1.2, 0.75);
    for (const it of this.items()) {
      const col = KIND_COLOR[it.kind] || it.color || KIND_COLOR.accent;
      if (it.type === "polyline") poly(it.points, col, it.width || 2);
      else if (it.type === "ring" && it.center) {
        const s = this.toScreen(it.center[0], it.center[2]);
        ctx.strokeStyle = col; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(s[0], s[1], Math.max(3, it.radius * t.s), 0, Math.PI * 2); ctx.stroke();
      } else if (it.type === "point") {
        const s = this.toScreen(it.pos[0], it.pos[2]);
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(s[0], s[1], 2.8, 0, Math.PI * 2); ctx.fill();
      }
    }
    if (this.target) {
      const s = this.toScreen(this.target[0], this.target[1]);
      ctx.strokeStyle = "#e7e4dd"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(s[0] - 6, s[1]); ctx.lineTo(s[0] + 6, s[1]); ctx.moveTo(s[0], s[1] - 6); ctx.lineTo(s[0], s[1] + 6); ctx.stroke();
    }
    if (this.drone) {
      const p = this.drone.position, q = this.drone.q;
      const s = this.toScreen(p[0], p[2]);
      // camera looks along its local -Z: rotate (0,0,-1) by q, keep the horizontal part
      const [x, y, z, w] = q;
      const fx = -(2 * (x * z + w * y)), fz = -(1 - 2 * (x * x + y * y));
      const a = Math.atan2(fx, -fz);
      ctx.save();
      ctx.translate(s[0], s[1]);
      ctx.rotate(a);
      ctx.fillStyle = "#c9a45c";
      ctx.beginPath(); ctx.moveTo(0, -8); ctx.lineTo(5, 5); ctx.lineTo(0, 2); ctx.lineTo(-5, 5); ctx.closePath(); ctx.fill();
      ctx.restore();
    }
  }
}
