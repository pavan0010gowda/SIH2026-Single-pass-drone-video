// Small DOM and formatting helpers shared by all panels.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// el("div", {class: "card", onclick: fn}, [children or text])
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  }
  return node;
}

export function toast(message, kind = "") {
  const host = $("#toasts");
  const t = el("div", { class: `toast ${kind}` }, message);
  host.appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .3s"; }, 4200);
  setTimeout(() => t.remove(), 4600);
}

export function openModal(id) { $(`#${id}`).classList.add("open"); }
export function closeModal(id) { $(`#${id}`).classList.remove("open"); }

export function kv(rows) {
  const dl = el("dl", { class: "kv" });
  for (const [k, v] of rows) {
    if (v === undefined || v === null || v === "") continue;
    dl.appendChild(el("dt", {}, k));
    dl.appendChild(el("dd", {}, v));
  }
  return dl;
}

export function stat(label, value, unit = "") {
  return el("div", { class: "stat" }, [el("div", { class: "k" }, label), el("div", { class: "v", html: `${value}${unit ? ` <small>${unit}</small>` : ""}` })]);
}

export function badge(text, kind = "") {
  return el("span", { class: `badge ${kind}` }, text);
}

export function group(title, children) {
  return el("div", { class: "group" }, [title ? el("div", { class: "group-title" }, title) : null, ...[].concat(children)]);
}

export function callout(text, kind = "info") {
  return el("div", { class: `callout ${kind}` }, text);
}

export function spinnerRow(text) {
  return el("div", { class: "row note" }, [el("span", { class: "spinner" }), text]);
}

export const fmt = {
  m(v, d = 2) { return v === null || v === undefined || !isFinite(v) ? "—" : `${Number(v).toFixed(d)} m`; },
  cm(v, d = 1) { return v === null || v === undefined || !isFinite(v) ? "—" : `${Number(v).toFixed(d)} cm`; },
  pct(v, d = 0) { return v === null || v === undefined || !isFinite(v) ? "—" : `${Number(v).toFixed(d)}%`; },
  n(v) { return v === null || v === undefined ? "—" : Number(v).toLocaleString(); },
  deg(v, d = 1) { return v === null || v === undefined || !isFinite(v) ? "—" : `${Number(v).toFixed(d)}°`; },
  ll(lat, lon) {
    if (lat === null || lat === undefined || lon === null || lon === undefined) return "—";
    return `${Math.abs(lat).toFixed(6)}° ${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(6)}° ${lon >= 0 ? "E" : "W"}`;
  },
  time(sec) {
    if (!isFinite(sec)) return "—";
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  },
  area(v) { return v === null || v === undefined ? "—" : v >= 10000 ? `${(v / 10000).toFixed(2)} ha` : `${Math.round(v).toLocaleString()} m²`; },
  len(v) { return v === null || v === undefined ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} km` : `${Math.round(v)} m`; },
  plusminus(v, s, d = 2) { return `${Number(v).toFixed(d)} ± ${Number(s).toFixed(d)} m`; },
};

export function download(filename, text, mime = "application/json") {
  const blob = new Blob([text], { type: mime });
  const a = el("a", { href: URL.createObjectURL(blob), download: filename });
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}

// 1.5 px line icons (24 x 24)
const P = {
  overview: '<path d="M4 5h16v14H4z"/><path d="M4 9h16M9 9v10"/>',
  measure: '<path d="M4 17 17 4l3 3L7 20z"/><path d="m8 13 2 2M11 10l2 2M14 7l2 2"/>',
  height: '<path d="M12 3v18M8 7l4-4 4 4M8 17l4 4 4-4"/><path d="M4 21h16"/>',
  inventory: '<path d="M4 20V10l5-4 5 4v10"/><path d="M14 20v-6l3-2 3 2v6"/><path d="M3 20h18"/>',
  ceiling: '<path d="M3 6h18" stroke-dasharray="3 2"/><path d="M6 20v-9M12 20V8M18 20v-6"/>',
  roads: '<path d="M8 3 5 21M16 3l3 18"/><path d="M12 5v2M12 11v2M12 17v2"/>',
  sites: '<circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/><path d="m5 5 1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19"/>',
  route: '<circle cx="5" cy="18" r="2"/><circle cx="19" cy="6" r="2"/><path d="M7 18h6a3 3 0 0 0 0-6H11a3 3 0 0 1 0-6h6"/>',
  compare: '<path d="M8 3v18M16 3v18"/><path d="M3 8h5M16 16h5"/><path d="m4 11 4-3M20 13l-4 3"/>',
  quality: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
  exportIcon: '<path d="M12 4v11M8 11l4 4 4-4"/><path d="M4 17v3h16v-3"/>',
  clearIcon: '<path d="M4 20h16"/><path d="m14.5 4.5 5 5L11 18H6l-2-2z"/><path d="m9.5 9.5 5 5"/>',
  reset: '<path d="M4 12a8 8 0 1 0 2.3-5.6"/><path d="M4 4v4h4"/>',
  top: '<path d="M12 3v6M9 6l3-3 3 3"/><rect x="5" y="11" width="14" height="10" rx="1"/>',
};

export function icon(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${P[name] || ""}</svg>`;
}

export const SEVERITY_KIND = { HIGH: "danger", MEDIUM: "caution", LOW: "info" };
export const CONFIDENCE_KIND = { HIGH: "ok", MEDIUM: "caution", LOW: "danger" };
