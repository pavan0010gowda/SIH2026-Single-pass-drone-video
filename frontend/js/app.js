// PRISM dashboard: application shell.
import * as api from "./api.js";
import { $, $$, el, toast, fmt, icon, CONFIDENCE_KIND } from "./ui.js";
import { setOrigin, toLatLon, hasOrigin } from "./geo.js";
import { Viewer } from "./viewer.js";
import { VideoPanel } from "./video.js";
import { MiniMap } from "./minimap.js";
import { initMissions } from "./missions.js";
import overview from "./panels/overview.js";
import measure from "./panels/measure.js";
import height from "./panels/height.js";
import inventory from "./panels/inventory.js";
import roads from "./panels/roads.js";
import sites from "./panels/sites.js";
import route from "./panels/route.js";
import ceiling from "./panels/ceiling.js";
import compare from "./panels/compare.js";
import quality from "./panels/quality.js";
import { initExports } from "./exports.js";

const PANELS = { overview, measure, height, inventory, ceiling, roads, sites, route, compare, quality };
const RAIL = ["overview", "measure", "height", "inventory", "ceiling", "|", "roads", "sites", "route", "compare", "|", "quality", "export", "|", "clear", "reset", "top"];
const RAIL_TIPS = { overview: "Survey overview", measure: "Measure distance (M)", height: "Measure height (H)",
  inventory: "Structures: roofs, facades, trees, vehicles", ceiling: "Height ceiling", roads: "Roads & potholes", sites: "Base site finder",
  route: "Covert route", compare: "Change detection", quality: "Quality & accuracy (Q)", export: "Export centre (Ctrl+E)", clear: "Clear all overlays (C)",
  reset: "Reset camera (R)", top: "Top view, north up (T)" };

const viewer = new Viewer({ canvas: $("#gl"), labelsEl: $("#labels"), compassNeedle: $("#compass-needle"),
  scaleLabel: $("#scale-label"), scaleBar: $("#scale-bar") });

const annotations = {};            // panel id -> [{type, ...}] shared by the video overlay and the map
const allItems = () => Object.values(annotations).flat();
const video = new VideoPanel({ video: $("#video"), canvas: $("#video-ar"), empty: $("#video-empty"),
  onPose: (pose) => { viewer.setDronePose(pose); minimap.setDrone(pose); } });
video.items = allItems;
const minimap = new MiniMap({ canvas: $("#minimap"), onClick: (x, z) => viewer.focus([x, viewer.groundAt(x, z), z], 90) });
minimap.items = allItems;

const state = {
  survey: null,          // {kind: "active"|"baseline", id, name, pointsFile, meshFile, info, telemetry, calibration}
  terrainGrid: null,
  layers: {},
  analysis: {},          // results kept for the report export
  activeTool: null,
  pick: null,
};

// --------------------------------------------------------------------------------------------- context
const ctx = {
  api, viewer, video, minimap, state, toast,
  bid() { return state.survey && state.survey.kind === "baseline" ? state.survey.id : null; },
  bq(prefix = "?") { const b = ctx.bid(); return b ? `${prefix}baseline_id=${encodeURIComponent(b)}` : ""; },
  setAnnotations(id, items) { annotations[id] = items || []; video.redraw(); minimap.draw(); },
  keep(key, value) { state.analysis[key] = value; },
  async terrainGrid() {
    if (!state.terrainGrid) {
      state.terrainGrid = await api.get(`/api/terrain/grid${ctx.bq()}`);
      viewer.setTerrainGrid(state.terrainGrid);
    }
    return state.terrainGrid;
  },
  async layer(name) {
    if (!state.layers[name]) state.layers[name] = await api.get(`/api/terrain/layer/${name}${ctx.bq()}`);
    return state.layers[name];
  },
  requestPick({ prompt, onPick, continuous = false }) {
    cancelPick();
    state.pick = { onPick, continuous };
    viewer.setPicking(true);
    const b = $("#pick-banner");
    b.innerHTML = `${prompt} <kbd>Esc</kbd> to stop`;
    b.classList.remove("hidden");
  },
  cancelPick,
  setStatus,
  openTool,
  metric() { return !!(state.survey && state.survey.calibration && state.survey.calibration.metric); },
};

function cancelPick() {
  state.pick = null;
  viewer.setPicking(false);
  $("#pick-banner").classList.add("hidden");
}

viewer.onPick = (p) => {
  if (!state.pick) return;
  const cb = state.pick.onPick;
  if (!state.pick.continuous) cancelPick();
  cb([p.x, p.y, p.z]);
};

viewer.onHover = (p) => {
  if (!p) return;
  $("#c-xyz").textContent = `x ${p.x.toFixed(2)}  y ${p.y.toFixed(2)}  z ${p.z.toFixed(2)} m`;
  const ll = toLatLon(p.x, p.z);
  $("#c-ll").textContent = ll ? fmt.ll(ll.lat, ll.lon) : "";
  $("#c-hag").textContent = state.terrainGrid ? `${(p.y - viewer.groundAt(p.x, p.z)).toFixed(2)} m above ground` : "";
};

// --------------------------------------------------------------------------------------------- status
function setStatus(text, kind = "") {
  $("#sb-text").textContent = text;
  $("#sb-dot").className = `status-dot ${kind}`;
}

function setJobProgress(label, pct) {
  const p = $("#sb-progress");
  if (label === null) { p.classList.add("hidden"); $("#sb-job").textContent = ""; return; }
  p.classList.remove("hidden");
  p.firstElementChild.style.width = `${Math.max(2, Math.min(100, pct))}%`;
  $("#sb-job").textContent = label;
}
ctx.setJobProgress = setJobProgress;

function showCalibration(cal) {
  const dot = $("#cal-dot"), text = $("#cal-text");
  if (!cal || cal.status !== "ok") {
    dot.className = "status-dot danger";
    text.textContent = "Not metric: heights unavailable";
    $("#sb-cal").textContent = "Scale: uncalibrated";
    return;
  }
  const kind = CONFIDENCE_KIND[cal.confidence] || "caution";
  dot.className = `status-dot ${kind}`;
  const unc = cal.scale_rel_uncertainty != null ? ` ±${(100 * cal.scale_rel_uncertainty).toFixed(2)}%` : "";
  const mode = { GPS_SIM3: "GPS", GPS_4DOF: "GPS", GPS_LINE: "GPS (line)", ALTITUDE_AGL: "altitude", ASSUMED_ALTITUDE: "assumed altitude" }[cal.mode] || cal.mode || "GPS";
  text.textContent = `Scale from ${mode} · ${String(cal.confidence || "").toLowerCase()} confidence`;
  $("#sb-cal").textContent = `Metric scale${unc}${cal.latlon_verified ? " · north verified" : ""}`;
}

// --------------------------------------------------------------------------------------------- survey loading
async function loadSurvey(spec) {
  const veil = $("#veil"), bar = $("#veil-bar");
  veil.classList.remove("hidden");
  $("#veil-text").textContent = "Loading survey…";
  bar.style.width = "3%";
  cancelPick();
  for (const id of Object.keys(annotations)) annotations[id] = [];
  for (const name of Object.keys(viewer.layers)) viewer.clearLayer(name);
  state.terrainGrid = null;
  state.layers = {};
  state.analysis = {};
  state.survey = spec;
  for (const p of Object.values(PANELS)) if (p.reset) p.reset();
  viewer.hag = null;
  viewer.classImage = null;
  viewer.ceiling = null;
  viewer.grid = null;
  try {
    let pointsFile, meshFile, telemetry = {}, frames = [], videoSrc = null, pv = 0, mv = 0;
    if (spec.kind === "active") {
      const info = await api.get("/api/model/info");
      if (!info.exists) {
        veil.querySelector("#veil-text").textContent = "No survey yet — start a new mission.";
        bar.style.width = "0";
        $("#mission-name").textContent = "No survey loaded";
        openTool("overview");
        return;
      }
      spec.info = info;
      pointsFile = info.points_file;
      meshFile = info.has_mesh ? "data/models/actionable_threat_mesh.ply" : null;
      pv = Math.round(info.last_modified || 0);
      mv = Math.round(info.mesh_last_modified || 0);
      telemetry = await api.get("/api/telemetry").catch(() => ({}));
      frames = (await api.get("/api/cameras").catch(() => ({ frames: [] }))).frames;
      videoSrc = api.url("data/raw_videos/drone_flight.mp4") + `?v=${Math.round(info.last_modified || 0)}`;
      spec.calibration = info.calibration;
      spec.name = spec.name || describeLocation(telemetry, info);
    } else {
      const b = spec.meta;
      pointsFile = `data/baselines/${b.id}/model.ply`;
      meshFile = b.has_mesh ? `data/baselines/${b.id}/mesh.ply` : null;
      pv = b.model_version || 0;
      mv = b.mesh_version || 0;
      telemetry = await api.get(`/data/baselines/${b.id}/telemetry.json`).catch(() => ({}));
      const fi = await api.get(`/data/baselines/${b.id}/frame_index.json`).catch(() => null);
      frames = Array.isArray(fi) ? fi.filter((f) => f.position && f.quaternion_xyzw).map((f) => ({ t: f.timestamp_sec, position: f.position, q: f.quaternion_xyzw, fov_y: f.fov_y_deg })) : [];
      videoSrc = b.video_url ? api.url(b.video_url) : null;
      spec.calibration = b.calibration || (telemetry.georeference ? { ...telemetry.georeference, metric: !!telemetry.georeference.metric } : null);
      spec.name = b.name || b.id;
    }
    spec.pointsFile = pointsFile;
    spec.telemetry = telemetry;
    const geo = (telemetry && telemetry.georeference) || {};
    setOrigin(geo.origin || (spec.calibration && spec.calibration.origin));
    $("#mission-name").textContent = spec.kind === "baseline" ? `Baseline · ${spec.name}` : spec.name;
    showCalibration(spec.calibration);
    $$('[data-action="back-to-active"]').forEach((b) => b.classList.toggle("hidden", spec.kind !== "baseline"));

    const n = await viewer.loadPoints(`${api.url(pointsFile)}?v=${pv}`, (f) => { bar.style.width = `${5 + 80 * f}%`; });
    viewer.frameAll();
    $("#sb-model").textContent = `${fmt.n(n)} points`;
    veil.classList.add("hidden");
    video.setSource(videoSrc);
    video.setCameras(frames);
    video.setTelemetry(telemetry);
    const path = telemetry.camera_trajectory || frames.map((f) => f.position);
    viewer.setFlightPath(path);
    viewer.setLayerVisible("flightpath", viewState.overlays.flightpath);
    minimap.setPath(path);
    setStatus("Survey loaded", "ok");
    if (meshFile) {
      setStatus("Loading surface mesh…", "busy");
      viewer.loadMesh(`${api.url(meshFile)}?v=${mv}`).then((faces) => {
        $("#sb-model").textContent = `${fmt.n(n)} points · ${fmt.n(Math.round(faces))} faces`;
        setStatus("Survey loaded", "ok");
        syncViewButtons();
        if (state.activeTool === "overview") openTool("overview");      // show the mesh in the model summary
      }).catch(() => setStatus("Mesh could not be loaded", "caution"));
    }
    syncViewButtons();
    // terrain products (built once per model on the server, then cached)
    if (ctx.metric()) {
      setStatus("Analysing terrain…", "busy");
      Promise.all([ctx.terrainGrid(), ctx.layer("ortho")]).then(([grid, ortho]) => {
        minimap.setLayer(ortho);
        viewer.applyColorMode();
        setStatus("Ready", "ok");
        if (state.activeTool && PANELS[state.activeTool].onTerrain) PANELS[state.activeTool].onTerrain(ctx);
      }).catch((e) => setStatus(`Terrain analysis unavailable: ${e.message}`, "caution"));
    }
    openTool(state.activeTool || "overview");
  } catch (e) {
    $("#veil-text").textContent = `Could not load the survey: ${e.message}`;
    setStatus("Load failed", "danger");
  }
}
ctx.loadSurvey = loadSurvey;
ctx.reloadActive = () => loadSurvey({ kind: "active" });

function describeLocation(telem, info) {
  const lat = telem.center_latitude ?? info.center_latitude, lon = telem.center_longitude ?? info.center_longitude;
  if (lat != null && lon != null) return `Survey at ${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;
  return "Current survey";
}

// --------------------------------------------------------------------------------------------- view state
const viewState = { render: "points", color: "photo", overlays: { exposure: false, flightpath: true } };

function syncViewButtons() {
  const hasMesh = !!viewer.mesh;
  $$("[data-render]").forEach((b) => {
    const on = b.dataset.render === viewState.render;
    b.classList.toggle("on", on);
    const chk = b.querySelector(".check");
    if (chk) chk.textContent = on ? "✓" : "";
    b.disabled = b.dataset.render !== "points" && !hasMesh;
    if (b.disabled) b.title = "Build the surface mesh first (Mission ▸ Build surface mesh)";
  });
  $$("[data-color]").forEach((b) => {
    const on = b.dataset.color === viewState.color;
    b.classList.toggle("on", on);
    const chk = b.querySelector(".check");
    if (chk) chk.textContent = on ? "✓" : "";
    b.disabled = b.dataset.color !== "photo" && !ctx.metric();
  });
  $$("[data-overlay]").forEach((b) => { const chk = b.querySelector(".check"); if (chk) chk.textContent = viewState.overlays[b.dataset.overlay] ? "✓" : ""; });
}

async function setRender(mode) {
  if (mode !== "points" && !viewer.mesh) { toast("No surface mesh yet. Use Mission ▸ Build surface mesh."); return; }
  viewState.render = mode;
  viewer.setRenderMode(mode);
  syncViewButtons();
}

async function setColor(mode) {
  viewState.color = mode;
  syncViewButtons();
  try {
    if (mode === "height" && !viewer.hag) {
      setStatus("Computing heights above ground…", "busy");
      await ctx.terrainGrid();
      const buf = await api.getBinary(`/api/analysis/hag?file=${encodeURIComponent(state.survey.pointsFile)}${ctx.bq("&")}`);
      viewer.setHeightAboveGround(new Float32Array(buf));
      setStatus("Colour: height above ground (0–20 m)", "ok");
    }
    if (mode === "cover" && !viewer.classImage) {
      setStatus("Loading land cover…", "busy");
      const lay = await ctx.layer("classes");
      viewer.setClassImage(await rasterData(lay));
      setStatus("Colour: land cover", "ok");
    }
  } catch (e) {
    toast(`Colour mode unavailable: ${e.message}`, "danger");
  }
  viewer.setColorMode(mode);
}

function rasterData(lay) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = lay.nx; c.height = lay.nz;
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0);
      resolve({ data: g.getImageData(0, 0, lay.nx, lay.nz).data, x0: lay.x0, z0: lay.z0, res: lay.res, nx: lay.nx, nz: lay.nz });
    };
    img.src = lay.image;
  });
}

async function toggleOverlay(name) {
  viewState.overlays[name] = !viewState.overlays[name];
  const on = viewState.overlays[name];
  syncViewButtons();
  if (name === "flightpath") { viewer.setLayerVisible("flightpath", on); viewer.setLayerVisible("drone", on); return; }
  if (name === "exposure") {
    if (!on) { viewer.clearLayer("exposure"); return; }
    try {
      setStatus("Computing visibility from buildings…", "busy");
      const [grid, lay] = await Promise.all([ctx.terrainGrid(), ctx.layer("exposure")]);
      viewer.drape("exposure", lay, grid, { opacity: 0.55 });
      setStatus("Overlay: probability of being seen from buildings (blue low → red high)", "ok");
    } catch (e) { toast(e.message, "danger"); viewState.overlays.exposure = false; syncViewButtons(); }
  }
}

// --------------------------------------------------------------------------------------------- tools / inspector
function buildRail() {
  const rail = $("#rail");
  for (const id of RAIL) {
    if (id === "|") { rail.appendChild(el("div", { class: "sep" })); continue; }
    const iconName = id === "export" ? "exportIcon" : id === "clear" ? "clearIcon" : id;
    const b = el("button", { type: "button", "data-tip": RAIL_TIPS[id], "data-rail": id, html: icon(iconName) });
    b.addEventListener("click", () => {
      if (id === "reset") viewer.frameAll();
      else if (id === "top") viewer.topView();
      else if (id === "export") exportsUi.open();
      else if (id === "clear") clearOverlays();
      else openTool(id);
    });
    rail.appendChild(b);
  }
}

let mounted = null;
function openTool(id) {
  const p = PANELS[id];
  if (!p) return;
  if (mounted && mounted.unmount) mounted.unmount(ctx);
  cancelPick();
  state.activeTool = id;
  mounted = p;
  $("#workspace").classList.remove("no-inspector");
  $("#sb-show-inspector").classList.add("hidden");
  $("#insp-title").textContent = p.title;
  const body = $("#insp-body");
  body.innerHTML = "";
  $$("[data-rail]").forEach((b) => b.classList.toggle("on", b.dataset.rail === id));
  if (!state.survey || (state.survey.kind === "active" && !state.survey.info)) {
    body.appendChild(el("div", { class: "empty" }, "No survey is loaded. Start a new mission or open a saved survey."));
    if (id !== "overview") return;
  }
  if (p.needsMetric && state.survey && !ctx.metric()) {
    body.appendChild(el("div", { class: "callout danger" },
      "This model has no metric calibration, so measurements would be meaningless. Recalibrate it with its flight log (Mission ▸ Recalibrate) or re-run the mission with the .SRT file."));
    return;
  }
  p.mount(ctx, body);
}

$("#insp-close").addEventListener("click", () => {
  $("#workspace").classList.add("no-inspector");
  $("#sb-show-inspector").classList.remove("hidden");
});
$("#sb-show-inspector").addEventListener("click", () => openTool(state.activeTool || "overview"));

// --------------------------------------------------------------------------------------------- menus & actions
function closeMenus() { $$(".menu").forEach((m) => m.classList.remove("open")); }
$$(".menu > button").forEach((b) => b.addEventListener("click", (e) => {
  e.stopPropagation();
  const m = b.parentElement, was = m.classList.contains("open");
  closeMenus();
  if (!was) m.classList.add("open");
}));
document.addEventListener("click", closeMenus);

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-action],[data-tool],[data-render],[data-color],[data-overlay]");
  if (!t || t.disabled) return;
  if (t.dataset.tool) openTool(t.dataset.tool);
  else if (t.dataset.render) setRender(t.dataset.render);
  else if (t.dataset.color) setColor(t.dataset.color);
  else if (t.dataset.overlay) toggleOverlay(t.dataset.overlay);
  else if (t.dataset.action) runAction(t.dataset.action);
});

async function runAction(a) {
  switch (a) {
    case "new-mission": missions.openNewMission(); break;
    case "open-survey": missions.openSurveys(); break;
    case "save-survey": missions.openSave(); break;
    case "back-to-active": loadSurvey({ kind: "active" }); break;
    case "recalibrate": recalibrate(); break;
    case "build-mesh": buildMesh(); break;
    case "reset-view": viewer.frameAll(); break;
    case "top-view": viewer.topView(); break;
    case "export-centre": exportsUi.open(); break;
    case "clear-overlays": clearOverlays(); break;
    default: break;
  }
}

// every result drawn on the model (routes, sites, labels, change boxes, markers...) off in one go;
// the analyses themselves stay cached, so any of them can be shown again instantly
function clearOverlays() {
  cancelPick();
  for (const p of Object.values(PANELS)) if (p.clear) { try { p.clear(ctx); } catch (_) { /* keep clearing the rest */ } }
  const core = new Set(["flightpath", "drone"]);
  for (const name of Object.keys(viewer.layers)) if (!core.has(name)) viewer.clearLayer(name);
  for (const id of Object.keys(annotations)) annotations[id] = [];
  viewer.setCeiling(null);
  viewState.overlays.exposure = false;
  video.redraw();
  minimap.draw();
  syncViewButtons();
  if (state.activeTool) openTool(state.activeTool);
  setStatus("All overlays cleared", "ok");
}

async function recalibrate() {
  if (state.survey && state.survey.kind === "baseline") { toast("Recalibration applies to the current mission. Go back to it first."); return; }
  setStatus("Recalibrating with the flight log…", "busy");
  try {
    const r = await api.post("/api/model/recalibrate", { force: true });
    toast(`Recalibrated: scale ×${r.scale_correction}, GPS fit ${r.gps_rmse_m} m, confidence ${String(r.confidence).toLowerCase()}.`, "ok");
    loadSurvey({ kind: "active" });
  } catch (e) {
    setStatus("Recalibration failed", "danger");
    toast(`Recalibration not possible: ${e.message}`, "danger");
  }
}

async function buildMesh() {
  try {
    await api.post("/api/model/meshify", { baseline_id: ctx.bid() });
  } catch (e) { toast(e.message, "danger"); return; }
  setJobProgress("Building mesh", 2);
  setStatus("Building gap-free surface mesh…", "busy");
  const poll = setInterval(async () => {
    let s;
    try { s = await api.get("/api/model/meshify/status"); } catch (_) { return; }
    setJobProgress(s.message || "Building mesh", s.percent || 0);
    if (s.status === "completed") {
      clearInterval(poll);
      setJobProgress(null);
      const r = s.result || {};
      toast(`Surface mesh ready: ${fmt.n(r.triangles)} faces in ${r.seconds} s.`, "ok");
      await viewer.loadMesh(`${api.url(r.mesh_ply_url)}?v=${Date.now()}`);
      if (state.survey && state.survey.meta) state.survey.meta.has_mesh = true;
      setRender("surface");
      setStatus("Surface mesh loaded", "ok");
    } else if (s.status === "failed") {
      clearInterval(poll);
      setJobProgress(null);
      setStatus("Mesh failed", "danger");
      toast(`Mesh failed: ${s.error}`, "danger");
    }
  }, 1200);
}

// --------------------------------------------------------------------------------------------- keyboard
document.addEventListener("keydown", (e) => {
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
  if (e.key === "Escape") { cancelPick(); closeMenus(); $$(".modal-back.open").forEach((m) => m.classList.remove("open")); return; }
  if (e.ctrlKey || e.metaKey) {
    const k = e.key.toLowerCase();
    if (k === "s") { e.preventDefault(); missions.openSave(); }
    if (k === "o") { e.preventDefault(); missions.openSurveys(); }
    if (k === "n") { e.preventDefault(); missions.openNewMission(); }
    if (k === "e") { e.preventDefault(); exportsUi.open(); }
    return;
  }
  const k = e.key.toLowerCase();
  if (k === "m") openTool("measure");
  else if (k === "h") openTool("height");
  else if (k === "q") openTool("quality");
  else if (k === "c") clearOverlays();
  else if (k === "r") viewer.frameAll();
  else if (k === "t") viewer.topView();
});

// --------------------------------------------------------------------------------------------- misc wiring
$("#ar-toggle").addEventListener("change", (e) => { video.enabled = e.target.checked; video.redraw(); });
$$("#map-seg button").forEach((b) => b.addEventListener("click", async () => {
  $$("#map-seg button").forEach((x) => x.classList.toggle("on", x === b));
  try { minimap.setLayer(await ctx.layer(b.dataset.layer)); } catch (e) { toast(e.message, "danger"); }
}));
setInterval(() => { const t = viewer.controls.target; minimap.setTarget(t.x, t.z); }, 400);

const missions = initMissions(ctx);
const exportsUi = initExports(ctx);
buildRail();
syncViewButtons();
loadSurvey({ kind: "active" });
missions.resumeIfRunning();
