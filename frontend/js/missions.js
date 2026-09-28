// Mission ingestion (local pipeline or Colab package), saved surveys, baselines.
import * as api from "./api.js";
import { $, $$, el, toast, openModal, closeModal, fmt, badge, CONFIDENCE_KIND } from "./ui.js";

export function initMissions(ctx) {
  const m = { source: "video", quality: "medium", media: null, log: null, rtk: null, cal: null, imu: null, poll: null, t0: 0 };

  // ------------------------------------------------------------------ modal plumbing
  $$(".modal-back").forEach((back) => {
    back.addEventListener("click", (e) => { if (e.target === back || e.target.closest("[data-close]")) back.classList.remove("open"); });
  });

  function setSource(s) {
    m.source = s;
    $$("#mm-source .choice").forEach((c) => c.classList.toggle("on", c.dataset.source === s));
    $("#mm-media-title").textContent = s === "video" ? "Drop the drone video here, or click to choose"
      : s === "frames" ? "Drop a ZIP of photos here, or click to choose" : "Drop prism_colab_bundle.zip here, or click to choose";
    $("#mm-media").accept = s === "video" ? ".mp4,.mov,.avi,.mkv,.m4v" : ".zip";
    $("#mm-drop-log").classList.toggle("hidden", s === "colab");
    $("#mm-quality-field").classList.toggle("hidden", s === "colab");
    $("#mm-optional").classList.toggle("hidden", s === "colab");
    $("#mm-colab-note").classList.toggle("hidden", s !== "colab");
    $("#mm-start").textContent = s === "colab" ? "Import package" : "Start reconstruction";
  }
  $$("#mm-source .choice").forEach((c) => c.addEventListener("click", () => setSource(c.dataset.source)));
  $$("#mm-quality .choice").forEach((c) => c.addEventListener("click", () => {
    m.quality = c.dataset.quality;
    $$("#mm-quality .choice").forEach((x) => x.classList.toggle("on", x === c));
  }));

  function bindDrop(zone, input, onFile) {
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("over");
      if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => { if (input.files.length) onFile(input.files[0]); });
  }
  const mb = (f) => `${(f.size / 1e6).toFixed(f.size > 1e8 ? 0 : 1)} MB`;
  bindDrop($("#mm-drop-media"), $("#mm-media"), (f) => {
    m.media = f;
    $("#mm-media-file").textContent = `${f.name} · ${mb(f)}`;
    if (/\.zip$/i.test(f.name) && /(colab|bundle|prism)/i.test(f.name)) setSource("colab");
  });
  bindDrop($("#mm-drop-log"), $("#mm-log"), (f) => {
    m.log = f;
    $("#mm-log-file").textContent = `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
  });
  for (const [key, zone, input, label] of [["rtk", "#mm-drop-rtk", "#mm-rtk", "#mm-rtk-file"], ["cal", "#mm-drop-cal", "#mm-cal", "#mm-cal-file"],
    ["imu", "#mm-drop-imu", "#mm-imu", "#mm-imu-file"]]) {
    bindDrop($(zone), $(input), (f) => { m[key] = f; $(label).textContent = `${f.name} · ${(f.size / 1024).toFixed(0)} KB`; });
  }

  function showProgress(stage, pct, lines) {
    $("#mm-progress").classList.remove("hidden");
    $("#mm-stage").textContent = stage;
    $("#mm-bar").style.width = `${Math.max(2, pct)}%`;
    if (lines) {
      const c = $("#mm-console");
      c.textContent = lines.join("\n");
      c.scrollTop = c.scrollHeight;
    }
    $("#mm-timer").textContent = fmt.time((Date.now() - m.t0) / 1000);
  }

  async function start() {
    if (!m.media) { toast("Choose the video, photo ZIP or Colab package first."); return; }
    const btn = $("#mm-start");
    btn.disabled = true;
    m.t0 = Date.now();
    try {
      if (m.source === "colab") {
        const form = new FormData();
        form.append("bundle_file", m.media);
        showProgress("Uploading package", 1, ["Uploading…"]);
        const r = await api.upload("/api/pipeline/import-colab", form, (f) => showProgress(`Uploading package (${Math.round(f * 100)}%)`, f * 70));
        const recal = r.recalibration && r.recalibration.status === "ok"
          ? [`GPS check: scale ×${r.recalibration.scale_correction}, fit ${r.recalibration.gps_rmse_m} m, ${r.recalibration.latlon_swapped ? "lat/lon order corrected" : "lat/lon order verified"}`] : [];
        showProgress("Imported", 100, [`Deployed: ${r.deployed_files.join(", ")}`, `${fmt.n(r.vertex_count)} points, ${fmt.n(r.face_count)} mesh faces`, ...recal]);
        toast("Colab package imported.", "ok");
        setTimeout(() => { closeModal("m-mission"); ctx.reloadActive(); }, 900);
      } else {
        const form = new FormData();
        form.append("input_type", m.source === "frames" ? "frames" : "video");
        form.append("quality", m.quality);
        form.append("has_telemetry", m.log ? "true" : "false");
        form.append("media_file", m.media);
        if (m.log) form.append("telemetry_file", m.log);
        if (m.rtk) form.append("rtk_file", m.rtk);
        if (m.cal) form.append("intrinsics_file", m.cal);
        if (m.imu) form.append("imu_file", m.imu);
        form.append("dynamic_masks", $("#mm-dynamic").checked ? "true" : "false");
        if (!m.log && !m.imu) toast("No flight log: the scale will be estimated and heights are low confidence.", "danger");
        showProgress("Uploading", 1, ["Uploading media…"]);
        await api.upload("/api/pipeline/start", form, (f) => showProgress(`Uploading (${Math.round(f * 100)}%)`, f * 5));
        followPipeline();
      }
    } catch (e) {
      toast(e.message, "danger");
      showProgress("Failed", 0, [String(e.message)]);
    } finally {
      btn.disabled = false;
    }
  }
  $("#mm-start").addEventListener("click", start);
  $("#mm-cancel").addEventListener("click", async () => {
    if (!confirm("Stop the running reconstruction?")) return;
    await api.post("/api/pipeline/cancel").catch(() => {});
  });

  function followPipeline() {
    if (m.poll) clearInterval(m.poll);
    $("#mm-cancel").classList.remove("hidden");
    ctx.setStatus("Reconstruction running…", "busy");
    m.poll = setInterval(async () => {
      let s;
      try { s = await api.get("/api/pipeline/status"); } catch (_) { return; }
      const stage = s.current_stage ? s.current_stage.charAt(0) + s.current_stage.slice(1).toLowerCase() : "Working";
      showProgress(stage, s.progress_percent || 0, s.logs || []);
      ctx.setJobProgress(`Reconstruction ${s.progress_percent || 0}%`, s.progress_percent || 0);
      if (s.status === "completed" || s.status === "failed") {
        clearInterval(m.poll);
        m.poll = null;
        $("#mm-cancel").classList.add("hidden");
        ctx.setJobProgress(null);
        if (s.status === "completed") {
          toast("Reconstruction finished.", "ok");
          ctx.reloadActive();
        } else {
          ctx.setStatus("Reconstruction failed", "danger");
          toast(`Reconstruction failed: ${s.error || "see the log"}`, "danger");
        }
      }
    }, 1500);
  }

  // ------------------------------------------------------------------ saved surveys
  async function openSurveys() {
    openModal("m-open");
    const list = $("#mo-list");
    list.innerHTML = "";
    list.appendChild(el("div", { class: "note" }, "Loading…"));
    let data;
    try { data = await api.get("/api/baselines"); } catch (e) { list.innerHTML = ""; list.appendChild(el("div", { class: "callout danger" }, e.message)); return; }
    list.innerHTML = "";
    if (!data.baselines.length) {
      list.appendChild(el("div", { class: "empty" }, "No saved surveys yet. Use Mission ▸ Save as baseline."));
      return;
    }
    for (const b of data.baselines) {
      const cal = b.calibration || {};
      const bounds = b.telemetry_bounds;
      const card = el("div", { class: "card" }, [
        el("div", { class: "card-row" }, [
          el("span", { class: "card-title" }, b.name || b.id),
          el("span", { class: "spacer" }),
          cal.metric ? badge(`scale: ${String(cal.confidence || "?").toLowerCase()}`, CONFIDENCE_KIND[cal.confidence] || "") : badge("not metric", "danger"),
          b.has_mesh ? badge("mesh") : null,
          b.has_video ? badge("video") : null,
        ]),
        el("div", { class: "note", style: { marginTop: "4px" } },
          `${b.created_at ? new Date(b.created_at).toLocaleString() : ""} · ${fmt.n(b.vertex_count)} points` +
          (bounds ? ` · ${bounds.center_lat.toFixed(4)}, ${bounds.center_lon.toFixed(4)}` : "")),
        el("div", { class: "row", style: { marginTop: "8px" } }, [
          el("button", { class: "btn small primary", type: "button", onclick: () => { closeModal("m-open"); ctx.loadSurvey({ kind: "baseline", id: b.id, meta: b }); } }, "Open"),
          el("button", { class: "btn small", type: "button", onclick: () => { closeModal("m-open"); ctx.state.compareBaseline = b.id; ctx.openTool("compare"); } }, "Compare with current"),
          el("span", { class: "spacer grow" }),
          el("button", { class: "btn small danger", type: "button", onclick: async () => {
            if (!confirm(`Delete the saved survey "${b.name || b.id}"? This removes its model and log permanently.`)) return;
            try { await api.del(`/api/baseline/${encodeURIComponent(b.id)}`); toast("Survey deleted."); openSurveys(); } catch (e) { toast(e.message, "danger"); }
          } }, "Delete"),
        ]),
      ]);
      list.appendChild(card);
    }
  }

  // ------------------------------------------------------------------ save baseline
  function openSave() {
    if (!ctx.state.survey || ctx.state.survey.kind !== "active") { toast("Only the current mission can be saved as a baseline."); return; }
    const now = new Date();
    $("#ms-name").value = `Survey ${now.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" })}`;
    const warn = $("#ms-warn");
    warn.classList.toggle("hidden", ctx.metric());
    warn.textContent = "This model is not metrically calibrated: it can be viewed but not used for change detection.";
    openModal("m-save");
    setTimeout(() => $("#ms-name").select(), 50);
  }
  $("#ms-save").addEventListener("click", async () => {
    try {
      const r = await api.post("/api/baseline/save", { name: $("#ms-name").value.trim() || undefined, keep_video: $("#ms-video").checked });
      toast(r.message, "ok");
      closeModal("m-save");
    } catch (e) { toast(e.message, "danger"); }
  });

  async function resumeIfRunning() {
    try {
      const s = await api.get("/api/pipeline/status");
      if (s.status === "running") {
        m.t0 = Date.now() - 1000 * (s.elapsed_seconds || 0);
        followPipeline();
        toast("A reconstruction is running on the server; progress is shown in the status bar.");
      }
    } catch (_) { /* server not ready */ }
  }

  return {
    openNewMission() { setSource(m.source); openModal("m-mission"); },
    openSurveys, openSave, resumeIfRunning,
  };
}
