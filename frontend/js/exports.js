// Export centre: every deliverable of the loaded survey (3-D, rasters, vectors, reports) in one place.
import { $, el, toast, openModal, badge, spinnerRow } from "./ui.js";

const GROUP_NOTE = {
  "Point cloud": "LAS is georeferenced (UTM) with colour and ground / vegetation / building / road / water classes.",
  Mesh: "Local metric frame (x east, y up, z south, metres); the georeference sheet gives its origin.",
  Raster: "Resampled on an exact UTM grid: open directly in QGIS, ArcGIS or Global Mapper.",
  Vector: "GeoJSON in WGS 84; CityJSON buildings (LoD 1.2 solids) in UTM.",
  Document: "Accuracy, completeness and processing time against the problem-statement targets.",
  Package: "All available products in one ZIP.",
};

export function initExports(ctx) {
  const list = $("#mx-list");
  const offset = $("#mx-offset");
  let catalog = null;

  async function open() {
    if (!ctx.state.survey || (ctx.state.survey.kind === "active" && !ctx.state.survey.info)) { toast("Load a survey first."); return; }
    openModal("m-export");
    list.innerHTML = "";
    list.appendChild(spinnerRow("Reading the survey…"));
    try {
      catalog = await ctx.api.get(`/api/export/catalog${ctx.bq()}`);
    } catch (e) {
      list.innerHTML = "";
      list.appendChild(el("div", { class: "callout danger" }, e.message));
      return;
    }
    render();
  }

  function render() {
    list.innerHTML = "";
    const g = catalog.georeference;
    $("#mx-crs").textContent = g ? `${g.crs} (EPSG:${g.epsg}) · origin ${g.origin_wgs84.lat.toFixed(6)}, ${g.origin_wgs84.lon.toFixed(6)}` : "Not georeferenced";
    $("#mx-vertical").textContent = g ? `Heights: ${g.vertical}.` : "";
    const groups = {};
    for (const p of catalog.products) (groups[p.group] = groups[p.group] || []).push(p);
    for (const [name, items] of Object.entries(groups)) {
      const box = el("div", { class: "export-group" }, [
        el("div", { class: "row" }, [el("span", { class: "group-title" }, name), el("span", { class: "spacer grow" }),
          el("span", { class: "faint export-note" }, GROUP_NOTE[name] || "")]),
      ]);
      for (const p of items) {
        const btn = el("button", { class: `btn small${p.key === "package" ? " primary" : ""}`, type: "button", disabled: !p.available }, "Download");
        btn.addEventListener("click", () => fetchProduct(p, btn));
        box.appendChild(el("div", { class: `export-row${p.available ? "" : " off"}` }, [
          el("div", { class: "grow" }, [el("div", {}, p.label), el("div", { class: "faint mono export-file" }, p.available ? p.file : p.reason)]),
          p.ready ? badge(`${p.size_mb} MB`, "ok") : null,
          btn,
        ]));
      }
      list.appendChild(box);
    }
  }

  async function fetchProduct(p, btn) {
    const q = new URLSearchParams();
    if (ctx.bid()) q.set("baseline_id", ctx.bid());
    const v = parseFloat(offset.value);
    if (isFinite(v)) q.set("vertical_offset", String(v));
    const label = btn.textContent;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    ctx.setStatus(`Preparing ${p.label}…`, "busy");
    try {
      const res = await fetch(ctx.api.url(`api/export/file/${p.key}?${q.toString()}`));
      if (!res.ok) {
        let msg = `${res.status}`;
        try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
        throw new Error(msg);
      }
      const blob = await res.blob();
      const cd = res.headers.get("content-disposition") || "";
      const m = /filename="?([^";]+)"?/i.exec(cd);
      const a = el("a", { href: URL.createObjectURL(blob), download: m ? m[1] : p.file });
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
      ctx.setStatus(`${p.label}: ${(blob.size / 1e6).toFixed(1)} MB`, "ok");
      p.ready = true;
      p.size_mb = +(blob.size / 1e6).toFixed(2);
      render();
    } catch (e) {
      toast(`${p.label}: ${e.message}`, "danger");
      ctx.setStatus("Export failed", "danger");
      btn.disabled = false;
      btn.textContent = label;
    }
  }

  offset.addEventListener("change", () => { if (catalog) render(); });
  return { open };
}
