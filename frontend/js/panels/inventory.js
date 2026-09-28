import { el, group, fmt, toast, spinnerRow, kv, badge, stat } from "../ui.js";
import { heightCard, drawHeight } from "./height.js";

let data = null;          // quick inventory (all kinds)
let blds = null;          // detailed building analysis (roofs, facades, blocks)
let filter = "building";
let blocksOn = false;
let labelsOn = true;

const ROOF_KIND = { FLAT: "", SHED: "info", GABLE: "accent", HIPPED: "accent", COMPLEX: "caution", IRREGULAR: "caution" };

export default {
  id: "inventory",
  title: "Structures",
  needsMetric: true,
  mount(ctx, body) {
    const head = el("div", { class: "group" }, [
      el("div", { class: "note" }, "Buildings with rooftop form, eaves, storeys and how much of their facades the flight actually saw; trees and vehicles with quick heights. Select one for the precise measurement."),
    ]);
    const seg = el("div", { class: "group" });
    const segBtns = el("div", { class: "seg" });
    const summary = el("div");
    const list = el("div");
    const detail = el("div");
    body.append(head, seg, summary, detail, list);

    const draw3d = () => {
      ctx.viewer.clearLayer("inventory");
      ctx.setAnnotations("inventory", []);
      if (!data || !labelsOn) return;
      const ann = [];
      if (filter === "building" && blds) {
        for (const b of blds.buildings.slice(0, 40)) {
          const top = b.base_y + b.height_m;
          ctx.viewer.label("inventory", [b.centroid[0], top + 1.2, b.centroid[2]], `${b.id} · ${b.height_m.toFixed(1)} m`);
          ann.push({ type: "point", pos: [b.centroid[0], top, b.centroid[2]], label: `${b.height_m.toFixed(0)} m`, kind: "info" });
        }
      } else {
        for (const it of data.items.filter((i) => i.kind === filter).slice(0, 40)) {
          ctx.viewer.label("inventory", [it.position[0], it.position[1] + it.approx_height_m + 1.2, it.position[2]], `${it.id} · ${it.approx_height_m.toFixed(1)} m`);
          ann.push({ type: "point", pos: [it.position[0], it.position[1] + it.approx_height_m, it.position[2]], label: `${it.approx_height_m.toFixed(0)} m`, kind: "info" });
        }
      }
      ctx.setAnnotations("inventory", ann);
    };

    const drawBlocks = () => {
      ctx.viewer.clearLayer("twin");
      if (!blocksOn || !blds) return;
      for (const b of blds.buildings) {
        if (!b.footprint || b.footprint.length < 3) continue;
        const lod1 = b.roof_type === "FLAT" || !b.eaves_m ? b.height_m : 0.5 * (b.eaves_m + b.height_m);
        ctx.viewer.block("twin", b.footprint, b.base_y, b.base_y + lod1, 0x7fa6c9, 0.22);
      }
    };

    const render = () => {
      seg.innerHTML = "";
      const tools = el("div", { class: "row" }, [
        el("label", { class: "check grow" }, [el("input", { type: "checkbox", checked: labelsOn || undefined, onchange: (e) => { labelsOn = e.target.checked; draw3d(); } }),
          "Show labels on the model"]),
        el("button", { class: "btn small", type: "button", onclick: () => { clearStructures(ctx); render(); } }, "Clear from the model"),
      ]);
      seg.appendChild(tools);
      segBtns.innerHTML = "";
      seg.appendChild(segBtns);
      const counts = data ? data.counts : {};
      for (const [k, label] of [["building", "Buildings"], ["tree", "Trees"], ["vehicle", "Vehicles"]]) {
        const n = k === "building" && blds ? blds.summary.count : counts[k] || 0;
        segBtns.appendChild(el("button", { type: "button", class: filter === k ? "on" : "", onclick: () => { filter = k; render(); draw3d(); } },
          `${label} ${data ? n : ""}`));
      }
      summary.innerHTML = "";
      list.innerHTML = "";
      if (!data) return;
      if (filter === "building") return renderBuildings();
      const rows = data.items.filter((i) => i.kind === filter);
      if (!rows.length) { list.appendChild(el("div", { class: "empty" }, "None detected.")); return; }
      const tb = el("tbody");
      for (const it of rows.slice(0, 200)) {
        tb.appendChild(el("tr", { class: "click", onclick: () => measure(it.position, it.approx_height_m) }, [
          el("td", { class: "mono" }, it.id),
          el("td", { class: "num" }, `${it.approx_height_m.toFixed(1)} m`),
          el("td", { class: "num" }, fmt.area(it.footprint_m2)),
        ]));
      }
      list.appendChild(el("table", { class: "data" }, [el("thead", {}, el("tr", {}, [el("th", {}, "ID"), el("th", { class: "num" }, "Height ≈"), el("th", { class: "num" }, "Footprint")])), tb]));
    };

    const renderBuildings = () => {
      if (!blds) { list.appendChild(spinnerRow("Analysing rooftops and facades…")); return; }
      const s = blds.summary;
      if (!s.count) { list.appendChild(el("div", { class: "empty" }, "No buildings detected.")); return; }
      const roofs = Object.entries(s.roof_types || {}).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(" · ");
      summary.append(group("", [
        el("div", { class: "stat-row" }, [stat("Buildings", s.count), stat("Tallest", s.tallest_m.toFixed(1), "m"),
          stat("Facades seen", s.median_facade_coverage_pct != null ? s.median_facade_coverage_pct : "—", "%")]),
        el("div", { class: "note" }, `Roofs: ${roofs}. Built volume ${fmt.n(s.total_volume_m3)} m³ on ${fmt.area(s.total_footprint_m2)}.`),
        el("label", { class: "check" }, [el("input", { type: "checkbox", checked: blocksOn || undefined, onchange: (e) => { blocksOn = e.target.checked; drawBlocks(); } }),
          "Show digital-twin blocks (LoD 1, exported as CityJSON)"]),
      ]));
      const tb = el("tbody");
      for (const b of blds.buildings) {
        tb.appendChild(el("tr", { class: "click", onclick: () => selectBuilding(b) }, [
          el("td", { class: "mono" }, b.id),
          el("td", { class: "num" }, `${b.height_m.toFixed(2)}`),
          el("td", {}, badge(b.roof_type.toLowerCase(), ROOF_KIND[b.roof_type] || "")),
          el("td", { class: "num" }, String(b.storeys_est)),
          el("td", { class: "num" }, b.facade_coverage_pct != null ? `${Math.round(b.facade_coverage_pct)}%` : "—"),
        ]));
      }
      list.appendChild(el("table", { class: "data" }, [el("thead", {}, el("tr", {}, [el("th", {}, "ID"), el("th", { class: "num" }, "Height m"),
        el("th", {}, "Roof"), el("th", { class: "num" }, "Storeys"), el("th", { class: "num" }, "Facades")])), tb]));
    };

    const selectBuilding = async (b) => {
      await measure([b.centroid[0], b.base_y, b.centroid[2]], b.height_m, b);
    };

    const measure = async (pos, h, b = null) => {
      ctx.viewer.focus([pos[0], pos[1] + h / 2, pos[2]], 40);
      detail.innerHTML = "";
      if (b) detail.appendChild(roofCard(b));
      const slot = el("div");
      detail.appendChild(slot);
      slot.appendChild(spinnerRow("Measuring precisely…"));
      ctx.viewer.clearLayer("inventory-sel");
      if (b && b.footprint) ctx.viewer.outline("inventory-sel", b.footprint, 0xc9a45c);
      try {
        const r = await ctx.api.post("/api/measure/height", { x: pos[0], z: pos[2], baseline_id: ctx.bid() });
        slot.innerHTML = "";
        if (r.status !== "ok") { slot.appendChild(el("div", { class: "note" }, r.message || "Could not measure this object.")); return; }
        drawHeight(ctx, r, "inventory-sel");
        slot.appendChild(heightCard(ctx, r));
      } catch (e) { slot.innerHTML = ""; toast(e.message, "danger"); }
    };

    const loadBuildings = () => ctx.api.get(`/api/analysis/buildings${ctx.bq()}`).then((d) => {
      blds = d;
      ctx.keep("buildings", d.summary);
      render();
      draw3d();
      drawBlocks();
    }).catch((e) => { toast(`Building analysis: ${e.message}`, "danger"); });

    if (data) { render(); draw3d(); drawBlocks(); if (!blds) loadBuildings(); return; }
    list.appendChild(spinnerRow("Detecting structures…"));
    ctx.api.get(`/api/analysis/inventory${ctx.bq()}`).then((d) => {
      data = d;
      ctx.keep("inventory", d);
      render();
      draw3d();
      loadBuildings();
    }).catch((e) => { list.innerHTML = ""; list.appendChild(el("div", { class: "callout danger" }, e.message)); });
  },
  unmount(ctx) { /* labels and blocks stay until cleared (here or View > Clear all overlays) */ },
  clear(ctx) { clearStructures(ctx); },
  reset() { data = null; blds = null; blocksOn = false; labelsOn = true; },
};

function clearStructures(ctx) {
  labelsOn = false;
  blocksOn = false;
  for (const l of ["inventory", "inventory-sel", "twin"]) ctx.viewer.clearLayer(l);
  ctx.setAnnotations("inventory", []);
}

function roofCard(b) {
  const planes = (b.roof_planes || []).filter((p) => p.share_pct >= 8)
    .map((p) => `${p.slope_deg.toFixed(0)}° facing ${compass(p.aspect_deg)} (${Math.round(p.share_pct)}%)`).join(", ");
  return el("div", { class: "card" }, [
    el("div", { class: "card-row" }, [el("span", { class: "card-title" }, `${b.id} · rooftop & facades`), el("span", { class: "spacer" }),
      badge(b.roof_label, ROOF_KIND[b.roof_type] || "")]),
    kv([
      ["Roof pitch", b.roof_pitch_deg ? fmt.deg(b.roof_pitch_deg) : "flat"],
      ["Roof faces", planes || "—"],
      ["Eaves / ridge", `${b.eaves_m != null ? b.eaves_m.toFixed(2) : "—"} / ${b.height_m.toFixed(2)} m`],
      ["Storeys (est.)", String(b.storeys_est)],
      ["Footprint", `${fmt.area(b.footprint_m2)} · ${b.length_m != null ? b.length_m.toFixed(1) : "—"} × ${b.width_m != null ? b.width_m.toFixed(1) : "—"} m`],
      ["Roof area", fmt.area(b.roof_area_m2)],
      ["Wall area", b.wall_area_m2 != null ? fmt.area(b.wall_area_m2) : "—"],
      ["Facades reconstructed", b.facade_coverage_pct != null ? `${b.facade_coverage_pct.toFixed(0)}%` : "—"],
      ["Volume", `${fmt.n(b.volume_m3)} m³`],
      ["Location", fmt.ll(b.gps_lat, b.gps_lon)],
    ]),
  ]);
}

function compass(deg) {
  return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(((deg % 360) + 360) % 360 / 45) % 8];
}
