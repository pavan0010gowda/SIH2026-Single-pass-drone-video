// Change detection: day 1 vs day 2 (any two surveys of the same place). New tents, nets, vehicles, masts,
// trenches and removed objects are found by differencing the two surface models after both flights are
// aligned on unchanged ground; nothing below the measured noise floor is reported.
import { el, group, kv, fmt, toast, badge, callout, spinnerRow, download } from "../ui.js";
import { COLORS } from "../viewer.js";

let result = null;
let surveys = null;
let sel = { earlier: null, later: null, minH: 0.5, minA: 2.0, showMap: true };

const SEV_KIND = { HIGH: "danger", ELEVATED: "caution", ADVISORY: "info" };
const SEV_COLOR = { HIGH: COLORS.danger, ELEVATED: COLORS.caution, ADVISORY: COLORS.info };

function loadedId(ctx) { return ctx.bid() || "active"; }

function distKm(a, b) {
  if (!a || !b) return null;
  const r = Math.PI / 180, dla = (b.lat - a.lat) * r, dlo = (b.lon - a.lon) * r;
  const h = Math.sin(dla / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dlo / 2) ** 2;
  return 12742 * Math.asin(Math.min(1, Math.sqrt(h)));
}

async function draw(ctx) {
  const v = ctx.viewer;
  v.clearLayer("compare");
  v.clearLayer("change");
  if (!result || result.status !== "completed" || result.frame !== loadedId(ctx)) { ctx.setAnnotations("compare", []); return; }
  const ann = [];
  for (const a of result.alerts) {
    const col = SEV_COLOR[a.severity] || COLORS.accent;
    const p = a.position;
    if (a.kind === "group") {
      v.disc("compare", p, a.radius_m, col, 0.08);
      v.label("compare", [p[0], p[1] + 6, p[2]], `${a.id} · possible camp`, "danger");
      ann.push({ type: "point", pos: p, label: a.id, kind: "danger" });
      continue;
    }
    if (a.kind === "gain") {
      v.block("compare", a.footprint, a.base_y, Math.max(a.base_y + 0.3, p[1]), col, 0.3);
      v.label("compare", [p[0], p[1] + 1.2, p[2]], `${a.id} · ${a.category.toLowerCase()} +${a.height_change_m.toFixed(1)} m`, SEV_KIND[a.severity]);
    } else {
      v.outline("compare", a.footprint, COLORS.info);
      v.label("compare", [p[0], p[1] + 1.2, p[2]], `${a.id} · ${a.category.toLowerCase()} ${a.height_change_m.toFixed(1)} m`, "info");
    }
    ann.push({ type: "point", pos: p, label: `${a.id} ${a.height_change_m > 0 ? "+" : ""}${a.height_change_m.toFixed(1)} m`, kind: SEV_KIND[a.severity] });
  }
  ctx.setAnnotations("compare", ann);
  if (sel.showMap && result.change_layer) {
    try { v.drape("change", result.change_layer, await ctx.terrainGrid(), { opacity: 0.85, lift: 0.3 }); } catch (_) { /* no terrain grid */ }
  }
}

function geojson(r) {
  const feats = r.alerts.filter((a) => a.gps_lat != null).map((a) => ({
    type: "Feature", geometry: { type: "Point", coordinates: [a.gps_lon, a.gps_lat] },
    properties: { id: a.id, category: a.category, severity: a.severity, height_change_m: a.height_change_m, area_m2: a.area_m2,
      reasons: (a.reasons || []).join("; "), day1: r.earlier.name, day2: r.later.name },
  }));
  return JSON.stringify({ type: "FeatureCollection", name: `PRISM changes ${r.earlier.name} -> ${r.later.name}`, features: feats }, null, 1);
}

export default {
  id: "compare",
  title: "Change detection",
  needsMetric: true,
  mount(ctx, body) {
    const pick = el("div", { class: "group" }, [spinnerRow("Loading saved surveys…")]);
    const out = el("div", { class: "group" });
    body.appendChild(group("", [
      el("div", { class: "note" }, "Compares two flights over the same place (day 1 and day 2) and alerts on anything that appeared, disappeared or was dug in between: tents, camouflage nets, vehicles, masts, trenches. Both flights are aligned on unchanged ground first, so the GPS difference between days does not create false alarms."),
    ]));
    body.appendChild(pick);
    body.appendChild(out);

    const segRow = (values, current, fmtv, onPick) => {
      const s = el("div", { class: "seg" });
      for (const x of values) {
        s.appendChild(el("button", { type: "button", class: x === current ? "on" : "", onclick: (e) => {
          onPick(x);
          [...s.children].forEach((b) => b.classList.toggle("on", b === e.currentTarget));
        } }, fmtv(x)));
      }
      return s;
    };

    const renderPick = () => {
      pick.innerHTML = "";
      const here = loadedId(ctx);
      const opts = [{ id: "active", name: "Current mission", date: null, centre: ctx.state.survey && ctx.state.survey.kind === "active" && ctx.state.survey.telemetry
        && ctx.state.survey.telemetry.georeference && ctx.state.survey.telemetry.georeference.origin }]
        .concat((surveys || []).map((b) => ({ id: b.id, name: b.name || b.id, date: b.created_at, metric: b.calibration && b.calibration.metric,
          centre: b.telemetry_bounds ? { lat: b.telemetry_bounds.center_lat, lon: b.telemetry_bounds.center_lon } : null })));
      if (!sel.later) sel.later = here;
      const laterOpt = opts.find((o) => o.id === sel.later) || opts[0];
      if (!sel.earlier || sel.earlier === sel.later) {
        // default day 1: the closest saved survey to day 2's location (then the most recent)
        const cands = opts.filter((o) => o.id !== sel.later);
        cands.sort((a, b) => (distKm(laterOpt.centre, a.centre) ?? 1e9) - (distKm(laterOpt.centre, b.centre) ?? 1e9) || String(b.date).localeCompare(String(a.date)));
        sel.earlier = cands.length ? cands[0].id : null;
      }
      const select = (key) => {
        const s = el("select", { class: "input" });
        for (const o of opts) {
          const d = distKm(laterOpt.centre, o.centre);
          const txt = `${o.name}${o.date ? ` · ${new Date(o.date).toLocaleDateString()}` : ""}${key === "earlier" && d != null && o.id !== sel.later ? ` · ${d < 1 ? `${Math.round(d * 1000)} m` : `${d.toFixed(0)} km`} away` : ""}`;
          const op = el("option", { value: o.id }, txt);
          if (o.id === sel[key]) op.selected = true;
          s.appendChild(op);
        }
        s.addEventListener("change", () => { sel[key] = s.value; if (key === "later") sel.earlier = null; renderPick(); });
        return s;
      };
      pick.append(
        el("div", { class: "field" }, [el("label", {}, "Day 1 (earlier survey)"), select("earlier")]),
        el("div", { class: "field" }, [el("label", {}, "Day 2 (later survey)"), select("later")]),
        el("div", { class: "field" }, [el("label", {}, "Smallest height change"), segRow([0.3, 0.5, 1.0, 2.0], sel.minH, (x) => `${x} m`, (x) => { sel.minH = x; })]),
        el("div", { class: "field" }, [el("label", {}, "Smallest object"), segRow([1, 2, 5, 10], sel.minA, (x) => `${x} m²`, (x) => { sel.minA = x; })]),
        el("div", { class: "row" }, [
          el("button", { class: "btn primary grow", type: "button", onclick: run }, "Compare"),
          el("button", { class: "btn", type: "button", onclick: () => { clearAll(ctx); render(); } }, "Clear from the model"),
        ]),
      );
    };

    const run = async () => {
      if (!sel.earlier || !sel.later) { toast("Choose the two surveys."); return; }
      out.innerHTML = "";
      out.appendChild(spinnerRow("Checking the location, aligning the two flights on unchanged ground and differencing the surfaces…"));
      try {
        result = await ctx.api.post("/api/diff/compare", { earlier: sel.earlier, later: sel.later, min_height_m: sel.minH, min_area_m2: sel.minA });
        ctx.keep("change_detection", { ...result, change_layer: undefined });
        const high = (result.alerts || []).filter((a) => a.severity === "HIGH");
        if (high.length) toast(`ALERT: ${high.length} high-priority change(s) between the two flights.`, "danger");
        render();
        await draw(ctx);
      } catch (e) { out.innerHTML = ""; out.appendChild(callout(e.message, "danger")); }
    };

    const render = () => {
      out.innerHTML = "";
      if (!result) return;
      const loc = result.location || {};
      if (result.status !== "completed") {
        out.appendChild(callout(loc.message || result.message || "The two surveys cannot be compared.", "caution"));
        return;
      }
      out.appendChild(callout(loc.message, "ok"));
      const sc = result.severity_counts || {};
      const high = result.alerts.filter((a) => a.severity === "HIGH");
      if (high.length) {
        out.appendChild(el("div", { class: "callout danger alert-banner" }, [
          el("b", {}, `${high.length} high-priority change${high.length > 1 ? "s" : ""}: `),
          high.map((a) => `${a.id} ${a.category.toLowerCase()}`).join(", "),
        ]));
      } else if (!result.alerts.length) {
        out.appendChild(callout(`No change above ${result.noise_floor_ground_m.toFixed(2)} m (the noise floor) or ${result.min_height_m} m between ${result.earlier.name} and ${result.later.name}.`, "ok"));
      }
      out.appendChild(el("div", { class: "stat-row" }, [
        el("div", { class: "stat" }, [el("div", { class: "k" }, "High"), el("div", { class: "v" }, String(sc.HIGH || 0))]),
        el("div", { class: "stat" }, [el("div", { class: "k" }, "Elevated"), el("div", { class: "v" }, String(sc.ELEVATED || 0))]),
        el("div", { class: "stat" }, [el("div", { class: "k" }, "Advisory"), el("div", { class: "v" }, String(sc.ADVISORY || 0))]),
      ]));
      const reg = result.registration || {};
      out.appendChild(kv([
        ["Compared", `${fmt.area(result.compared_area_m2)} seen on both days`],
        ["Noise floor (95 %)", `${result.noise_floor_ground_m.toFixed(2)} m on open ground`],
        ["GPS difference corrected", `${reg.horizontal_shift_total_m} m horizontal, ${reg.rotation_deg}° rotation, ${reg.vertical_offset_m} m vertical`],
        ["Alignment residual", reg.registration_sigma_m != null ? `${(100 * reg.registration_sigma_m).toFixed(0)} cm on ${fmt.n(reg.ground_cells)} ground cells` : "—"],
      ]));
      if (result.frame !== loadedId(ctx)) {
        out.appendChild(el("button", { class: "btn primary", type: "button", onclick: async () => {
          const spec = result.frame === "active" ? { kind: "active" } : { kind: "baseline", id: result.frame, meta: (surveys || []).find((b) => b.id === result.frame) };
          await ctx.loadSurvey(spec);
        } }, `Open ${result.later.name} to see the changes in 3-D`));
      }
      for (const a of result.alerts) {
        out.appendChild(el("div", { class: "card selectable", onclick: () => { if (result.frame === loadedId(ctx)) ctx.viewer.focus(a.position, a.kind === "group" ? 90 : 35); } }, [
          el("div", { class: "card-row" }, [
            el("span", { class: "card-title mono" }, a.id),
            el("span", {}, a.category.toLowerCase()),
            el("span", { class: "spacer" }),
            badge(a.severity.toLowerCase(), SEV_KIND[a.severity]),
          ]),
          a.kind === "group" ? null : kv([
            ["Height change", `${a.height_change_m > 0 ? "+" : ""}${a.height_change_m.toFixed(2)} ± ${a.uncertainty_m.toFixed(2)} m`],
            ["Size", `${a.length_m.toFixed(1)} × ${a.width_m.toFixed(1)} m (${fmt.area(a.area_m2)})`],
          ]),
          el("div", { class: "note", style: { marginTop: "4px" } }, (a.reasons || []).join(" · ")),
          el("div", { class: "faint", style: { fontSize: "11px", marginTop: "2px" } }, fmt.ll(a.gps_lat, a.gps_lon)),
        ]));
      }
      const mapChk = el("input", { type: "checkbox", checked: sel.showMap || undefined, onchange: (e) => { sel.showMap = e.target.checked; draw(ctx); } });
      out.appendChild(el("label", { class: "check" }, [mapChk, "Show the change map (red = higher on day 2, blue = lower)"]));
      if (result.alerts.length) {
        out.appendChild(el("button", { class: "btn small", type: "button", onclick: () => download(`prism_changes_${Date.now()}.geojson`, geojson(result), "application/geo+json") }, "Export alerts (GeoJSON)"));
      }
    };

    const clearAll = (c) => { result = null; c.viewer.clearLayer("compare"); c.viewer.clearLayer("change"); c.setAnnotations("compare", []); };

    ctx.api.get("/api/baselines").then((d) => {
      surveys = d.baselines;
      renderPick();
    }).catch((e) => { pick.innerHTML = ""; pick.appendChild(callout(e.message, "danger")); });
    if (result) { render(); draw(ctx); }
  },
  clear(ctx) { result = null; ctx.viewer.clearLayer("compare"); ctx.viewer.clearLayer("change"); ctx.setAnnotations("compare", []); },
  // results are kept when another survey is opened: day 2 is often opened right after the comparison
  reset() { if (!result) { sel.later = null; sel.earlier = null; } },
};
