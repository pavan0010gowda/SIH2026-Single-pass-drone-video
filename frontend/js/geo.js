// Model frame <-> geographic coordinates.
// Model frame: metres, x = East, y = Up, z = South, origin = georeference.origin (lat, lon, alt).
// Over a few kilometres the local tangent-plane conversion below is accurate to a few centimetres.

let origin = null;

export function setOrigin(o) {
  origin = o && isFinite(o.lat) && isFinite(o.lon) ? { lat: +o.lat, lon: +o.lon, alt: +(o.alt || 0) } : null;
}

export function hasOrigin() { return origin !== null; }

function metersPerDegree(latDeg) {
  const phi = (latDeg * Math.PI) / 180;
  const a = 6378137.0, e2 = 6.69437999014e-3;
  const mLat = 111132.954 - 559.822 * Math.cos(2 * phi) + 1.175 * Math.cos(4 * phi);
  const mLon = (Math.PI / 180) * a * Math.cos(phi) / Math.sqrt(1 - e2 * Math.sin(phi) ** 2);
  return { mLat, mLon };
}

export function toLatLon(x, z) {
  if (!origin) return null;
  const { mLat, mLon } = metersPerDegree(origin.lat);
  return { lat: origin.lat + -z / mLat, lon: origin.lon + x / mLon };
}

export function fromLatLon(lat, lon) {
  if (!origin) return null;
  const { mLat, mLon } = metersPerDegree(origin.lat);
  return { x: (lon - origin.lon) * mLon, z: -(lat - origin.lat) * mLat };
}

export function haversine(lat1, lon1, lat2, lon2) {
  const r = 6371000, p1 = (lat1 * Math.PI) / 180, p2 = (lat2 * Math.PI) / 180;
  const dp = p2 - p1, dl = ((lon2 - lon1) * Math.PI) / 180;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * r * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function bearing(lat1, lon1, lat2, lon2) {
  const p1 = (lat1 * Math.PI) / 180, p2 = (lat2 * Math.PI) / 180, dl = ((lon2 - lon1) * Math.PI) / 180;
  const y = Math.sin(dl) * Math.cos(p2);
  const x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

// turbo colormap (Google, polynomial approximation) -> [r, g, b] in 0..1
export function turbo(t) {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [Math.min(1, Math.max(0, r)), Math.min(1, Math.max(0, g)), Math.min(1, Math.max(0, b))];
}

export const CLASS_COLORS = {
  0: [0.35, 0.35, 0.35], 1: [0.63, 0.59, 0.47], 2: [0.47, 0.67, 0.35], 3: [0.16, 0.43, 0.2],
  4: [0.78, 0.35, 0.24], 5: [0.24, 0.47, 0.78], 6: [0.9, 0.78, 0.47], 7: [0.9, 0.24, 0.24],
};
export const CLASS_LABELS = ["No data", "Open ground", "Grass / reeds", "Trees", "Buildings", "Water / no return", "Roads", "Vehicles"];
