export const DEFAULT_HSV = {
  h1_low: 0,
  s1_low: 100,
  v1_low: 40,
  h1_high: 12,
  s1_high: 255,
  v1_high: 200,
  h2_low: 165,
  h2_high: 180,
  morph_open: 3,
  morph_close: 7,
};

export const HSV_PRESETS = {
  "Red wine": { h1_low: 0, s1_low: 80, v1_low: 30, h1_high: 15, s1_high: 255, v1_high: 220, h2_low: 160, h2_high: 180 },
  "Coffee / tea": { h1_low: 5, s1_low: 60, v1_low: 20, h1_high: 25, s1_high: 255, v1_high: 180, h2_low: 165, h2_high: 180 },
  "Grass / green": { h1_low: 35, s1_low: 40, v1_low: 30, h1_high: 85, s1_high: 255, v1_high: 200, h2_low: 165, h2_high: 180 },
  "Blue ink": { h1_low: 90, s1_low: 50, v1_low: 30, h1_high: 130, s1_high: 255, v1_high: 200, h2_low: 165, h2_high: 180 },
  "Dark / grease": { h1_low: 0, s1_low: 30, v1_low: 0, h1_high: 180, s1_high: 255, v1_high: 80, h2_low: 165, h2_high: 180 },
};

export const METRIC_ROWS = [
  { key: "area_cm2", label: "Area (cm²)" },
  { key: "area_m2", label: "Area (m²)" },
  { key: "perimeter_cm", label: "Perimeter (cm)" },
  { key: "bbox_w_cm", label: "BBox width (cm)" },
  { key: "bbox_h_cm", label: "BBox height (cm)" },
  { key: "centroid_x_cm", label: "Centroid X (cm)" },
  { key: "centroid_y_cm", label: "Centroid Y (cm)" },
  { key: "circularity", label: "Circularity" },
  { key: "aspect_ratio", label: "Aspect ratio" },
];

export function formatValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Math.abs(value) < 0.001 ? value.toExponential(3) : value.toFixed(4);
  }
  return String(value);
}

export function formatDelta(value) {
  if (value === null || value === undefined) return "—";
  const formatted = formatValue(value);
  if (formatted === "—") return formatted;
  return value > 0 ? `+${formatted}` : formatted;
}
