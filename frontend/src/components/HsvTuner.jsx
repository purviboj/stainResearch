import { DEFAULT_HSV, HSV_PRESETS } from "../constants";

const SLIDERS = [
  { key: "h1_low", label: "H1 low", max: 180 },
  { key: "s1_low", label: "S1 low", max: 255 },
  { key: "v1_low", label: "V1 low", max: 255 },
  { key: "h1_high", label: "H1 high", max: 180 },
  { key: "s1_high", label: "S1 high", max: 255 },
  { key: "v1_high", label: "V1 high", max: 255 },
  { key: "h2_low", label: "H2 low", max: 180 },
  { key: "h2_high", label: "H2 high", max: 180 },
  { key: "morph_open", label: "Morph open", max: 20 },
  { key: "morph_close", label: "Morph close", max: 30 },
];

export default function HsvTuner({ hsv, onChange }) {
  const applyPreset = (preset) => {
    onChange({ ...hsv, ...preset });
  };

  return (
    <div className="panel">
      <h2>HSV tuning</h2>
      <p className="subtitle">
        Adjust sliders until only the stain is selected. Hue wraps at 360° for reds.
      </p>

      <div className="preset-row">
        {Object.keys(HSV_PRESETS).map((name) => (
          <button
            key={name}
            type="button"
            className="chip"
            onClick={() => applyPreset(HSV_PRESETS[name])}
          >
            {name}
          </button>
        ))}
        <button
          type="button"
          className="chip chip-secondary"
          onClick={() => onChange({ ...DEFAULT_HSV })}
        >
          Reset defaults
        </button>
      </div>

      <div className="slider-grid">
        {SLIDERS.map(({ key, label, max }) => (
          <label key={key} className="slider-field">
            <span>
              {label} <strong>{hsv[key]}</strong>
            </span>
            <input
              type="range"
              min={0}
              max={max}
              value={hsv[key]}
              onChange={(e) => onChange({ ...hsv, [key]: Number(e.target.value) })}
            />
          </label>
        ))}
      </div>
    </div>
  );
}
