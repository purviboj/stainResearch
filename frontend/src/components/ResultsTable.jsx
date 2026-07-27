import { formatDelta, formatValue, METRIC_ROWS } from "../constants";

export default function ResultsTable({ results }) {
  if (!results) {
    return (
      <div className="panel muted">
        Upload a before/after pair and run analysis to see measurements here.
      </div>
    );
  }

  const { pre, post, delta } = results;

  return (
    <div className="panel">
      <h2>Measurement results</h2>
      <p className="subtitle">
        Before vs. after stain metrics using the current HSV range.
      </p>
      <div className="table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Before</th>
              <th>After</th>
              <th>Delta</th>
            </tr>
          </thead>
          <tbody>
            {METRIC_ROWS.map(({ key, label }) => (
              <tr key={key}>
                <td>{label}</td>
                <td>{formatValue(pre?.[key])}</td>
                <td>{formatValue(post?.[key])}</td>
                <td className="delta">{formatDelta(delta?.[key])}</td>
              </tr>
            ))}
            <tr className="highlight-row">
              <td>Δ Area (alias)</td>
              <td colSpan={2} className="muted-cell">post − pre</td>
              <td className="delta">{formatDelta(delta?.delta_area)}</td>
            </tr>
            <tr className="highlight-row">
              <td>Δ Perimeter (alias)</td>
              <td colSpan={2} className="muted-cell">post − pre</td>
              <td className="delta">{formatDelta(delta?.delta_perimeter)}</td>
            </tr>
            <tr className="highlight-row">
              <td>Δ Circularity (alias)</td>
              <td colSpan={2} className="muted-cell">post − pre</td>
              <td className="delta">{formatDelta(delta?.delta_circularity)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
