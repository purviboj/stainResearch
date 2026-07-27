import { useState } from "react";
import { analyzeStainPair } from "./api";
import ImageUpload from "./components/ImageUpload";
import HsvTuner from "./components/HsvTuner";
import ResultsTable from "./components/ResultsTable";
import { DEFAULT_HSV } from "./constants";

export default function App() {
  const [preFile, setPreFile] = useState(null);
  const [postFile, setPostFile] = useState(null);
  const [hsv, setHsv] = useState({ ...DEFAULT_HSV });
  const [pxPerCm, setPxPerCm] = useState(120);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runAnalysis = async () => {
    if (!preFile || !postFile) {
      setError("Upload both before and after images.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const payload = await analyzeStainPair({
        preFile,
        postFile,
        hsv,
        pxPerCm,
        mode: "hsv",
      });
      setResults(payload);
    } catch (err) {
      setResults(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <h1>Stain Detection & Measurement</h1>
        <p>
          Upload a before/after pair, tune the HSV color range to isolate the stain,
          and view quantitative measurements from the OpenCV pipeline.
        </p>
      </header>

      <main className="layout">
        <section className="left-column">
          <ImageUpload
            preFile={preFile}
            postFile={postFile}
            onPreChange={setPreFile}
            onPostChange={setPostFile}
          />

          <div className="panel controls-panel">
            <label className="inline-field">
              <span>Pixels per centimeter</span>
              <input
                type="number"
                min={0}
                step={0.1}
                value={pxPerCm}
                onChange={(e) => setPxPerCm(Number(e.target.value))}
              />
            </label>
            <button
              type="button"
              className="primary-button"
              onClick={runAnalysis}
              disabled={loading}
            >
              {loading ? "Analyzing…" : "Run analysis"}
            </button>
            {error && <p className="error">{error}</p>}
          </div>

          <HsvTuner hsv={hsv} onChange={setHsv} />
        </section>

        <section className="right-column">
          <ResultsTable results={results} />

          <div className="panel image-key">
            <h2>Image key (Real Stain Images 3)</h2>
            <p>
              In order, each group of four images is: side view before cleaning,
              top-down before cleaning, side view after cleaning, top-down after cleaning.
              This repeats four times for pre-cooking times of 0, 10, 20, and 30 minutes.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
