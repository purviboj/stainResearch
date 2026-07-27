const API_BASE = import.meta.env.VITE_API_URL || "";

export async function analyzeStainPair({ preFile, postFile, hsv, pxPerCm, mode = "hsv" }) {
  const formData = new FormData();
  formData.append("pre", preFile);
  formData.append("post", postFile);
  formData.append("px_per_cm", String(pxPerCm));
  formData.append("mode", mode);

  Object.entries(hsv).forEach(([key, value]) => {
    formData.append(key, String(value));
  });

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Analysis failed");
  }
  return payload;
}
