"""
app.py — Adhesive Stain Analysis Web App
=========================================
A self-contained Flask application that lets a user upload a "pre" image
and (optionally) a "post" image of an adhesive stain and get back the same
measurement set produced by stain_tracker.py — area, perimeter, circularity,
bounding box, centroid, aspect ratio, plus before/after deltas — without
needing a camera, CLI flags, or a fixed file path.

This file is INTENTIONALLY SEPARATE from stain_tracker.py. It re-implements
the same detection math (thresholding -> morphology -> contour selection ->
measurement) but reworked to operate on uploaded image bytes in memory,
since the web app has no local file paths, no argparse, and no OpenCV GUI.

Run locally:
    pip install flask opencv-python numpy
    python app.py
    # then open http://localhost:5000

API:
    POST /api/analyze
        multipart/form-data:
          pre        (file, required)  - pre-scrub image
          post       (file, optional)  - post-scrub image
          mode       (str,  optional)  - "hsv" (default) or "lab"
          px_per_cm  (float, optional) - manual calibration
          use_grid   ("true"/"false")  - auto-detect 1cm grid instead
          roi        ("x,y,w,h")       - optional crop, pixels

        -> JSON: { pre: {...}, post: {...}|null, delta: {...}|null }

    GET /
        -> minimal upload UI (functional placeholder — swap in your own
           frontend design once you send it over; this just wires the
           upload -> results flow so your PhD student can use it today)
"""

import base64
import math
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ===========================================================================
# TUNABLE DEFAULTS — same values as stain_tracker.py, kept in sync manually
# ===========================================================================

GAUSSIAN_BLUR_K  = 5
USE_CLAHE        = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

HSV_LOW1  = (0,   100, 40)
HSV_HIGH1 = (12,  255, 200)
HSV_LOW2  = (165, 100, 40)
HSV_HIGH2 = (180, 255, 200)

LAB_L_LOW, LAB_L_HIGH = 20, 180
LAB_A_LOW, LAB_A_HIGH = 135, 255
LAB_B_LOW, LAB_B_HIGH = 120, 255

MORPH_OPEN_K  = 3
MORPH_CLOSE_K = 7

MIN_CONTOUR_AREA_PX       = 200
MIN_CONTOUR_AREA_FRACTION = 0.0005
MAX_CONTOUR_AREA_FRACTION = 0.35
MAX_BBOX_IMAGE_FRACTION   = 0.85
BORDER_TOUCH_MARGIN_PX    = 3

HOUGH_RHO       = 1
HOUGH_THETA     = np.pi / 180
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE  = 40
HOUGH_MAX_GAP   = 10
CANNY_LOW       = 20
CANNY_HIGH      = 60

COLOR_PRE  = (0, 220, 0)
COLOR_POST = (0, 0, 220)
COLOR_BOX  = (255, 180, 0)
COLOR_ELLIPSE = (255, 100, 255)
COLOR_TEXT = (255, 255, 255)

API_METRIC_KEYS = (
    "area_cm2", "area_m2", "perimeter_cm", "bbox_w_cm", "bbox_h_cm",
    "centroid_x_cm", "centroid_y_cm", "circularity", "aspect_ratio",
)

MAX_UPLOAD_DIMENSION = 2000  # guard against giant phone photos stalling the pipeline


# ===========================================================================
# IMAGE LOADING (from uploaded bytes, not disk paths)
# ===========================================================================

def decode_upload(file_storage):
    """Read a Flask uploaded file into a BGR OpenCV image, or None if invalid."""
    if file_storage is None or file_storage.filename == "":
        return None
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > MAX_UPLOAD_DIMENSION:
        scale = MAX_UPLOAD_DIMENSION / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def parse_roi(roi_str):
    if not roi_str:
        return None
    try:
        parts = [int(x) for x in roi_str.split(",")]
        if len(parts) != 4:
            return None
        return tuple(parts)
    except ValueError:
        return None


def preprocess(img, roi=None):
    """Same steps as stain_tracker.load_and_preprocess, minus the disk read."""
    if roi is not None:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]

    if USE_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        img_eq = img.copy()

    k = GAUSSIAN_BLUR_K if GAUSSIAN_BLUR_K % 2 == 1 else GAUSSIAN_BLUR_K + 1
    blurred = cv2.GaussianBlur(img_eq, (k, k), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab_blur = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    return img, blurred, hsv, lab_blur


# ===========================================================================
# MASKING / CONTOUR / MEASUREMENT — same math as stain_tracker.py
# ===========================================================================

def build_stain_mask_hsv(hsv):
    mask1 = cv2.inRange(hsv, np.array(HSV_LOW1), np.array(HSV_HIGH1))
    mask2 = cv2.inRange(hsv, np.array(HSV_LOW2), np.array(HSV_HIGH2))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def build_stain_mask_lab(lab):
    l, a, b = cv2.split(lab)
    mask_l = cv2.inRange(l, LAB_L_LOW, LAB_L_HIGH)
    mask_a = cv2.inRange(a, LAB_A_LOW, LAB_A_HIGH)
    mask_b = cv2.inRange(b, LAB_B_LOW, LAB_B_HIGH)
    mask = cv2.bitwise_and(mask_l, cv2.bitwise_and(mask_a, mask_b))

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def detect_grid_spacing(gray):
    """Auto px/cm from a 1cm grid backdrop. Returns None if not found."""
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA,
                             HOUGH_THRESHOLD, None, HOUGH_MIN_LINE, HOUGH_MAX_GAP)
    if lines is None:
        return None

    h_gaps, v_gaps = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        if angle < 10:
            h_gaps.append(y1)
        elif angle > 80:
            v_gaps.append(x1)

    spacings = []
    for gaps in [sorted(h_gaps), sorted(v_gaps)]:
        if len(gaps) > 1:
            diffs = [gaps[i + 1] - gaps[i] for i in range(len(gaps) - 1)
                      if 10 < gaps[i + 1] - gaps[i] < 200]
            if diffs:
                spacings.append(np.median(diffs))

    if not spacings:
        return None
    return float(np.mean(spacings))


def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    img_h, img_w = mask.shape[:2]
    img_area = float(img_h * img_w)
    min_area = max(MIN_CONTOUR_AREA_PX, img_area * MIN_CONTOUR_AREA_FRACTION)
    valid = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches_border = (
            x <= BORDER_TOUCH_MARGIN_PX or
            y <= BORDER_TOUCH_MARGIN_PX or
            x + w >= img_w - BORDER_TOUCH_MARGIN_PX or
            y + h >= img_h - BORDER_TOUCH_MARGIN_PX
        )
        if touches_border:
            continue
        if area / img_area > MAX_CONTOUR_AREA_FRACTION:
            continue
        if max(w / img_w, h / img_h) > MAX_BBOX_IMAGE_FRACTION:
            continue

        valid.append(contour)

    if not valid:
        return None
    return max(valid, key=cv2.contourArea)


def measure_stain(contour, px_per_cm):
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    area_cm2 = area_px / (px_per_cm ** 2)
    perimeter_cm = perimeter_px / px_per_cm
    circularity = (4 * math.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    bbox_w_cm = w / px_per_cm
    bbox_h_cm = h / px_per_cm
    aspect_ratio = w / h if h > 0 else 0

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2, y + h / 2

    CM_TO_M = 0.01

    return {
        "area_cm2": round(area_cm2, 4),
        "perimeter_cm": round(perimeter_cm, 4),
        "bbox_w_cm": round(bbox_w_cm, 4),
        "bbox_h_cm": round(bbox_h_cm, 4),
        "centroid_x_cm": round(cx / px_per_cm, 4),
        "centroid_y_cm": round(cy / px_per_cm, 4),
        "area_m2": round(area_cm2 * (CM_TO_M ** 2), 8),
        "perimeter_m": round(perimeter_cm * CM_TO_M, 6),
        "bbox_w_m": round(bbox_w_cm * CM_TO_M, 6),
        "bbox_h_m": round(bbox_h_cm * CM_TO_M, 6),
        "centroid_x_m": round((cx / px_per_cm) * CM_TO_M, 6),
        "centroid_y_m": round((cy / px_per_cm) * CM_TO_M, 6),
        "aspect_ratio": round(aspect_ratio, 4),
        "circularity": round(circularity, 4),
        "centroid_x_px": round(cx, 1),
        "centroid_y_px": round(cy, 1),
        "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
    }


def annotate_image(img, contour, metrics, label="", color=None):
    color = color or COLOR_PRE
    out = img.copy()
    cv2.drawContours(out, [contour], -1, color, 2)

    x, y, w, h = metrics["bbox_x"], metrics["bbox_y"], metrics["bbox_w"], metrics["bbox_h"]
    cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_BOX, 2)

    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(out, ellipse, COLOR_ELLIPSE, 2)
        except cv2.error:
            pass

    text_lines = [
        f"{label}",
        f"Area:  {metrics['area_cm2']} cm2",
        f"BBox:  {metrics['bbox_w_cm']} x {metrics['bbox_h_cm']} cm",
        f"Circ:  {metrics['circularity']}",
        f"Perim: {metrics['perimeter_cm']} cm",
    ]
    tx, ty = max(x - 5, 5), max(y - 10 - 18 * len(text_lines), 5)
    for i, line in enumerate(text_lines):
        cv2.putText(out, line, (tx, ty + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (tx, ty + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    return out


def encode_image_b64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def metrics_for_api(metrics):
    if not metrics:
        return None
    return {key: metrics[key] for key in API_METRIC_KEYS if key in metrics}


def compute_delta(pre_metrics, post_metrics):
    pre = metrics_for_api(pre_metrics) or {}
    post = metrics_for_api(post_metrics) or {}
    delta = {}
    for key in API_METRIC_KEYS:
        if key in pre and key in post:
            delta[key] = round(post[key] - pre[key], 8 if key == "area_m2" else 4)
    delta["delta_area"] = delta.get("area_cm2")
    delta["delta_perimeter"] = delta.get("perimeter_cm")
    delta["delta_circularity"] = delta.get("circularity")
    return delta


def run_pipeline(img, px_per_cm, mode, roi, label, color):
    """Full single-image pipeline; also returns elapsed processing time in seconds."""
    started_at = time.time()
    original, blurred, hsv, lab = preprocess(img, roi=roi)

    if px_per_cm is None:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        px_per_cm = detect_grid_spacing(gray)
        if px_per_cm is None:
            return None, None, "Could not auto-detect a calibration grid. Provide px_per_cm manually.", time.time() - started_at

    mask = build_stain_mask_lab(lab) if mode == "lab" else build_stain_mask_hsv(hsv)
    contour = get_largest_contour(mask)
    if contour is None:
        return None, None, "No stain contour found in this image.", time.time() - started_at

    metrics = measure_stain(contour, px_per_cm)
    annotated = annotate_image(original, contour, metrics, label=label, color=color)
    return metrics, encode_image_b64(annotated), None, time.time() - started_at


# ===========================================================================
# ROUTES
# ===========================================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():
    pre_img = decode_upload(request.files.get("pre"))
    if pre_img is None:
        return jsonify({"error": "A valid 'pre' image is required."}), 400

    post_img = decode_upload(request.files.get("post"))

    mode = request.form.get("mode", "hsv")
    if mode not in ("hsv", "lab"):
        mode = "hsv"

    use_grid = request.form.get("use_grid", "false").lower() == "true"
    roi = parse_roi(request.form.get("roi"))

    px_per_cm = None
    if not use_grid:
        raw = request.form.get("px_per_cm")
        if raw:
            try:
                px_per_cm = float(raw)
            except ValueError:
                return jsonify({"error": "px_per_cm must be a number."}), 400
        else:
            px_per_cm = 1.0  # no calibration supplied -> report in pixel-derived units

    pre_metrics, pre_img_b64, pre_err, pre_time = run_pipeline(
        pre_img, px_per_cm, mode, roi, "PRE-SCRUB", COLOR_PRE
    )
    result = {
        "pre": {"metrics": metrics_for_api(pre_metrics), "annotated_image": pre_img_b64, "error": pre_err},
        "post": None,
        "delta": None,
        "benchmark": {"sample_count": 1, "processing_time_s": round(pre_time, 4)},
    }

    if post_img is not None:
        post_metrics, post_img_b64, post_err, post_time = run_pipeline(
            post_img, px_per_cm, mode, roi, "POST-SCRUB", COLOR_POST
        )
        result["post"] = {"metrics": metrics_for_api(post_metrics), "annotated_image": post_img_b64, "error": post_err}
        if pre_metrics and post_metrics:
            result["delta"] = compute_delta(pre_metrics, post_metrics)
        result["benchmark"] = {
            "sample_count": 2,
            "processing_time_s": round(pre_time + post_time, 4),
            "average_processing_time_s": round((pre_time + post_time) / 2, 4),
        }

    raw_manual_time = request.form.get("manual_time_minutes")
    if raw_manual_time:
        try:
            manual_time_minutes = float(raw_manual_time)
            if manual_time_minutes < 0:
                raise ValueError
        except ValueError:
            return jsonify({"error": "manual_time_minutes must be a non-negative number."}), 400
        manual_seconds = manual_time_minutes * 60 * result["benchmark"]["sample_count"]
        result["benchmark"]["manual_time_minutes_per_image"] = manual_time_minutes
        result["benchmark"]["estimated_time_saved_s"] = round(
            manual_seconds - result["benchmark"]["processing_time_s"], 4
        )

    return jsonify(result)


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stain Analysis</title>
<style>
  :root { --ink:#1c1c1c; --muted:#6b6b6b; --line:#e4e2dd; --accent:#2f5d50; --bg:#faf9f6; --card:#ffffff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; background:var(--bg); color:var(--ink); }
  header { padding: 28px 32px 8px; }
  header h1 { margin:0; font-size: 22px; letter-spacing: -0.01em; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
  main { max-width: 980px; margin: 0 auto; padding: 24px 32px 64px; }
  .upload-row { display:flex; gap:16px; flex-wrap:wrap; }
  .upload-card { flex:1; min-width:260px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }
  .upload-card h3 { margin:0 0 10px; font-size:14px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
  .upload-card input[type=file] { width:100%; font-size: 13px; }
  .preview { margin-top:10px; max-width:100%; border-radius:6px; display:none; }
  .options { margin-top:18px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; display:flex; gap:20px; flex-wrap:wrap; align-items:end; }
  .field { display:flex; flex-direction:column; gap:6px; font-size:13px; color:var(--muted); }
  .field input, .field select { padding:7px 9px; border:1px solid var(--line); border-radius:6px; font-size:13px; }
  button.run { margin-top:18px; background:var(--accent); color:white; border:none; padding:11px 22px; border-radius:8px; font-size:14px; cursor:pointer; }
  button.run:disabled { opacity:.5; cursor:default; }
  #status { margin-top:12px; font-size:13px; color:var(--muted); }
  .results { margin-top:28px; display:none; }
  .metric-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap:12px; margin-top:12px; }
  .metric { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
  .metric .label { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  .metric .value { font-size:18px; margin-top:4px; }
  .metric .value.up { color:#b3452e; }
  .metric .value.down { color:var(--accent); }
  .stage-title { margin: 26px 0 4px; font-size:15px; font-weight:600; }
  .annotated { max-width:100%; border-radius:8px; border:1px solid var(--line); margin-top:8px; }
  .error { color:#b3452e; font-size:13px; margin-top:8px; }
</style>
</head>
<body>
<header>
  <h1>Adhesive Stain Analysis</h1>
  <p>Upload a pre-scrub image (and optionally a post-scrub image) to measure the stain.</p>
</header>
<main>
  <div class="upload-row">
    <div class="upload-card">
      <h3>Pre-scrub image</h3>
      <input type="file" id="preFile" accept="image/*">
      <img class="preview" id="prePreview">
    </div>
    <div class="upload-card">
      <h3>Post-scrub image (optional)</h3>
      <input type="file" id="postFile" accept="image/*">
      <img class="preview" id="postPreview">
    </div>
  </div>

  <div class="options">
    <div class="field">
      <label>Mode</label>
      <select id="mode"><option value="hsv">HSV (red/brown stains)</option><option value="lab">LAB</option></select>
    </div>
    <div class="field">
      <label>Calibration</label>
      <select id="calibMode"><option value="manual">Manual px/cm</option><option value="grid">Auto-detect grid</option><option value="none">None (pixel units)</option></select>
    </div>
    <div class="field">
      <label>px per cm</label>
      <input type="number" id="pxPerCm" step="0.1" placeholder="e.g. 45.2">
    </div>
    <div class="field">
      <label>Manual time per image (min, optional)</label>
      <input type="number" id="manualTime" min="0" step="0.1" placeholder="e.g. 5">
    </div>
  </div>

  <button class="run" id="runBtn">Analyze</button>
  <div id="status"></div>

  <div class="results" id="results">
    <div class="stage-title">Pre-scrub</div>
    <div class="metric-grid" id="preMetrics"></div>
    <img class="annotated" id="preAnnotated">

    <div id="postSection" style="display:none">
      <div class="stage-title">Post-scrub</div>
      <div class="metric-grid" id="postMetrics"></div>
      <img class="annotated" id="postAnnotated">

      <div class="stage-title">Change (post − pre)</div>
      <div class="metric-grid" id="deltaMetrics"></div>
    </div>
  </div>
</main>

<script>
const preFile = document.getElementById('preFile');
const postFile = document.getElementById('postFile');
const prePreview = document.getElementById('prePreview');
const postPreview = document.getElementById('postPreview');
const calibMode = document.getElementById('calibMode');
const pxPerCmField = document.getElementById('pxPerCm');
const manualTimeField = document.getElementById('manualTime');

function wirePreview(input, img) {
  input.addEventListener('change', () => {
    if (input.files && input.files[0]) {
      img.src = URL.createObjectURL(input.files[0]);
      img.style.display = 'block';
    }
  });
}
wirePreview(preFile, prePreview);
wirePreview(postFile, postPreview);

const METRIC_LABELS = {
  area_cm2: 'Area (cm²)', perimeter_cm: 'Perimeter (cm)',
  bbox_w_cm: 'Width (cm)', bbox_h_cm: 'Height (cm)',
  circularity: 'Circularity', aspect_ratio: 'Aspect ratio',
  centroid_x_cm: 'Centroid X (cm)', centroid_y_cm: 'Centroid Y (cm)',
};
const DELTA_LABELS = {
  area_cm2: 'Δ Area (cm²)', perimeter_cm: 'Δ Perimeter (cm)', circularity: 'Δ Circularity',
  bbox_w_cm: 'Δ Width (cm)', bbox_h_cm: 'Δ Height (cm)',
};

function renderMetrics(container, metrics, labels, signed) {
  container.innerHTML = '';
  for (const key in labels) {
    if (!(key in metrics)) continue;
    const v = metrics[key];
    const div = document.createElement('div');
    div.className = 'metric';
    const cls = signed ? (v > 0 ? 'up' : v < 0 ? 'down' : '') : '';
    div.innerHTML = `<div class="label">${labels[key]}</div><div class="value ${cls}">${signed && v > 0 ? '+' : ''}${v}</div>`;
    container.appendChild(div);
  }
}

document.getElementById('runBtn').addEventListener('click', async () => {
  const statusEl = document.getElementById('status');
  if (!preFile.files[0]) { statusEl.textContent = 'Please choose a pre-scrub image.'; return; }

  const form = new FormData();
  form.append('pre', preFile.files[0]);
  if (postFile.files[0]) form.append('post', postFile.files[0]);
  form.append('mode', document.getElementById('mode').value);

  if (calibMode.value === 'grid') {
    form.append('use_grid', 'true');
  } else if (calibMode.value === 'manual' && pxPerCmField.value) {
    form.append('px_per_cm', pxPerCmField.value);
  }
  if (manualTimeField.value) form.append('manual_time_minutes', manualTimeField.value);

  statusEl.textContent = 'Analyzing...';
  document.getElementById('runBtn').disabled = true;

  try {
    const resp = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) { statusEl.textContent = data.error || 'Something went wrong.'; return; }

    statusEl.textContent = '';
    if (data.benchmark) {
      const b = data.benchmark;
      statusEl.textContent = `Processed ${b.sample_count} image(s) in ${b.processing_time_s}s.` +
        (b.estimated_time_saved_s !== undefined ? ` Estimated time saved: ${(b.estimated_time_saved_s / 60).toFixed(2)} min.` : '');
    }
    document.getElementById('results').style.display = 'block';

    if (data.pre.error) {
      statusEl.textContent = 'Pre-scrub: ' + data.pre.error;
    } else {
      renderMetrics(document.getElementById('preMetrics'), data.pre.metrics, METRIC_LABELS, false);
      document.getElementById('preAnnotated').src = data.pre.annotated_image;
    }

    const postSection = document.getElementById('postSection');
    if (data.post) {
      postSection.style.display = 'block';
      if (data.post.error) {
        statusEl.textContent = 'Post-scrub: ' + data.post.error;
      } else {
        renderMetrics(document.getElementById('postMetrics'), data.post.metrics, METRIC_LABELS, false);
        document.getElementById('postAnnotated').src = data.post.annotated_image;
      }
      if (data.delta) {
        renderMetrics(document.getElementById('deltaMetrics'), data.delta, DELTA_LABELS, true);
      }
    } else {
      postSection.style.display = 'none';
    }
  } catch (e) {
    statusEl.textContent = 'Request failed: ' + e;
  } finally {
    document.getElementById('runBtn').disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
