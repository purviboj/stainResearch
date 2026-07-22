"""
stainresearch.py — Adhesive Stain Edge Detection & Tracking Pipeline
=====================================================================
Usage:
  # Single image (interactive tuning mode):
  python src/stainresearch.py --pre pre.jpg

  # Before/after comparison:
  python src/stainresearch.py --pre pre.jpg --post post.jpg

  # With 1cm grid paper for auto-calibration:
  python src/stainresearch.py --pre pre.jpg --post post.jpg --use-grid

  # Manual scale (measure grid manually once per camera setup):
  python src/stainresearch.py --pre pre.jpg --post post.jpg --px-per-cm 45.2

  # Crop to region of interest (x,y,width,height in pixels):
  python src/stainresearch.py --pre pre.jpg --post post.jpg --roi 80,60,400,350

  # Batch mode (pairs.txt = tab-separated pre/post paths, one pair per line):
  python src/stainresearch.py --batch data/pairs/pairs.txt --no-gui --px-per-cm 45.2

  # LAB-based thresholding instead of HSV:
  python src/stainresearch.py --pre pre.jpg --mode lab

Dependencies:
  pip install opencv-python numpy

Project summary:
  This script turns before/after stain photos into quantitative measurements.
  It thresholds the image by color, filters out background-like detections,
  selects the stain contour, measures that contour, and saves annotated images
  plus CSV rows that can be combined later with combine_results.py.

This file is CLI/GUI only. The web-app version (upload-and-analyze via
Flask) lives separately in app.py — that file has its own copy of the
detection math reworked to run on uploaded image bytes instead of file
paths, GUI windows, and argparse flags.
"""

import argparse
import csv
import math
import os
import time

import cv2
import numpy as np

# ===========================================================================
# TUNABLE DEFAULTS — adjust these at the top rather than hunting through code
# ===========================================================================

# --- Preprocessing ---
GAUSSIAN_BLUR_K     = 5
USE_CLAHE           = True
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# --- HSV Stain Thresholds ---
HSV_LOW1  = (0,   100, 40)
HSV_HIGH1 = (12,  255, 200)
HSV_LOW2  = (165, 100, 40)
HSV_HIGH2 = (180, 255, 200)

# --- LAB Stain Thresholds ---
LAB_L_LOW  = 20
LAB_L_HIGH = 180
LAB_A_LOW  = 135
LAB_A_HIGH = 255
LAB_B_LOW  = 120
LAB_B_HIGH = 255

# --- Morphological Operations ---
MORPH_OPEN_K  = 3
MORPH_CLOSE_K = 7

# --- Contour Filtering ---
MIN_CONTOUR_AREA_PX = 200
MIN_CONTOUR_AREA_FRACTION = 0.0005
MAX_CONTOUR_AREA_FRACTION = 0.35
MAX_BBOX_IMAGE_FRACTION = 0.85
BORDER_TOUCH_MARGIN_PX = 3

# --- Grid Detection ---
HOUGH_RHO         = 1
HOUGH_THETA       = np.pi / 180
HOUGH_THRESHOLD   = 80
HOUGH_MIN_LINE    = 40
HOUGH_MAX_GAP     = 10
CANNY_LOW         = 20
CANNY_HIGH        = 60

# --- Visualization Colors (BGR) ---
COLOR_PRE     = (0, 220, 0)
COLOR_POST    = (0, 0, 220)
COLOR_BOX     = (255, 180, 0)
COLOR_ELLIPSE = (255, 100, 255)
COLOR_TEXT    = (255, 255, 255)

# Reused across every image instead of being rebuilt on each call.
_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


# ===========================================================================
# CSV DEDUPLICATION HELPER
# ===========================================================================

def load_existing_csv_keys(csv_path):
    """
    Read the existing CSV and return a set of (image, stage) tuples
    that have already been saved. Used to prevent duplicate rows when
    the script is run multiple times on the same images.
    """
    existing = set()
    if not os.path.isfile(csv_path):
        return existing
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img   = row.get("image", "").strip()
                stage = row.get("stage", "").strip()
                if img and stage:
                    existing.add((img, stage))
    except Exception as e:
        print(f"[WARNING] Could not read existing CSV for dedup check: {e}")
    return existing


def save_results_to_csv(results, csv_path, existing_keys=None):
    """
    Append results to CSV, skipping any rows whose (image, stage) key
    already exists in the file. Prints a skip notice for duplicates.

    existing_keys: pass in a set built once (e.g. via load_existing_csv_keys)
    when calling this repeatedly in a loop — batch mode does this so it
    doesn't re-read the whole CSV from disk on every single pair. If not
    provided, it's loaded fresh from csv_path (fine for a one-off call).

    Returns the updated existing_keys set so a caller can carry it into
    the next call.
    """
    if existing_keys is None:
        existing_keys = load_existing_csv_keys(csv_path)

    if not results:
        return existing_keys

    new_rows = []
    for row in results:
        key = (row.get("image", "").strip(), row.get("stage", "").strip())
        if key in existing_keys:
            print(f"[SKIP] Already in CSV — {key[0]} ({key[1]}), skipping duplicate.")
        else:
            new_rows.append(row)
            existing_keys.add(key)   # guard against duplicates within the same batch run

    if not new_rows:
        print("[INFO] No new rows to write — all results already saved.")
        return existing_keys

    fieldnames  = list(new_rows[0].keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"[SAVED] CSV results ({len(new_rows)} new row(s)): {csv_path}")

    return existing_keys


# ===========================================================================
# BENCHMARKING / VALIDATION HELPERS
# ===========================================================================

BENCHMARK_FIELDS = (
    "image", "stage", "detected_count", "actual_count", "correct",
    "processing_time_s", "detection_status",
)


def load_ground_truth(csv_path):
    """Read a CSV with `image` and `actual_count` columns for validation."""
    if not csv_path:
        return {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "actual_count"}.issubset(reader.fieldnames):
            raise ValueError("Ground-truth CSV must contain image and actual_count columns.")
        truth = {}
        for row in reader:
            try:
                truth[row["image"].strip()] = int(row["actual_count"])
            except (KeyError, TypeError, ValueError):
                print(f"[WARNING] Skipping invalid ground-truth row: {row}")
    return truth


def build_benchmark_row(path, stage, metrics, processing_time_s, ground_truth):
    """Create one validation row. This pipeline is a single-primary-stain detector."""
    actual = ground_truth.get(path)
    if actual is None:
        actual = ground_truth.get(os.path.basename(path))
    detected = 1 if metrics is not None else 0
    return {
        "image": path,
        "stage": stage,
        "detected_count": detected,
        "actual_count": "" if actual is None else actual,
        "correct": "" if actual is None else int(detected == actual),
        "processing_time_s": round(processing_time_s, 4),
        "detection_status": "detected" if detected else "not_detected",
    }


def save_benchmark_results(rows, csv_path):
    """Append unique per-image benchmark rows to a separate, stable CSV."""
    existing = load_existing_csv_keys(csv_path)
    new_rows = [row for row in rows if (row["image"], row["stage"]) not in existing]
    if not new_rows:
        return
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"[SAVED] Benchmark log ({len(new_rows)} new row(s)): {csv_path}")


def print_benchmark_summary(rows, manual_time_minutes=None):
    """Print speed and labelled detection-accuracy metrics for this run."""
    if not rows:
        return
    total_time = sum(row["processing_time_s"] for row in rows)
    labelled = [row for row in rows if row["correct"] != ""]
    print("\n--- BENCHMARK SUMMARY ---")
    print(f"  Samples/images analysed: {len(rows)}")
    print(f"  Processing time: {total_time:.4f} s total; {total_time / len(rows):.4f} s/image")
    if labelled:
        accuracy = sum(row["correct"] for row in labelled) / len(labelled) * 100
        print(f"  Detection accuracy: {accuracy:.2f}% ({len(labelled)} labelled images)")
    else:
        print("  Detection accuracy: not calculated (no ground truth supplied)")
    if manual_time_minutes is not None:
        manual_seconds = manual_time_minutes * 60 * len(rows)
        saved_seconds = manual_seconds - total_time
        print(f"  Estimated time saved vs manual: {saved_seconds / 60:.2f} minutes")


# ===========================================================================
# CORE PIPELINE FUNCTIONS
# ===========================================================================

def load_and_preprocess(path, roi=None):
    """
    Load one image and prepare color spaces used by the detector.
    Returns (original_img, blurred_img, hsv, lab_blur).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if roi is not None:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]

    if USE_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        lab = cv2.merge([l, a, b])
        img_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        img_eq = img.copy()

    k = GAUSSIAN_BLUR_K if GAUSSIAN_BLUR_K % 2 == 1 else GAUSSIAN_BLUR_K + 1
    blurred = cv2.GaussianBlur(img_eq, (k, k), 0)

    hsv      = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab_blur = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    return img, blurred, hsv, lab_blur


def build_stain_mask_hsv(hsv, low1=None, high1=None, low2=None, high2=None,
                          morph_open=None, morph_close=None):
    """Dual-range HSV threshold for red stains. Returns cleaned binary mask."""
    low1  = low1  or HSV_LOW1
    high1 = high1 or HSV_HIGH1
    low2  = low2  or HSV_LOW2
    high2 = high2 or HSV_HIGH2
    mk    = morph_open  or MORPH_OPEN_K
    ck    = morph_close or MORPH_CLOSE_K

    mask1 = cv2.inRange(hsv, np.array(low1), np.array(high1))
    mask2 = cv2.inRange(hsv, np.array(low2), np.array(high2))
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def build_stain_mask_lab(lab, l_range=None, a_range=None, b_range=None,
                          morph_open=None, morph_close=None):
    """LAB-based mask: isolates reddish-brown stain colors."""
    l_range = l_range or (LAB_L_LOW, LAB_L_HIGH)
    a_range = a_range or (LAB_A_LOW, LAB_A_HIGH)
    b_range = b_range or (LAB_B_LOW, LAB_B_HIGH)
    mk = morph_open or MORPH_OPEN_K
    ck = morph_close or MORPH_CLOSE_K

    l, a, b = cv2.split(lab)
    mask_l = cv2.inRange(l, l_range[0], l_range[1])
    mask_a = cv2.inRange(a, a_range[0], a_range[1])
    mask_b = cv2.inRange(b, b_range[0], b_range[1])
    mask   = cv2.bitwise_and(mask_l, cv2.bitwise_and(mask_a, mask_b))

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def detect_grid_spacing(gray):
    """
    Attempt to detect 1cm grid lines via HoughLinesP.
    Returns px_per_cm (float) or None if detection fails.
    """
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA,
                             HOUGH_THRESHOLD, None,
                             HOUGH_MIN_LINE, HOUGH_MAX_GAP)
    if lines is None:
        print("[WARNING] Grid lines not detected. Use --px-per-cm for manual calibration.")
        return None

    h_gaps = []
    v_gaps = []

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
            diffs = [gaps[i+1] - gaps[i] for i in range(len(gaps)-1)
                     if 10 < gaps[i+1] - gaps[i] < 200]
            if diffs:
                spacings.append(np.median(diffs))

    if not spacings:
        print("[WARNING] Could not compute grid spacing from detected lines.")
        return None

    px_per_cm = float(np.mean(spacings))
    print(f"[INFO] Grid calibration: {px_per_cm:.2f} px/cm")
    return px_per_cm


def get_largest_contour(mask, min_area=None):
    """
    Find and return the largest stain-like contour in the binary mask,
    filtering out background edges, table borders, and tiny specks.

    Uses CHAIN_APPROX_SIMPLE (compresses straight contour segments down to
    their endpoints) instead of CHAIN_APPROX_NONE (every point along the
    boundary). Confirmed this gives identical area/perimeter/moments for
    our shapes while being cheaper for findContours and every function
    downstream that walks the contour (moments, drawContours, fitEllipse).
    """
    min_area = min_area or MIN_CONTOUR_AREA_PX
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_h, img_w = mask.shape[:2]
    img_area = float(img_h * img_w)
    min_area = max(min_area, img_area * MIN_CONTOUR_AREA_FRACTION)
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
    """Compute all stain metrics. Returns a dict."""
    area_px      = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    area_cm2     = area_px / (px_per_cm ** 2)
    perimeter_cm = perimeter_px / px_per_cm
    circularity  = (4 * math.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0

    x, y, w, h   = cv2.boundingRect(contour)
    bbox_w_cm    = w / px_per_cm
    bbox_h_cm    = h / px_per_cm
    aspect_ratio = w / h if h > 0 else 0

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2, y + h / 2

    CM_TO_M = 0.01

    return {
        "area_cm2":      round(area_cm2, 4),
        "perimeter_cm":  round(perimeter_cm, 4),
        "bbox_w_cm":     round(bbox_w_cm, 4),
        "bbox_h_cm":     round(bbox_h_cm, 4),
        "centroid_x_cm": round(cx / px_per_cm, 4),
        "centroid_y_cm": round(cy / px_per_cm, 4),
        "area_m2":       round(area_cm2 * (CM_TO_M ** 2), 8),
        "perimeter_m":   round(perimeter_cm * CM_TO_M, 6),
        "bbox_w_m":      round(bbox_w_cm * CM_TO_M, 6),
        "bbox_h_m":      round(bbox_h_cm * CM_TO_M, 6),
        "centroid_x_m":  round((cx / px_per_cm) * CM_TO_M, 6),
        "centroid_y_m":  round((cy / px_per_cm) * CM_TO_M, 6),
        "aspect_ratio":  round(aspect_ratio, 4),
        "circularity":   round(circularity, 4),
        "centroid_x_px": round(cx, 1),
        "centroid_y_px": round(cy, 1),
        "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
    }


def annotate_image(img, contour, metrics, label="", color=None):
    """Draw contour, bounding box, ellipse, and measurement text on image."""
    color = color or COLOR_PRE
    out   = img.copy()

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
    # Automatically draw the 3-inch scale overlay if measurements exist
    if metrics and metrics.get("area_m2", 0) > 0 and metrics.get("area_px", 0) > 0:
        px_per_m = (metrics["area_px"] / metrics["area_m2"]) ** 0.5
        out = draw_3in_reference_overlay(out, px_per_m)
    return out


def process_image(path, px_per_cm, roi=None, mode="hsv",
                  hsv_params=None, label="", color=None):
    """
    Full pipeline for one image.
    Returns (annotated_img, metrics_dict, contour) or (None, None, None) on failure.
    """
    img, blurred, hsv, lab = load_and_preprocess(path, roi=roi)

    if px_per_cm is None:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        px_per_cm = detect_grid_spacing(gray)
        if px_per_cm is None:
            print(f"[ERROR] No px_per_cm available for {path}.")
            return None, None, None

    params = dict(hsv_params or {})
    selected_mode = params.pop("mode", mode)
    params.pop("px_per_cm", None)  # <-- ADD THIS LINE

    if selected_mode == "lab":
        mask = build_stain_mask_lab(
            lab,
            l_range=params.get("L"),
            a_range=params.get("A"),
            b_range=params.get("B"),
            morph_open=params.get("morph_open"),
            morph_close=params.get("morph_close"),
        )
    else:
        if params:
            mask = build_stain_mask_hsv(hsv, **params)
        else:
            mask = build_stain_mask_hsv(hsv)

    contour = get_largest_contour(mask)
    if contour is None:
        print(f"[WARNING] No stain contour found in {path}.")
        return img, None, None

    metrics  = measure_stain(contour, px_per_cm)
    annotated = annotate_image(img, contour, metrics, label=label, color=color)

    return annotated, metrics, contour


# ===========================================================================
# WEB API ADAPTERS
# ===========================================================================

API_METRIC_KEYS = (
    "area_cm2", "area_m2", "perimeter_cm", "bbox_w_cm", "bbox_h_cm",
    "centroid_x_cm", "centroid_y_cm", "circularity", "aspect_ratio",
)


def _api_int(data, key, default, low, high):
    """Read and clamp one integer tuning value supplied by the web form."""
    value = data.get(key, default)
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def build_hsv_params_from_request(data, mode="hsv"):
    """Convert web form fields into the parameter format used by process_image."""
    supplied = data.get("hsv_params")
    if isinstance(supplied, dict):
        # Accept a JSON object as well as individual multipart form fields.
        data = {**data, **supplied}

    open_k = _api_int(data, "morph_open", MORPH_OPEN_K, 1, 99)
    close_k = _api_int(data, "morph_close", MORPH_CLOSE_K, 1, 99)

    if mode == "lab":
        return {
            "mode": "lab",
            "L": (_api_int(data, "l_low", LAB_L_LOW, 0, 255),
                  _api_int(data, "l_high", LAB_L_HIGH, 0, 255)),
            "A": (_api_int(data, "a_low", LAB_A_LOW, 0, 255),
                  _api_int(data, "a_high", LAB_A_HIGH, 0, 255)),
            "B": (_api_int(data, "b_low", LAB_B_LOW, 0, 255),
                  _api_int(data, "b_high", LAB_B_HIGH, 0, 255)),
            "morph_open": open_k,
            "morph_close": close_k,
        }

    s_low = _api_int(data, "s1_low", HSV_LOW1[1], 0, 255)
    v_low = _api_int(data, "v1_low", HSV_LOW1[2], 0, 255)
    s_high = _api_int(data, "s1_high", HSV_HIGH1[1], 0, 255)
    v_high = _api_int(data, "v1_high", HSV_HIGH1[2], 0, 255)
    return {
        "mode": "hsv",
        "low1": (_api_int(data, "h1_low", HSV_LOW1[0], 0, 180), s_low, v_low),
        "high1": (_api_int(data, "h1_high", HSV_HIGH1[0], 0, 180), s_high, v_high),
        "low2": (_api_int(data, "h2_low", HSV_LOW2[0], 0, 180), s_low, v_low),
        "high2": (_api_int(data, "h2_high", HSV_HIGH2[0], 0, 180), s_high, v_high),
        "morph_open": open_k,
        "morph_close": close_k,
    }


def metrics_for_api(metrics):
    """Return only the stable, frontend-facing measurement fields."""
    if not metrics:
        return None
    return {key: metrics[key] for key in API_METRIC_KEYS if key in metrics}


def compute_api_deltas(pre_metrics, post_metrics):
    """Calculate post-minus-pre values used by the web results table."""
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


# ===========================================================================
# INTERACTIVE TUNING UI
# ===========================================================================

def _make_hsv_window(win_name):
    """Create an HSV-only slider window."""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 640)
    cv2.createTrackbar("H1_low",     win_name, HSV_LOW1[0],  180, lambda x: None)
    cv2.createTrackbar("S1_low",     win_name, HSV_LOW1[1],  255, lambda x: None)
    cv2.createTrackbar("V1_low",     win_name, HSV_LOW1[2],  255, lambda x: None)
    cv2.createTrackbar("H1_high",    win_name, HSV_HIGH1[0], 180, lambda x: None)
    cv2.createTrackbar("S1_high",    win_name, HSV_HIGH1[1], 255, lambda x: None)
    cv2.createTrackbar("V1_high",    win_name, HSV_HIGH1[2], 255, lambda x: None)
    cv2.createTrackbar("H2_low",     win_name, HSV_LOW2[0],  180, lambda x: None)
    cv2.createTrackbar("H2_high",    win_name, HSV_HIGH2[0], 180, lambda x: None)
    cv2.createTrackbar("MorphOpen",  win_name, MORPH_OPEN_K,  20, lambda x: None)
    cv2.createTrackbar("MorphClose", win_name, MORPH_CLOSE_K, 30, lambda x: None)


def _make_lab_window(win_name):
    """Create a LAB-only slider window."""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 640)
    cv2.createTrackbar("L_low",      win_name, LAB_L_LOW,  255, lambda x: None)
    cv2.createTrackbar("L_high",     win_name, LAB_L_HIGH, 255, lambda x: None)
    cv2.createTrackbar("A_low",      win_name, LAB_A_LOW,  255, lambda x: None)
    cv2.createTrackbar("A_high",     win_name, LAB_A_HIGH, 255, lambda x: None)
    cv2.createTrackbar("B_low",      win_name, LAB_B_LOW,  255, lambda x: None)
    cv2.createTrackbar("B_high",     win_name, LAB_B_HIGH, 255, lambda x: None)
    cv2.createTrackbar("MorphOpen",  win_name, MORPH_OPEN_K,  20, lambda x: None)
    cv2.createTrackbar("MorphClose", win_name, MORPH_CLOSE_K, 30, lambda x: None)


def interactive_tune(path, roi=None, mode="hsv", current_px_per_cm=1.0):
    """
    Launch interactive tuner with SEPARATE windows for HSV and LAB.

    Only the sliders for the active mode are visible at a time.
    Press 'm' to toggle between modes — the old window closes and the
    new one opens so you only ever see relevant sliders.

    Press:
      'm' → switch between HSV and LAB windows
      'g' → toggle 1cm auto-grid detection ON/OFF
      's' → print current params to terminal
      'q' or 'n' → accept settings and advance to NEXT image
    """
    img, blurred, hsv, lab = load_and_preprocess(path, roi=roi)

    WIN_HSV  = "[ HSV MODE ]  M=switch | G=grid | S=save | N=next image"
    WIN_LAB  = "[ LAB MODE ]  M=switch | G=grid | S=save | N=next image"
    PREVIEW  = "Stain Preview  (left=detected contour | right=mask)"

    current_mode = mode
    saved_params = {}
    last_params_key = None
    last_combined = None
    grid_active = False
    active_px_per_cm = current_px_per_cm

    # Open the starting window based on mode arg
    if current_mode == "hsv":
        _make_hsv_window(WIN_HSV)
        cv2.imshow(WIN_HSV, np.zeros((100, 500, 3), dtype=np.uint8))
    else:
        _make_lab_window(WIN_LAB)
        cv2.imshow(WIN_LAB, np.zeros((100, 500, 3), dtype=np.uint8))

    cv2.namedWindow(PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW, 1200, 560)

    while True:
        active_win = WIN_HSV if current_mode == "hsv" else WIN_LAB

        mk = max(cv2.getTrackbarPos("MorphOpen",  active_win), 1)
        ck = max(cv2.getTrackbarPos("MorphClose", active_win), 1)

        if current_mode == "hsv":
            h1l = cv2.getTrackbarPos("H1_low",  WIN_HSV)
            s1l = cv2.getTrackbarPos("S1_low",  WIN_HSV)
            v1l = cv2.getTrackbarPos("V1_low",  WIN_HSV)
            h1h = cv2.getTrackbarPos("H1_high", WIN_HSV)
            s1h = cv2.getTrackbarPos("S1_high", WIN_HSV)
            v1h = cv2.getTrackbarPos("V1_high", WIN_HSV)
            h2l = cv2.getTrackbarPos("H2_low",  WIN_HSV)
            h2h = cv2.getTrackbarPos("H2_high", WIN_HSV)

            saved_params = dict(
                mode="hsv",
                low1=(h1l, s1l, v1l), high1=(h1h, s1h, v1h),
                low2=(h2l, s1l, v1l), high2=(h2h, s1h, v1h),
                morph_open=mk, morph_close=ck
            )
        else:
            ll = cv2.getTrackbarPos("L_low",  WIN_LAB)
            lh = cv2.getTrackbarPos("L_high", WIN_LAB)
            al = cv2.getTrackbarPos("A_low",  WIN_LAB)
            ah = cv2.getTrackbarPos("A_high", WIN_LAB)
            bl = cv2.getTrackbarPos("B_low",  WIN_LAB)
            bh = cv2.getTrackbarPos("B_high", WIN_LAB)

            saved_params = dict(
                mode="lab",
                L=(ll, lh), A=(al, ah), B=(bl, bh),
                morph_open=mk, morph_close=ck
            )

        # Recompute preview if sliders moved or grid state changed.
        params_key = tuple(sorted(saved_params.items())) + (active_px_per_cm, grid_active)
        if params_key != last_params_key:
            if saved_params["mode"] == "hsv":
                mask = build_stain_mask_hsv(
                    hsv,
                    low1=saved_params["low1"], high1=saved_params["high1"],
                    low2=saved_params["low2"], high2=saved_params["high2"],
                    morph_open=saved_params["morph_open"], morph_close=saved_params["morph_close"]
                )
            else:
                mask = build_stain_mask_lab(
                    lab,
                    l_range=saved_params["L"], a_range=saved_params["A"], b_range=saved_params["B"],
                    morph_open=saved_params["morph_open"], morph_close=saved_params["morph_close"],
                )

            preview          = img.copy()
            selected_contour = get_largest_contour(mask)
            selected_mask    = np.zeros_like(mask)
            if selected_contour is not None:
                cv2.drawContours(preview, [selected_contour], -1, COLOR_PRE, 2)
                cv2.drawContours(selected_mask, [selected_contour], -1, 255, -1)

            mask_rgb = cv2.cvtColor(selected_mask, cv2.COLOR_GRAY2BGR)

            bar_h    = 40
            combined = np.hstack([
                cv2.resize(preview,  (600, 500)),
                cv2.resize(mask_rgb, (600, 500))
            ])
            info_canvas = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
            cv2.rectangle(info_canvas, (0, 0), (combined.shape[1], bar_h), (30, 30, 30), -1)
            label_color = (0, 255, 255) if current_mode == "hsv" else (255, 180, 0)
            
            grid_status = f"GRID: {'ON' if grid_active else 'OFF'} ({active_px_per_cm:.1f} px/cm)"
            cv2.putText(info_canvas,
                        f"MODE: {current_mode.upper()}  |  {grid_status}  |  M=switch  G=grid  S=save  N=next image",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)
            combined = np.vstack([info_canvas, combined])

            last_combined = combined
            last_params_key = params_key
        else:
            combined = last_combined

        cv2.imshow(PREVIEW, combined)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('m'):
            if current_mode == "hsv":
                cv2.destroyWindow(WIN_HSV)
                current_mode = "lab"
                _make_lab_window(WIN_LAB)
                cv2.imshow(WIN_LAB, np.zeros((100, 500, 3), dtype=np.uint8))
                print("[MODE SWITCH] Now using: LAB")
            else:
                cv2.destroyWindow(WIN_LAB)
                current_mode = "hsv"
                _make_hsv_window(WIN_HSV)
                cv2.imshow(WIN_HSV, np.zeros((100, 500, 3), dtype=np.uint8))
                print("[MODE SWITCH] Now using: HSV")
            last_params_key = None

        elif key == ord('g'):
            grid_active = not grid_active
            if grid_active:
                gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
                detected = detect_grid_spacing(gray)
                if detected:
                    active_px_per_cm = detected
                    print(f"[GRID TOGGLE] Auto-grid scale detected: {active_px_per_cm:.2f} px/cm")
                else:
                    print("[GRID TOGGLE] Could not detect grid lines in this image.")
            else:
                active_px_per_cm = current_px_per_cm
                print(f"[GRID TOGGLE] Grid off. Reverted scale: {active_px_per_cm:.2f} px/cm")
            last_params_key = None

        elif key == ord('s'):
            print("[SAVED PARAMS]", saved_params)

        elif key == ord('q') or key == ord('n'):
            break

    cv2.destroyAllWindows()
    saved_params["px_per_cm"] = active_px_per_cm
    return saved_params


# ===========================================================================
# SHARED REPORTING / OUTPUT HELPERS
# (previously duplicated separately inside compare_pre_post and run_batch)
# ===========================================================================

def build_result_rows(pre_path, post_path, metrics_pre, metrics_post):
    """
    Print pre/post metrics and the post-minus-pre delta to the console,
    and return the CSV-ready row list for this pair.
    """
    results = []

    if metrics_pre:
        print("\n--- PRE-SCRUB METRICS ---")
        for k, v in metrics_pre.items():
            print(f"  {k}: {v}")
        results.append({"image": pre_path, "stage": "pre", **metrics_pre})

    if metrics_post:
        print("\n--- POST-SCRUB METRICS ---")
        for k, v in metrics_post.items():
            print(f"  {k}: {v}")
        results.append({"image": post_path, "stage": "post", **metrics_post})

    if metrics_pre and metrics_post:
        print("\n--- DELTA (post - pre) ---")
        print("  [cm units]")
        for key in ["area_cm2", "perimeter_cm", "circularity", "bbox_w_cm", "bbox_h_cm"]:
            delta = round(metrics_post[key] - metrics_pre[key], 4)
            pct   = round(delta / metrics_pre[key] * 100, 2) if metrics_pre[key] != 0 else "N/A"
            print(f"  Δ{key}: {delta:+.4f}  ({pct}%)")
        dx = round(metrics_post["centroid_x_cm"] - metrics_pre["centroid_x_cm"], 4)
        dy = round(metrics_post["centroid_y_cm"] - metrics_pre["centroid_y_cm"], 4)
        print(f"  Centroid shift: Δx={dx:+.4f} cm, Δy={dy:+.4f} cm")
        print("  [meter units — matches Rocky simulation CSV]")
        for key in ["area_m2", "perimeter_m", "bbox_w_m", "bbox_h_m"]:
            delta = round(metrics_post[key] - metrics_pre[key], 8)
            pct   = round(delta / metrics_pre[key] * 100, 2) if metrics_pre[key] != 0 else "N/A"
            print(f"  Δ{key}: {delta:+.8f}  ({pct}%)")
        dx_m = round(metrics_post["centroid_x_m"] - metrics_pre["centroid_x_m"], 6)
        dy_m = round(metrics_post["centroid_y_m"] - metrics_pre["centroid_y_m"], 6)
        print(f"  Centroid shift: Δx={dx_m:+.6f} m, Δy={dy_m:+.6f} m")

    return results


def build_comparison_canvas(ann_pre, ann_post, contour_post=None):
    """
    Build the side-by-side pre/post annotated canvas, with the post
    contour overlaid on the pre image for a direct visual comparison.
    """
    if ann_pre is None or ann_post is None:
        return None

    h = max(ann_pre.shape[0], ann_post.shape[0])
    w = ann_pre.shape[1] + ann_post.shape[1]
    canvas = np.zeros((h + 30, w, 3), dtype=np.uint8)
    canvas[:ann_pre.shape[0],  :ann_pre.shape[1]]  = ann_pre
    canvas[:ann_post.shape[0], ann_pre.shape[1]:]  = ann_post
    cv2.rectangle(canvas, (0, h), (w, h + 30), (40, 40, 40), -1)
    cv2.putText(canvas, "PRE-SCRUB",  (10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRE,  2)
    cv2.putText(canvas, "POST-SCRUB", (ann_pre.shape[1] + 10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_POST, 2)

    if contour_post is not None:
        overlay_img = ann_pre.copy()
        cv2.drawContours(overlay_img, [contour_post], -1, COLOR_POST, 2)
        overlay = cv2.addWeighted(overlay_img, 0.5, ann_pre, 0.5, 0)
        canvas[:overlay.shape[0], :overlay.shape[1]] = overlay

    return canvas


def save_pair_outputs(output_dir, pre_path, post_path, ann_pre, ann_post,
                       contour_post, comparison_filename, results, existing_keys):
    """
    Save the comparison canvas, annotated pre/post images, and dedup-safe
    CSV rows for one pre/post pair. Returns the updated existing_keys set
    so the caller can pass it straight into the next pair without
    re-reading the CSV from disk.
    """
    canvas = build_comparison_canvas(ann_pre, ann_post, contour_post)
    if canvas is not None:
        out_img_path = os.path.join(output_dir, comparison_filename)
        cv2.imwrite(out_img_path, canvas)
        print(f"\n[SAVED] Comparison image: {out_img_path}")

    pre_base  = os.path.splitext(os.path.basename(pre_path))[0]
    post_base = os.path.splitext(os.path.basename(post_path))[0]
    if ann_pre is not None:
        p = os.path.join(output_dir, f"{pre_base}_annotated.png")
        cv2.imwrite(p, ann_pre)
        print(f"[SAVED] Annotated pre: {p}")
    if ann_post is not None:
        p = os.path.join(output_dir, f"{post_base}_annotated.png")
        cv2.imwrite(p, ann_post)
        print(f"[SAVED] Annotated post: {p}")

    csv_path = os.path.join(output_dir, "stain_results.csv")
    return save_results_to_csv(results, csv_path, existing_keys=existing_keys)


# ===========================================================================
# BEFORE / AFTER COMPARISON (single pair)
# ===========================================================================

def compare_pre_post(pre_path, post_path, px_per_cm, roi=None,
                     mode="hsv", hsv_params=None, output_dir=".",
                     ground_truth=None, manual_time_minutes=None):
    """
    Run pipeline on both images, generate side-by-side comparison,
    print delta metrics, save results (deduped).
    """
    print(f"\n[PRE]  Processing: {pre_path}")
    pre_start = time.time()
    ann_pre, metrics_pre, contour_pre = process_image(
        pre_path, px_per_cm, roi=roi, mode=mode,
        hsv_params=hsv_params, label="PRE-SCRUB", color=COLOR_PRE
    )
    pre_time = time.time() - pre_start

    print(f"[POST] Processing: {post_path}")
    post_start = time.time()
    ann_post, metrics_post, contour_post = process_image(
        post_path, px_per_cm, roi=roi, mode=mode,
        hsv_params=hsv_params, label="POST-SCRUB", color=COLOR_POST
    )
    post_time = time.time() - post_start

    results = build_result_rows(pre_path, post_path, metrics_pre, metrics_post)
    save_pair_outputs(output_dir, pre_path, post_path, ann_pre, ann_post,
                       contour_post, "comparison.png", results, existing_keys=None)

    benchmark_rows = [
        build_benchmark_row(pre_path, "pre", metrics_pre, pre_time, ground_truth or {}),
        build_benchmark_row(post_path, "post", metrics_post, post_time, ground_truth or {}),
    ]
    save_benchmark_results(benchmark_rows, os.path.join(output_dir, "benchmark_results.csv"))
    print_benchmark_summary(benchmark_rows, manual_time_minutes)

    return results


# ===========================================================================
# BATCH MODE
# ===========================================================================

def run_batch(pairs_file, px_per_cm, roi=None, mode="hsv", output_dir=".",
              tune_each=False, ground_truth=None, manual_time_minutes=None):
    """Process all pre/post pairs listed in a tab-separated text file."""
    with open(pairs_file) as f:
        pairs = [line.strip().split("\t") for line in f if line.strip()]

    # Load the dedup key set ONCE for the whole batch instead of re-reading
    # the growing CSV from scratch on every pair (was O(pairs^2) before).
    csv_path = os.path.join(output_dir, "stain_results.csv")
    existing_keys = load_existing_csv_keys(csv_path)

    all_results = []
    benchmark_rows = []
    for i, pair in enumerate(pairs):
        if len(pair) < 2:
            print(f"[SKIP] Bad line in batch file: {pair}")
            continue
        pre_path, post_path = pair[0], pair[1]

        hsv_params_pre  = None
        hsv_params_post = None

        if tune_each:
            info_pre = f"Pair {i+1}/{len(pairs)} - PRE"
            print(f"\n[TUNER] {info_pre}: {pre_path}")
            hsv_params_pre = interactive_tune(pre_path, roi=roi, mode=mode, current_px_per_cm=px_per_cm or 1.0, batch_info=info_pre)
            if "px_per_cm" in hsv_params_pre:
                px_per_cm = hsv_params_pre["px_per_cm"]

            info_post = f"Pair {i+1}/{len(pairs)} - POST"
            print(f"\n[TUNER] {info_post}: {post_path}")
            hsv_params_post = interactive_tune(post_path, roi=roi, mode=mode, current_px_per_cm=px_per_cm or 1.0, batch_info=info_post)
            if "px_per_cm" in hsv_params_post:
                px_per_cm = hsv_params_post["px_per_cm"]

        print(f"\n[PRE]  Processing: {pre_path}")
        pre_start = time.time()
        ann_pre, metrics_pre, contour_pre = process_image(
            pre_path, px_per_cm, roi=roi, mode=mode,
            hsv_params=hsv_params_pre, label="PRE-SCRUB", color=COLOR_PRE
        )
        pre_time = time.time() - pre_start

        print(f"[POST] Processing: {post_path}")
        post_start = time.time()
        ann_post, metrics_post, contour_post = process_image(
            post_path, px_per_cm, roi=roi, mode=mode,
            hsv_params=hsv_params_post, label="POST-SCRUB", color=COLOR_POST
        )
        post_time = time.time() - post_start

        results = build_result_rows(pre_path, post_path, metrics_pre, metrics_post)

        pair_name = os.path.splitext(os.path.basename(pre_path))[0]
        existing_keys = save_pair_outputs(
            output_dir, pre_path, post_path, ann_pre, ann_post, contour_post,
            f"comparison_{pair_name}.png", results, existing_keys=existing_keys
        )

        all_results.extend(results)
        benchmark_rows.extend([
            build_benchmark_row(pre_path, "pre", metrics_pre, pre_time, ground_truth or {}),
            build_benchmark_row(post_path, "post", metrics_post, post_time, ground_truth or {}),
        ])

    print(f"\n[BATCH DONE] Processed {len(pairs)} pairs.")
    save_benchmark_results(benchmark_rows, os.path.join(output_dir, "benchmark_results.csv"))
    print_benchmark_summary(benchmark_rows, manual_time_minutes)
    return all_results


# ===========================================================================
# MAIN
# ===========================================================================

def parse_roi(s):
    """Parse a command-line crop rectangle written as x,y,width,height."""
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 4, "ROI must be x,y,width,height"
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Adhesive Stain Edge Detection & Tracking")
    parser.add_argument("--pre",        help="Pre-scrub image path")
    parser.add_argument("--post",       help="Post-scrub image path (optional)")
    parser.add_argument("--batch",      help="Batch mode: path to tab-separated pairs.txt")
    parser.add_argument("--use-grid",   action="store_true")
    parser.add_argument("--px-per-cm",  type=float, default=None)
    parser.add_argument("--roi",        type=parse_roi, default=None)
    parser.add_argument("--mode",       choices=["hsv", "lab"], default="hsv")
    parser.add_argument("--no-gui",     action="store_true")
    parser.add_argument("--tune-only",  action="store_true")
    parser.add_argument("--tune-each",  action="store_true")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--ground-truth-csv", help="CSV with image,actual_count columns")
    parser.add_argument("--manual-time-minutes", type=float,
                        help="Manual analysis time per image, for time-saved estimate")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    px_per_cm = args.px_per_cm
    if args.manual_time_minutes is not None and args.manual_time_minutes < 0:
        parser.error("--manual-time-minutes must be zero or greater")
    try:
        ground_truth = load_ground_truth(args.ground_truth_csv)
    except ValueError as exc:
        parser.error(str(exc))

    if args.use_grid and args.pre:
        img, blurred, _, _ = load_and_preprocess(args.pre, roi=args.roi)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        px_per_cm = detect_grid_spacing(gray)

    # 2. Interactive 3-inch calibration line check (NEW)
    if px_per_cm is None and not args.no_gui and args.pre and not args.batch:
        raw_img = cv2.imread(args.pre)
        if raw_img is not None:
            line_px = interactive_draw_3in_line(raw_img)
            if line_px:
                px_per_m = scale_from_reference_line(line_px, real_world_inches=3.0)
                px_per_cm = px_per_m / 100.0  # Convert px/m to px/cm for internal math

    if px_per_cm is None and not args.use_grid:
        print("[WARNING] No px/cm calibration set. Measurements will be in pixels.")
        px_per_cm = 1.0

    hsv_params = None
    if not args.no_gui and not args.tune_only and args.pre:
        print("\n[TUNER] Launching interactive HSV tuner. Adjust sliders, press 'n' when done.")
        hsv_params = interactive_tune(args.pre, roi=args.roi, mode=args.mode, current_px_per_cm=px_per_cm)
        if "px_per_cm" in hsv_params:
            px_per_cm = hsv_params["px_per_cm"]

    if args.tune_only:
        if args.pre:
            interactive_tune(args.pre, roi=args.roi, mode=args.mode, current_px_per_cm=px_per_cm)
        return

    if args.batch:
        run_batch(args.batch, px_per_cm, roi=args.roi,
                  mode=args.mode, output_dir=args.output_dir,
                  tune_each=args.tune_each, ground_truth=ground_truth,
                  manual_time_minutes=args.manual_time_minutes)
        return

    if args.pre:
        compare_pre_post(
            args.pre,
            args.post or args.pre,
            px_per_cm,
            roi=args.roi,
            mode=args.mode,
            hsv_params=hsv_params,
            output_dir=args.output_dir,
            ground_truth=ground_truth,
            manual_time_minutes=args.manual_time_minutes,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# =====================================================================
# 3-INCH REFERENCE LINE & SCALE OVERLAY
# =====================================================================

def interactive_draw_3in_line(image):
    """
    Opens an OpenCV window where the user can click and drag a 3-inch reference line.
    Returns the pixel length of the drawn line.
    """
    line_points = []
    drawing = False
    temp_img = image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal line_points, drawing, temp_img
        if event == cv2.EVENT_LBUTTONDOWN:
            line_points = [(x, y)]
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_img = image.copy()
            cv2.line(temp_img, line_points[0], (x, y), (0, 255, 255), 2)
            cv2.imshow("Draw 3-Inch Scale Line (Click & Drag, Press ENTER)", temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            line_points.append((x, y))
            drawing = False
            cv2.line(temp_img, line_points[0], line_points[1], (0, 255, 0), 2)
            cv2.imshow("Draw 3-Inch Scale Line (Click & Drag, Press ENTER)", temp_img)

    win_name = "Draw 3-Inch Scale Line (Click & Drag, Press ENTER)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)
    cv2.imshow(win_name, temp_img)

    print("[INFO] Click and drag across a 3-inch object in the photo, then press ENTER.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 27:  # ENTER (13) or ESC (27)
            break

    cv2.destroyWindow(win_name)

    if len(line_points) == 2:
        p1, p2 = line_points[0], line_points[1]
        pixel_length = float(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
        return pixel_length
    return None


def draw_3in_reference_overlay(image, pixels_per_meter, position=(30, 40)):
    """
    Draws a flat horizontal 3-inch (0.0762m) reference line with a bounding border box 
    on the annotated image for visual verification.
    """
    if pixels_per_meter is None or pixels_per_meter <= 0:
        return image

    THREE_INCHES_IN_METERS = 3.0 * 0.0254  # 0.0762 m
    line_px = int(THREE_INCHES_IN_METERS * pixels_per_meter)
    
    x, y = position
    pad = 8
    
    # Draw dark background box for readability
    cv2.rectangle(image, (x - pad, y - pad), (x + line_px + pad, y + 25), (30, 30, 30), -1)
    
    # Draw horizontal 3-inch line (bright cyan)
    cv2.line(image, (x, y), (x + line_px, y), (255, 255, 0), 2)
    
    # Draw start/end vertical tick marks
    cv2.line(image, (x, y - 4), (x, y + 4), (255, 255, 0), 2)
    cv2.line(image, (x + line_px, y - 4), (x + line_px, y + 4), (255, 255, 0), 2)
    
    # Draw border outline
    cv2.rectangle(image, (x - pad, y - pad), (x + line_px + pad, y + 25), (255, 255, 0), 1)
    
    # Label text
    cv2.putText(image, "3.0 in (0.0762m) scale", (x, y + 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image

# =====================================================================
# ADDITIONAL CALIBRATION & EXPORT HELPERS
# =====================================================================

def scale_from_reference_line(pixel_length, real_world_inches=3.0):
    """Converts pixel line length to Pixels per Meter scale factor."""
    meters = real_world_inches * 0.0254
    pixels_per_meter = pixel_length / meters
    print(f"[INFO] 3-inch line registered: {pixel_length}px = {meters:.4f}m.")
    return pixels_per_meter


def undistort_image(image, camera_matrix=None, dist_coeffs=None):
    """Removes lens distortion if camera matrix is provided."""
    if camera_matrix is None or dist_coeffs is None:
        print("[INFO] No camera calibration data provided. Skipping undistort.")
        return image
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]


def find_grid_homography(image, pattern_size=(9, 6)):
    """Perspective transform to flatten grid perspective."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        print("[WARNING] Reference grid not found. Returning original image.")
        return image, None
    obj_points = np.zeros((pattern_size[0] * pattern_size[1], 2), np.float32)
    obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    obj_points *= 0.05
    H, _ = cv2.findHomography(corners, obj_points)
    warped = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    return warped, H


def export_contour_points(contour, pixels_per_meter, output_dir="data", base_filename="stain_edge"):
    """Exports contour coordinates to JSON and CSV in meters, cm, and inches."""
    import json
    os.makedirs(output_dir, exist_ok=True)
    M_TO_CM, M_TO_IN = 100.0, 39.3700787
    data_out = []
    for point in contour:
        x_px, y_px = point[0]
        x_m = x_px / pixels_per_meter
        y_m = y_px / pixels_per_meter
        data_out.append({
            "pixel_x": int(x_px), "pixel_y": int(y_px),
            "m_x": float(x_m), "m_y": float(y_m),
            "cm_x": float(x_m * M_TO_CM), "cm_y": float(y_m * M_TO_CM),
            "in_x": float(x_m * M_TO_IN), "in_y": float(y_m * M_TO_IN)
        })
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(data_out, f, indent=4)
    csv_path = os.path.join(output_dir, f"{base_filename}_plot.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_out[0].keys())
        writer.writeheader()
        writer.writerows(data_out)
    return json_path, csv_path