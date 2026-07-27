"""Flask API — connects the React frontend to stainresearch.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stainresearch import (  # noqa: E402
    COLOR_POST,
    COLOR_PRE,
    build_hsv_params_from_request,
    compute_api_deltas,
    metrics_for_api,
    process_image,
)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _parse_form_data():
    data = request.form.to_dict()
    if "hsv_params" in data:
        try:
            data["hsv_params"] = json.loads(data["hsv_params"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid hsv_params JSON: {exc}") from exc
    return data


def _float_param(data, key, default):
    val = data.get(key)
    if val is None or val == "":
        return default
    return float(val)


def _save_upload(upload, dest_dir, label):
    if not upload or not upload.filename:
        raise ValueError("Missing image upload")
    filename = secure_filename(upload.filename)
    if not _allowed_file(filename):
        raise ValueError(f"Unsupported file type: {filename}")
    # The two uploads may legitimately have the same original filename.
    # Prefixing prevents the post file from overwriting the pre file.
    path = dest_dir / f"{label}_{filename}"
    upload.save(path)
    return path


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/analyze")
def analyze():
    """
    Accept pre/post image uploads plus HSV/LAB tuning params.
    Returns measurement JSON for both images and deltas.
    """
    try:
        data = _parse_form_data()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    pre_file = request.files.get("pre") or request.files.get("pre_image")
    post_file = request.files.get("post") or request.files.get("post_image")
    if not pre_file or not post_file:
        return jsonify({"error": "Both pre and post images are required."}), 400

    mode = data.get("mode", "hsv")
    if mode not in {"hsv", "lab"}:
        return jsonify({"error": "mode must be 'hsv' or 'lab'."}), 400

    try:
        px_per_cm = _float_param(data, "px_per_cm", 120.0)
    except ValueError:
        return jsonify({"error": "px_per_cm must be a number."}), 400
    if px_per_cm <= 0:
        return jsonify({"error": "px_per_cm must be greater than zero."}), 400
    threshold_params = build_hsv_params_from_request(data, mode=mode)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            pre_path = _save_upload(pre_file, tmp_dir, "pre")
            post_path = _save_upload(post_file, tmp_dir, "post")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        _, metrics_pre, _ = process_image(
            str(pre_path),
            px_per_cm,
            mode=mode,
            hsv_params=threshold_params,
            label="PRE-SCRUB",
            color=COLOR_PRE,
        )
        _, metrics_post, _ = process_image(
            str(post_path),
            px_per_cm,
            mode=mode,
            hsv_params=threshold_params,
            label="POST-SCRUB",
            color=COLOR_POST,
        )

    pre = metrics_for_api(metrics_pre)
    post = metrics_for_api(metrics_post)

    if pre is None and post is None:
        return jsonify({
            "error": "No stain contour detected in either image. Adjust HSV sliders and try again.",
            "pre": None,
            "post": None,
            "delta": None,
        }), 422

    delta = compute_api_deltas(metrics_pre, metrics_post)

    return jsonify({
        "pre": pre,
        "post": post,
        "delta": delta,
        "px_per_cm": px_per_cm,
        "mode": mode,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
