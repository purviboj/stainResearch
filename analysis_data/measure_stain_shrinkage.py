import csv
import argparse
import re
import statistics
from pathlib import Path

import cv2
import numpy as np


INPUT_ROOT = Path("analysis_data/Visualization-PusherV3")
OUTPUT_ROOT = Path("analysis_results")
ROI = (450, 400, 1550, 780)


def stain_bbox(image):
    x0, y0, x1, y1 = ROI
    blue, green, red = cv2.split(image)
    mask = (
        (red.astype(np.int16) > green.astype(np.int16) + 35)
        & (red.astype(np.int16) > blue.astype(np.int16) + 35)
        & (red > 50)
    ).astype(np.uint8) * 255
    restricted = np.zeros_like(mask)
    restricted[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    kernel = np.ones((3, 3), np.uint8)
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_OPEN, kernel)
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(restricted)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[index]
    return int(x), int(y), int(width), int(height), int(area)


def read_frame(video_path, index):
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index if index >= 0 else frame_count - 1))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def annotate(image, bbox, label):
    result = image.copy()
    if bbox is not None:
        x, y, width, height, area = bbox
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 3)
        text = f"{label}: {width} x {height} px"
    else:
        text = f"{label}: no detection"
    cv2.putText(result, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    return result


def video_sort_key(video_path):
    numbers = re.findall(r"\d+", video_path.stem)
    return tuple(int(number) for number in numbers), video_path.name.lower()


def find_videos(root):
    videos = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".mov", ".mp4"}),
        key=video_sort_key,
    )
    return videos


def main():
    parser = argparse.ArgumentParser(description="Measure stain size change between first and last video frames.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    frames_root = output_root / "frames"
    pairs_root = output_root / "pairs"
    frames_root.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)
    rows = []
    videos = find_videos(input_root)
    for video_path in videos:
        first = read_frame(video_path, 0)
        last = read_frame(video_path, -1)
        if first is None or last is None:
            rows.append({"trial": video_path.parent.name, "video": video_path.name, "status": "frame_read_failed"})
            continue
        first_bbox = stain_bbox(first)
        last_bbox = stain_bbox(last)
        trial = video_path.parent.name if video_path.parent != input_root else input_root.name
        stem = f"{trial[:1]}_{video_path.stem}"
        cv2.imwrite(str(frames_root / f"{stem}_first_annotated.jpg"), annotate(first, first_bbox, "BEFORE"))
        cv2.imwrite(str(frames_root / f"{stem}_last_annotated.jpg"), annotate(last, last_bbox, "AFTER"))
        first_small = cv2.resize(annotate(first, first_bbox, "BEFORE"), (960, 540))
        last_small = cv2.resize(annotate(last, last_bbox, "AFTER"), (960, 540))
        cv2.imwrite(str(pairs_root / f"{stem}_before_after.jpg"), np.hstack((first_small, last_small)))
        row = {"trial": trial, "video": video_path.name, "status": "ok"}
        if first_bbox is None or last_bbox is None:
            row["status"] = "review_detection"
        else:
            fx, fy, fw, fh, fa = first_bbox
            lx, ly, lw, lh, la = last_bbox
            row.update({
                "before_width_px": fw,
                "before_height_px": fh,
                "after_width_px": lw,
                "after_height_px": lh,
                "x_ratio_after_over_before": round(lw / fw, 5),
                "y_ratio_after_over_before": round(lh / fh, 5),
                "before_mask_area_px": fa,
                "after_mask_area_px": la,
                "area_ratio_after_over_before": round(la / fa, 5),
                "review_flag": "yes" if min(fw, fh, lw, lh) < 20 else "no",
            })
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    fieldnames.remove("trial")
    fieldnames.remove("video")
    fieldnames = ["trial", "video"] + fieldnames
    with (output_root / "shrinkage_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for trial in sorted({row["trial"] for row in rows}):
        valid = [row for row in rows if row["trial"] == trial and row["status"] == "ok"]
        if not valid:
            continue
        x_values = [float(row["x_ratio_after_over_before"]) for row in valid]
        y_values = [float(row["y_ratio_after_over_before"]) for row in valid]
        summary_rows.append({
            "trial": trial,
            "video_count": len(valid),
            "x_ratio_mean": round(statistics.mean(x_values), 5),
            "x_ratio_median": round(statistics.median(x_values), 5),
            "y_ratio_mean": round(statistics.mean(y_values), 5),
            "y_ratio_median": round(statistics.median(y_values), 5),
        })
    with (output_root / "trial_summary.csv").open("w", newline="") as handle:
        summary_fields = ["trial", "video_count", "x_ratio_mean", "x_ratio_median", "y_ratio_mean", "y_ratio_median"]
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Processed {len(rows)} videos")


if __name__ == "__main__":
    main()
