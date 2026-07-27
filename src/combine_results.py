#!/usr/bin/env python3
"""Combine clean batch result CSVs into one portable results CSV."""

import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCES = [
    ("images1", PROJECT_DIR / "results" / "images1" / "stain_results.csv"),
    ("images2", PROJECT_DIR / "results" / "images2" / "stain_results.csv"),
]
OUTPUT = PROJECT_DIR / "results" / "stain_results_combined.csv"

PREFERRED_COLUMNS = [
    "image_group",
    "image",
    "stage",
    "area_cm2",
    "perimeter_cm",
    "bbox_w_cm",
    "bbox_h_cm",
    "centroid_x_cm",
    "centroid_y_cm",
    "area_m2",
    "perimeter_m",
    "bbox_w_m",
    "bbox_h_m",
    "centroid_x_m",
    "centroid_y_m",
    "aspect_ratio",
    "circularity",
    "centroid_x_px",
    "centroid_y_px",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
]


def read_rows():
    rows = []
    seen = set()

    for image_group, path in SOURCES:
        if not path.exists():
            print(f"[SKIP] Missing CSV: {path}")
            continue

        with path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                row["image_group"] = image_group
                key = (row.get("image_group"), row.get("image"), row.get("stage"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    return rows


def main():
    rows = read_rows()
    if not rows:
        raise SystemExit("[ERROR] No rows found. Run both batches first.")

    columns = [column for column in PREFERRED_COLUMNS if any(column in row for row in rows)]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] Combined CSV: {OUTPUT}")
    print(f"[INFO] Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
