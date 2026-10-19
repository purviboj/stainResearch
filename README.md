# Adhesive Stain Detection Project

OpenCV research pipeline and web interfaces for measuring changes in adhesive stains between before/after photos.

## Project structure

```text
finalstain-project/
├── src/                         # Reusable research and data-processing code
│   ├── stainresearch.py          # CLI pipeline, tuner, measurements, benchmarking
│   └── combine_results.py        # CSV-combining utility
├── backend/                      # Flask API used by the React application
├── frontend/                     # React + Vite upload interface
├── webapp/                       # Standalone Flask upload interface
├── scripts/                      # Convenience commands
│   └── stainrun.sh               # Runs the original two CLI batches
├── data/
│   ├── images/processed/         # Analysis-ready source images
│   ├── raw/                      # Original unprocessed image exports
│   ├── pairs/                    # Before/after pair definitions
│   ├── templates/                # Ground-truth labelling template
│   └── archives/                 # Supplied image ZIP files
├── docs/                         # Dataset and workflow documentation
│   └── IMAGE_KEY.md              # Real Stain Images 3 reference
├── results/                      # Generated annotations, comparisons, and CSVs
└── archives/                     # Project handoff archive
```

## Setup

```bash
pip install opencv-python numpy
pip install -r backend/requirements.txt
```

For the React interface:

```bash
cd frontend
npm install
```

## Run the web interfaces

The React interface uses the Flask API:

```bash
python3 backend/app.py
```

In a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` and upload a before/after pair. The standalone Flask interface is also available with:

```bash
python3 webapp/app.py
```

## Run the local research pipeline

Run the original batches:

```bash
./scripts/stainrun.sh
```

Or run a batch directly:

```bash
python3 src/stainresearch.py \
  --batch data/pairs/pairs1.txt \
  --px-per-cm 120 \
  --tune-each \
  --mode lab \
  --output-dir results/images1/
```

Use the interactive tuner to isolate the stain, then press `s` to print settings and `q` or `n` to continue. For difficult photos, use `--roi x,y,width,height` to crop analysis to the stain area.

## Measure video stain shrinkage

To analyze the first and last frame of the PusherV3 videos, download and extract the supplied `Visualization-PusherV3.zip` into `analysis_data/`, then run:

```bash
python3 analysis_data/measure_stain_shrinkage.py
```

The script writes per-video measurements to `analysis_results/shrinkage_measurements.csv`, trial averages to `analysis_results/trial_summary.csv`, and annotated before/after images to `analysis_results/`. The X and Y ratios are final pixel dimensions divided by initial pixel dimensions, so `0.5` means the stain ended at half its starting size. The ZIP data and generated results are intentionally kept outside Git because the video archive is large.

For a flat video folder or a separate dataset, specify the input and output folders explicitly:

```bash
python3 analysis_data/measure_stain_shrinkage.py \
  --input-root /absolute/path/to/videos \
  --output-root /absolute/path/to/results
```

Video filenames are processed in numerical order, and both flat folders and nested trial folders are supported.

## Benchmarking and validation

Each CLI run writes `benchmark_results.csv` alongside `stain_results.csv`. The benchmark log records one row per image: detection status, processing time, actual stain count, and correctness.

1. Fill in [data/templates/ground_truth_template.csv](data/templates/ground_truth_template.csv) with supervisor-provided `actual_count` values.
2. Run a labelled batch:

```bash
python3 src/stainresearch.py \
  --batch data/pairs/pairs3_top.txt \
  --px-per-cm 120 \
  --mode hsv \
  --ground-truth-csv data/templates/ground_truth_template.csv \
  --manual-time-minutes 5 \
  --output-dir results/benchmark_top/
```

The terminal reports image count, average processing time, detection accuracy for labelled images, and estimated time saved over manual analysis. This is a single-primary-stain detector, so the present accuracy metric is appropriate when actual count is `0` or `1`.

See [docs/IMAGE_KEY_1_2.md](docs/IMAGE_KEY_1_2.md) and [docs/IMAGE_KEY.md](docs/IMAGE_KEY.md) for the dataset keys and pair definitions.
