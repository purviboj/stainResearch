#!/bin/bash
set -e

PYTHON=${PYTHON:-python3}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"

echo "Running Batch 1..."
"$PYTHON" src/stainresearch.py \
  --batch data/pairs/pairs1.txt \
  --px-per-cm 120 \
  --tune-each \
  --mode lab \
  --output-dir results/images1/

echo "Batch 1 done."

echo "Running Batch 2..."
"$PYTHON" src/stainresearch.py \
  --batch data/pairs/pairs2.txt \
  --px-per-cm 120 \
  --tune-each \
  --mode lab \
  --output-dir results/images2/

echo "All batches complete. Results are in results/images1/ and results/images2/."
