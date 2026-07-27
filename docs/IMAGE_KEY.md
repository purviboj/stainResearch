# Key for Real Stain Images 3

In order, the images represent:

1. Side view of the stain **before** cleaning
2. Top-down view of the stain **before** cleaning
3. Side view of the stain **after** cleaning
4. Top-down view of the stain **after** cleaning

This pattern repeats **4 times**, for progressively increased pre-cooking times:

| Group | Pre-cook time | Files |
|-------|---------------|-------|
| 1 | 0 minutes | `00min_side_pre`, `00min_top_pre`, `00min_side_post`, `00min_top_post` |
| 2 | 10 minutes | `10min_side_pre`, `10min_top_pre`, `10min_side_post`, `10min_top_post` |
| 3 | 20 minutes | `20min_side_pre`, `20min_top_pre`, `20min_side_post`, `20min_top_post` |
| 4 | 30 minutes | `30min_side_pre`, `30min_top_pre`, `30min_side_post`, `30min_top_post` |

All 16 analysis-ready images live in `data/images/processed/real_stain_images_3/`. The filenames and `manifest.csv` follow the supplied Box-folder order: `IMG_2669`, `IMG_2667`, `IMG_2670`, `IMG_2668`, then the next three groups of four in that same displayed order.

## Batch pair files

**Side-view before/after pairs** (`data/pairs/pairs3_side.txt`):

```text
00min_side_pre  →  00min_side_post
10min_side_pre  →  10min_side_post
20min_side_pre  →  20min_side_post
30min_side_pre  →  30min_side_post
```

**Top-down before/after pairs** (`data/pairs/pairs3_top.txt`):

```text
00min_top_pre  →  00min_top_post
10min_top_pre  →  10min_top_post
20min_top_pre  →  20min_top_post
30min_top_pre  →  30min_top_post
```

## Run analysis on Images 3

```bash
python3 src/stainresearch.py \
  --batch data/pairs/pairs3_top.txt \
  --px-per-cm 120 \
  --tune-each \
  --mode hsv \
  --output-dir results/images3_top/

python3 src/stainresearch.py \
  --batch data/pairs/pairs3_side.txt \
  --px-per-cm 120 \
  --tune-each \
  --mode hsv \
  --output-dir results/images3_side/
```

Side views use the grid paper in the background for scale calibration. Top-down views use the square metal plate; adjust `--px-per-cm` if needed.
