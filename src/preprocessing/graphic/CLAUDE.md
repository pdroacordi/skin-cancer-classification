# src/preprocessing/graphic/CLAUDE.md

Image-level preprocessing applied **offline** to dermoscopic images before training.
Controlled by `USE_GRAPHIC_PREPROCESSING`, `USE_HAIR_REMOVAL`, and
`USE_ENHANCED_CONTRAST` in `config.py`.

## When to Use

Run offline once using `preprocess_dataset.py` to produce preprocessed copies of all
images. After that, set `USE_GRAPHIC_PREPROCESSING=True` in `config.py` and the
pipeline will automatically read from `res/preprocessed_*_files.txt`.

```bash
# From src/
python preprocessing/preprocess_dataset.py \
  --metadata path/to/metadata.csv \
  --images-dir1 path/to/HAM10000_images_part_1 \
  --images-dir2 path/to/HAM10000_images_part_2 \
  --output-dir path/to/preprocessed_images \
  --hair-removal --contrast-enhancement
```

## Pipeline (`pipeline.py`)

`PreprocessingPipeline` chains steps defined in `config.PreprocessingConfig`:

1. **HairRemovalStep** (if `use_hair_removal=True`) — deep-learning based; uses a
   singleton `HairRemover` shared across all images to avoid reloading the model.
2. **ContrastEnhancer** (if `use_contrast_enhancement=True`) — applies CLAHE or
   similar contrast normalization.
3. **ColorNormalizationStep** (if `use_color_normalization=True`) — Reinhard LAB
   color normalization; requires pre-fitted stats at `color_norm_stats_path`.

Entry point for a single image:

```python
from preprocessing.graphic.pipeline import apply_graphic_preprocessing
out = apply_graphic_preprocessing(
    img_bgr,
    use_hair_removal=True,
    use_contrast_enhancement=False,
    use_color_normalization=False,
    color_norm_stats_path=None,
)
```

## Color Normalization (`color_normalization.py`)

Reinhard (2001) LAB-space per-image normalization. Standardizes color appearance
across images acquired with different dermoscopes and skin tones — analogous to stain
normalization in histopathology.

**Functions:**
- `compute_reference_stats(image_paths)` — compute mean/std per LAB channel over a
  set of training images. Call once offline; save result with `joblib` to
  `res/color_norm_stats.joblib`.
- `reinhard_normalize(img_bgr, reference_mean, reference_std)` — z-score source
  channels, rescale to reference statistics.
- `ColorNormalizationStep` — `ImagePreprocessor`-compatible step that loads stats from
  `color_norm_stats_path` on first use (lazy-loaded).

**Setup (offline, run once before enabling `USE_COLOR_NORMALIZATION`):**
```python
from preprocessing.graphic.color_normalization import compute_reference_stats
import joblib, glob
train_paths = open('res/train_files.txt').read().splitlines()
train_paths = [line.split('\t')[0] for line in train_paths]
stats = compute_reference_stats(train_paths)
joblib.dump(stats, 'res/color_norm_stats.joblib')
```

## Hair Removal (`hair_removal/`)

| File | Role |
|---|---|
| `model.py` | `SEResBlock` custom Keras layer (Squeeze-and-Excitation ResNet block) used in the segmentation network |
| `inference.py` | `HairRemover` — loads weights, runs forward pass, inpaints masked region |
| `config.py` | `HairRemovalConfig` — image size 448×448, weights at `results/chimera/best_weights.h5`, TTA enabled |
| `training/` | Dataset, trainer, and metrics for training the hair removal model |

**Weight path:** `results/chimera/best_weights.h5` (relative to repo root).
The model must be trained or downloaded before `USE_HAIR_REMOVAL=True` will work.

## Directory Layout

```
graphic/
  base/preprocessor.py       # ImagePreprocessor abstract base
  config.py                  # PreprocessingConfig dataclass
  pipeline.py                # PreprocessingPipeline + apply_graphic_preprocessing()
  color_normalization.py     # Reinhard LAB normalization step
  preprocess_dataset.py      # Offline batch preprocessing script
  steps/
    hair_removal.py          # HairRemovalStep (wraps HairRemover as a singleton)
    contrast_enhancer.py     # ContrastEnhancer step
  hair_removal/
    config.py                # HairRemovalConfig
    model.py                 # SEResBlock Keras layer
    inference.py             # HairRemover inference class
    training/                # Training utilities for the hair removal network
```
