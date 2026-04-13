# Skin Cancer Classification

This project implements and compares two approaches for skin cancer lesion classification using the HAM10000 dataset:
1. End-to-end CNN classification
2. CNN feature extraction + classical machine learning classifiers

## Project Structure

```
src/
  ├── config.py                          # Central configuration (all flags)
  ├── main.py                            # CLI entry point
  ├── models/
  │   ├── cnn_models.py                  # CNN loading, focal loss, Grad-CAM
  │   └── classical_models.py            # Classical ML classifiers
  ├── pipelines/
  │   ├── cnn_classifier.py              # End-to-end CNN training + evaluation
  │   └── feature_extraction.py          # CNN feature extractor + classical ML
  ├── preprocessing/
  │   ├── graphic/
  │   │   ├── pipeline.py                # Image preprocessing orchestrator
  │   │   ├── color_normalization.py     # Reinhard LAB color normalization
  │   │   └── steps/                     # Hair removal, contrast enhancement
  │   └── feature/
  │       └── algorithm/                 # Per-algorithm preprocessing pipelines
  ├── utils/
  │   ├── data_loaders.py                # Image loading + Mixup augmentation
  │   ├── metadata_extractor.py          # Patient metadata features
  │   ├── calibration.py                 # Expected Calibration Error (ECE)
  │   └── fold_utils.py                  # K-fold result persistence
  └── analysis/
      └── aggregate_results.py           # Consolidate results across experiments
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- OpenCV
- albumentations
- numpy
- pandas
- matplotlib
- seaborn
- joblib
- shap

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd skin-cancer-classification

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the HAM10000 dataset, which contains 10,000 dermatoscopic images of pigmented skin lesions across 7 diagnostic categories:
- Actinic keratoses and intraepithelial carcinoma (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

### Dataset Preparation

To prepare the dataset:

1. Download the HAM10000 dataset from [ISIC Archive](https://challenge.isic-archive.com/data/) or [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
2. Extract the images and metadata
3. Run the dataset splitting script:

```bash
python src/main.py --create-splits \
  --metadata path/to/HAM10000_metadata.csv \
  --images-dir1 path/to/HAM10000_images_part_1 \
  --images-dir2 path/to/HAM10000_images_part_2
```

This will create train/validation/test splits in the `res/` directory.

## Usage

### Basic Usage

Run both pipelines with default settings:

```bash
python src/main.py
```

### Running Specific Pipeline

Run only the CNN classifier pipeline:

```bash
python src/main.py --pipeline cnn
```

Run only the feature extraction + classical ML pipeline:

```bash
python src/main.py --pipeline feature-extraction
```

### Cross-Validation

Run with k-fold cross-validation:

```bash
python src/main.py --cv
```

### Custom Data Paths

Specify custom paths to data files:

```bash
python src/main.py --train-files path/to/train_files.txt \
  --val-files path/to/val_files.txt \
  --test-files path/to/test_files.txt
```

## Configuration

All experiment toggles live in `src/config.py`:

| Key | Options / Notes |
|---|---|
| `CNN_MODEL` | `'VGG19'`, `'Inception'`, `'ResNet'`, `'Xception'`, `'EfficientNet'` |
| `CLASSICAL_CLASSIFIER_MODEL` | `'RandomForest'`, `'XGBoost'`, `'AdaBoost'`, `'ExtraTrees'`, `'SVM'` |
| `BATCH_SIZE` / `NUM_EPOCHS` | Training hyperparameters |
| `USE_FINE_TUNING` | Unfreeze CNN layers for fine-tuning |
| `USE_GRAPHIC_PREPROCESSING` | Redirect data paths to `res/preprocessed_*` |
| `USE_DATA_AUGMENTATION` | Image-level augmentation during CNN training |
| `USE_FEATURE_AUGMENTATION` | Feature-space augmentation for classical ML |
| `USE_FEATURE_PREPROCESSING` | Per-algorithm feature preprocessing pipeline |
| `USE_METADATA` | Append patient metadata (age, location, sex) to features |
| `USE_HAIR_REMOVAL` | Deep-learning hair removal (requires `USE_GRAPHIC_PREPROCESSING`) |
| `USE_COLOR_NORMALIZATION` | Reinhard LAB color normalization across dermoscopes |
| `USE_FOCAL_LOSS` | Focal loss instead of cross-entropy (addresses class imbalance) |
| `LABEL_SMOOTHING` | Label smoothing value (0.1 recommended); `0.0` disables |
| `USE_MIXUP` | Mixup augmentation during CNN training (Zhang et al., 2018) |
| `USE_TTA` | Test-Time Augmentation: average N augmented predictions |
| `TTA_N_STEPS` | Number of augmented copies for TTA (default 8) |
| `USE_MC_DROPOUT` | Monte Carlo Dropout uncertainty quantification |
| `MC_DROPOUT_STEPS` | Stochastic passes per sample for MC Dropout (default 50) |
| `NUM_KFOLDS` / `NUM_ITERATIONS` | Cross-validation folds and repetitions |

## Results

Results are saved in the `results/` directory, with subdirectories for each run:

- `models/`: Saved models
- `features/`: Extracted features (for feature extraction pipeline)
- `plots/`: Visualizations including confusion matrices
- Evaluation metrics in text files

## Key Features

1. **Patient-Level Data Splits**
   - Splits by `lesion_id` so no patient's images appear in both train and test
   - Prevents ~3-5 point metric inflation from cross-patient leakage

2. **Multiple CNN Backbones**
   - VGG19, InceptionV3, ResNet50, Xception — each at its canonical resolution
   - EfficientNetB4 (380×380) for state-of-the-art performance

3. **Class Imbalance Handling**
   - Class-weighted loss at CNN training time
   - Optional focal loss (Lin et al., 2017) to focus on hard minority examples
   - Optional label smoothing to regularize overconfident predictions
   - Balanced class weights in all tree-based classifiers

4. **Advanced Image Preprocessing**
   - Deep-learning hair removal (SEResNet segmentation model)
   - CLAHE contrast enhancement
   - Reinhard LAB color normalization — standardizes cross-device color bias

5. **Advanced Data Augmentation**
   - Albumentations-based geometric and color augmentation
   - Mixup training augmentation (Zhang et al., 2018, ICLR)

6. **Two Complete Pipelines**
   - End-to-end CNN classification
   - CNN feature extraction + classical ML (RandomForest, XGBoost, ExtraTrees, AdaBoost, SVM)

7. **Comprehensive Evaluation**
   - K-fold cross-validation with per-fold and aggregated metrics
   - Macro AUC-ROC, Expected Calibration Error (ECE), macro-F1
   - Test-Time Augmentation (TTA) for improved inference
   - Monte Carlo Dropout uncertainty quantification
   - Grad-CAM visualizations for CNN interpretability
   - TreeSHAP feature importance for classical models