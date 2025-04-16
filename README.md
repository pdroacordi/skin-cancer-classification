# Skin Cancer Classification

This project implements and compares two approaches for skin cancer lesion classification using the HAM10000 dataset:
1. End-to-end CNN classification
2. CNN feature extraction + classical machine learning classifiers

## Project Structure

```
src/
  ├── config.py                    # Central configuration
  ├── utils/
  │   ├── data_loaders.py          # Image loading utilities 
  │   ├── preprocessing.py         # Image preprocessing (graphic)
  │   └── augmentation.py          # Data augmentation strategies
  ├── models/
  │   ├── cnn_models.py            # CNN model definitions and loading
  │   └── classical_models.py      # Classical ML model definitions
  ├── pipelines/
  │   ├── cnn_classifier.py        # End-to-end CNN classification
  │   └── feature_extraction.py    # CNN feature extractor + classical ML
  └── main.py                      # Entry point
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

The central configuration can be modified in `src/config.py`. Key parameters include:

- `CNN_MODEL`: CNN architecture to use ('VGG19', 'Inception', 'ResNet', 'Xception')
- `CLASSICAL_CLASSIFIER_MODEL`: Classical ML model ('RandomForest', 'XGBoost', 'AdaBoost', 'ExtraTrees', 'SVM')
- `BATCH_SIZE`: Batch size for training
- `NUM_EPOCHS`: Number of epochs for training
- `USE_FINE_TUNING`: Whether to fine-tune the CNN
- `USE_GRAPHIC_PREPROCESSING`: Whether to apply graphics preprocessing
- `USE_DATA_AUGMENTATION`: Whether to apply data augmentation
- `NUM_KFOLDS`: Number of folds for cross-validation

## Results

Results are saved in the `results/` directory, with subdirectories for each run:

- `models/`: Saved models
- `features/`: Extracted features (for feature extraction pipeline)
- `plots/`: Visualizations including confusion matrices
- Evaluation metrics in text files

## Key Features

1. **Memory-Efficient Data Processing**
   - Batch processing of images to prevent OOM errors
   - Efficient data generators

2. **Advanced Image Preprocessing**
   - Hair removal
   - Contrast enhancement
   - GVF-based segmentation

3. **Robust Data Augmentation**
   - Multiple augmentation strategies
   - Class balancing

4. **Two Complete Pipelines**
   - End-to-end CNN classification
   - CNN feature extraction + classical ML

5. **Comprehensive Evaluation**
   - K-fold cross-validation
   - Detailed metrics and visualizations