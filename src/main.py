"""
Main entry point for skin cancer classification project.
Handles command-line arguments and runs the appropriate pipeline.

All config flags can be overridden from the command line — no editing of
config.py required between experiments.  Example:

    python main.py --cnn-model Xception --use-augmentation
    python main.py --cnn-model ResNet --classifier RandomForest
    python main.py --cnn-model EfficientNet --use-focal-loss --label-smoothing 0.1
    python main.py --batch-size 32 --num-kfolds 3 --no-use-fine-tuning
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Only the module is imported here so that apply_cli_overrides() can mutate
# its globals before any pipeline module is imported (pipeline modules bind
# config values at their own import time).
import config as _config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Skin Cancer Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Data preparation                                                     #
    # ------------------------------------------------------------------ #
    parser.add_argument("--create-splits", action="store_true",
                        help="Create new patient-level dataset splits")
    parser.add_argument("--metadata", type=str,
                        help="Path to HAM10000 metadata CSV")
    parser.add_argument("--images-dir1", type=str,
                        help="First directory containing images")
    parser.add_argument("--images-dir2", type=str,
                        help="Second directory containing images")

    # ------------------------------------------------------------------ #
    # Pipeline selection                                                   #
    # ------------------------------------------------------------------ #
    parser.add_argument("--pipeline",
                        choices=["cnn", "feature-extraction", "both"],
                        default="both",
                        help="Pipeline(s) to run")
    parser.add_argument("--cv", action="store_true",
                        help="Run k-fold cross-validation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (evaluation only)")
    parser.add_argument("--train-segmentation", action="store_true",
                        help="Train the lesion segmentation model")

    # ------------------------------------------------------------------ #
    # Data paths (override the paths derived from config flags)           #
    # ------------------------------------------------------------------ #
    parser.add_argument("--train-files", type=str, default=None,
                        help="Override path to training file list")
    parser.add_argument("--val-files", type=str, default=None,
                        help="Override path to validation file list")
    parser.add_argument("--test-files", type=str, default=None,
                        help="Override path to test file list")

    # ------------------------------------------------------------------ #
    # Model selection                                                      #
    # ------------------------------------------------------------------ #
    parser.add_argument("--cnn-model",
                        choices=["VGG19", "Inception", "ResNet", "Xception", "EfficientNet"],
                        default=None, metavar="BACKBONE",
                        help="CNN backbone (default: config.CNN_MODEL)")
    parser.add_argument("--classifier",
                        choices=["RandomForest", "XGBoost", "AdaBoost", "ExtraTrees", "SVM"],
                        default=None,
                        help="Classical ML classifier (default: config.CLASSICAL_CLASSIFIER_MODEL)")

    # ------------------------------------------------------------------ #
    # Numeric hyperparameters                                              #
    # ------------------------------------------------------------------ #
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--num-epochs",      type=int,   default=None)
    parser.add_argument("--num-kfolds",      type=int,   default=None)
    parser.add_argument("--num-iterations",  type=int,   default=None)
    parser.add_argument("--num-final-models",type=int,   default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--tta-n-steps",     type=int,   default=None)
    parser.add_argument("--mc-dropout-steps",type=int,   default=None)

    # ------------------------------------------------------------------ #
    # Boolean pipeline flags  (--flag / --no-flag)                        #
    # ------------------------------------------------------------------ #
    def _bool(name, dest=None, help_text=""):
        kwargs = dict(action=argparse.BooleanOptionalAction, default=None)
        if dest:
            kwargs["dest"] = dest
        parser.add_argument(f"--{name}", help=help_text, **kwargs)

    _bool("use-fine-tuning",           dest="use_fine_tuning")
    _bool("use-augmentation",          dest="use_data_augmentation",
          help_text="Image-level augmentation during CNN training")
    _bool("use-feature-augmentation",  dest="use_feature_augmentation")
    _bool("use-feature-preprocessing", dest="use_feature_preprocessing")
    _bool("use-graphic-preprocessing", dest="use_graphic_preprocessing",
          help_text="Hair removal / contrast enhancement (offline, see CLAUDE.md)")
    _bool("use-metadata",              dest="use_metadata")
    _bool("use-hair-removal",          dest="use_hair_removal")
    _bool("use-enhanced-contrast",     dest="use_enhanced_contrast")
    _bool("use-color-normalization",   dest="use_color_normalization")
    _bool("use-focal-loss",            dest="use_focal_loss")
    _bool("use-tta",                   dest="use_tta",
          help_text=f"Test-Time Augmentation (default steps: {_config.TTA_N_STEPS})")
    _bool("use-mixup",                 dest="use_mixup")
    _bool("use-mc-dropout",            dest="use_mc_dropout",
          help_text=f"MC-Dropout uncertainty ({_config.MC_DROPOUT_STEPS} passes)")

    return parser


def setup_environment():
    """Configure GPU and set random seeds."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found. Running on CPU.")
    np.random.seed(42)
    tf.random.set_seed(42)


def create_dataset_splits(metadata_path, image_dir_1, image_dir_2, output_dir='./res'):
    """
    Create patient-level train/val/test splits and save them as tab-separated
    text files under *output_dir*.

    Splits at the lesion_id level to prevent the same patient's images from
    appearing in both train and test sets (HAM10000 has ~7,470 unique patients
    but 10,015 images).
    """
    print("Creating dataset splits...")
    os.makedirs(output_dir, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    metadata['image_file'] = metadata['image_id'] + ".jpg"
    metadata['image_path'] = metadata['image_file'].apply(
        lambda x: os.path.join(image_dir_1, x) if os.path.exists(os.path.join(image_dir_1, x))
        else os.path.join(image_dir_2, x)
    )

    missing = metadata[~metadata['image_path'].apply(os.path.exists)]
    if not missing.empty:
        print(f"Warning: {len(missing)} images not found (skipping).")
        metadata = metadata[metadata['image_path'].apply(os.path.exists)]

    class_names = sorted(metadata['dx'].unique())
    class_to_label = {cls: idx for idx, cls in enumerate(class_names)}
    metadata['label'] = metadata['dx'].map(class_to_label)

    # Patient-level split to prevent leakage.
    unique_patients = (
        metadata.groupby('lesion_id')['dx']
        .first()
        .reset_index()
        .rename(columns={'dx': 'primary_dx'})
    )
    train_val_pts, test_pts = train_test_split(
        unique_patients, test_size=0.15, random_state=42,
        stratify=unique_patients['primary_dx']
    )
    train_pts, val_pts = train_test_split(
        train_val_pts, test_size=0.15, random_state=42,
        stratify=train_val_pts['primary_dx']
    )
    train_meta = metadata[metadata['lesion_id'].isin(train_pts['lesion_id'])]
    val_meta   = metadata[metadata['lesion_id'].isin(val_pts['lesion_id'])]
    test_meta  = metadata[metadata['lesion_id'].isin(test_pts['lesion_id'])]

    for split, df in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        df[['image_path', 'label']].to_csv(
            os.path.join(output_dir, f"{split}_files.txt"),
            index=False, header=False, sep='\t'
        )

    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        for cls, idx in class_to_label.items():
            f.write(f"{idx}\t{cls}\n")

    print(f"  Training:   {len(train_meta)} images")
    print(f"  Validation: {len(val_meta)} images")
    print(f"  Test:       {len(test_meta)} images")
    print(f"  Total:      {len(metadata)} images")
    for cls in class_names:
        tr = (train_meta['dx'] == cls).sum()
        va = (val_meta['dx'] == cls).sum()
        te = (test_meta['dx'] == cls).sum()
        print(f"  {cls}: {tr} train / {va} val / {te} test")

    return class_to_label


def get_class_names(class_names_path):
    if not os.path.exists(class_names_path):
        return None
    try:
        names = []
        with open(class_names_path) as f:
            for line in f:
                _, name = line.strip().split('\t')
                names.append(name)
        return names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None


def print_experiment_configuration():
    """Print the active configuration (after any CLI overrides)."""
    import config as cfg
    print("\nExperiment Configuration:")
    print(f"  CNN Model:              {cfg.CNN_MODEL}")
    print(f"  Classical Classifier:   {cfg.CLASSICAL_CLASSIFIER_MODEL}")
    print(f"  Image size:             {cfg.IMG_SIZE[:2]}")
    print(f"  Batch size:             {cfg.BATCH_SIZE}")
    print(f"  Epochs:                 {cfg.NUM_EPOCHS}")
    print(f"  K-folds × iterations:   {cfg.NUM_KFOLDS} × {cfg.NUM_ITERATIONS}")
    print(f"  Fine-tuning:            {cfg.USE_FINE_TUNING}")
    print(f"  Data augmentation:      {cfg.USE_DATA_AUGMENTATION}")
    print(f"  Graphic preprocessing:  {cfg.USE_GRAPHIC_PREPROCESSING}")
    print(f"  Focal loss:             {cfg.USE_FOCAL_LOSS}  label_smoothing={cfg.LABEL_SMOOTHING}")
    print(f"  TTA:                    {cfg.USE_TTA}  (steps={cfg.TTA_N_STEPS})")
    print(f"  MC Dropout:             {cfg.USE_MC_DROPOUT}  (steps={cfg.MC_DROPOUT_STEPS})")
    print(f"  Mixup:                  {cfg.USE_MIXUP}")
    print(f"  Metadata:               {cfg.USE_METADATA}")
    print(f"  Feature preprocessing:  {cfg.USE_FEATURE_PREPROCESSING}")
    print(f"  Feature augmentation:   {cfg.USE_FEATURE_AUGMENTATION}")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Apply CLI overrides to config module globals BEFORE any pipeline module
    # is imported, so pipeline-level `from config import X` binds the
    # overridden values.
    import config as cfg
    cfg.apply_cli_overrides(args)
    cfg.validate_config()

    # NOW it is safe to import pipelines (after overrides are applied).
    from pipelines.cnn.runner import run_cnn_classifier_pipeline
    from pipelines.feature_extraction.runner import run_feature_extraction_pipeline

    setup_environment()

    if args.create_splits:
        if not (args.metadata and args.images_dir1 and args.images_dir2):
            parser.error("--create-splits requires --metadata, --images-dir1, --images-dir2")
        create_dataset_splits(
            metadata_path=args.metadata,
            image_dir_1=args.images_dir1,
            image_dir_2=args.images_dir2,
        )

    # Resolve data paths: explicit CLI paths override config defaults.
    train_files = args.train_files or cfg.TRAIN_FILES_PATH
    val_files   = args.val_files   or cfg.VAL_FILES_PATH
    test_files  = args.test_files  or cfg.TEST_FILES_PATH

    class_names_path = os.path.join(os.path.dirname(train_files), "class_names.txt")
    class_names = get_class_names(class_names_path)
    if class_names:
        print("\nClass Names:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")

    print_experiment_configuration()

    results = {}

    if args.pipeline in ["cnn", "both"]:
        print("\n" + "=" * 50)
        print("Running CNN Classifier Pipeline")
        print("=" * 50)
        results["cnn"] = run_cnn_classifier_pipeline(
            train_files_path=train_files,
            val_files_path=val_files,
            test_files_path=test_files,
            run_kfold=args.cv,
            class_names=class_names,
            skip_training=args.skip_train,
        )
        gc.collect()

    if args.pipeline in ["feature-extraction", "both"]:
        print("\n" + "=" * 50)
        print("Running Feature Extraction + Classical ML Pipeline")
        print("=" * 50)
        results["feature_extraction"] = run_feature_extraction_pipeline(
            train_files_path=train_files,
            val_files_path=val_files,
            test_files_path=test_files,
            use_kfold=args.cv,
            tune_hyperparams=False,
            class_names=class_names,
        )

    print("\n" + "=" * 50)
    print("Experiment completed successfully!")
    print("=" * 50)
    return results


if __name__ == "__main__":
    main()
