"""
Feature extraction + classical ML pipeline runner.

Orchestration only.  Composes: feature extraction, optional preprocessing,
classifier training, evaluation, SHAP analysis.  All persistence is
delegated to save_run_results().

Entry point used by main.py:
    run_feature_extraction_pipeline(train_files_path, val_files_path,
                                    test_files_path, use_kfold,
                                    tune_hyperparams, class_names)
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.backend import clear_session

from core.run_context import RunContext
from core.types import EvalResult, FoldResult, RunArtifact
from models.classical_models import create_ml_pipeline, save_model
from models.cnn_models import get_feature_extractor_from_cnn
from models.dynamic_ensemble import DynamicEnsembleSelector
from pipelines.feature_extraction.stages.evaluate import evaluate_classifier
from pipelines.feature_extraction.stages.extract import load_or_extract_features
from pipelines.feature_extraction.stages.train import train_classifier
from preprocessing.data.augmentation import AugmentationFactory
from preprocessing.feature.pipeline import apply_feature_preprocessing
from utils.data_loaders import load_paths_labels
from utils.fold_utils import save_fold_results
from utils.gpu_utils import setup_gpu_memory
from utils.metadata_extractor import (
    MetadataFeatureExtractor,
    extract_metadata_for_paths,
)
from utils.metrics import aggregate_eval_results
from utils.persistence import save_run_results, save_multiple_model_stats
from utils.result_naming import feature_extraction_experiment_dir


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #

def run_feature_extraction_pipeline(
    train_files_path: str,
    val_files_path: str,
    test_files_path: str,
    use_kfold: bool = True,
    tune_hyperparams: bool = True,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run the complete feature-extraction + classical ML pipeline.

    Flow:
    1. Load feature extractor (from trained CNN or ImageNet weights).
    2. Fit metadata extractor on train+val (prevents test-set leakage).
    3. K-fold CV: extract per-fold features → preprocess → train → evaluate.
    4. Train NUM_FINAL_MODELS on all train+val data → evaluate on test.
    5. SHAP analysis on best final model.
    6. Write all results via save_run_results().

    Args:
        train_files_path: Path to tab-separated train file list.
        val_files_path:   Path to tab-separated val file list.
        test_files_path:  Path to tab-separated test file list.
        use_kfold:        Whether to run K-fold cross-validation.
        tune_hyperparams: Whether to use GridSearchCV for each fold.
        class_names:      Optional class label strings.

    Returns:
        Dict with keys: fold_results, final_metrics, result_dir.
    """
    import config  # lazy import ensures apply_cli_overrides() has already run

    setup_gpu_memory()

    train_paths, train_labels = load_paths_labels(train_files_path)
    val_paths,   val_labels   = load_paths_labels(val_files_path)
    test_paths,  test_labels  = load_paths_labels(test_files_path)

    result_dir = Path(feature_extraction_experiment_dir())
    result_dir.mkdir(parents=True, exist_ok=True)

    # When DES is active, results live under dynamic_ensemble/ so they are
    # clearly distinguishable from single-classifier runs in aggregate_results.
    subdir_name = "dynamic_ensemble" if config.USE_DYNAMIC_ENSEMBLE else config.CLASSICAL_CLASSIFIER_MODEL.lower()
    classifier_dir = result_dir / subdir_name
    classifier_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext.create(
        pipeline="feature_extraction",
        result_dir=str(classifier_dir),
        config_snapshot=_build_config_snapshot(config),
    )
    print(f"Results will be saved to: {classifier_dir}")
    print(f"Run ID: {ctx.run_id}")

    # ---- Load or create feature extractor ---- #
    extractor_path = str(result_dir / "models" / f"{config.CNN_MODEL.lower()}_feature_extractor.h5")
    (result_dir / "models").mkdir(parents=True, exist_ok=True)
    feature_extractor, _ = get_feature_extractor_from_cnn(extractor_path)

    # ---- Metadata extractor (fit on train+val only) ---- #
    metadata_extractor, metadata_df = _setup_metadata_extractor(
        train_paths, val_paths, config
    )

    # ---- Augmentation pipelines (for K-fold feature extraction) ---- #
    # Feature augmentation is disabled during K-fold to prevent the
    # augmented-duplicate data-leakage bug: if augmented copies of the
    # same image appear in both train and val splits, the classifier
    # memorises the duplicates rather than learning generalizable features.
    aug_pipelines_for_kfold: Optional[List] = None
    aug_pipelines_for_final = (
        AugmentationFactory.get_feature_extraction_augmentation()
        if config.USE_FEATURE_AUGMENTATION
        else None
    )

    img_size = tuple(config.IMG_SIZE[:2])

    # ---- K-fold cross-validation ---- #
    fold_results: List[FoldResult] = []

    if use_kfold:
        all_paths  = np.concatenate([train_paths, val_paths])
        all_labels = np.concatenate([train_labels, val_labels])

        fold_results = _run_kfold_cv(
            all_paths=all_paths,
            all_labels=all_labels,
            ctx=ctx,
            config=config,
            feature_extractor=feature_extractor,
            metadata_extractor=metadata_extractor,
            metadata_df=metadata_df,
            img_size=img_size,
            aug_pipelines=aug_pipelines_for_kfold,
            tune_hyperparams=tune_hyperparams,
            class_names=class_names,
        )

        save_fold_results(
            fold_results=[fr.to_legacy_dict() for fr in fold_results],
            result_dir=str(ctx.result_dir),
            classifier_name=config.CLASSICAL_CLASSIFIER_MODEL,
        )

    # ---- Train final models ---- #
    all_paths  = np.concatenate([train_paths, val_paths])
    all_labels = np.concatenate([train_labels, val_labels])

    model_evals, final_models_dir = _train_and_evaluate_final_models(
        all_paths=all_paths,
        all_labels=all_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        ctx=ctx,
        config=config,
        feature_extractor=feature_extractor,
        metadata_extractor=metadata_extractor,
        metadata_df=metadata_df,
        img_size=img_size,
        aug_pipelines=aug_pipelines_for_final,
        tune_hyperparams=tune_hyperparams,
        class_names=class_names,
    )

    final_metrics = aggregate_eval_results(model_evals)

    if len(model_evals) > 1:
        save_multiple_model_stats(model_evals, final_models_dir, class_names)

    artifacts = _list_artifacts(ctx, final_models_dir)

    save_run_results(
        ctx=ctx,
        fold_results=fold_results,
        final_metrics=final_metrics,
        artifacts=artifacts,
    )

    clear_session()
    gc.collect()

    return {
        "fold_results":  fold_results,
        "final_metrics": final_metrics,
        "result_dir":    str(ctx.result_dir),
    }


# ------------------------------------------------------------------ #
#  K-fold cross-validation                                            #
# ------------------------------------------------------------------ #

def _run_kfold_cv(
    all_paths: np.ndarray,
    all_labels: np.ndarray,
    ctx: RunContext,
    config,
    feature_extractor,
    metadata_extractor: Optional[MetadataFeatureExtractor],
    metadata_df: Optional[pd.DataFrame],
    img_size: Tuple,
    aug_pipelines: Optional[List],
    tune_hyperparams: bool,
    class_names: Optional[List[str]],
) -> List[FoldResult]:
    """Run NUM_ITERATIONS × NUM_KFOLDS folds and return all FoldResult objects."""
    fold_results: List[FoldResult] = []
    features_dir = ctx.result_dir / "features_by_fold"
    features_dir.mkdir(parents=True, exist_ok=True)

    stratify_labels = (
        np.argmax(all_labels, axis=1)
        if all_labels.ndim > 1 and all_labels.shape[1] > 1
        else all_labels
    )

    for iteration in range(config.NUM_ITERATIONS):
        print(f"\n{'=' * 50}")
        print(f"Iteration {iteration + 1}/{config.NUM_ITERATIONS}")
        print(f"{'=' * 50}")

        iter_dir = ctx.result_dir / f"iteration_{iteration + 1}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_features_dir = features_dir / f"iteration_{iteration + 1}"
        iter_features_dir.mkdir(parents=True, exist_ok=True)

        skf = StratifiedKFold(
            n_splits=config.NUM_KFOLDS, shuffle=True, random_state=42 + iteration
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(all_paths, stratify_labels), start=1
        ):
            print(f"\n{'=' * 40}")
            print(f"Iteration {iteration + 1}, Fold {fold}/{config.NUM_KFOLDS}")
            print(f"{'=' * 40}")

            fold_dir = iter_dir / f"fold_{fold}"
            (fold_dir / "models").mkdir(parents=True, exist_ok=True)

            train_cache = str(iter_features_dir / f"train_fold{fold}.npz")
            val_cache   = str(iter_features_dir / f"val_fold{fold}.npz")

            # Metadata for this fold's training split
            train_meta = _get_metadata_features(
                all_paths[train_idx], metadata_df, metadata_extractor
            )
            val_meta = _get_metadata_features(
                all_paths[val_idx], metadata_df, metadata_extractor
            )

            try:
                train_feats, train_labs = load_or_extract_features(
                    feature_extractor=feature_extractor,
                    paths=all_paths[train_idx],
                    labels=all_labels[train_idx],
                    img_size=img_size,
                    model_name=config.CNN_MODEL,
                    batch_size=config.BATCH_SIZE,
                    features_save_path=train_cache,
                    apply_augmentation=False,  # augmentation disabled in K-fold
                    augmentation_pipelines=None,
                    metadata_features=train_meta,
                )

                val_feats, val_labs = load_or_extract_features(
                    feature_extractor=feature_extractor,
                    paths=all_paths[val_idx],
                    labels=all_labels[val_idx],
                    img_size=img_size,
                    model_name=config.CNN_MODEL,
                    batch_size=config.BATCH_SIZE,
                    features_save_path=val_cache,
                    apply_augmentation=False,
                    augmentation_pipelines=None,
                    metadata_features=val_meta,
                )

                # Feature preprocessing (optional per-algorithm pipeline).
                # Skipped for DES: raw features preserve pool diversity.
                if config.USE_FEATURE_PREPROCESSING and not config.USE_DYNAMIC_ENSEMBLE:
                    pipe_save_path = str(fold_dir / "models" / "feat_preprocessing.joblib")
                    train_feats, train_labs, feat_pipe = apply_feature_preprocessing(
                        features=train_feats,
                        labels=train_labs,
                        algorithm=config.CLASSICAL_CLASSIFIER_MODEL,
                        training=True,
                        save_path=pipe_save_path,
                    )
                    val_feats, val_labs = feat_pipe.transform(val_feats, val_labs)

                if config.USE_DYNAMIC_ENSEMBLE:
                    model, eval_result = _train_des_ensemble(
                        train_feats=train_feats,
                        train_labs=train_labs,
                        val_feats=val_feats,
                        val_labs=val_labs,
                        config=config,
                    )
                    model_save_path = str(
                        fold_dir / "models" / f"des_{config.DES_ALGORITHM}_fold{fold}.joblib"
                    )
                    model.save(model_save_path)
                else:
                    model, eval_result = train_classifier(
                        train_features=train_feats,
                        train_labels=train_labs,
                        val_features=val_feats,
                        val_labels=val_labs,
                        classifier_name=config.CLASSICAL_CLASSIFIER_MODEL,
                        tune_hyperparams=tune_hyperparams,
                    )
                    model_save_path = str(
                        fold_dir / "models" /
                        f"{config.CLASSICAL_CLASSIFIER_MODEL.lower()}_fold{fold}.joblib"
                    )
                    save_model(model, model_save_path)

                fold_results.append(
                    FoldResult(
                        iteration=iteration + 1,
                        fold=fold,
                        eval_result=eval_result,
                        model_path=model_save_path,
                    )
                )

            except Exception as exc:
                print(f"Error in iteration {iteration + 1}, fold {fold}: {exc}")
                import traceback
                traceback.print_exc()

            finally:
                gc.collect()

    return fold_results


# ------------------------------------------------------------------ #
#  Final model training                                               #
# ------------------------------------------------------------------ #

def _train_and_evaluate_final_models(
    all_paths: np.ndarray,
    all_labels: np.ndarray,
    test_paths: np.ndarray,
    test_labels: np.ndarray,
    ctx: RunContext,
    config,
    feature_extractor,
    metadata_extractor: Optional[MetadataFeatureExtractor],
    metadata_df: Optional[pd.DataFrame],
    img_size: Tuple,
    aug_pipelines: Optional[List],
    tune_hyperparams: bool,
    class_names: Optional[List[str]],
) -> Tuple[List[EvalResult], Path]:
    """Train NUM_FINAL_MODELS classifiers on all data and evaluate on test."""
    print(f"\n{'=' * 60}")
    print(f"Training {config.NUM_FINAL_MODELS} final {config.CLASSICAL_CLASSIFIER_MODEL} model(s)")
    print(f"{'=' * 60}")

    final_models_dir = ctx.result_dir / "final_models"
    final_models_dir.mkdir(parents=True, exist_ok=True)

    # Extract training features once; reused for all NUM_FINAL_MODELS
    all_meta   = _get_metadata_features(all_paths, metadata_df, metadata_extractor)
    test_meta  = _get_metadata_features(test_paths, metadata_df, metadata_extractor)

    train_feats, train_labs = load_or_extract_features(
        feature_extractor=feature_extractor,
        paths=all_paths,
        labels=all_labels,
        img_size=img_size,
        model_name=config.CNN_MODEL,
        batch_size=config.BATCH_SIZE,
        features_save_path=str(ctx.result_dir / "features" / "all_features.npz"),
        apply_augmentation=config.USE_FEATURE_AUGMENTATION,
        augmentation_pipelines=aug_pipelines,
        metadata_features=all_meta,
    )

    test_feats, test_labs = load_or_extract_features(
        feature_extractor=feature_extractor,
        paths=test_paths,
        labels=test_labels,
        img_size=img_size,
        model_name=config.CNN_MODEL,
        batch_size=config.BATCH_SIZE,
        features_save_path=str(ctx.result_dir / "features" / "test_features.npz"),
        apply_augmentation=False,
        augmentation_pipelines=None,
        metadata_features=test_meta,
    )

    # Feature preprocessing for final models.
    # Skipped when USE_DYNAMIC_ENSEMBLE=True: the pool contains classifiers
    # with different optimal preprocessing requirements, and applying a single
    # algorithm-specific pipeline (based on CLASSICAL_CLASSIFIER_MODEL) could
    # hurt pool diversity. Raw CNN features work well for all tree-based classifiers.
    if config.USE_FEATURE_PREPROCESSING and not config.USE_DYNAMIC_ENSEMBLE:
        final_pipe_path = str(final_models_dir / "final_feat_preprocessing.joblib")
        train_feats, train_labs, feat_pipe = apply_feature_preprocessing(
            features=train_feats,
            labels=train_labs,
            algorithm=config.CLASSICAL_CLASSIFIER_MODEL,
            training=True,
            save_path=final_pipe_path,
        )
        test_feats, test_labs = feat_pipe.transform(test_feats, test_labs)

    # ---- DES branch: train pool + calibrate DES + evaluate ---- #
    if config.USE_DYNAMIC_ENSEMBLE:
        return _train_and_evaluate_des_final(
            train_feats=train_feats,
            train_labs=train_labs,
            test_feats=test_feats,
            test_labs=test_labs,
            config=config,
            class_names=class_names,
            final_models_dir=final_models_dir,
        ), final_models_dir

    # ---- Single-classifier branch (original path) ---- #
    model_evals: List[EvalResult] = []
    best_model = None

    for model_idx in range(1, config.NUM_FINAL_MODELS + 1):
        print(f"\n--- Final model {model_idx}/{config.NUM_FINAL_MODELS} ---")

        model_path = str(
            final_models_dir / f"{config.CLASSICAL_CLASSIFIER_MODEL.lower()}_model{model_idx}.joblib"
        )

        try:
            model, _ = train_classifier(
                train_features=train_feats,
                train_labels=train_labs,
                val_features=test_feats,   # val is test for final models
                val_labels=test_labs,
                classifier_name=config.CLASSICAL_CLASSIFIER_MODEL,
                tune_hyperparams=tune_hyperparams,
            )
            save_model(model, model_path)

            _, _, _, eval_result = evaluate_classifier(
                model=model,
                test_features=test_feats,
                test_labels=test_labs,
                class_names=class_names,
            )
            model_evals.append(eval_result)

            # Track best model for SHAP (highest macro-F1)
            if best_model is None or eval_result.macro_f1 > model_evals[0].macro_f1:
                best_model = model

            print(f"  Test — acc={eval_result.accuracy:.4f}  "
                  f"macro-F1={eval_result.macro_f1:.4f}  "
                  f"AUC={eval_result.macro_auc_roc:.4f}")

        except Exception as exc:
            print(f"Error training final model {model_idx}: {exc}")
            import traceback
            traceback.print_exc()

        finally:
            gc.collect()

    # SHAP analysis on the best final model
    if best_model is not None:
        _compute_and_save_shap(best_model, test_feats, final_models_dir, config)

    return model_evals, final_models_dir


# ------------------------------------------------------------------ #
#  SHAP analysis                                                      #
# ------------------------------------------------------------------ #

def _compute_and_save_shap(
    model,
    test_features: np.ndarray,
    output_dir: Path,
    config,
) -> None:
    """
    Compute TreeSHAP values for tree-based classifiers and save to .npy.

    Skipped when:
    - USE_DYNAMIC_ENSEMBLE=True (DES is an ensemble; per-base SHAP is a separate analysis)
    - Classifier is SVM or HistGradientBoosting (not supported by shap.TreeExplainer)
    """
    if config.USE_DYNAMIC_ENSEMBLE:
        return  # DES: no single tree model to explain

    tree_based = {"RandomForest", "XGBoost", "LightGBM", "ExtraTrees"}
    if config.CLASSICAL_CLASSIFIER_MODEL not in tree_based:
        return

    try:
        import shap
        print("\nComputing SHAP values...")
        # Use a background sample to keep computation tractable
        n_background = min(100, len(test_features))
        background = shap.sample(test_features, n_background)
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(test_features)
        np.save(str(output_dir / "shap_values.npy"), shap_values)
        print(f"SHAP values saved to {output_dir / 'shap_values.npy'}")
    except Exception as exc:
        print(f"SHAP computation skipped: {exc}")


# ------------------------------------------------------------------ #
#  Private helpers                                                     #
# ------------------------------------------------------------------ #

def _setup_metadata_extractor(
    train_paths: np.ndarray,
    val_paths: np.ndarray,
    config,
) -> Tuple[Optional[MetadataFeatureExtractor], Optional[pd.DataFrame]]:
    """
    Load or fit the metadata extractor.

    The extractor is fitted ONLY on train+val image IDs to prevent the
    StandardScaler (age normalization) from seeing test-set statistics.
    """
    if not config.USE_METADATA:
        return None, None

    metadata_df = pd.read_csv(config.METADATA_PATH)

    extractor_path = os.path.join(
        config.RESULTS_DIR, "metadata_extractor", "metadata_extractor.joblib"
    )

    if os.path.exists(extractor_path):
        print("Loading existing metadata extractor...")
        return MetadataFeatureExtractor.load(extractor_path), metadata_df

    print("Fitting metadata extractor on train+val images...")
    train_val_ids = {
        os.path.splitext(os.path.basename(p))[0]
        for p in list(train_paths) + list(val_paths)
    }
    train_val_meta = metadata_df[metadata_df["image_id"].isin(train_val_ids)]

    extractor = MetadataFeatureExtractor()
    extractor.fit(train_val_meta)

    os.makedirs(os.path.dirname(extractor_path), exist_ok=True)
    extractor.save(extractor_path)

    return extractor, metadata_df


def _get_metadata_features(
    paths: np.ndarray,
    metadata_df: Optional[pd.DataFrame],
    metadata_extractor: Optional[MetadataFeatureExtractor],
) -> Optional[np.ndarray]:
    """Return metadata feature matrix for paths, or None when not configured."""
    if metadata_df is None or metadata_extractor is None:
        return None
    return extract_metadata_for_paths(
        image_paths=paths,
        metadata_df=metadata_df,
        metadata_extractor=metadata_extractor,
    )


def _build_config_snapshot(config) -> dict:
    """Capture all active experiment flags as a plain dict for metadata.json."""
    return {
        "cnn_model":                  config.CNN_MODEL,
        "classifier":                 "dynamic_ensemble" if config.USE_DYNAMIC_ENSEMBLE else config.CLASSICAL_CLASSIFIER_MODEL,
        "batch_size":                 config.BATCH_SIZE,
        "num_kfolds":                 config.NUM_KFOLDS,
        "num_iterations":             config.NUM_ITERATIONS,
        "num_final_models":           config.NUM_FINAL_MODELS,
        "use_fine_tuning":            config.USE_FINE_TUNING,
        "use_data_augmentation":      config.USE_DATA_AUGMENTATION,
        "use_feature_augmentation":   config.USE_FEATURE_AUGMENTATION,
        "use_feature_preprocessing":  config.USE_FEATURE_PREPROCESSING,
        "use_graphic_preprocessing":  config.USE_GRAPHIC_PREPROCESSING,
        "use_hair_removal":           config.USE_HAIR_REMOVAL,
        "use_enhanced_contrast":      config.USE_ENHANCED_CONTRAST,
        "use_color_normalization":    config.USE_COLOR_NORMALIZATION,
        "use_metadata":               config.USE_METADATA,
        "use_dynamic_ensemble":       config.USE_DYNAMIC_ENSEMBLE,
        "des_algorithm":              config.DES_ALGORITHM if config.USE_DYNAMIC_ENSEMBLE else None,
        "des_pool_classifiers":       config.DES_POOL_CLASSIFIERS if config.USE_DYNAMIC_ENSEMBLE else None,
    }


# ------------------------------------------------------------------ #
#  DES helpers                                                         #
# ------------------------------------------------------------------ #

def _build_pool_classifiers(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    pool_names: List[str],
) -> List[Tuple[str, object]]:
    """
    Train each classifier in *pool_names* on (X_tr, y_tr).

    Returns a list of (name, fitted_pipeline) tuples.  Each pipeline
    contains a single 'classifier' step (from create_ml_pipeline).
    class_weight='balanced' is embedded in every classifier's params,
    so no separate sample_weight handling is needed here.
    """
    pool = []
    for name in pool_names:
        print(f"  Training pool classifier: {name}...")
        clf = create_ml_pipeline(classifier_name=name)
        clf.fit(X_tr.astype(np.float32), y_tr)
        pool.append((name, clf))
    return pool


def _train_des_ensemble(
    train_feats: np.ndarray,
    train_labs: np.ndarray,
    val_feats: np.ndarray,
    val_labs: np.ndarray,
    config,
) -> Tuple[DynamicEnsembleSelector, EvalResult]:
    """
    Train DES pool on a stratified subset of (train_feats, train_labs) and
    calibrate on the remaining DSEL split.  Evaluate on (val_feats, val_labs).

    The DSEL split (DES_DSEL_FRACTION of the fold training data) is kept out
    of pool-classifier training so DES competence estimates are unbiased.

    Returns (fitted DynamicEnsembleSelector, EvalResult on val set).
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from pipelines.feature_extraction.stages.evaluate import evaluate_classifier

    print(f"  Building DES pool ({config.DES_ALGORITHM}, "
          f"k={config.DES_K_NEIGHBORS}, "
          f"dsel_frac={config.DES_DSEL_FRACTION})...")

    # Split training data into pool-training (X_tr) and DSEL (X_dsel).
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.DES_DSEL_FRACTION,
        random_state=42,
    )
    tr_idx, dsel_idx = next(splitter.split(train_feats, train_labs))
    X_tr,   y_tr   = train_feats[tr_idx],   train_labs[tr_idx]
    X_dsel, y_dsel = train_feats[dsel_idx], train_labs[dsel_idx]

    print(f"  Pool training: {len(X_tr)} samples | DSEL: {len(X_dsel)} samples")

    pool = _build_pool_classifiers(X_tr, y_tr, config.DES_POOL_CLASSIFIERS)

    des = DynamicEnsembleSelector(
        algorithm=config.DES_ALGORITHM,
        k_neighbors=config.DES_K_NEIGHBORS,
    )
    des.fit(pool, X_dsel, y_dsel)

    _, _, _, eval_result = evaluate_classifier(
        model=des,
        test_features=val_feats,
        test_labels=val_labs,
    )

    print(f"  Val — acc={eval_result.accuracy:.4f}  "
          f"macro-F1={eval_result.macro_f1:.4f}  "
          f"AUC={eval_result.macro_auc_roc:.4f}")

    return des, eval_result


def _train_and_evaluate_des_final(
    train_feats: np.ndarray,
    train_labs: np.ndarray,
    test_feats: np.ndarray,
    test_labs: np.ndarray,
    config,
    class_names: Optional[List[str]],
    final_models_dir: Path,
) -> List[EvalResult]:
    """
    Train NUM_FINAL_MODELS DES ensembles on all training data and evaluate
    on the test set.  Each model uses a fresh stratified DSEL split.

    Multiple models are trained identically to the single-classifier path so
    result variance can be measured across seeds.  Each DSEL split uses a
    different random_state (42 + model_idx) to obtain distinct calibration sets.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from pipelines.feature_extraction.stages.evaluate import evaluate_classifier

    print(f"\n{'=' * 60}")
    print(f"Training {config.NUM_FINAL_MODELS} final DES "
          f"({config.DES_ALGORITHM}) ensemble(s)")
    print(f"Pool: {config.DES_POOL_CLASSIFIERS}")
    print(f"{'=' * 60}")

    model_evals: List[EvalResult] = []

    for model_idx in range(1, config.NUM_FINAL_MODELS + 1):
        print(f"\n--- Final DES model {model_idx}/{config.NUM_FINAL_MODELS} ---")

        try:
            # Use a different random_state per model so DSEL splits vary.
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=config.DES_DSEL_FRACTION,
                random_state=42 + model_idx,
            )
            tr_idx, dsel_idx = next(splitter.split(train_feats, train_labs))
            X_tr,   y_tr   = train_feats[tr_idx],   train_labs[tr_idx]
            X_dsel, y_dsel = train_feats[dsel_idx], train_labs[dsel_idx]

            print(f"  Pool training: {len(X_tr)} | DSEL: {len(X_dsel)}")

            pool = _build_pool_classifiers(X_tr, y_tr, config.DES_POOL_CLASSIFIERS)

            des = DynamicEnsembleSelector(
                algorithm=config.DES_ALGORITHM,
                k_neighbors=config.DES_K_NEIGHBORS,
            )
            des.fit(pool, X_dsel, y_dsel)

            des_save_path = str(
                final_models_dir / f"des_{config.DES_ALGORITHM}_model{model_idx}.joblib"
            )
            des.save(des_save_path)

            _, _, _, eval_result = evaluate_classifier(
                model=des,
                test_features=test_feats,
                test_labels=test_labs,
                class_names=class_names,
            )
            model_evals.append(eval_result)

            print(f"  Test — acc={eval_result.accuracy:.4f}  "
                  f"macro-F1={eval_result.macro_f1:.4f}  "
                  f"AUC={eval_result.macro_auc_roc:.4f}")

        except Exception as exc:
            print(f"Error training DES model {model_idx}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()

    return model_evals


def _list_artifacts(ctx: RunContext, final_models_dir: Path) -> List[RunArtifact]:
    """Enumerate result artifacts that exist on disk after training."""
    artifacts = []
    candidates = [
        ("cv_results.csv",               "Per-fold cross-validation metrics"),
        ("fold_results_summary.csv",      "Fold results in legacy format"),
        ("model_performance_summary.csv", "Aggregated final model test metrics"),
        ("per_class_metrics.csv",         "Per-class F1/precision/recall"),
    ]
    for rel_path, description in candidates:
        if (ctx.result_dir / rel_path).exists():
            artifacts.append(RunArtifact(path=rel_path, description=description))

    if final_models_dir.exists():
        shap_path = final_models_dir / "shap_values.npy"
        if shap_path.exists():
            artifacts.append(RunArtifact(
                path=str(shap_path.relative_to(ctx.result_dir)),
                description="TreeSHAP feature importance values",
            ))
        artifacts.append(RunArtifact(
            path=str(final_models_dir.relative_to(ctx.result_dir)),
            description="Trained final classifier models (.joblib)",
        ))

    return artifacts
