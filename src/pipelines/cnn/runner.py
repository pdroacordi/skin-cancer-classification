"""
CNN classifier pipeline runner.

Orchestration only.  This module contains no business logic — it composes
the three stages (train, predict, evaluate) and delegates all persistence
to save_run_results().

Entry point used by main.py:
    run_cnn_classifier_pipeline(train_files_path, val_files_path, test_files_path,
                                run_kfold, class_names, skip_training)
"""

from __future__ import annotations

import datetime
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.backend import clear_session

from core.run_context import RunContext
from core.types import EvalResult, FoldResult, RunArtifact
from models.cnn_models import (
    create_model_name,
    save_gradcam_visualizations,
)
from pipelines.cnn.stages.evaluate import evaluate_predictions
from pipelines.cnn.stages.predict import predict_batch
from pipelines.cnn.stages.train import train_one_fold
from preprocessing.data.augmentation import AugmentationFactory
from utils.data_loaders import load_paths_labels
from utils.fold_utils import save_fold_results
from utils.gpu_utils import setup_gpu_memory
from utils.metrics import aggregate_eval_results
from utils.persistence import save_run_results, save_multiple_model_stats
from utils.result_naming import cnn_result_dir


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #

def run_cnn_classifier_pipeline(
    train_files_path: str,
    val_files_path: str,
    test_files_path: str,
    run_kfold: bool = True,
    class_names: Optional[List[str]] = None,
    skip_training: bool = False,
) -> Dict:
    """
    Run the complete CNN classifier pipeline.

    Args:
        train_files_path: Path to tab-separated train file list.
        val_files_path:   Path to tab-separated val file list.
        test_files_path:  Path to tab-separated test file list.
        run_kfold:        Whether to run K-fold cross-validation before training
                          final models.
        class_names:      Optional list of class label strings.
        skip_training:    When True, skip K-fold and go straight to training
                          NUM_FINAL_MODELS on all train+val data.

    Returns:
        Dict with keys: fold_results, final_metrics, result_dir.
    """
    import config  # lazy import ensures apply_cli_overrides() has already run

    setup_gpu_memory()

    train_paths, train_labels = load_paths_labels(train_files_path)
    val_paths,   val_labels   = load_paths_labels(val_files_path)
    test_paths,  test_labels  = load_paths_labels(test_files_path)

    result_dir = Path(cnn_result_dir())
    result_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext.create(
        pipeline="cnn_classifier",
        result_dir=str(result_dir),
        config_snapshot=_build_config_snapshot(config),
    )
    print(f"Results will be saved to: {result_dir}")
    print(f"Run ID: {ctx.run_id}")

    fold_results: List[FoldResult] = []

    # ---- K-fold cross-validation ---- #
    if run_kfold and not skip_training:
        all_cv_paths  = np.concatenate([train_paths, val_paths])
        all_cv_labels = np.concatenate([train_labels, val_labels])

        fold_results = _run_kfold_cv(
            all_cv_paths, all_cv_labels, ctx, config, class_names
        )

        save_fold_results(
            fold_results=[fr.to_legacy_dict() for fr in fold_results],
            result_dir=str(ctx.result_dir),
            classifier_name="CNN",
        )

    # ---- Final model training on all train+val data ---- #
    all_paths  = np.concatenate([train_paths, val_paths])
    all_labels = np.concatenate([train_labels, val_labels])

    model_evals, final_models_dir = _train_and_evaluate_final_models(
        all_paths, all_labels, test_paths, test_labels, ctx, config, class_names
    )

    # Aggregate across NUM_FINAL_MODELS
    final_metrics = aggregate_eval_results(model_evals)

    # Persist statistical summary for multiple models
    if len(model_evals) > 1:
        save_multiple_model_stats(model_evals, final_models_dir, class_names)

    # Collect result artifacts
    artifacts = _list_artifacts(ctx, final_models_dir)

    save_run_results(
        ctx=ctx,
        fold_results=fold_results,
        final_metrics=final_metrics,
        artifacts=artifacts,
    )

    return {
        "fold_results":   fold_results,
        "final_metrics":  final_metrics,
        "result_dir":     str(ctx.result_dir),
    }


# ------------------------------------------------------------------ #
#  K-fold cross-validation                                            #
# ------------------------------------------------------------------ #

def _run_kfold_cv(
    all_paths: np.ndarray,
    all_labels: np.ndarray,
    ctx: RunContext,
    config,
    class_names: Optional[List[str]],
) -> List[FoldResult]:
    """Run NUM_ITERATIONS × NUM_KFOLDS folds and return all FoldResult objects."""
    fold_results: List[FoldResult] = []

    for iteration in range(config.NUM_ITERATIONS):
        print(f"\n{'=' * 50}")
        print(f"Iteration {iteration + 1}/{config.NUM_ITERATIONS}")
        print(f"{'=' * 50}")

        iter_dir = ctx.result_dir / f"iteration_{iteration + 1}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Different random state per iteration for independent splits
        skf = StratifiedKFold(
            n_splits=config.NUM_KFOLDS, shuffle=True, random_state=42 + iteration
        )

        # StratifiedKFold requires integer labels
        stratify_labels = (
            np.argmax(all_labels, axis=1)
            if all_labels.ndim > 1 and all_labels.shape[1] > 1
            else all_labels
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(all_paths, stratify_labels), start=1
        ):
            print(f"\n{'=' * 40}")
            print(f"Iteration {iteration + 1}, Fold {fold}/{config.NUM_KFOLDS}")
            print(f"{'=' * 40}")

            fold_dir = iter_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            model_name = create_model_name(
                base_model_name=config.CNN_MODEL,
                mode="classifier",
                use_fine_tuning=config.USE_FINE_TUNING,
                use_preprocessing=config.USE_GRAPHIC_PREPROCESSING,
            )
            model_save_path = str(
                iter_dir / "models" / f"{model_name}_iter{iteration+1}_fold{fold}.h5"
            )
            (iter_dir / "models").mkdir(parents=True, exist_ok=True)

            log_dir = str(
                fold_dir / "logs" / datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            )

            augment_fn = _make_augment_fn(config.USE_DATA_AUGMENTATION)

            try:
                model, _ = train_one_fold(
                    train_paths=all_paths[train_idx],
                    train_labels=all_labels[train_idx],
                    val_paths=all_paths[val_idx],
                    val_labels=all_labels[val_idx],
                    model_save_path=model_save_path,
                    log_dir=log_dir,
                    cnn_model=config.CNN_MODEL,
                    batch_size=config.BATCH_SIZE,
                    num_epochs=config.NUM_EPOCHS,
                    use_fine_tuning=config.USE_FINE_TUNING,
                    augment_fn=augment_fn,
                )

                # Evaluate on the validation split
                y_true, y_pred, y_prob, _ = predict_batch(
                    model=model,
                    paths=all_paths[val_idx],
                    labels=all_labels[val_idx],
                    cnn_model=config.CNN_MODEL,
                    batch_size=config.BATCH_SIZE,
                )

                eval_result = evaluate_predictions(y_true, y_pred, y_prob, class_names)

                print(f"\nIteration {iteration + 1}, Fold {fold} — "
                      f"acc={eval_result.accuracy:.4f}  "
                      f"macro-F1={eval_result.macro_f1:.4f}")

                fold_results.append(
                    FoldResult(
                        iteration=iteration + 1,
                        fold=fold,
                        eval_result=eval_result,
                        model_path=model_save_path,
                    )
                )

            except Exception as exc:
                print(f"Error in iteration {iteration+1}, fold {fold}: {exc}")

            finally:
                # Release Keras session and GPU memory between folds
                clear_session()
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
    class_names: Optional[List[str]],
) -> Tuple[List[EvalResult], Path]:
    """
    Train NUM_FINAL_MODELS CNNs on all train+val data, evaluate each on the
    test set, and save Grad-CAM visualizations for the last model.

    Returns (list_of_eval_results, final_models_dir).
    """
    print(f"\n{'=' * 60}")
    print(f"Training {config.NUM_FINAL_MODELS} final CNN model(s)")
    print(f"{'=' * 60}")

    final_models_dir = ctx.result_dir / "final_models"
    final_models_dir.mkdir(parents=True, exist_ok=True)

    augment_fn = _make_augment_fn(config.USE_DATA_AUGMENTATION)
    tta_augment_fn = (
        _make_augment_fn(True) if config.USE_TTA else None
    )

    model_evals: List[EvalResult] = []

    for model_idx in range(1, config.NUM_FINAL_MODELS + 1):
        print(f"\n{'=' * 50}")
        print(f"Final model {model_idx}/{config.NUM_FINAL_MODELS}")
        print(f"{'=' * 50}")

        model_dir = final_models_dir / f"model_{model_idx}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = str(model_dir / "final_cnn_model.h5")
        log_dir = str(model_dir / "logs" / datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))

        # Use a small validation subset from training data to monitor progress.
        # This subset is NOT used for model selection — only to track loss curves.
        rng = np.random.RandomState(42 + model_idx)
        val_size = min(1000, len(all_paths) // 5)
        val_idx  = rng.choice(len(all_paths), val_size, replace=False)
        monitor_paths  = all_paths[val_idx]
        monitor_labels = all_labels[val_idx]

        try:
            model, _ = train_one_fold(
                train_paths=all_paths,
                train_labels=all_labels,
                val_paths=monitor_paths,
                val_labels=monitor_labels,
                model_save_path=model_save_path,
                log_dir=log_dir,
                cnn_model=config.CNN_MODEL,
                batch_size=config.BATCH_SIZE,
                num_epochs=config.NUM_EPOCHS,
                use_fine_tuning=config.USE_FINE_TUNING,
                augment_fn=augment_fn,
            )

            y_true, y_pred, y_prob, uncertainty = predict_batch(
                model=model,
                paths=test_paths,
                labels=test_labels,
                cnn_model=config.CNN_MODEL,
                batch_size=config.BATCH_SIZE,
                use_mc_dropout=config.USE_MC_DROPOUT,
                mc_dropout_steps=config.MC_DROPOUT_STEPS,
                use_tta=config.USE_TTA,
                tta_n_steps=config.TTA_N_STEPS,
                tta_augment_fn=tta_augment_fn,
            )

            eval_result = evaluate_predictions(y_true, y_pred, y_prob, class_names)
            model_evals.append(eval_result)

            print(f"Model {model_idx} — test acc={eval_result.accuracy:.4f}  "
                  f"macro-F1={eval_result.macro_f1:.4f}  "
                  f"AUC={eval_result.macro_auc_roc:.4f}")

            # Save MC Dropout uncertainty array when available
            if uncertainty is not None:
                np.save(str(model_dir / "mc_dropout_uncertainty.npy"), uncertainty)
                print(f"MC Dropout uncertainty: mean={uncertainty.mean():.4f}")

            # Grad-CAM visualizations for interpretability
            # Only run for the last model to avoid redundant computation
            if model_idx == config.NUM_FINAL_MODELS:
                try:
                    save_gradcam_visualizations(
                        model=model,
                        model_name=config.CNN_MODEL,
                        image_paths=test_paths,
                        y_true=y_true,
                        y_pred=y_pred,
                        y_pred_prob=y_prob,
                        class_names=class_names,
                        result_dir=str(ctx.result_dir),
                    )
                except Exception as exc:
                    print(f"Grad-CAM skipped: {exc}")

        except Exception as exc:
            print(f"Error training final model {model_idx}: {exc}")
            import traceback
            traceback.print_exc()

        finally:
            clear_session()
            gc.collect()

    return model_evals, final_models_dir


# ------------------------------------------------------------------ #
#  Private helpers                                                     #
# ------------------------------------------------------------------ #

def _make_augment_fn(enabled: bool) -> Optional[callable]:
    """Return an albumentations augmentation function if enabled, else None."""
    if not enabled:
        return None
    augmentation = AugmentationFactory.get_medium_augmentation()
    return lambda img: augmentation(image=img)["image"]


def _build_config_snapshot(config) -> dict:
    """Capture all active experiment flags as a plain dict for metadata.json."""
    return {
        "cnn_model":                  config.CNN_MODEL,
        "batch_size":                 config.BATCH_SIZE,
        "num_epochs":                 config.NUM_EPOCHS,
        "num_kfolds":                 config.NUM_KFOLDS,
        "num_iterations":             config.NUM_ITERATIONS,
        "num_final_models":           config.NUM_FINAL_MODELS,
        "use_fine_tuning":            config.USE_FINE_TUNING,
        "use_data_augmentation":      config.USE_DATA_AUGMENTATION,
        "use_graphic_preprocessing":  config.USE_GRAPHIC_PREPROCESSING,
        "use_hair_removal":           config.USE_HAIR_REMOVAL,
        "use_enhanced_contrast":      config.USE_ENHANCED_CONTRAST,
        "use_color_normalization":    config.USE_COLOR_NORMALIZATION,
        "use_focal_loss":             config.USE_FOCAL_LOSS,
        "label_smoothing":            config.LABEL_SMOOTHING,
        "use_mixup":                  config.USE_MIXUP,
        "use_tta":                    config.USE_TTA,
        "tta_n_steps":                config.TTA_N_STEPS,
        "use_mc_dropout":             config.USE_MC_DROPOUT,
        "mc_dropout_steps":           config.MC_DROPOUT_STEPS,
    }


def _list_artifacts(ctx: RunContext, final_models_dir: Path) -> List[RunArtifact]:
    """Enumerate result artifacts that exist on disk after training."""
    artifacts = []
    candidates = [
        ("cv_results.csv",              "Per-fold cross-validation metrics"),
        ("fold_results_summary.csv",    "Fold results in legacy format"),
        ("model_performance_summary.csv","Aggregated final model test metrics"),
        ("per_class_metrics.csv",       "Per-class F1/precision/recall"),
        ("gradcam",                     "Grad-CAM heatmap overlays per class"),
    ]
    for rel_path, description in candidates:
        full_path = ctx.result_dir / rel_path
        if full_path.exists():
            artifacts.append(RunArtifact(path=rel_path, description=description))

    # Final models directory
    if final_models_dir.exists():
        artifacts.append(RunArtifact(
            path=str(final_models_dir.relative_to(ctx.result_dir)),
            description="Trained final CNN model checkpoints (.h5)",
        ))

    return artifacts
