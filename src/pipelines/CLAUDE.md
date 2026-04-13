# src/pipelines/CLAUDE.md

## cnn_classifier.py

End-to-end CNN training pipeline. Entry point: called from `main.py`.

**Training flow:**
1. `setup_gpu_memory()` — enables memory growth to avoid OOM.
2. `create_result_directories()` — builds the output path from config flags.
3. K-fold cross-validation loop (`NUM_ITERATIONS × NUM_KFOLDS` folds total):
   - Images are loaded lazily via `MemoryEfficientDataGenerator` (never all in RAM).
   - Class weights computed via `sklearn.utils.class_weight.compute_class_weight('balanced')` and passed to `model.fit()`.
   - Each fold trains with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
   - Per-fold metrics saved by `save_fold_results()` → `iteration_results.txt`.
4. After CV, trains `NUM_FINAL_MODELS` on the full train set and selects the best
   by macro-F1; summary written to `final_models/model_performance_summary.csv`.

**Evaluation features** (in `evaluate_model()`):
- Macro AUC-ROC via `roc_auc_score` (one-vs-rest, macro average).
- Expected Calibration Error (ECE) from `utils.calibration`.
- **TTA** (`USE_TTA=True`): averages `TTA_N_STEPS` augmented predictions per sample.
- **MC Dropout** (`USE_MC_DROPOUT=True`): `MC_DROPOUT_STEPS` stochastic forward passes; per-sample uncertainty saved to `mc_dropout_uncertainty.npy`.
- **Grad-CAM**: saves heatmap overlays per class to `gradcam/` subdirectory.

**Output structure:**
```
results/cnn_classifier_<MODEL>_<flags>/
    models/                        # per-fold checkpoints
    final_models/
        model_<i>/final_cnn_model.h5
        model_performance_summary.csv
    model_performance_summary.csv  # CV aggregated metrics (includes macro_auc_roc, ece)
    per_class_metrics.csv
    gradcam/                       # Grad-CAM overlays per class
    mc_dropout_uncertainty.npy     # (if USE_MC_DROPOUT=True)
```

## feature_extraction.py

CNN feature extraction → classical ML pipeline. Entry point: called from `main.py`.

**Training flow:**
1. Loads (or creates) a feature extractor via `get_feature_extractor_from_cnn()`.
   - Prefers a CNN already trained by `cnn_classifier.py`; falls back to ImageNet weights.
2. Extracts feature vectors from train/val/test images (batch inference, no full-image RAM load).
3. When `USE_METADATA=True`: `MetadataFeatureExtractor.fit()` is called **only on train+val images** to prevent scaler leakage, then appended via `combine_cnn_and_metadata_features()`.
4. If `USE_FEATURE_PREPROCESSING=True`, runs `apply_feature_preprocessing()` (per-algorithm pipeline).
5. If `USE_FEATURE_AUGMENTATION=True`, applies feature-space augmentation before fitting.
6. Trains the classical ML classifier; evaluates on test set; saves model with `joblib`.

**Evaluation features:**
- Macro AUC-ROC and ECE computed alongside accuracy/F1.
- **SHAP**: TreeSHAP values computed after training and saved as `shap_values.npy`.

**Legacy guard:** `run_kfold_cross_validation()` raises `RuntimeError` when
`USE_FEATURE_AUGMENTATION=True` to prevent the 68k-row memorisation bug (augmented
duplicates leaking across K-fold splits).

**Output structure:**
```
results/feature_extraction_<CNN>_<flags>/<classifier>/
    iteration_<i>/
        fold_<j>/
            classifier.joblib
            feature_preprocessing_pipeline.joblib
            iteration_results.txt
    model_performance_summary.csv  # includes macro_auc_roc, ece
    per_class_metrics.csv
    shap_values.npy
```

## Shared behaviour

- Both pipelines call `save_fold_results()` from `utils/fold_utils.py` to persist
  per-fold metrics in a standard format consumed by `analysis/aggregate_results.py`.
- GPU memory growth is configured identically in both pipelines at startup.
- `clear_session()` and `gc.collect()` are called between folds to free Keras/GPU memory.
