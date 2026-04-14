# src/pipelines/CLAUDE.md

## Structure

Each pipeline lives in its own sub-package with three stages plus a runner:

```
pipelines/
  cnn/
    runner.py          ← public entry point called by main.py
    stages/
      train.py         ← train_one_fold()
      predict.py       ← predict_batch() (TTA, MC Dropout)
      evaluate.py      ← evaluate_predictions()
  feature_extraction/
    runner.py          ← public entry point called by main.py
    stages/
      extract.py       ← load_or_extract_features()
      train.py         ← train_classifier()
      evaluate.py      ← evaluate_classifier()
```

## cnn/runner.py

End-to-end CNN training pipeline.

**Training flow:**
1. `setup_gpu_memory()` — enables memory growth to avoid OOM.
2. `cnn_result_dir()` — builds the output path from `cfg` flags.
3. Optional K-fold CV (`cfg.num_iterations × cfg.num_kfolds` folds total):
   - Images are loaded lazily via `MemoryEfficientDataGenerator`.
   - Class weights computed via `sklearn.utils.class_weight.compute_class_weight('balanced')`.
   - Each fold trains with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
4. Trains `cfg.num_final_models` on the full train+val set; Grad-CAM run on the last model.
5. All results written via `save_run_results()`.

**Model checkpoints** are saved as `.keras` (not `.h5`).

**Evaluation features** (in `predict_batch()`):
- **TTA** (`cfg.use_tta=True`): averages `cfg.tta_n_steps` augmented predictions per sample.
- **MC Dropout** (`cfg.use_mc_dropout=True`): `cfg.mc_dropout_steps` stochastic forward passes; per-sample uncertainty saved to `mc_dropout_uncertainty.npy`.
- **Grad-CAM**: saves heatmap overlays per class to `gradcam/` subdirectory.

**Output structure:**
```
results/cnn_classifier_<MODEL>_<flags>/
    iteration_<i>/
        fold_<j>/
            logs/
        models/                        # per-fold checkpoints (.keras)
    final_models/
        model_<i>/
            final_cnn_model.keras
            mc_dropout_uncertainty.npy  (if use_mc_dropout=True)
            logs/
        model_performance_summary.csv
    fold_results_summary.csv
    per_class_metrics.csv
    gradcam/                           # Grad-CAM overlays per class
    metadata.json                      # full config snapshot + metrics
```

## feature_extraction/runner.py

CNN feature extraction → classical ML pipeline.

**Training flow:**
1. Loads (or creates) a feature extractor via `get_feature_extractor_from_cnn()`.
   - Prefers a CNN already trained by the CNN pipeline; falls back to ImageNet weights.
2. Fits `MetadataFeatureExtractor` **only on train+val images** when `cfg.use_metadata=True`.
3. Optional K-fold CV: extract per-fold features → preprocess → train → evaluate.
   - Feature augmentation is **disabled** during K-fold to prevent augmented-duplicate leakage.
4. Trains `cfg.num_final_models` final classifiers on all train+val data → evaluate on test.
5. SHAP analysis (TreeSHAP) on the best final model for tree-based classifiers.
6. When `cfg.use_dynamic_ensemble=True`, trains a DES pool instead of a single classifier.

**Output structure:**
```
results/feature_extraction_<CNN>_<flags>/
    models/
        <cnn>_feature_extractor.keras
    <classifier>/                       # or dynamic_ensemble/
        iteration_<i>/
            fold_<j>/
                models/
                    <classifier>_fold<j>.joblib
                    feat_preprocessing.joblib  (if use_feature_preprocessing=True)
        features/
            all_features.npz
            test_features.npz
        final_models/
            <classifier>_model<i>.joblib
            final_feat_preprocessing.joblib
            shap_values.npy
        fold_results_summary.csv
        model_performance_summary.csv
        per_class_metrics.csv
        metadata.json
```

## Shared behaviour

- Both runners call `save_fold_results()` from `utils/fold_utils.py` to persist
  per-fold metrics in a standard format consumed by `analysis/aggregate_results.py`.
- GPU memory growth is configured at startup via `setup_gpu_memory()`.
- `clear_session()` and `gc.collect()` are called between folds to free Keras/GPU memory.
- `_build_config_snapshot()` in each runner captures `cfg` at run time into `metadata.json`.
