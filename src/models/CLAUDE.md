# src/models/CLAUDE.md

## cnn_models.py

Loads and configures Keras pretrained CNNs (VGG19, InceptionV3, ResNet50, Xception,
EfficientNetB4).

**Two modes:**
- `'classifier'` — adds `GlobalAveragePooling2D → Dense(512) → Dropout(0.5) → Dense(7, softmax)`. Compiled with Adam(lr=1e-4) and configurable loss (see below).
- `'extractor'` — returns the raw base model (no head). Call `create_feature_extractor()` to attach a `GlobalAveragePooling2D` at a model-specific layer.

**Feature extraction layers per model:**

| Model | Layer | Input size |
|---|---|---|
| VGG19 | `block5_pool` | 224×224 |
| Inception | `mixed10` | 299×299 |
| ResNet | `avg_pool` | 224×224 |
| Xception | `avg_pool` | 299×299 |
| EfficientNet | `top_activation` | 380×380 |

**Loss function:** resolved at `load_or_create_cnn()` call time from `cfg`:
- `cfg.use_focal_loss=True` → focal loss (Lin et al., 2017) with γ=2.0.
- `cfg.label_smoothing > 0` → `CategoricalCrossentropy(label_smoothing=...)`.
- Otherwise → plain `'categorical_crossentropy'`.

**Fine-tuning:** layers before `FINE_TUNING_AT_LAYER[model]` (module-level constant in `config.py`) are frozen; later layers are trainable. Set `fine_tune=False` to freeze the entire base model.

**Key functions:**
- `load_or_create_cnn(model_name, mode, fine_tune, save_path)` — loads from disk if `save_path` exists, otherwise creates fresh.
- `get_feature_extractor_from_cnn(save_path, cnn_model_path)` — prefers a pre-trained CNN (selected by best macro-F1 in `final_models/model_performance_summary.csv`); falls back to ImageNet weights.
- `find_trained_cnn_model(results_dir)` — scans the CNN results dir for the best model by macro-F1.
- `get_callbacks(save_path)` — returns EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
- `compute_gradcam(model, img_array, class_idx, layer_name)` — GradientTape-based Grad-CAM heatmap for a single image and target class.
- `save_gradcam_visualizations(...)` — saves heatmap overlays for the highest-confidence correct and wrong prediction per class under `result_dir/gradcam/`.

**`GRADCAM_LAYER` dict** maps each backbone to its last convolutional activation layer.

## classical_models.py

Defines sklearn/XGBoost classifiers for the feature extraction pipeline.

**Classifiers and notable settings:**

| Name | Key hyperparameters |
|---|---|
| `RandomForest` | 200 trees, `class_weight='balanced'`, OOB scoring |
| `XGBoost` | 200 trees, lr=0.05, max_depth=4, `tree_method='hist'` |
| `LightGBM` | 300 iters, lr=0.05, num_leaves=63, `class_weight='balanced'` |
| `HistGradientBoosting` | 300 iters, lr=0.05, max_leaf_nodes=63, `class_weight='balanced'` |
| `ExtraTrees` | 200 trees, `max_features='sqrt'`, `class_weight='balanced'` |
| `SVM` | RBF kernel, C=10, γ=0.01, `class_weight='balanced'` |

All classifiers are wrapped in a `sklearn.pipeline.Pipeline` by `create_ml_pipeline()`.
Models are persisted/loaded via `joblib` (`save_model` / `load_model`).

`tune_hyperparameters()` runs `GridSearchCV` on a stratified subsample (default 50%) 
scored by `balanced_accuracy` — use sparingly as it is expensive.

## dynamic_ensemble.py

Wraps DESlib (0.3.7) in an sklearn-compatible interface for the feature extraction
pipeline.  Enabled by `cfg.use_dynamic_ensemble=True` (CLI: `--use-dynamic-ensemble`).

**Class:** `DynamicEnsembleSelector(algorithm, k_neighbors)`

| Algorithm | Description | When to use |
|---|---|---|
| `knorau` | KNN Oracle Union — selects classifiers correct on ≥1 DSEL neighbor | Default; robust |
| `desmi` | DES Multiple Instances — designed for imbalanced multiclass | Best fit for HAM10000 |
| `metades` | META-DES — meta-classifier learns local competence | Powerful; needs large DSEL |
| `singlebest` | Static: picks globally best classifier on DSEL | Fast baseline |

**Interface:** `fit(pool_classifiers, X_dsel, y_dsel)` · `predict(X)` · `predict_proba(X)` · `save(path)` · `load(path)`

Pool classifiers are pre-fitted sklearn Pipelines (output of `create_ml_pipeline`).
DSEL is a stratified 30% split of training data (configurable via `cfg.des_dsel_fraction`).
Results are saved under `results/feature_extraction_.../dynamic_ensemble/`.
