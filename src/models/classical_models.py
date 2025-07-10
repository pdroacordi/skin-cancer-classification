"""
Classical machine learning model definitions for the feature extraction pipeline.
"""

import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.append('..')
from config import NUM_CLASSES


def get_classifier(classifier_name, random_state=42):
    """
    Get a classifier instance based on the specified name.

    Args:
        classifier_name (str): Name of the classifier.
        random_state (int): Random state for reproducibility.

    Returns:
        object: Classifier instance.
    """
    if classifier_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            oob_score=True,
            n_jobs=-1,
            random_state=random_state,
        )
    elif classifier_name == "XGBoost":
        # Import XGBoost only if needed to avoid dependency issues
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1,
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multi:softprob',
            num_class=NUM_CLASSES,
            tree_method='hist',
            random_state=random_state,
            eval_metric='mlogloss',
        )
    elif classifier_name == "AdaBoost":
        return AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.1,
            algorithm='SAMME',
            random_state=random_state
        )
    elif classifier_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        )
    elif classifier_name == "SVM":
        return SVC(
            C=10.0,
            kernel='rbf',
            gamma=0.01,
            probability=True,
            class_weight='balanced',
            random_state=random_state
        )
    else:
        raise ValueError(
            f"Unsupported classifier: {classifier_name}. Choose from 'RandomForest', 'XGBoost', 'AdaBoost', 'ExtraTrees', or 'SVM'")


def create_ml_pipeline(classifier_name, random_state=42):
    """
    Create a machine learning pipeline with preprocessing and a classifier.

    Args:
        classifier_name (str): Name of the classifier.
        random_state (int): Random state for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: ML pipeline.
    """
    steps = []

    # Add classifier
    steps.append(('classifier', get_classifier(classifier_name, random_state=random_state)))

    return Pipeline(steps)


def tune_hyperparameters(pipeline, X, y, param_grid, cv=5, n_jobs=-1, subset_fraction=0.5, verbose=1, class_weights=None):
    """
    Tune hyperparameters using grid search cross-validation.

    Args:
        pipeline (sklearn.pipeline.Pipeline): ML pipeline.
        X (numpy.array): Feature matrix.
        y (numpy.array): Target labels.
        param_grid (dict): Parameter grid for grid search.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of parallel jobs.
        subset_fraction (float): Fraction of training data to be used as subset.
        verbose (int): Verbosity level.
        class_weights (dict, optional): Dictionary mapping class indices to weights.

    Returns:
        sklearn.model_selection.GridSearchCV: Fitted grid search object.
    """
    from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

    # Create stratified sample
    sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_fraction, random_state=42)
    for _, subset_idx in sss.split(X, y):
        X_subset = X[subset_idx]
        y_subset = y[subset_idx]

    print(f"Using {len(X_subset)} samples ({subset_fraction * 100:.0f}%) for hyperparameter tuning")

    # Configure fit parameters if class weights are provided
    fit_params = {}
    if class_weights is not None:
        sample_weights = np.array([class_weights[label] for label in y_subset])
        fit_params = {'classifier__sample_weight': sample_weights}
        print("Using class weights during hyperparameter tuning")

    # Create the grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring='balanced_accuracy',  # Use balanced accuracy for imbalanced data
        return_train_score=True
    )

    # Fit with or without fit_params
    if fit_params:
        grid_search.fit(X_subset, y_subset, **fit_params)
    else:
        grid_search.fit(X_subset, y_subset)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search


def get_default_param_grid(classifier_name):
    """
    Get a default parameter grid for hyperparameter tuning.

    Args:
        classifier_name (str): Name of the classifier.

    Returns:
        dict: Parameter grid.
    """
    if classifier_name == "RandomForest":
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 15, 30],
            'classifier__min_samples_split': [2, 10],
            'classifier__min_samples_leaf': [1, 4]
        }
    elif classifier_name == "XGBoost":
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.7, 0.8, 0.9],
            'classifier__colsample_bytree': [0.7, 0.8, 0.9]
        }
    elif classifier_name == "AdaBoost":
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1.0]
        }
    elif classifier_name == "ExtraTrees":
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif classifier_name == "SVM":
        return {
            'classifier__C': [0.1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'classifier__kernel': ['rbf', 'linear', 'poly']
        }
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")


def save_model(model, save_path):
    """
    Save a trained model to disk.

    Args:
        model: Trained model.
        save_path (str): Path to save the model.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Save the model
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")


def load_model(load_path):
    """
    Load a trained model from disk.

    Args:
        load_path (str): Path to load the model from.

    Returns:
        object: Loaded model.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")

    model = joblib.load(load_path)
    print(f"Model loaded from: {load_path}")

    return model