"""
Data preprocessing utilities for the skin cancer classification project.
Includes techniques like class weighting and resampling for handling class imbalance.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def compute_balanced_class_weights(labels):
    """
    Compute class weights inversely proportional to class frequencies.

    Args:
        labels (numpy.array): Class labels (integers).

    Returns:
        dict: Dictionary mapping class indices to weights.
    """
    # Get unique classes
    unique_classes = np.unique(labels)

    # Compute weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )

    # Create dictionary mapping class indices to weights
    class_weights = {i: w for i, w in zip(unique_classes, weights)}

    print(f"Computed class weights: {class_weights}")
    return class_weights

def apply_smote(features, labels, random_state=42, sampling_strategy='auto'):
    """
    Apply SMOTE to balance the dataset by generating synthetic samples for minority classes.

    Args:
        features (numpy.array): Feature matrix.
        labels (numpy.array): Target labels.
        random_state (int): Random state for reproducibility.
        sampling_strategy (str or dict): Sampling strategy for SMOTE.
            'auto': All classes except majority class are resampled to match majority class.
            'not majority': Only minority classes are resampled to match majority class.
            dict: Specify target number of samples for each class.

    Returns:
        tuple: (balanced_features, balanced_labels)
    """
    print(f"Applying SMOTE with sampling strategy: {sampling_strategy}")
    print(f"Original class distribution: {np.bincount(labels)}")

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    print(f"SMOTE applied. New class distribution: {np.bincount(labels_resampled)}")

    # Return the resampled data
    return features_resampled, labels_resampled

def apply_random_undersampling(features, labels, random_state=42):
    """
    Apply random undersampling to balance the dataset.

    Args:
        features (numpy.array): Feature matrix.
        labels (numpy.array): Target labels.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (balanced_features, balanced_labels)
    """
    print(f"Applying random undersampling")
    print(f"Original class distribution: {np.bincount(labels)}")

    # Apply random undersampling
    undersampler = RandomUnderSampler(random_state=random_state)
    features_resampled, labels_resampled = undersampler.fit_resample(features, labels)

    print(f"Random undersampling applied. New class distribution: {np.bincount(labels_resampled)}")

    return features_resampled, labels_resampled

def apply_hybrid_sampling(features, labels, random_state=42):
    """
    Apply a hybrid approach: undersample majority class and apply SMOTE to minority classes.

    Args:
        features (numpy.array): Feature matrix.
        labels (numpy.array): Target labels.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (balanced_features, balanced_labels)
    """
    print(f"Applying hybrid sampling approach")
    print(f"Original class distribution: {np.bincount(labels)}")

    # Step 1: Moderate undersampling of the majority class
    class_counts = np.bincount(labels)
    majority_class = np.argmax(class_counts)

    # Target count for majority class: twice the second most frequent class
    second_largest = np.partition(class_counts, -2)[-2]  # Second largest count
    target_count_majority = min(class_counts[majority_class], 2 * second_largest)

    sampling_strategy = {i: count for i, count in enumerate(class_counts)}
    sampling_strategy[majority_class] = target_count_majority

    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    features_temp, labels_temp = undersampler.fit_resample(features, labels)

    print(f"After undersampling: {np.bincount(labels_temp)}")

    # Step 2: Apply SMOTE to create a balanced dataset
    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    features_resampled, labels_resampled = smote.fit_resample(features_temp, labels_temp)

    print(f"After hybrid sampling: {np.bincount(labels_resampled)}")

    return features_resampled, labels_resampled

def apply_data_preprocessing(features, labels, method="class_weight", random_state=42):
    """
    Apply data preprocessing techniques to handle class imbalance.

    Args:
        features (numpy.array): Feature matrix.
        labels (numpy.array): Target labels.
        method (str): Preprocessing method to use:
            - "class_weight": Return original data with class weight dict
            - "smote": Apply SMOTE oversampling
            - "undersampling": Apply random undersampling
            - "hybrid": Apply hybrid approach (undersampling + SMOTE)
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Depends on method:
            - "class_weight": (features, labels, class_weights)
            - other methods: (resampled_features, resampled_labels)
    """
    print(f"Applying {method} for handling class imbalance")
    print(f"Original class distribution: {np.bincount(labels)}")

    if method == "class_weight":
        class_weights = compute_balanced_class_weights(labels)
        return features, labels, class_weights

    elif method == "smote":
        return apply_smote(features, labels, random_state=random_state), None

    elif method == "undersampling":
        return apply_random_undersampling(features, labels, random_state=random_state), None

    elif method == "hybrid":
        return apply_hybrid_sampling(features, labels, random_state=random_state), None

    else:
        print(f"Unknown method: {method}, returning original data")
        return features, labels, None