"""
Model calibration utilities.

Provides Expected Calibration Error (ECE) computation for evaluating
how well a model's confidence scores match observed frequencies.

ECE reference: Guo et al. (2017, ICML, "On Calibration of Modern Neural Networks").
Medical imaging context: Nixon et al. (2019, CVPR Workshop).
"""

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the gap between a model's confidence and its actual accuracy,
    averaged across confidence bins. A well-calibrated model predicting 80%
    confidence should be correct 80% of the time.

    Args:
        y_true: 1-D integer array of true class indices, shape (n_samples,).
        y_prob: 2-D float array of softmax probabilities, shape (n_samples, n_classes).
        n_bins: Number of equally-spaced confidence bins (default: 15).

    Returns:
        float: ECE in [0, 1]. Lower is better. 0 = perfectly calibrated.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += n_in_bin * abs(bin_accuracy - bin_confidence)

    return ece / len(y_true)
