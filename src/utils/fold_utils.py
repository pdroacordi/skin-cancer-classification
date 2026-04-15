import os


def save_fold_results(fold_results, result_dir, classifier_name):
    """
    Save detailed results for each fold in a structured format.

    Note: the primary fold_results_summary.csv is written by
    persistence.save_run_results(). This function writes supplementary
    files only: JSON detail dump and per-iteration summary statistics.

    Args:
        fold_results: List of dictionaries containing fold results
        result_dir: Base directory for results
        classifier_name: Name of the classifier
    """
    import pandas as pd
    import json

    fold_data = []
    for result in fold_results:
        fold_data.append({
            'iteration': result['iteration'],
            'fold': result['fold'],
            'accuracy': result['accuracy'],
            'precision': result['macro_avg_precision'],
            'recall': result['macro_avg_recall'],
            'f1_score': result['macro_avg_f1']
        })

    df_folds = pd.DataFrame(fold_data)

    # Save detailed results as JSON (not produced by save_run_results)
    json_path = os.path.join(result_dir, "fold_results_detailed.json")
    with open(json_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    print(f"Detailed fold results saved to: {json_path}")

    # Per-iteration summary statistics (not produced by save_run_results)
    if not df_folds.empty:
        iteration_summary = df_folds.groupby('iteration').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std', 'min', 'max'],
            'recall': ['mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std', 'min', 'max']
        }).round(4)
        iter_summary_path = os.path.join(result_dir, "iteration_summary_stats.csv")
        iteration_summary.to_csv(iter_summary_path)

    return df_folds