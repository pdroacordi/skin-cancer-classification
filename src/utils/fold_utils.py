import os


def save_fold_results(fold_results, result_dir, classifier_name):
    """
    Save detailed results for each fold in a structured format.

    Args:
        fold_results: List of dictionaries containing fold results
        result_dir: Base directory for results
        classifier_name: Name of the classifier
    """
    import pandas as pd
    import json

    # Create a DataFrame from fold results
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

    # Save as CSV
    csv_path = os.path.join(result_dir, "fold_results_summary.csv")
    df_folds.to_csv(csv_path, index=False)
    print(f"Fold results saved to: {csv_path}")

    # Save detailed results as JSON
    json_path = os.path.join(result_dir, "fold_results_detailed.json")
    with open(json_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    print(f"Detailed fold results saved to: {json_path}")

    # Create summary statistics by iteration
    iteration_summary = df_folds.groupby('iteration').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max']
    }).round(4)

    # Save iteration summary
    iter_summary_path = os.path.join(result_dir, "iteration_summary_stats.csv")
    iteration_summary.to_csv(iter_summary_path)

    # Create summary statistics across all folds
    overall_summary = {
        'total_folds': len(fold_results),
        'num_iterations': df_folds['iteration'].nunique(),
        'folds_per_iteration': len(fold_results) // df_folds['iteration'].nunique(),
        'metrics': {
            'accuracy': {
                'mean': df_folds['accuracy'].mean(),
                'std': df_folds['accuracy'].std(),
                'min': df_folds['accuracy'].min(),
                'max': df_folds['accuracy'].max()
            },
            'precision': {
                'mean': df_folds['precision'].mean(),
                'std': df_folds['precision'].std(),
                'min': df_folds['precision'].min(),
                'max': df_folds['precision'].max()
            },
            'recall': {
                'mean': df_folds['recall'].mean(),
                'std': df_folds['recall'].std(),
                'min': df_folds['recall'].min(),
                'max': df_folds['recall'].max()
            },
            'f1_score': {
                'mean': df_folds['f1_score'].mean(),
                'std': df_folds['f1_score'].std(),
                'min': df_folds['f1_score'].min(),
                'max': df_folds['f1_score'].max()
            }
        }
    }
    return df_folds