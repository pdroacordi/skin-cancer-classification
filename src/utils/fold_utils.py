import os

import numpy as np


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

    # Save overall summary
    summary_path = os.path.join(result_dir, "overall_fold_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Cross-Validation Fold Results Summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Classifier: {classifier_name}\n")
        f.write(f"Total folds evaluated: {overall_summary['total_folds']}\n")
        f.write(f"Number of iterations: {overall_summary['num_iterations']}\n")
        f.write(f"Folds per iteration: {overall_summary['folds_per_iteration']}\n\n")

        f.write("Overall Metrics (Mean ± Std):\n")
        for metric, stats in overall_summary['metrics'].items():
            f.write(f"  {metric.upper()}:\n")
            f.write(f"    Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"    Min/Max: {stats['min']:.4f} / {stats['max']:.4f}\n\n")

    # Create visualization of fold results
    plot_fold_results(df_folds, result_dir)

    return df_folds


def plot_fold_results(df_folds, result_dir):
    """
    Create visualizations for fold results.

    Args:
        df_folds: DataFrame with fold results
        result_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    sns.set_style("whitegrid")

    # 1. Line plot showing performance across folds for each iteration
    plt.figure(figsize=(12, 8))

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['blue', 'green', 'red', 'purple']

    for iteration in df_folds['iteration'].unique():
        iter_data = df_folds[df_folds['iteration'] == iteration].sort_values('fold')

        for metric, color in zip(metrics, colors):
            plt.plot(iter_data['fold'], iter_data[metric],
                     marker='o', label=f'{metric} (Iter {iteration})',
                     alpha=0.7, linestyle='--' if iteration > 1 else '-')

    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Performance Metrics Across Folds by Iteration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "fold_performance_by_iteration.png"), dpi=300)
    plt.close()

    # 2. Box plot of metrics across all folds
    plt.figure(figsize=(10, 6))

    # Prepare data for box plot
    plot_data = []
    for metric in metrics:
        for value in df_folds[metric]:
            plot_data.append({'Metric': metric.capitalize(), 'Value': value})

    import pandas as pd
    df_plot = pd.DataFrame(plot_data)

    sns.boxplot(data=df_plot, x='Metric', y='Value')
    plt.title('Distribution of Metrics Across All Folds')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "fold_metrics_distribution.png"), dpi=300)
    plt.close()

    # 3. Heatmap of fold results
    plt.figure(figsize=(10, 8))

    # Reshape data for heatmap
    heatmap_data = df_folds.pivot_table(
        index='fold',
        columns='iteration',
        values='f1_score',
        aggfunc='mean'
    )

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                cbar_kws={'label': 'F1-Score'})
    plt.title('F1-Score Heatmap: Fold vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "fold_f1_heatmap.png"), dpi=300)
    plt.close()

def plot_metric_distributions(stats_results, result_dir, raw_metrics=None):
    """
    Create box plots for metric distributions across multiple models.

    Args:
        stats_results: Dictionary with statistical results
        result_dir: Directory to save plots
        raw_metrics: Dictionary with raw metric values for each model
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if raw_metrics is None:
        print("Warning: No raw metrics provided for box plot. Skipping distribution plots.")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    metric_names = ['accuracy', 'f1_score', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        if metric_name in raw_metrics:
            values = raw_metrics[metric_name]

            # Create box plot
            axes[idx].boxplot(values, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2))

            # Add individual points
            x = np.ones(len(values))
            axes[idx].scatter(x, values, alpha=0.5, s=30, color='darkblue')

            # Add mean line
            mean_val = np.mean(values)
            axes[idx].axhline(y=mean_val, color='green', linestyle='--',
                              label=f'Mean: {mean_val:.4f}')

            # Customize plot
            axes[idx].set_ylabel(metric_label)
            axes[idx].set_xticklabels([''])
            axes[idx].set_title(f'{metric_label} Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "metric_distributions.png"), dpi=300)
    plt.close()

    # Create a combined violin plot
    plt.figure(figsize=(10, 6))

    # Prepare data for violin plot
    plot_data = []
    for metric_name, metric_label in zip(metric_names, metric_labels):
        if metric_name in raw_metrics:
            for value in raw_metrics[metric_name]:
                plot_data.append({'Metric': metric_label, 'Value': value})

    if plot_data:
        import pandas as pd
        df_plot = pd.DataFrame(plot_data)

        sns.violinplot(data=df_plot, x='Metric', y='Value', inner='box')
        plt.title('Distribution of Metrics Across Multiple Models')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "metric_violin_plots.png"), dpi=300)
        plt.close()