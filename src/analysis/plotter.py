from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import (
    COLOR_PALETTE,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    OUTPUT_DIR,
    CLASSES,
    HEATMAP_CMAP
)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
})


class Plotter:
    """
    Enhanced plotter that reads CSV data and calculates statistics across multiple model runs.

    This class reads from standardized CSV files containing model performance data
    and generates plots with error bars showing mean ± standard deviation.
    Now includes statistical testing analysis capabilities.
    """

    def __init__(self, general_csv_path: str, per_class_csv_path: str,
                 stat_tests_csv_path: str = None, out_dir: Path = OUTPUT_DIR):
        """
        Initialize the plotter with CSV file paths.

        Parameters
        ----------
        general_csv_path : str
            Path to the CSV file containing general model metrics
        per_class_csv_path : str
            Path to the CSV file containing per-class model metrics
        stat_tests_csv_path : str, optional
            Path to the CSV file containing statistical test results
        out_dir : Path
            Directory where plots will be saved
        """
        self.general_csv_path = general_csv_path
        self.per_class_csv_path = per_class_csv_path
        self.stat_tests_csv_path = stat_tests_csv_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Load and process the data
        self._load_data()
        self._process_data()

        # Load statistical tests if provided
        if stat_tests_csv_path:
            self._load_stat_tests()
            self._process_stat_tests()

    def _load_data(self):
        """Load CSV data into pandas DataFrames."""
        try:
            self.general_df = pd.read_csv(self.general_csv_path)
            self.per_class_df = pd.read_csv(self.per_class_csv_path)

            # Clean column names
            self.general_df.columns = self.general_df.columns.str.strip()
            self.per_class_df.columns = self.per_class_df.columns.str.strip()

            print(f"Loaded general data: {len(self.general_df)} rows")
            print(f"Loaded per-class data: {len(self.per_class_df)} rows")

        except Exception as e:
            raise ValueError(f"Error loading CSV files: {e}")

    def _load_stat_tests(self):
        """Load statistical tests CSV data."""
        try:
            self.stat_tests_df = pd.read_csv(self.stat_tests_csv_path)
            self.stat_tests_df.columns = self.stat_tests_df.columns.str.strip()

            # Convert string booleans to actual booleans
            self.stat_tests_df['significant'] = self.stat_tests_df['significant'].map({
                'True': True, 'true': True, True: True,
                'False': False, 'false': False, False: False
            })

            print(f"Loaded statistical tests data: {len(self.stat_tests_df)} rows")

        except Exception as e:
            print(f"Warning: Could not load statistical tests: {e}")
            self.stat_tests_df = pd.DataFrame()

    def _process_data(self):
        """Process the loaded data and calculate statistics."""
        # Create model identifiers
        self.general_df['model_id'] = self.general_df.apply(
            lambda row: self._create_model_id(row), axis=1
        )
        self.per_class_df['model_id'] = self.per_class_df.apply(
            lambda row: self._create_model_id(row), axis=1
        )

        # Calculate statistics for general metrics
        self.general_stats = self.general_df.groupby('model_id').agg({
            'accuracy': ['mean', 'std', 'count'],
            'macro_avg_precision': ['mean', 'std', 'count'],
            'macro_avg_recall': ['mean', 'std', 'count'],
            'macro_avg_f1': ['mean', 'std', 'count']
        }).round(4)

        # Flatten column names
        self.general_stats.columns = [f'{col[0]}_{col[1]}' for col in self.general_stats.columns]

        # Calculate statistics for per-class metrics
        self.per_class_stats = self.per_class_df.groupby(['model_id', 'class_name']).agg({
            'precision': ['mean', 'std', 'count'],
            'recall': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std', 'count']
        }).round(4)

        # Flatten column names for per-class stats
        self.per_class_stats.columns = [f'{col[0]}_{col[1]}' for col in self.per_class_stats.columns]
        self.per_class_stats = self.per_class_stats.reset_index()

    def _process_stat_tests(self):
        """Process statistical tests data and calculate wins/losses."""
        if self.stat_tests_df.empty:
            self.stat_wins = pd.DataFrame()
            return

        # Filter significant tests only
        significant_tests = self.stat_tests_df[self.stat_tests_df['significant'] == True].copy()

        if significant_tests.empty:
            self.stat_wins = pd.DataFrame()
            return

        # For each comparison, determine the winner based on which model performed better
        # We need to cross-reference with the general performance data
        wins_data = []

        for _, test_row in significant_tests.iterrows():
            model_a = test_row['model_a']
            model_b = test_row['model_b']
            metric = test_row['metric']

            # Map metric names to our column names
            metric_map = {
                'precision': 'macro_avg_precision',
                'recall': 'macro_avg_recall',
                'f1_score': 'macro_avg_f1'
            }

            if metric in metric_map:
                metric_col = metric_map[metric]

                # Get performance values (we'll use means from our processed data)
                # Convert model names to our format
                model_a_id = self._convert_model_name(model_a)
                model_b_id = self._convert_model_name(model_b)

                if model_a_id in self.general_stats.index and model_b_id in self.general_stats.index:
                    perf_a = self.general_stats.loc[model_a_id, f'{metric_col}_mean']
                    perf_b = self.general_stats.loc[model_b_id, f'{metric_col}_mean']

                    # Determine winner (higher value wins)
                    if perf_a > perf_b:
                        winner = model_a_id
                        loser = model_b_id
                    else:
                        winner = model_b_id
                        loser = model_a_id

                    wins_data.append({
                        'metric': metric,
                        'winner': winner,
                        'loser': loser,
                        'p_value': test_row['p'],
                        'winner_score': max(perf_a, perf_b),
                        'loser_score': min(perf_a, perf_b)
                    })

        # Create wins DataFrame
        self.wins_df = pd.DataFrame(wins_data)

        # Calculate win statistics
        if not self.wins_df.empty:
            self.stat_wins = self.wins_df.groupby(['winner', 'metric']).size().reset_index(name='wins')
            self.stat_losses = self.wins_df.groupby(['loser', 'metric']).size().reset_index(name='losses')
            self.stat_losses.rename(columns={'loser': 'model'}, inplace=True)
            self.stat_wins.rename(columns={'winner': 'model'}, inplace=True)

            # Merge wins and losses
            all_models = list(set(self.stat_wins['model'].tolist() + self.stat_losses['model'].tolist()))
            all_metrics = self.wins_df['metric'].unique()

            # Create complete summary
            summary_data = []
            for model in all_models:
                for metric in all_metrics:
                    wins = self.stat_wins[(self.stat_wins['model'] == model) &
                                          (self.stat_wins['metric'] == metric)]['wins'].sum()
                    losses = self.stat_losses[(self.stat_losses['model'] == model) &
                                              (self.stat_losses['metric'] == metric)]['losses'].sum()

                    summary_data.append({
                        'model': model,
                        'metric': metric,
                        'wins': wins,
                        'losses': losses,
                        'net_wins': wins - losses
                    })

            self.stat_summary = pd.DataFrame(summary_data)
        else:
            self.stat_summary = pd.DataFrame()

    @staticmethod
    def _convert_model_name(model_name: str) -> str:
        """Convert model name from statistical tests format to our internal format."""
        # Handle the conversion from stat tests format to our format
        if 'feature_extractor' in model_name:
            parts = model_name.split('_')
            if len(parts) >= 3:
                net = parts[0]
                algorithm = parts[-1]  # Last part is the algorithm
                return f"{net}_feature_extractor_{algorithm}"
        elif 'classifier' in model_name:
            parts = model_name.split('_')
            if len(parts) >= 2:
                net = parts[0]
                return f"{net}_classifier_none"

        return model_name

    @staticmethod
    def _create_model_id(row) -> str:
        """Create a unique model identifier from row data."""
        net = str(row['net']).strip()
        kind = str(row['kind']).strip()
        algorithm = str(row.get('algorithm', '')).strip()

        # Handle null/nan values properly
        if algorithm and algorithm.lower() not in ['nan', '', 'null']:
            return f"{net}_{kind}_{algorithm}"
        else:
            return f"{net}_{kind}_none"

    def _extract_model_metadata(self):
        """Extract model metadata for easier plotting."""
        self.cnn_models = {}
        self.ensemble_models = {}

        for model_id in self.general_stats.index:
            parts = model_id.split('_')
            net = parts[0]
            kind = parts[1]

            if kind == 'classifier':
                self.cnn_models[net] = model_id
            elif kind == 'feature' and len(parts) >= 4:  # feature_extractor_algorithm
                algorithm = parts[3]  # Now it's the 4th part
                if net not in self.ensemble_models:
                    self.ensemble_models[net] = {}
                self.ensemble_models[net][algorithm] = model_id

    def make_all_figures(self):
        """Generate all plots with enhanced statistics."""
        print("Generating enhanced plots with statistics...")

        self._fig_01_cnn_vs_ensembles()
        self._fig_02_best_per_ensemble()
        self._fig_03_f1_per_class_heatmap()
        self._fig_04_cnn_metric_bars()

        if hasattr(self, 'stat_summary') and not self.stat_summary.empty:
            print("Generating statistical analysis plots...")
            self._fig_05_statistical_wins_bars()
            self._generate_latex_table()

        print("All enhanced plots generated successfully!")

    def _get_metric_stats(self, model_id: str, metric: str) -> Tuple[float, float]:
        """Get mean and std for a metric of a specific model."""
        if model_id in self.general_stats.index:
            mean = self.general_stats.loc[model_id, f'{metric}_mean']
            std = self.general_stats.loc[model_id, f'{metric}_std']
            return mean, std
        return 0.0, 0.0

    def _fig_01_cnn_vs_ensembles(self):
        """Enhanced CNN vs Ensembles comparison with error bars."""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # Prepare data - match exact CSV structure
        networks = ['Inception', 'Resnet', 'Vgg19', 'Xception']  # Match CSV naming
        algorithms = ['adaboost', 'extratrees', 'randomforest', 'xgboost']

        bar_width = 0.15
        x_pos = np.arange(len(networks))

        # CNN scores (classifier + none algorithm)
        cnn_means = []
        cnn_stds = []
        for net in networks:
            model_id = f"{net}_classifier_none"  # Match the actual data structure
            mean, std = self._get_metric_stats(model_id, 'macro_avg_f1')
            cnn_means.append(mean)
            cnn_stds.append(std)

        # Ensemble scores (feature_extractor + specific algorithm)
        ensemble_data = {}
        for alg in algorithms:
            means = []
            stds = []
            for net in networks:
                model_id = f"{net}_feature_extractor_{alg}"  # Match the actual data structure
                mean, std = self._get_metric_stats(model_id, 'macro_avg_f1')
                means.append(mean)
                stds.append(std)
            ensemble_data[alg] = (means, stds)

        # Define colors matching your original constants
        colors = ['#4C72B0']  # Blue for CNN
        for alg in algorithms:
            if alg == 'randomforest':
                colors.append(COLOR_PALETTE.get('RandomForest', '#17becf'))
            elif alg == 'xgboost':
                colors.append(COLOR_PALETTE.get('XGBoost', '#bcbd22'))
            elif alg == 'adaboost':
                colors.append(COLOR_PALETTE.get('AdaBoost', '#ff9500'))
            elif alg == 'extratrees':
                colors.append(COLOR_PALETTE.get('ExtraTrees', '#e377c2'))
            else:
                colors.append('#666666')

        labels = ['CNN', 'RandomForest', 'XGBoost', 'AdaBoost', 'ExtraTrees']

        # Plot bars with error bars
        for i, (label, color) in enumerate(zip(labels, colors)):
            positions = x_pos + (i - 2) * bar_width

            if i == 0:  # CNN
                bars = ax.bar(positions, cnn_means, width=bar_width, label=label,
                              color=color, edgecolor='black', alpha=0.8,
                              yerr=cnn_stds, capsize=3)
                # Add value labels
                for bar, mean in zip(bars, cnn_means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            else:
                alg = algorithms[i - 1]
                means, stds = ensemble_data[alg]
                bars = ax.bar(positions, means, width=bar_width, label=label,
                              color=color, edgecolor='black', alpha=0.8,
                              yerr=stds, capsize=3)
                # Add value labels
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([net.replace('Resnet', 'ResNet').replace('Vgg19', 'VGG19')
                            for net in networks], size=16)
        ax.set_ylabel("F1-Score", size=10)
        ax.set_title("CNN end-to-end vs Ensembles")
        ax.legend(title="Arquitetura", fontsize=8, title_fontsize=9)

        # Calculate proper y-limit
        all_values = cnn_means + [val for means, _ in ensemble_data.values() for val in means]
        ax.set_ylim(0, max(all_values) * 1.15)
        ax.tick_params(labelsize=14)

        self._save(fig, "01_cnn_vs_ensembles_enhanced.png")

    def _fig_02_best_per_ensemble(self):
        """Best performing CNN for each ensemble algorithm."""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        algorithms = ['adaboost', 'extratrees', 'randomforest', 'xgboost']
        networks = ['Inception', 'Resnet', 'Vgg19', 'Xception']

        best_scores = []
        best_networks = []
        best_stds = []

        for alg in algorithms:
            best_score = 0
            best_net = ""
            best_std = 0

            for net in networks:
                model_id = f"{net}_feature_extractor_{alg}"
                mean, std = self._get_metric_stats(model_id, 'macro_avg_f1')
                if mean > best_score:
                    best_score = mean
                    best_net = net
                    best_std = std

            best_scores.append(best_score)
            best_networks.append(best_net)
            best_stds.append(best_std)

        # Map algorithm names to proper display names and colors
        display_names = ['AdaBoost', 'ExtraTrees', 'RandomForest', 'XGBoost']
        colors = [COLOR_PALETTE.get(name, '#666666') for name in display_names]

        bars = ax.bar(range(len(algorithms)), best_scores,
                      color=colors, edgecolor="black", alpha=0.8,
                      yerr=best_stds, capsize=5)

        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(display_names)
        ax.set_ylabel("F1-Score")
        ax.set_title("Melhor Desempenho por Algoritmo Ensemble")
        ax.set_ylim(0, max(best_scores) * 1.15)
        ax.grid(True, alpha=0.3)

        # Annotate with best CNN for each algorithm
        for i, (bar, net, score, std) in enumerate(zip(bars, best_networks, best_scores, best_stds)):
            # Network name below bar
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    net.replace('Resnet', 'ResNet').replace('Vgg19', 'VGG19'),
                    ha="center", va="bottom", fontsize=10, rotation=90, color='white', weight='bold')
            # Score above bar
            ax.text(bar.get_x() + bar.get_width() / 2, score + std + 0.01,
                    f'{score:.3f}±{std:.3f}', ha="center", va="bottom", fontsize=10)

        self._save(fig, "02_best_ensemble_per_algo_enhanced.png")

    def _fig_03_f1_per_class_heatmap(self):
        """Heatmap of F1-scores per class with mean values."""
        if self.per_class_stats.empty:
            print("No per-class data available for heatmap.")
            return

        # Create pivot table with mean F1 scores
        pivot_data = self.per_class_stats.pivot_table(
            index='model_id',
            columns='class_name',
            values='f1_score_mean',
            fill_value=0
        )

        # Ensure all classes are present
        for class_name in CLASSES:
            if class_name not in pivot_data.columns:
                pivot_data[class_name] = 0

        pivot_data = pivot_data[CLASSES]  # Reorder columns

        fig, ax = plt.subplots(figsize=(14, max(8, len(pivot_data) * 0.4)))
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
                    vmin=0, vmax=1, linewidths=0.5,
                    cbar_kws={"label": "Mean F1-Score"}, ax=ax)

        ax.set_title("Mean F1-Score per Class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Model Configuration")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        self._save(fig, "03_f1_class_heatmap_enhanced.png")

    def _fig_04_cnn_metric_bars(self):
        """Enhanced CNN metrics comparison with error bars."""
        networks = ['Inception', 'Resnet', 'Vgg19', 'Xception']
        metrics = ['macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']
        metric_labels = ['Precisão', 'Recall', 'F1-Score']

        # Prepare data
        data = {}
        for metric in metrics:
            means = []
            stds = []
            for net in networks:
                model_id = f"{net}_classifier_none"
                mean, std = self._get_metric_stats(model_id, metric)
                means.append(mean)
                stds.append(std)
            data[metric] = (means, stds)

        # Create grouped bar chart
        x = np.arange(len(networks))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each metric

        for i, (metric, (means, stds)) in enumerate(data.items()):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, means, width, label=metric_labels[i],
                         color=colors[i], alpha=0.8, yerr=stds, capsize=3)

            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('CNN')
        ax.set_ylabel('Score')
        ax.set_title('Comparação das métricas das CNNs puras')
        ax.set_xticks(x)
        ax.set_xticklabels([net.replace('Resnet', 'ResNet').replace('Vgg19', 'VGG19') for net in networks])
        ax.legend(title="Métrica")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        self._save(fig, "04_cnn_metrics_comparison_enhanced.png")

    def _fig_05_statistical_wins_bars(self):
        """Bar chart showing statistical wins per model."""
        if not hasattr(self, 'stat_summary') or self.stat_summary.empty:
            print("No statistical data available for wins chart.")
            return

        # Calculate total wins per model across all metrics
        total_wins = self.stat_summary.groupby('model')['wins'].sum().sort_values(ascending=False)
        total_losses = self.stat_summary.groupby('model')['losses'].sum()
        total_net_wins = self.stat_summary.groupby('model')['net_wins'].sum().sort_values(ascending=False)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Total wins
        models = total_wins.index
        wins = total_wins.values
        losses_vals = [total_losses.get(model, 0) for model in models]

        x_pos = np.arange(len(models))
        width = 0.35

        bars1 = ax1.bar(x_pos - width / 2, wins, width, label='Vitórias', color='green', alpha=0.7)
        bars2 = ax1.bar(x_pos + width / 2, losses_vals, width, label='Derrotas', color='red', alpha=0.7)

        ax1.set_xlabel('Modelo')
        ax1.set_ylabel('Número de Comparações Significativas')
        ax1.set_title('Vitórias vs Derrotas Estatisticamente Significativas')
        ax1.set_xticks(x_pos)

        # Format model names for display
        display_names = [self._format_model_name_for_display(model) for model in models]
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.annotate(f'{int(height)}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=9)

        # Plot 2: Net wins (wins - losses)
        models_net = total_net_wins.index
        net_wins = total_net_wins.values
        colors = ['green' if x >= 0 else 'red' for x in net_wins]

        bars3 = ax2.bar(range(len(models_net)), net_wins, color=colors, alpha=0.7)
        ax2.set_xlabel('Modelo')
        ax2.set_ylabel('Vitórias Líquidas (Vitórias - Derrotas)')
        ax2.set_title('Ranking por Vitórias Líquidas')
        ax2.set_xticks(range(len(models_net)))

        display_names_net = [self._format_model_name_for_display(model) for model in models_net]
        ax2.set_xticklabels(display_names_net, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3 if height >= 0 else -15),
                         textcoords="offset points",
                         ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        self._save(fig, "05_statistical_wins_bars.png")

    def _generate_latex_table(self):
        """Generate LaTeX table in SBC format for top performing models."""
        if not hasattr(self, 'stat_summary') or self.stat_summary.empty:
            print("No statistical data available for LaTeX table.")
            return

        # Calculate ranking based on total net wins
        model_ranking = self.stat_summary.groupby('model').agg({
            'wins': 'sum',
            'losses': 'sum',
            'net_wins': 'sum'
        }).sort_values('net_wins', ascending=False)

        top_models = model_ranking.head(10)

        latex_content = [
            "\\begin{table}[htb]",
            "\\centering",
            "\\caption{Ranking dos Modelos por Desempenho Estatístico}",
            "\\label{tab:statistical_ranking}",
            "\\begin{tabular}{|l|c|c|c|c|c|}",
            "\\hline",
            "\\textbf{Modelo} & \\textbf{Vitórias} & \\textbf{Derrotas} & \\textbf{Saldo} & \\textbf{F1-Score} & \\textbf{Acurácia} \\\\",
            "\\hline"
        ]

        for idx, (model, stats) in enumerate(top_models.iterrows()):
            # Format model name
            display_name = self._format_model_name_for_display(model)

            # Get performance metrics
            if model in self.general_stats.index:
                f1_mean = self.general_stats.loc[model, 'macro_avg_f1_mean']
                f1_std = self.general_stats.loc[model, 'macro_avg_f1_std']
                acc_mean = self.general_stats.loc[model, 'accuracy_mean']
                acc_std = self.general_stats.loc[model, 'accuracy_std']

                f1_str = f"{f1_mean:.3f}±{f1_std:.3f}"
                acc_str = f"{acc_mean:.3f}±{acc_std:.3f}"
            else:
                f1_str = "N/A"
                acc_str = "N/A"

            latex_content.append(
                f"{display_name} & {int(stats['wins'])} & {int(stats['losses'])} & "
                f"{int(stats['net_wins'])} & {f1_str} & {acc_str} \\\\"
            )

            if idx < len(top_models) - 1:
                latex_content.append("\\hline")

        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")

        # Save to file
        latex_file = self.out_dir / "statistical_ranking_table.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_content))

        print(f"✅ LaTeX table saved: {latex_file}")

    @staticmethod
    def _format_model_name_for_display(model_name: str) -> str:
        """Format model name for nice display in plots and tables."""
        # Convert internal format to display format
        parts = model_name.split('_')

        if len(parts) >= 2:
            net = parts[0]
            kind = parts[1]

            # Format network name
            net_display = net.replace('Resnet', 'ResNet').replace('Vgg19', 'VGG19')

            if kind == 'classifier' and len(parts) >= 3 and parts[2] == 'none':
                return f"{net_display}"
            elif kind == 'feature' and len(parts) >= 4:
                algorithm = parts[3]
                alg_display = {
                    'randomforest': 'RandomForest',
                    'xgboost': 'XGBoost',
                    'adaboost': 'AdaBoost',
                    'extratrees': 'ExtraTrees'
                }.get(algorithm, algorithm.title())
                return f"{net_display}+{alg_display}"

        return model_name

    def _save(self, fig: plt.Figure, filename: str):
        """Save figure to output directory."""
        output_path = self.out_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ figure saved: {filename}")


# Usage example
def create_plots(general_csv_path: str, per_class_csv_path: str,
                 stat_tests_csv_path: str = None, output_dir: str = "../../figures"):
    """
    Create all enhanced plots from CSV files.

    Parameters
    ----------
    general_csv_path : str
        Path to all_models_general.csv
    per_class_csv_path : str
        Path to all_models_per_class.csv
    stat_tests_csv_path : str
        Path to stat_tests.csv
    output_dir : str
        Directory to save plots
    """
    plotter = Plotter(general_csv_path, per_class_csv_path, stat_tests_csv_path, Path(output_dir))
    plotter.make_all_figures()
    return plotter


if __name__ == "__main__":
    create_plots(
        general_csv_path="D:\\PIBIC\\python\\skincancer\\skincancer\\res\\all_models_general.csv",
        per_class_csv_path="D:\\PIBIC\\python\\skincancer\\skincancer\\res\\all_models_per_class.csv",
        stat_tests_csv_path="D:\\PIBIC\\python\\skincancer\\skincancer\\res\\stat_tests_pairs.csv",
    )