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
    HEATMAP_CMAP,
    ALG_NICE,
    NET_NICE
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
        """Carrega o CSV de testes estatísticos, aceitando formatos antigo e novo."""
        try:
            df = pd.read_csv(self.stat_tests_csv_path)

            # normaliza cabeçalhos
            df.columns = df.columns.str.strip().str.lower()

            # mapeia nomes alternativos ➜ nomes esperados
            rename_map = {
                "model_1": "model_a",
                "model_2": "model_b",
                "model_x": "model_a",
                "model_y": "model_b",
                "p_value": "p",
                "p_adj": "p",
                "p_corrected": "p",
            }
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
                      inplace=True)

            # garante colunas mínimas
            required = {"metric", "model_a", "model_b", "p"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Stat-tests CSV sem colunas obrigatórias: {missing}")

            # se não existir, cria coluna `significant` (α = 0.05)
            if "significant" not in df.columns:
                df["significant"] = df["p"] < 0.05

            # força booleano (lida com strings)
            df["significant"] = (
                df["significant"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
                .astype(bool)
            )

            self.stat_tests_df = df
            print(f"✅  Loaded statistical tests: {len(df)} rows")

        except Exception as err:
            print(f"⚠️  Could not load statistical tests: {err}")
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

    def _process_stat_tests(self, top_n_if_none: int = 50):
        """
        Cria quadro de vitórias/derrotas a partir do CSV.
        • Se houver linhas `significant == True` usa só elas (comportamento antigo).
        • Se NÃO houver, usa os `top_n_if_none` menores p-values corrigidos.
        """
        if self.stat_tests_df.empty:
            self.stat_summary = pd.DataFrame()
            return

        # ── 1) Escolhe subconjunto a usar ────────────────────────────────────────
        sig = self.stat_tests_df[self.stat_tests_df["significant"]]

        use_df = sig if not sig.empty else (
            self.stat_tests_df.sort_values("p").head(top_n_if_none)
            .assign(significant=False)  # marca como falso, só p/ compatibilidade
        )

        if use_df.empty:
            self.stat_summary = pd.DataFrame()
            return

        # ── 2) Mapeia métrica ➜ coluna de média ───────
        metric_to_col = {
            "precision":      "macro_avg_precision",
            "recall":         "macro_avg_recall",
            "f1_score":       "macro_avg_f1",
            "f1":             "macro_avg_f1",
            "macro_avg_f1":   "macro_avg_f1",
            "macro_f1":       "macro_avg_f1",
            "accuracy":       "accuracy",
        }

        wins = []
        for _, row in use_df.iterrows():
            metric_key = str(row["metric"]).lower()
            metric_col = metric_to_col.get(metric_key)
            if metric_col is None:
                continue

            col_mean = f"{metric_col}_mean" if f"{metric_col}_mean" in self.general_stats.columns else metric_col

            # resolve IDs (usa seu helper _convert_model_name + _resolve se já tiver)
            a_id = self._convert_model_name(row["model_a"])
            b_id = self._convert_model_name(row["model_b"])
            if a_id not in self.general_stats.index or b_id not in self.general_stats.index:
                continue

            score_a = self.general_stats.loc[a_id, col_mean]
            score_b = self.general_stats.loc[b_id, col_mean]

            winner, loser = (a_id, b_id) if score_a > score_b else (b_id, a_id)
            wins.append(dict(
                metric=metric_key,
                winner=winner,
                loser=loser,
                p_value=row["p"],
                significant=row["significant"],
                winner_score=max(score_a, score_b),
                loser_score=min(score_a, score_b),
            ))

        self.wins_df = pd.DataFrame(wins)
        if self.wins_df.empty:
            self.stat_summary = pd.DataFrame()
            return

        # ── 3) Agrega vitórias/derrotas como antes ───────────────────────────────
        w = self.wins_df.groupby(["winner", "metric"]).size().reset_index(name="wins")
        l = self.wins_df.groupby(["loser", "metric"]).size().reset_index(name="losses")
        w.rename(columns={"winner": "model"}, inplace=True)
        l.rename(columns={"loser": "model"}, inplace=True)

        models  = set(w["model"]).union(l["model"])
        metrics = self.wins_df["metric"].unique()

        summary = []
        for m in models:
            for met in metrics:
                wins_   = w.query("model == @m and metric == @met")["wins"].sum()
                losses_ = l.query("model == @m and metric == @met")["losses"].sum()
                summary.append(dict(
                    model=m, metric=met,
                    wins=wins_, losses=losses_, net_wins=wins_ - losses_
                ))
        self.stat_summary = pd.DataFrame(summary)

    @staticmethod
    def _pretty(model_id: str, flag=False) -> str:
        """
        Converte IDs canônicos em rótulos legíveis.
            Resnet_classifier_none                → ResNet
            Resnet_feature_extractor_xgboost_aug  → ResNet + XGBoost (Aug)
        """
        parts = model_id.split("_")

        net_raw = parts[0]
        net = NET_NICE.get(net_raw, net_raw)  # Resnet→ResNet, Vgg19→VGG19 …

        # CNN
        if "_classifier" in model_id:
            return net

        # Ensembles
        alg = parts[3] if len(parts) >= 4 else "?"
        alg_disp = ALG_NICE.get(alg, alg.title())
        aug_flag = "(AF)" if model_id.endswith("_aug") else "(N)"
        return f"{net} + {alg_disp} {aug_flag if flag else ''}"

    @staticmethod
    def _convert_model_name(model_name: str) -> str:
        """
        Converte o nome de modelo vindo do CSV para o formato canônico usado no código,
        lidando tanto com ‘classifier’ quanto com ‘classifier_none’, e com
        ‘feature_extractor_algoritmo’.
        """
        # já está no formato canônico?
        if model_name.endswith("_none") or "_feature_extractor_" in model_name:
            return model_name

        # formato legado: "<Net>_classifier"
        if model_name.endswith("_classifier"):
            net = model_name.split("_")[0]
            return f"{net}_classifier_none"

        # formato legado: "<Net>_feature_extractor_<algo>"
        if "feature_extractor" in model_name:
            parts = model_name.split("_")
            if len(parts) >= 3:
                net, _, alg = parts[0], parts[1], parts[-1]
                return f"{net}_feature_extractor_{alg}"

        # fallback
        return model_name

    @staticmethod
    def _create_model_id(row) -> str:
        """
        Cria ID único do modelo.

        • CNN pura .................: <net>_classifier
        • Ensemble sem augmentation : <net>_feature_extractor_<alg>_noaug
        • Ensemble com augmentation : <net>_feature_extractor_<alg>_aug
        """
        net = str(row['net']).strip()
        kind = str(row['kind']).strip()
        algorithm = str(row.get('algorithm', '')).strip()

        if kind == 'feature_extractor':
            aug_flag = 'aug' if bool(row.get('feature_augmentation', False)) else 'noaug'
            return f"{net}_{kind}_{algorithm}_{aug_flag}"

        # CNN classifier
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
        self._fig_05_alg_aug_vs_noaug()

        if not self.stat_summary.empty:
            print("Generating statistical analysis plots...")
            self._fig_06_pvalue_hist()
            self._fig_07_pvalue_heatmap()

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
                model_id = f"{net}_feature_extractor_{alg}_aug"
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
                            f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
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
                            f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([net.replace('Resnet', 'ResNet').replace('Vgg19', 'VGG19')
                            for net in networks], size=16)
        ax.set_ylabel("F1-Score", size=12)
        ax.set_title("CNN end-to-end vs Ensembles (com feature augmentation)")
        ax.legend(title="Arquitetura", fontsize=9, title_fontsize=9)

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
                model_id = f"{net}_feature_extractor_{alg}_aug"
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

        pivot_data = (
            self.per_class_stats
            .pivot_table(index='model_id',
                         columns='class_name',
                         values='f1_score_mean',
                         fill_value=0)
        )

        # Converte índice (model_id) → rótulo amigável
        pivot_data.index = [
            self._pretty(mid, True)
            for mid in pivot_data.index
        ]

        # Garante ordem consistente das classes
        for cname in CLASSES:
            if cname not in pivot_data.columns:
                pivot_data[cname] = 0.0
        pivot_data = pivot_data[CLASSES]

        fig, ax = plt.subplots(figsize=(14, max(8, len(pivot_data) * 0.4)))
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
                    vmin=0, vmax=1, linewidths=0.5,
                    annot_kws={"size": 12},
                    cbar_kws={"label": "F1-Score Médio"}, ax=ax)

        ax.set_title("F1-Score Médio por Classe")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Modelo")
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
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=11)

        ax.set_xlabel('CNN')
        ax.set_ylabel('Score')
        ax.set_title('Comparação das métricas das CNNs puras')
        ax.set_xticks(x)
        ax.set_xticklabels([net.replace('Resnet', 'ResNet').replace('Vgg19', 'VGG19') for net in networks])
        ax.legend(title="Métrica")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        self._save(fig, "04_cnn_metrics_comparison_enhanced.png")

    def _fig_05_alg_aug_vs_noaug(self):
        """
        4 sub-plots (Inception, ResNet, VGG19, Xception) – comparação NoAug × Aug
        para cada algoritmo (AdaBoost, ExtraTrees, RandomForest, XGBoost),
        com rótulos “valor ± std”.
        """
        nets = ["Inception", "Resnet", "Vgg19", "Xception"]
        algs = ["adaboost", "extratrees", "randomforest", "xgboost"]
        names = dict(adaboost="AdaBoost", extratrees="ExtraTrees",
                     randomforest="RandomForest", xgboost="XGBoost")

        def _val(mid):  # -> (mean, std)
            return self._get_metric_stats(mid, "macro_avg_f1")

        fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharey=True)
        axes = axes.flatten()
        w = 0.35
        col_no, col_au = "#fdae61", "#d7191c"

        for idx, net in enumerate(nets):
            ax = axes[idx]
            x = np.arange(len(algs))

            means_no, std_no, means_au, std_au = [], [], [], []
            for alg in algs:
                m_no, s_no = _val(f"{net}_feature_extractor_{alg}_noaug")
                m_au, s_au = _val(f"{net}_feature_extractor_{alg}_aug")
                means_no.append(m_no)
                std_no.append(s_no)
                means_au.append(m_au)
                std_au.append(s_au)

            bars_no = ax.bar(x - w / 2, means_no, w, yerr=std_no, capsize=3,
                             color=col_no, edgecolor="black", label="Sem aumento de atributos")
            bars_au = ax.bar(x + w / 2, means_au, w, yerr=std_au, capsize=3,
                             color=col_au, edgecolor="black", label="Com aumento de atributos")

            # Rótulos para "sem aumento de atributos"
            for bar, m, s in zip(bars_no, means_no, std_no):
                ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015,
                        f"{m:.2f}±{s:.4f}", ha="center", va="bottom",
                        fontsize=10, rotation=90)

            # Rótulos para "com aumento de atributos"
            for bar, m, s in zip(bars_au, means_au, std_au):
                ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015,
                        f"{m:.2f}±{s:.4f}", ha="center", va="bottom",
                        fontsize=10, rotation=90)

            ax.set_xticks(x)
            ax.set_xticklabels([names[a] for a in algs], rotation=15)
            ax.set_title(net.replace("Resnet", "ResNet").replace("Vgg19", "VGG19"))
            ax.grid(axis="y", alpha=.25)

            max_val = max(max(means_no), max(means_au))
            ax.set_ylim(0, max_val + 0.2)

            if idx == 0:
                ax.legend(fontsize=8)
            if idx == 2:
                ax.set_ylabel("F1-Score")

        fig.suptitle("Ensembles – Com x Sem Aumento de Atributos (por Rede)", fontsize=14)
        fig.tight_layout(rect=[0, .03, 1, .95])
        self._save(fig, "05_alg_aug_vs_noaug.png")

    def _fig_06_pvalue_hist(self, bins: int = 30):
        """Histograma dos p-values corrigidos (todas as comparações)."""
        if self.stat_tests_df.empty:
            return
        pvals = self.stat_tests_df["p"].values
        fig, ax = plt.subplots(figsize=(6, 4))
        counts, edges, _ = ax.hist(pvals, bins=bins, edgecolor="black", alpha=0.7)
        for c, e in zip(counts, edges[:-1]):
            if c > 0:
                ax.text(e + (edges[1] - edges[0]) / 2, c + .5, int(c),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("p (corrigido, Holm)")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição dos p-values (Holm-Bonferroni)")
        ax.axvline(0.05, color="red", ls="--", label="α = 0.05")
        ax.legend()
        self._save(fig, "06_pvalue_hist.png")

    def _fig_07_pvalue_heatmap(self, max_models: int = 100):
        """
        Heatmap (p corrigido) para os 'max_models' modelos mais frequentes.
        Mostra só metade superior da matriz para evitar duplicidade.
        """
        if self.stat_tests_df.empty:
            return
        # pega modelos que aparecem mais vezes (garante matrix quadrada sólida)
        counts = pd.concat([self.stat_tests_df["model_a"],
                            self.stat_tests_df["model_b"]]).value_counts()
        models = counts.head(max_models).index.tolist()
        # constrói matriz
        models_pretty = [self._pretty(m) for m in models]
        mat = pd.DataFrame(1.0, index=models_pretty, columns=models_pretty)

        for _, row in self.stat_tests_df.iterrows():
            a_raw, b_raw, p = row["model_a"], row["model_b"], row["p"]
            if a_raw in models and b_raw in models:
                a, b = self._pretty(a_raw), self._pretty(b_raw)
                mat.loc[a, b] = mat.loc[b, a] = p

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(mat, cmap="viridis_r", vmax=.25, vmin=0,
                    square=True, linewidths=.4, cbar_kws={"label": "p"})
        plt.xticks(rotation=40, ha="right")
        ax.set_title("Heatmap de p-values corrigidos")
        self._save(fig, "07_pvalue_heatmap.png")

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