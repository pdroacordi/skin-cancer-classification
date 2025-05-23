"""Matplotlib wrappers that create every figure requested by the user.

This **rewrites** the previous implementation adding seven new plots that match the
latest specification without breaking backwards‑compatibility.  Existing helper
methods were kept and extended so *cli.py* continues to work transparently.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .constants import (
    CNN_MODELS,
    ML_CLASSIFIERS,
    COLOR_PALETTE,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    OUTPUT_DIR,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
})


class Plotter:
    """High‑level API that receives two pandas DataFrames (train, test) and writes
    *all* required figures to *out_dir*.

    Parameters
    ----------
    train_df, test_df
        DataFrames produced by :pyclass:`ResultsCollector`, **already** containing
        one row per (kind, network, classifier) with *aggregated* metrics.
    out_dir
        Where PNGs will be saved.  Defaults to the constant *OUTPUT_DIR* but the
        CLI wrapper allows overriding via command‑line.
    """

    # ---------------------------------------------------------------------
    # constructor & public helpers
    # ---------------------------------------------------------------------
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path = OUTPUT_DIR):
        self.train = train_df.copy()
        self.test = test_df.copy()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # derived – stores a single df for convenience
        self._combined = pd.concat([
            self.train.assign(split="train"),
            self.test.assign(split="test"),
        ], ignore_index=True)

    # The entry‑point used by *cli.py* – make sure to keep backwards compat.
    def make_all_figures(self) -> None:  # noqa: C901 (many small private methods)
        """Generate **every** chart needed for the *Results & Discussion* section.

        New figures were added at the *top* so legacy numbering stays intact and
        collaborators using the old file names won't be affected.
        """
        self._fig_01_cnn_vs_ensembles()
        self._fig_02_train_vs_test_split()
        self._fig_03_best_per_ensemble()
        self._fig_04_metrics_cnn_radar()
        self._fig_05_metrics_ensemble_per_cnn()
        self._fig_06_heatmap_f1()
        self._fig_07_boxplot_cv()

    # ------------------------------------------------------------------
    # Figure 1 – Geral CNN + Ensembles
    # ------------------------------------------------------------------
    def _fig_01_cnn_vs_ensembles(self) -> None:
        """Grouped bar chart (F1‑Score Test) – each CNN plus four ensembles."""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        bar_width = 0.15
        x_pos = list(range(len(CNN_MODELS)))

        bars: List[list[float]] = []
        labels = ["CNN"] + ML_CLASSIFIERS
        for lbl in labels:
            scores = []
            for net in CNN_MODELS:
                if lbl == "CNN":
                    scores.append(self._get_metric(net, "CNN", train=False, metric="f1_score"))
                else:
                    scores.append(self._get_metric(net, lbl, train=False, metric="f1_score"))
            bars.append(scores)

        # plot
        for i, (lbl, scores) in enumerate(zip(labels, bars)):
            positions = [p + (i - 2) * bar_width for p in x_pos]  # centre around group
            color = COLOR_PALETTE.get(lbl if lbl != "CNN" else CNN_MODELS[i % len(CNN_MODELS)], "#666666")
            # override colours for clarity
            if lbl == "CNN":
                color = "#4C72B0"
            ax.bar(positions, scores, width=bar_width, label=lbl, color=color, edgecolor="black")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(CNN_MODELS)
        ax.set_ylabel("F1‑Score (Teste)")
        ax.set_title("Comparativo Geral – CNN end‑to‑end vs +Ensembles")
        ax.legend(title="Arquitetura")
        ax.set_ylim(0, max(max(b) for b in bars) * 1.15)

        self._annotate_bars(ax)
        self._save(fig, "01_cnn_vs_ensembles.png")

    # ------------------------------------------------------------------
    # Figure 2 – Train vs Test split (CNNs *and* Ensembles)
    # ------------------------------------------------------------------
    def _fig_02_train_vs_test_split(self) -> None:
        """Two side‑by‑side grouped bar charts comparing F1 train vs test."""

        fig, axes = plt.subplots(1, 2, figsize=(DEFAULT_FIGSIZE[0] * 1.2, DEFAULT_FIGSIZE[1]))

        # --- (a) CNNs end‑to‑end ------------------------------------------------
        width = 0.35
        x = list(range(len(CNN_MODELS)))
        train_scores = [self._get_metric(net, "CNN", train=True) for net in CNN_MODELS]
        test_scores = [self._get_metric(net, "CNN", train=False) for net in CNN_MODELS]

        axes[0].bar([i - width/2 for i in x], train_scores, width=width, label="Treino", color=COLOR_PALETTE["train"], alpha=0.7)
        axes[0].bar([i + width/2 for i in x], test_scores,  width=width, label="Teste",  color=COLOR_PALETTE["test"],  alpha=0.9, edgecolor="black")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(CNN_MODELS, rotation=15)
        axes[0].set_title("CNN – Treino vs Teste")
        axes[0].set_ylabel("F1‑Score")
        axes[0].set_ylim(0, max(train_scores + test_scores) * 1.15)
        self._annotate_bars(axes[0])

        # --- (b) FE + Ensembles -------------------------------------------------
        all_labels = [f"{net}+{clf[:2]}" for net in CNN_MODELS for clf in ML_CLASSIFIERS]
        bar_x = list(range(len(all_labels)))
        ens_train = []
        ens_test = []
        for net in CNN_MODELS:
            for clf in ML_CLASSIFIERS:
                ens_train.append(self._get_metric(net, clf, train=True))
                ens_test.append(self._get_metric(net, clf, train=False))
        width_e = 0.4
        axes[1].bar([i - width_e/2 for i in bar_x], ens_train, width=width_e, label="Treino", color=COLOR_PALETTE["train"], alpha=0.7)
        axes[1].bar([i + width_e/2 for i in bar_x], ens_test,  width=width_e, label="Teste",  color=COLOR_PALETTE["test"],  alpha=0.9, edgecolor="black")
        axes[1].set_xticks(bar_x)
        axes[1].set_xticklabels(all_labels, rotation=90)
        axes[1].set_title("FE+Ensemble – Treino vs Teste")
        axes[1].set_ylim(0, max(ens_train + ens_test) * 1.15)
        axes[1].legend()
        self._annotate_bars(axes[1])

        fig.suptitle("Comparação F1‑Score – Treino x Teste", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save(fig, "02_train_vs_test_split.png")

    # ------------------------------------------------------------------
    # Figure 3 – Melhor ensemble por algoritmo
    # ------------------------------------------------------------------
    def _fig_03_best_per_ensemble(self) -> None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        best_scores: List[float] = []
        cnn_used: List[str] = []
        for clf in ML_CLASSIFIERS:
            subset = self.test[self.test["classifier"] == clf]
            if subset.empty:
                best_scores.append(0.0)
                cnn_used.append("-")
                continue
            best_row = subset.loc[subset["f1_score"].idxmax()]
            best_scores.append(best_row["f1_score"])
            cnn_used.append(best_row["network"])

        bars = ax.bar(ML_CLASSIFIERS, best_scores, color=[COLOR_PALETTE[c] for c in ML_CLASSIFIERS], edgecolor="black")
        ax.set_ylabel("F1‑Score (Teste)")
        ax.set_title("Melhor Desempenho por Algoritmo Ensemble")
        ax.set_ylim(0, max(best_scores) * 1.15)
        self._annotate_bars(ax)

        # annotate CNN extractor beneath each bar
        for bar, cnn in zip(bars, cnn_used):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.05, cnn, ha="center", va="bottom", fontsize=10, color="#333333", rotation=90)

        self._save(fig, "03_best_ensemble_per_algo.png")

    # ------------------------------------------------------------------
    # Figure 4 – Radar metrics (Accuracy, Precision, Recall, F1) per CNN
    # ------------------------------------------------------------------
    def _fig_04_metrics_cnn_radar(self) -> None:
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        angles = [n / float(len(metrics)) * 2 * 3.14159265359 for n in range(len(metrics))]
        angles += angles[:1]  # close loop

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

        for net in CNN_MODELS:
            values = [self._get_metric(net, "CNN", train=False, metric=m) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, label=net, linewidth=2)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_title("Métricas CNN end‑to‑end (Teste)")
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        self._save(fig, "04_cnn_radar_metrics.png")

    # ------------------------------------------------------------------
    # Figure 5 – Radar metrics for ensembles per CNN
    # ------------------------------------------------------------------
    def _fig_05_metrics_ensemble_per_cnn(self) -> None:
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        angles = [n / float(len(metrics)) * 2 * 3.14159265359 for n in range(len(metrics))]
        angles += angles[:1]

        for net in CNN_MODELS:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
            for clf in ML_CLASSIFIERS:
                values = [self._get_metric(net, clf, train=False, metric=m) for m in metrics]
                values += values[:1]
                ax.plot(angles, values, label=clf, linewidth=2)
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.capitalize() for m in metrics])
            ax.set_title(f"{net} – Métricas Ensemble (Teste)")
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 1.0)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            self._save(fig, f"05_{net.lower()}_ensemble_radar.png")

    # ------------------------------------------------------------------
    # Figure 6 – Heatmap F1‑Score (CNN x Ensembles)
    # ------------------------------------------------------------------
    def _fig_06_heatmap_f1(self) -> None:
        pivot = self.test[self.test["kind"] == "FE"].pivot_table(
            index="network", columns="classifier", values="f1_score")
        # ensure ordering & any missing values are shown as 0
        pivot = pivot.reindex(index=CNN_MODELS, columns=ML_CLASSIFIERS).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "F1‑Score"}, ax=ax, linewidths=0.5)
        ax.set_title("Heatmap – F1‑Score (CNN extractora x Algoritmo Ensemble)")
        ax.set_xlabel("Algoritmo Ensemble")
        ax.set_ylabel("CNN Extractora")
        self._save(fig, "06_heatmap_f1.png")

    # ------------------------------------------------------------------
    # Figure 7 – Boxplot CV (10 runs – 5x2)
    # ------------------------------------------------------------------
    def _fig_07_boxplot_cv(self) -> None:
        """Reads every *iteration_results.txt* under ./results and plots F1 distribution."""
        cv_records = []
        # naive but robust glob search – the files are light so speed is fine
        pattern = r"F1[\s\-]Score:\s*([0-9]*\.?[0-9]+)"
        f1_re = re.compile(pattern, flags=re.IGNORECASE)

        for file in Path(self.out_dir).parent.glob("results/**/iteration_results.txt"):
            try:
                content = file.read_text(encoding="utf-8", errors="ignore")
            except FileNotFoundError:
                continue
            # detect which network/cls this file refers to from its path
            parts = file.parts
            network = next((n for n in CNN_MODELS if n.lower() in " ".join(parts).lower()), "Unknown")
            classifier = "CNN" if "cnn_classifier" in file.as_posix() else next((c for c in ML_CLASSIFIERS if c.lower() in " ".join(parts).lower()), "Unknown")
            kind = "CNN" if classifier == "CNN" else "FE"

            for match in f1_re.finditer(content):
                try:
                    f1_val = float(match.group(1))
                except (ValueError, IndexError):
                    continue
                cv_records.append({
                    "kind": kind,
                    "network": network,
                    "classifier": classifier,
                    "f1": f1_val,
                    "label": f"{network}+{classifier}" if classifier != "CNN" else network,
                })

        if not cv_records:  # safety guard
            print("⚠️  Nenhum *iteration_results.txt* encontrado – Boxplot será pulado.")
            return

        cv_df = pd.DataFrame(cv_records)
        # order labels consistently
        label_order = [cnn for cnn in CNN_MODELS] + [f"{net}+{clf}" for net in CNN_MODELS for clf in ML_CLASSIFIERS]
        cv_df["label"] = pd.Categorical(cv_df["label"], categories=label_order, ordered=True)

        fig, ax = plt.subplots(figsize=(max(14, len(label_order) * 0.6), 8))
        sns.boxplot(data=cv_df, x="label", y="f1", palette="Set3", ax=ax)
        ax.set_title("Distribuição F1 – Validação Cruzada (5x2) 10 execuções")
        ax.set_xlabel("Combinação CNN + Classificador")
        ax.set_ylabel("F1‑Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        self._save(fig, "07_boxplot_cv.png")

    # ------------------------------------------------------------------
    # helpers – identical to old code so downstream remains unchanged
    # ------------------------------------------------------------------
    def _get_metric(self, network: str, classifier: str, *, train: bool, metric: str = "f1_score") -> float:
        """Return *metric* (defaults to *f1_score*) for the given combination."""
        df = self.train if train else self.test
        row = df[(df["network"] == network) & (df["classifier"] == classifier)]
        if row.empty:
            return 0.0
        value = float(row.iloc[0][metric])
        # guard‑rail: metric should be between 0 and 1
        return max(0.0, min(1.0, value))

    @staticmethod
    def _annotate_bars(ax: plt.Axes) -> None:
        for bar in ax.patches:
            if isinstance(bar.get_height(), (int, float)):
                height = bar.get_height()
            else:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    def _save(self, fig: plt.Figure, filename: str) -> None:
        out_path = (self.out_dir / filename).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        try:
            printable = out_path.relative_to(Path.cwd())
        except ValueError:
            printable = out_path
        print(f"✅ Figura salva em {printable}")
