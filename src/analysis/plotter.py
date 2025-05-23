"""Matplotlib wrappers that create exactly the figures the paper calls for."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .constants import (
    COLOR_PALETTE,
    CNN_MODELS,
    ML_CLASSIFIERS,
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
    """Holds both the data and the convenience methods to draw each figure."""

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path = OUTPUT_DIR):
        self.train = train_df
        self.test = test_df
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Public helpers (called from CLI)
    # ------------------------------------------------------------------
    def make_all_figures(self) -> None:
        self._fig_train_vs_test()
        self._fig_cnn_comparison()
        self._fig_ml_comparison()
        self._fig_algo_vs_cnn_per_network()

    # ------------------------------------------------------------------
    # Individual figures
    # ------------------------------------------------------------------
    def _fig_train_vs_test(self) -> None:
        """Side‑by‑side bars: each CNN – F1 train vs F1 test."""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        width = 0.35
        x = list(range(len(CNN_MODELS)))
        train_scores = [self._get_f1("CNN", net, "CNN", train=True) for net in CNN_MODELS]
        test_scores = [self._get_f1("CNN", net, "CNN", train=False) for net in CNN_MODELS]
        ax.bar([i - width/2 for i in x], train_scores, width=width, label="Treino (CV)", color=COLOR_PALETTE["train"], alpha=0.7)
        ax.bar([i + width/2 for i in x], test_scores,  width=width, label="Teste (Real)",  color=COLOR_PALETTE["test"],  alpha=0.9, edgecolor="black", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(CNN_MODELS)
        ax.set_ylabel("F1‑Score")
        ax.set_title("CNN end‑to‑end – Treino vs Teste")
        ax.legend()
        ax.set_ylim(0, max(train_scores + test_scores) * 1.15)
        self._annotate_bars(ax)
        self._save(fig, "train_vs_test.png")

    def _fig_cnn_comparison(self) -> None:
        """Simply compare the *test* F1 of every CNN backbone."""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        scores = [self._get_f1("CNN", net, "CNN", train=False) for net in CNN_MODELS]
        ax.barh(CNN_MODELS, scores, color=[COLOR_PALETTE[n] for n in CNN_MODELS], alpha=0.9)
        ax.set_xlabel("F1‑Score (Teste)")
        ax.set_title("Redes Convolucionais – Comparação direta (Teste)")
        self._annotate_bars(ax, horizontal=True)
        self._save(fig, "cnn_comparison.png")

    def _fig_ml_comparison(self) -> None:
        """Compare the *best* test score of each classical ML algorithm across all extractors."""
        best_scores = []
        for clf in ML_CLASSIFIERS:
            subset = self.test[self.test["classifier"] == clf]
            best_scores.append(subset["f1_score"].max() if not subset.empty else 0.0)
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.barh(ML_CLASSIFIERS, best_scores, color=[COLOR_PALETTE[c] for c in ML_CLASSIFIERS], alpha=0.9)
        ax.set_xlabel("Melhor F1‑Score (Teste)")
        ax.set_title("Algoritmos Clássicos – Melhor caso por algoritmo")
        self._annotate_bars(ax, horizontal=True)
        self._save(fig, "ml_comparison.png")

    def _fig_algo_vs_cnn_per_network(self) -> None:
        """For every backbone show how each classical algorithm stacks against the *same* backbone used as a classifier."""
        for net in CNN_MODELS:
            labels = ["CNN"] + ML_CLASSIFIERS
            scores = [self._get_f1("CNN", net, "CNN", train=False)]
            for clf in ML_CLASSIFIERS:
                scores.append(self._get_f1("FE", net, clf, train=False))
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
            ax.bar(labels, scores, color=[COLOR_PALETTE.get(l, "#999999") for l in labels], alpha=0.9, edgecolor="black")
            ax.set_ylabel("F1‑Score (Teste)")
            ax.set_title(f"{net}: CNN end‑to‑end vs FE + Clássicos (Teste)")
            self._annotate_bars(ax)
            self._save(fig, f"{net.lower()}_cnn_vs_algos.png")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_f1(self, kind: str, net: str, clf: str, train: bool) -> float:
        df = self.train if train else self.test
        row = df[(df["kind"] == kind) & (df["network"] == net) & (df["classifier"] == clf)]
        if row.empty:
            return 0.0
        return float(row.iloc[0]["f1_score"])

    @staticmethod
    def _annotate_bars(ax: plt.Axes, horizontal: bool = False) -> None:
        if horizontal:
            for bar in ax.patches:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.3f}", va="center", fontsize=10, fontweight="bold")
        else:
            for bar in ax.patches:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")

    def _save(self, fig: plt.Figure, filename: str) -> None:
        """Salva a figura e imprime o path de forma robusta."""
        out_path = (self.out_dir / filename).resolve()  # 1️⃣ usa self.out_dir e resolve()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        # 2️⃣ imprime caminho curto se pertencer ao CWD, senão absoluto
        try:
            printable = out_path.relative_to(Path.cwd())
        except ValueError:
            printable = out_path
        print(f"✅ Figura salva em {printable}")