"""Pipeline estatística completa
================================
* Entrada:
    - ``all_models_general.csv``   (não usado aqui ‑ testes precisam de repetições)
    - ``all_models_per_class.csv`` (7 classes = 7 repetições por modelo)
* Saída:
    - ``results/stat_tests_global.csv``   → 1 linha por métrica, 2 testes globais (ANOVA‑RM & Friedman)
    - ``results/stat_tests_pairs.csv``    → todas as combinações de modelos, p‑values corrigidos (Holm)

Testes implementados
--------------------
1. **AnovaRM** (paramétrico) – se todos os grupos passarem Shapiro (p>0.05)
2. **Friedman** (não‑param) – sempre calculado (robusto)
3. **Post‑hoc**
   * Se dados normais → *t‑test pareado* + Holm
   * Caso contrário   → *Wilcoxon signed‑rank* + Holm

Detalhes extras:
---------------
* Model ID = ``f"{net}_{kind}_{algorithm or 'none'}"``  – garante unicidade.
* Se qualquer grupo tiver variação zero, devolvemos estat=0, p=1 (sem falhar).
* Arquivos criados em ``<repo_root>/results``.
* Pode ser executado de qualquer pasta: ``python -m analysis.stat_tests``.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
import statsmodels.stats.multitest as smm
import scikit_posthocs as sp

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CSV_PER_CLASS = ROOT / "res" / "all_models_per_class.csv"
OUT_DIR = ROOT / "results"
OUT_GLOBAL = OUT_DIR / "stat_tests_global.csv"
OUT_PAIRS = OUT_DIR / "stat_tests_pairs.csv"
ALPHA = 0.05

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _numeric_metrics(df: pd.DataFrame) -> List[str]:
    """Return metric columns (float/int) excluding known metadata."""
    meta_cols = {
        "experiment",
        "net",
        "kind",
        "algorithm",
        "class",
        "scope",
    }
    return [c for c, dt in df.dtypes.items() if dt != "object" and c not in meta_cols]


def _build_model_id(row: pd.Series) -> str:
    alg = row.get("algorithm", "none") or "none"
    return f"{row['net']}_{row['kind']}_{alg}"


def _pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot seguro: média em caso de (classe, modelo) duplicado."""
    long = (
        df.assign(model_id=df.apply(_build_model_id, axis=1))[["class", "model_id", metric]]
        .dropna(subset=[metric])
        .astype({"class": str})
    )

    # ── agrega duplicates ──────────────────────────────────────────
    long = (
        long.groupby(["class", "model_id"], as_index=False)
            .mean(numeric_only=True)
    )

    pivot = long.pivot(index="class", columns="model_id", values=metric)

    # remove colunas sem variação
    pivot = pivot.loc[:, pivot.apply(lambda c: c.nunique() > 1)]
    return pivot


# ---------------------------------------------------------------------------
# CORE TESTS
# ---------------------------------------------------------------------------

def _shapiro_ok(arrays: List[np.ndarray]) -> bool:
    """True se *todos* grupos passaram Shapiro (p>0.05)."""
    for a in arrays:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Input data has range zero")
            _, p = st.shapiro(a)
        if p <= 0.05:
            return False
    return True


def _anova_rm(pivot: pd.DataFrame, metric: str):
    """Repeated‑measures ANOVA via statsmodels. Pode falhar se data singular."""
    df_long = (
        pivot.reset_index()
        .melt(id_vars="class", var_name="model", value_name=metric)
        .rename(columns={"class": "subject"})
    )
    try:
        aov = sm.stats.AnovaRM(df_long, depvar=metric, subject="subject", within=["model"]).fit()
        f_val = aov.anova_table["F Value"][0]
        p_val = aov.anova_table["Pr > F"][0]
    except Exception:  # singular matrix, etc.
        f_val, p_val = 0.0, 1.0
    return f_val, p_val


def _global_and_pairs(pivot: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (global_df, pairs_df) para uma métrica."""
    models = list(pivot.columns)
    arrays = [pivot[m].values for m in models]

    # ---------------- GLOBAL ----------------
    if _shapiro_ok(arrays):
        stat, p_anova = _anova_rm(pivot, metric)
        norm = True
    else:
        stat, p_anova = st.friedmanchisquare(*arrays)
        norm = False

    global_row = {
        "metric": metric,
        "test": "anova_rm" if norm else "friedman",
        "stat": stat,
        "p": p_anova,
    }
    global_df = pd.DataFrame([global_row])

    # ---------------- PAIRS ------------------
    records = []
    if p_anova <= ALPHA:  # só faz pós‑hoc se há diferença global
        combos = itertools.combinations(models, 2)
        for a, b in combos:
            if norm:
                stat_pair, p_pair = st.ttest_rel(pivot[a], pivot[b])
                test_name = "t_paired"
            else:
                stat_pair, p_pair = st.wilcoxon(pivot[a], pivot[b])
                test_name = "wilcoxon"
            records.append({
                "metric": metric,
                "test": test_name,
                "stat": stat_pair,
                "p_raw": p_pair,
                "model_a": a,
                "model_b": b,
            })
        if records:
            p_vals = [r["p_raw"] for r in records]
            _, p_adj, _, _ = smm.multipletests(p_vals, method="holm")
            for r, adj in zip(records, p_adj):
                r["p"] = adj
                r["significant"] = adj <= ALPHA
    pairs_df = pd.DataFrame(records)
    return global_df, pairs_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PER_CLASS)
    numeric_metrics = _numeric_metrics(df)

    globals_, pairs_ = [], []
    for m in numeric_metrics:
        pivot = _pivot(df, m)
        if pivot.shape[1] < 2:
            continue  # não há 2 modelos com variação
        g, p = _global_and_pairs(pivot, m)
        globals_.append(g)
        pairs_.append(p)

    pd.concat(globals_, ignore_index=True).to_csv(OUT_GLOBAL, index=False)
    pd.concat(pairs_, ignore_index=True).to_csv(OUT_PAIRS, index=False)
    print(
        f"✅ Estatísticas salvas: {OUT_GLOBAL.name} ({len(globals_)} métricas) | "
        f"{OUT_PAIRS.name} ({sum(len(x) for x in pairs_)} pares)"
    )


if __name__ == "__main__":
    main()