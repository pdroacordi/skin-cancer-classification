"""Executa testes estatísticos entre modelos.

Fluxo:
1. Carrega `all_models_aggregated.csv` gerado por `aggregate_results.py`.
2. Para cada métrica numérica (ex.: accuracy, f1, recall, ...):
   • Constrói um DataFrame "wide" (index = amostras; colunas = modelos).
     A amostra é definida por:
         - se a linha tem `class_label` → usa cada classe como repetição;
         - caso contrário, descarta (não há replicação suficiente).
   • Teste global para ≥3 modelos:
        - **Normalidade**: Shapiro–Wilk em cada coluna  (α = 0.05).
        - **Homogeneidade**: Levene (α = 0.05).
        - Se ambos os requisitos OK → **ANOVA de medidas repetidas (AnovaRM)**.
        - Caso contrário → **Friedman** (não‑paramétrico, repetidas).
   • Se só existirem 2 modelos →
        - Mesma checagem de premissas.
        - Se normal + homocedástico → **t‑teste pareado**.
        - Senão → **Wilcoxon signed‑rank**.
   • Pós‑hoc (se modelos ≥3 & p_global < 0.05):
        - Se ANOVA → t‑teste pareado com correção Holm.
        - Se Friedman → Wilcoxon pareado com correção Holm.
3. Os resultados são gravados em `results/model_stats.csv` com duas seções:
     A. Linhas `test = global` → estatística e p‑value do teste global por métrica.
     B. Linhas `test = pairwise` → comparação modelo A × B.
"""

from pathlib import Path
import itertools
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, friedmanchisquare, wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM

BASE = Path(__file__).resolve().parents[1]
CSV_PATH = BASE / "results" / "all_models_aggregated.csv"
OUTPUT_PATH = BASE / "results" / "model_stats.csv"
ALPHA = 0.05

META_COLS = [
    "experiment",
    "net",
    "kind",  # classifier | feature_extractor
    "class_label",  # NaN se métrica geral
    "data_augmentation",
    "feature_augmentation",
    "hair_removal",
    "segmentation",
]

def load_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Retorna DF wide com linhas=amostras, colunas=modelos para a métrica."""
    subset = (
        df[df["class_label"].notna()]  # só métricas por classe ⇒ repetição suficiente
        .pivot_table(
            index="class_label",
            columns=["net", "kind"],  # coluna multi‑índice para unicidade
            values=metric,
        )
    )
    # drop models com NaN em qualquer classe (amostra faltante)
    subset = subset.dropna(axis=1, how="any")
    return subset

def check_normality(df_wide: pd.DataFrame) -> bool:
    """True se TODAS colunas passam Shapiro (p>α)."""
    return all(shapiro(col).pvalue > ALPHA for col in df_wide.T.values)

def check_homoscedastic(df_wide: pd.DataFrame) -> bool:
    """True se passa Levene (p>α)."""
    stat, p = levene(*[df_wide[c].values for c in df_wide.columns])
    return p > ALPHA

def run_global_test(df_wide: pd.DataFrame, normal: bool, homo: bool):
    n_models = df_wide.shape[1]
    # Friedman / ANOVA / t‑test / Wilcoxon
    if n_models == 2:
        a, b = df_wide.columns
        if normal and homo:
            stat, p = ttest_rel(df_wide[a], df_wide[b])
            test_name = "paired_t"
        else:
            stat, p = wilcoxon(df_wide[a], df_wide[b])
            test_name = "wilcoxon"
        return test_name, stat, p
    else:
        if normal and homo:
            # Repeated‑measures ANOVA
            long = df_wide.melt(ignore_index=False, var_name=["net", "kind"], value_name="val").reset_index()
            long["subject"] = long["class_label"]  # repeated factor
            aov = AnovaRM(long, depvar="val", subject="subject", within=["net", "kind"]).fit()
            stat = aov.anova_table.loc["net:kind", "F Value"]
            p = aov.anova_table.loc["net:kind", "Pr > F"]
            test_name = "rm_anova"
        else:
            stat, p = friedmanchisquare(*[df_wide[c] for c in df_wide.columns])
            test_name = "friedman"
        return test_name, stat, p

def pairwise(df_wide: pd.DataFrame, normal: bool):
    """Pairwise comparação com correção Holm; retorna list(dict)."""
    comps = []
    combos = list(itertools.combinations(df_wide.columns, 2))
    stats = []
    pvals = []
    for a, b in combos:
        if normal:
            stat, p = ttest_rel(df_wide[a], df_wide[b])
            test = "paired_t"
        else:
            stat, p = wilcoxon(df_wide[a], df_wide[b])
            test = "wilcoxon"
        stats.append(stat)
        pvals.append(p)
        comps.append((a, b, test))
    # Holm correction
    reject, p_corr, _, _ = multipletests(pvals, method="holm", alpha=ALPHA)
    results = []
    for (a, b, test), stat, p_raw, p_adj, rej in zip(comps, stats, pvals, p_corr, reject):
        results.append({
            "test": "pairwise",
            "metric": current_metric,
            "model_a": "_".join(a),
            "model_b": "_".join(b),
            "test_name": test,
            "statistic": stat,
            "p_value": p_raw,
            "p_adj": p_adj,
            "significant": rej,
        })
    return results

def main():
    df = pd.read_csv(CSV_PATH)
    num_cols = [c for c in df.columns if c not in META_COLS]
    all_rows = []
    global current_metric
    for current_metric in num_cols:
        df_wide = load_wide(df, current_metric)
        if df_wide.empty or df_wide.shape[1] < 2:
            continue
        normal = check_normality(df_wide)
        homo = check_homoscedastic(df_wide) if df_wide.shape[1] > 2 else True
        test_name, stat, p = run_global_test(df_wide, normal, homo)
        all_rows.append({
            "test": "global",
            "metric": current_metric,
            "test_name": test_name,
            "statistic": stat,
            "p_value": p,
        })
        # Pairwise (se >=3 modelos ou 2 com global significativo?) — vamos sempre gerar
        all_rows.extend(pairwise(df_wide, normal and homo))
    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
    print(f"[stat_tests] saved → {OUTPUT_PATH.relative_to(BASE)}")

if __name__ == "__main__":
    main()