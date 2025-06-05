"""Pipeline estatística completa para comparação de modelos de ML

Análise estatística focada em F1-score usando:
* Entrada: all_models_general.csv (métricas já agregadas por execução)
* Testes: Wilcoxon signed-rank (pareado, não paramétrico)
* Correção: Holm-Bonferroni para múltiplas comparações
* Saída: stat_tests_pairs.csv com comparações e vencedores

Execução: python -m analysis.stat_tests
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.stats.multitest as smm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CSV_GENERAL = ROOT / "res" / "all_models_general.csv"  # MUDANÇA: usar arquivo geral
OUT_DIR = ROOT / "results"
OUT_PAIRS = OUT_DIR / "stat_tests_pairs.csv"
ALPHA = 0.05
MIN_VARIANCE_THRESHOLD = 1e-8

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _build_model_id(row: pd.Series) -> str:
    """Constrói ID único do modelo: net_kind_algorithm"""
    alg = row.get("algorithm", "none")
    if pd.isna(alg) or alg == "" or alg is None:
        alg = "none"
    return f"{row['net']}_{row['kind']}_{alg}"


def _prepare_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Prepara dados para análise estatística usando arquivo geral.

    Para cada modelo:
    1. Coleta valores de macro_avg_f1 por model_idx (execução)
    2. Verifica se há variação suficiente entre execuções
    3. Remove modelos com variância muito baixa

    Returns:
        Dict[model_id, np.array] com valores de F1-score por execução
    """
    df = df.copy()
    df['model_id'] = df.apply(_build_model_id, axis=1)

    # Usar diretamente a coluna macro_avg_f1 do arquivo geral
    df_clean = df.dropna(subset=['macro_avg_f1']).copy()

    model_data = {}
    excluded_count = 0

    for model_id in df_clean['model_id'].unique():
        # Obter valores de F1-score por execução diretamente
        model_scores = df_clean[df_clean['model_id'] == model_id]['macro_avg_f1'].values

        if len(model_scores) < 5:
            print(f"Modelo {model_id} excluído: apenas {len(model_scores)} execuções")
            excluded_count += 1
            continue

        variance = np.var(model_scores, ddof=1) if len(model_scores) > 1 else 0

        if variance < MIN_VARIANCE_THRESHOLD:
            print(f"Modelo {model_id} excluído: variância muito baixa ({variance:.2e})")
            excluded_count += 1
            continue

        model_data[model_id] = model_scores

    print(f"Modelos processados: {len(df_clean['model_id'].unique())}")
    print(f"Modelos excluídos: {excluded_count}")
    print(f"Modelos válidos: {len(model_data)}")

    return model_data


def _wilcoxon_comparison(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """Executa teste Wilcoxon signed-rank entre dois modelos."""
    try:
        differences = data1 - data2
        non_zero_diffs = differences[np.abs(differences) > 1e-10]

        if len(non_zero_diffs) < 3:
            return 0.0, 1.0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stat, p_val = st.wilcoxon(data1, data2, zero_method='wilcox', alternative='two-sided')

        if np.isnan(stat) or np.isnan(p_val) or np.isinf(p_val):
            return 0.0, 1.0

        return float(stat), float(p_val)

    except Exception:
        return 0.0, 1.0


def _identify_winner(model_a: str, model_b: str, data_a: np.ndarray, data_b: np.ndarray) -> str:
    """Identifica vencedor baseado na média de F1-score."""
    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)
    return model_a if mean_a > mean_b else model_b


def _perform_pairwise_comparisons(model_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Executa todas as comparações par a par usando Wilcoxon signed-rank."""
    models = list(model_data.keys())
    n_models = len(models)
    n_comparisons = n_models * (n_models - 1) // 2

    print(f"Executando {n_comparisons} comparações entre {n_models} modelos...")

    results = []

    for model_a, model_b in itertools.combinations(models, 2):
        data_a = model_data[model_a]
        data_b = model_data[model_b]

        min_len = min(len(data_a), len(data_b))
        data_a_trimmed = data_a[:min_len]
        data_b_trimmed = data_b[:min_len]

        stat, p_raw = _wilcoxon_comparison(data_a_trimmed, data_b_trimmed)
        winner = _identify_winner(model_a, model_b, data_a, data_b)

        results.append({
            'metric': 'f1_score',
            'test': 'wilcoxon',
            'stat': stat,
            'p_raw': p_raw,
            'model_a': model_a,
            'model_b': model_b,
            'winner': winner,
            'mean_a': np.mean(data_a),
            'mean_b': np.mean(data_b),
            'std_a': np.std(data_a, ddof=1),
            'std_b': np.std(data_b, ddof=1)
        })

    return pd.DataFrame(results)


def _apply_holm_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica correção Holm-Bonferroni aos p-valores."""
    df = df.copy()

    p_raw_values = df['p_raw'].values
    reject, p_corrected, alpha_sidak, alpha_bonf = smm.multipletests(
        p_raw_values,
        method='holm',
        alpha=ALPHA
    )

    df['p'] = p_corrected
    df['significant'] = reject

    return df


def _generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Gera estatísticas resumidas de vitórias/derrotas por modelo."""
    significant_df = df[df['significant'] == True].copy()

    if significant_df.empty:
        print("Nenhuma comparação estatisticamente significativa encontrada!")
        return pd.DataFrame()

    wins_count = significant_df['winner'].value_counts()
    all_models = set(df['model_a'].tolist() + df['model_b'].tolist())
    summary_data = []

    for model in all_models:
        wins = wins_count.get(model, 0)

        losses = len(significant_df[
            ((significant_df['model_a'] == model) | (significant_df['model_b'] == model)) &
            (significant_df['winner'] != model)
        ])

        model_comparisons = df[
            (df['model_a'] == model) | (df['model_b'] == model)
        ].copy()

        mean_f1 = model_comparisons.apply(
            lambda row: row['mean_a'] if row['model_a'] == model else row['mean_b'],
            axis=1
        ).mean()

        model_p_values = model_comparisons.apply(
            lambda row: row['p'] if row['model_a'] == model or row['model_b'] == model else np.nan,
            axis=1
        ).dropna()

        min_p = model_p_values.min() if not model_p_values.empty else np.nan

        summary_data.append({
            'model': model,
            'wins': wins,
            'losses': losses,
            'net_wins': wins - losses,
            'total_comparisons': len(model_comparisons),
            'significant_comparisons': len(significant_df[
                                               (significant_df['model_a'] == model) | (
                                                           significant_df['model_b'] == model)
                                               ]),
            'mean_f1_score': mean_f1,
            'win_rate': wins / len(model_comparisons) if len(model_comparisons) > 0 else 0,
            'min_p_value': min_p
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('net_wins', ascending=False)

    return summary_df


def main():
    """Função principal da análise estatística."""
    print("Iniciando análise estatística completa...")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Carregando dados de: {CSV_GENERAL}")
    try:
        df = pd.read_csv(CSV_GENERAL)
        print(f"Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    print("Preparando dados por modelo...")
    model_data = _prepare_data(df)

    if len(model_data) < 2:
        print("Menos de 2 modelos válidos para comparação!")
        return

    print("\nMODELOS VÁLIDOS PARA ANÁLISE:")
    for i, (model_id, data) in enumerate(model_data.items(), 1):
        variance = np.var(data, ddof=1)
        mean_f1 = np.mean(data)
        std_f1 = np.std(data, ddof=1)

        print(f"   {i:2d}. {model_id}")
        print(f"       {len(data)} execuções, F1 = {mean_f1:.4f} ± {std_f1:.4f} (var={variance:.2e})")

    print(f"\nExecutando comparações estatísticas...")
    comparisons_df = _perform_pairwise_comparisons(model_data)

    print("Aplicando correção Holm-Bonferroni...")
    comparisons_df = _apply_holm_correction(comparisons_df)

    print("Gerando estatísticas resumidas...")
    summary_df = _generate_summary_stats(comparisons_df)

    output_cols = ['metric', 'test', 'stat', 'p_raw', 'model_a', 'model_b', 'p', 'significant']
    comparisons_df[output_cols].to_csv(OUT_PAIRS, index=False)

    summary_path = OUT_DIR / "model_ranking_summary.csv"
    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
        print(f"Resumo salvo em: {summary_path}")

    n_significant = len(comparisons_df[comparisons_df['significant'] == True])
    n_total = len(comparisons_df)

    print(f"\nRESULTADOS FINAIS:")
    print(f"   Total de comparações: {n_total}")
    print(f"   Comparações significativas: {n_significant} ({n_significant/n_total*100:.1f}%)")
    print(f"   Resultados salvos em: {OUT_PAIRS}")

    if not summary_df.empty:
        print(f"\nTOP 5 MODELOS (por vitórias líquidas):")
        for i, (_, row) in enumerate(summary_df.head().iterrows(), 1):
            print(f"   {i}. {row['model']}: {row['net_wins']:+d} vitórias líquidas "
                  f"({row['wins']}W/{row['losses']}L), F1={row['mean_f1_score']:.4f}")
    else:
        print("Nenhuma diferença estatisticamente significativa encontrada.")
        print("Isso indica que os modelos têm desempenho muito similar.")

        if not comparisons_df.empty:
            top_comparisons = comparisons_df.nsmallest(5, 'p')
            print(f"\nTop 5 menores p-valores (mesmo não significativos):")
            for i, (_, row) in enumerate(top_comparisons.iterrows(), 1):
                print(f"     {i}. {row['model_a']} vs {row['model_b']}: "
                      f"p={row['p']:.4f}, vencedor={row['winner']}")

    print(f"\nAnálise estatística concluída!")


if __name__ == "__main__":
    main()