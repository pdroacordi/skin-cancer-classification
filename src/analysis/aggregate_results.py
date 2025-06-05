"""Consolida métricas em **dois** CSVs:

* `all_models_general.csv`   – **uma linha por modelo/experimento**
* `all_models_per_class.csv` – **uma linha por *classe***

Motivo: separar claramente desempenho global vs. granular.

Estrutura de saída (ambas as tabelas):
    experiment, net, kind, algorithm, class, <meta_flags>, <métricas>

* No CSV **geral**: `class` fica vazio e as métricas vêm prefixadas
  com `g_` (ex.: `g_accuracy`).
* No CSV **por classe**: `class` = `0,1,…`; métricas sem prefixo.

Uso:
```bash
python -m analysis.aggregate_results [-v]
```
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]  # <repo_root>
OUT_GENERAL_CSV = BASE / "res" / "all_models_general.csv"
OUT_PER_CLASS_CSV = BASE / "res" / "all_models_per_class.csv"
METADATA_FLAGS = [
    "feature_augmentation",
    "data_augmentation",
    "hair_removal",
    "segmentation",
]
GENERAL_CSV_BASENAME = "model_performance_summary.csv"
PER_CLASS_CSV_BASENAME = "per_class_metrics.csv"
KNOWN_NETS = {
    "resnet", "inception", "vgg19", "xception"
}
ALGO_NAMES = {"adaboost", "extratrees", "randomforest", "xgboost", "svm"}

# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def _vprint(verbose: bool, *msg) -> None:
    if verbose:
        print(*msg)


def _infer_meta_from_path(path: Path) -> Dict[str, bool]:
    """Marca *True* para cada *flag* reconhecida no path."""
    meta = {k: False for k in METADATA_FLAGS}
    p_str = str(path).lower()
    for flag in meta:
        if flag in p_str or flag.replace("_", "") in p_str:
            meta[flag] = True
    return meta


def _normalize_per_class_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garante coluna ``class`` mesmo quando o rótulo vem como índice."""
    if "class" in df.columns:
        return df
    df = df.reset_index()
    return df.rename(columns={df.columns[0]: "class"})


def _parse_experiment_name(name: str) -> Dict[str, str]:
    """Extrai ``experiment``, ``net`` e ``kind`` do nome do diretório‑pai."""
    parts = [p for p in name.strip("_").split("_") if p]
    parts_lower = [p.lower() for p in parts]

    # -------- kind ---------------------------------------------------------
    if "classifier" in parts_lower:
        kind = "classifier"
    elif "feature" in parts_lower and ("extraction" in parts_lower or "extractor" in parts_lower):
        kind = "feature_extractor"
    elif "extraction" in parts_lower or "extractor" in parts_lower:
        kind = "feature_extractor"
    else:
        kind = "unknown"

    # -------- net ----------------------------------------------------------
    net_token = next(
        (p for p in parts if p.lower() in KNOWN_NETS),
        next((p for p in parts if p and p[0].isupper()), parts[0]),
    )
    net = net_token.capitalize()

    return {"experiment": name, "net": net, "kind": kind}

# ---------------------------------------------------------------------------
# COLETA PRINCIPAL
# ---------------------------------------------------------------------------

def collect(verbose: bool = False) -> None:
    general_rows: List[Dict[str, str | float | int | bool]] = []
    per_rows: List[Dict[str, str | float | int | bool]] = []

    csv_paths = list(BASE.rglob(GENERAL_CSV_BASENAME))
    _vprint(verbose, f"Encontrados {len(csv_paths)} CSVs de performance.")

    for perf_csv in csv_paths:
        if perf_csv.parent.name != "final_models":
            continue  # estrutura inesperada

        # ---------------- pastas: exp_dir / algorithm ----------------------
        parent = perf_csv.parent.parent  # pode ser algoritmo *ou* experimento
        if parent.name.lower() in ALGO_NAMES:
            algorithm = parent.name
            exp_dir = parent.parent
        else:
            algorithm = ""
            exp_dir = parent

        exp_info = _parse_experiment_name(exp_dir.name)
        exp_info["algorithm"] = algorithm
        meta_flags = _infer_meta_from_path(exp_dir)

        # ---------------- CSV geral ---------------------------------------
        try:
            general_df = pd.read_csv(perf_csv)
        except pd.errors.EmptyDataError:
            _vprint(verbose, f"[WARN] {perf_csv} sem colunas ou vazio, pulando métricas gerais…")
            general_df = pd.DataFrame()
        if general_df.empty:
            _vprint(verbose, f"[WARN] {perf_csv} vazio, pulando…")
            continue

        for _, g in general_df.iterrows():
            metrics = {k: g[k] for k in general_df.columns if k.lower() != "class"}
            general_rows.append({
                **exp_info,
                **meta_flags,
                **metrics,
            })

        # ---------------- CSV por classe ----------------------------------
        per_class_csv = perf_csv.parent / PER_CLASS_CSV_BASENAME
        if not per_class_csv.exists():
            _vprint(verbose, f"[WARN] {per_class_csv} não encontrado, pulando métricas por classe…")
            continue

        per_df = _normalize_per_class_df(pd.read_csv(per_class_csv))
        metric_cols = [c for c in per_df.columns if c != "class"]

        for _, r in per_df.iterrows():
            row = {**exp_info, **meta_flags, "class": str(r["class"])}
            for m in metric_cols:
                row[m] = r[m]
            per_rows.append(row)

    # ---------------------- Salvando --------------------------------------
    if not general_rows:
        print("[ERROR] Nenhuma métrica geral encontrada!", file=sys.stderr)
        sys.exit(1)
    if not per_rows:
        print("[ERROR] Nenhuma métrica por classe encontrada!", file=sys.stderr)
        sys.exit(1)

    pd.DataFrame(general_rows).to_csv(OUT_GENERAL_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(per_rows).to_csv(OUT_PER_CLASS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print(
        f"✅ CSVs gerados com sucesso:\n  • {OUT_GENERAL_CSV.name}  →  {len(general_rows)} linhas"
        f"\n  • {OUT_PER_CLASS_CSV.name}  →  {len(per_rows)} linhas")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Consolida resultados de todos os experimentos")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostra detalhes do processamento")
    args = parser.parse_args()

    collect(verbose=args.verbose)



