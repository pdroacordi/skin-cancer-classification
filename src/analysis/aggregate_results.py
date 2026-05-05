"""Consolida métricas em **dois** CSVs:

* `all_models_general.csv`   – **uma linha por modelo/experimento**
* `all_models_per_class.csv` – **uma linha por *classe***

Suporta dois formatos de resultado:

**Novo formato** (experimentos com `metadata.json`):
  Flags e métricas são lidos diretamente do JSON — sem inferência por regex de path.

**Formato legado** (experimentos sem `metadata.json`):
  Mantém a lógica anterior: localiza `model_performance_summary.csv` dentro de
  `final_models/` e infere flags a partir do nome do diretório.

Uso:
```bash
python -m analysis.aggregate_results [-v]
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]  # <repo_root>
RESULTS_DIR = BASE / "results"
OUT_GENERAL_CSV = BASE / "res" / "all_models_general.csv"
OUT_PER_CLASS_CSV = BASE / "res" / "all_models_per_class.csv"

# Flag columns present in both output CSVs
METADATA_FLAGS = [
    "feature_augmentation",
    "data_augmentation",
    "hair_removal",
    "segmentation",
]

GENERAL_CSV_BASENAME = "model_performance_summary.csv"
PER_CLASS_CSV_BASENAME = "per_class_metrics.csv"

KNOWN_NETS = {"resnet", "inception", "vgg19", "xception", "efficientnet"}
ALGO_NAMES = {
    "extratrees", "randomforest", "xgboost", "lightgbm",
    "histgradientboosting", "svm",
    # DES variants — stored as dynamic_ensemble_<algorithm>
    "dynamic_ensemble_knorau",
    "dynamic_ensemble_knorau_ece",
    "dynamic_ensemble_knorau_temp",
    "dynamic_ensemble_conformal_targeted",
    "dynamic_ensemble_conformal_targeted_ece",
    "dynamic_ensemble_conformal_targeted_random",
    # Legacy (pre-per-algorithm naming)
    "dynamic_ensemble",
}

# Map metadata.json pipeline name → kind column value
_PIPELINE_TO_KIND = {
    "cnn_classifier":     "classifier",
    "feature_extraction": "feature_extractor",
}

# Map metadata.json parameter keys → METADATA_FLAGS column names
_PARAM_TO_FLAG: Dict[str, str] = {
    "use_feature_augmentation": "feature_augmentation",
    "use_data_augmentation":    "data_augmentation",
    "use_hair_removal":         "hair_removal",
}

# ---------------------------------------------------------------------------
# HELPERS PARTILHADOS
# ---------------------------------------------------------------------------

def _vprint(verbose: bool, *msg) -> None:
    if verbose:
        print(*msg)


def _normalize_per_class_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garante coluna ``class`` mesmo quando o rótulo vem como índice."""
    if "class_name" in df.columns:
        return df.rename(columns={"class_name": "class"})
    if "class" in df.columns:
        return df
    df = df.reset_index()
    return df.rename(columns={df.columns[0]: "class"})


def _empty_flags() -> Dict[str, bool]:
    return {k: False for k in METADATA_FLAGS}

# ---------------------------------------------------------------------------
# NOVO FORMATO — lê de metadata.json
# ---------------------------------------------------------------------------

def _collect_new_format(
    metadata_path: Path,
    verbose: bool,
) -> Tuple[List[dict], List[dict]]:
    """
    Extrai linhas de resultados de um arquivo metadata.json (novo formato).

    Retorna (general_rows, per_class_rows).
    """
    try:
        with open(metadata_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        _vprint(verbose, f"[WARN] Falha ao ler {metadata_path}: {exc}")
        return [], []

    if meta.get("status") not in ("success", None):
        _vprint(verbose, f"[SKIP] {metadata_path} — status={meta.get('status')}")
        return [], []

    pipeline = meta.get("pipeline", "")
    params   = meta.get("parameters", {})
    metrics  = meta.get("metrics", {})
    result_dir = metadata_path.parent

    kind      = _PIPELINE_TO_KIND.get(pipeline, pipeline)
    net       = str(params.get("cnn_model", "unknown")).capitalize()
    algorithm = str(params.get("classifier", ""))

    # For feature-extraction, metadata.json lives inside the classifier subdir
    # (e.g. results/feature_extraction_ResNet_.../extratrees/metadata.json).
    # The experiment name is the parent dir; algorithm is the leaf dir.
    if result_dir.name.lower() in ALGO_NAMES:
        exp_name = result_dir.parent.name
    else:
        exp_name = result_dir.name

    meta_flags = _empty_flags()
    for param_key, flag_name in _PARAM_TO_FLAG.items():
        meta_flags[flag_name] = bool(params.get(param_key, False))

    exp_info = {
        "experiment": exp_name,
        "net":        net,
        "kind":       kind,
        "algorithm":  algorithm,
    }

    # Normalize metric key names to the legacy CSV convention
    # (metadata.json uses macro_precision; CSVs use macro_avg_precision)
    normalized_metrics = _normalize_metric_keys(metrics)

    general_rows = [{**exp_info, **meta_flags, **normalized_metrics}]

    # Per-class metrics
    per_class_rows: List[dict] = []
    per_class_csv = result_dir / PER_CLASS_CSV_BASENAME
    if per_class_csv.exists():
        try:
            per_df = _normalize_per_class_df(pd.read_csv(per_class_csv))
            metric_cols = [
                c for c in per_df.columns
                if c not in ("class", "model_idx")
            ]
            for _, r in per_df.iterrows():
                row = {**exp_info, **meta_flags, "class": str(r["class"])}
                for m in metric_cols:
                    row[m] = r[m]
                per_class_rows.append(row)
        except Exception as exc:
            _vprint(verbose, f"[WARN] Falha ao ler {per_class_csv}: {exc}")

    return general_rows, per_class_rows


def _normalize_metric_keys(metrics: dict) -> dict:
    """
    Converte chaves do metadata.json para o formato de coluna do CSV legado.

    metadata.json usa:  macro_precision / macro_recall / macro_f1
    CSV legado usa:     macro_avg_precision / macro_avg_recall / macro_avg_f1
    """
    renames = {
        "macro_precision": "macro_avg_precision",
        "macro_recall":    "macro_avg_recall",
        "macro_f1":        "macro_avg_f1",
    }
    return {renames.get(k, k): v for k, v in metrics.items()}

# ---------------------------------------------------------------------------
# FORMATO LEGADO — lê de final_models/model_performance_summary.csv
# ---------------------------------------------------------------------------

def _infer_meta_from_path(path: Path) -> Dict[str, bool]:
    """Marca True para cada flag reconhecida no path (somente formato legado)."""
    meta = _empty_flags()
    p_str = str(path).lower()
    for flag in meta:
        if flag in p_str or flag.replace("_", "") in p_str:
            meta[flag] = True
    return meta


def _parse_experiment_name(name: str) -> Dict[str, str]:
    """Extrai experiment, net e kind do nome do diretório pai."""
    parts = [p for p in name.strip("_").split("_") if p]
    parts_lower = [p.lower() for p in parts]

    if "classifier" in parts_lower:
        kind = "classifier"
    elif "feature" in parts_lower and (
        "extraction" in parts_lower or "extractor" in parts_lower
    ):
        kind = "feature_extractor"
    elif "extraction" in parts_lower or "extractor" in parts_lower:
        kind = "feature_extractor"
    else:
        kind = "unknown"

    net_token = next(
        (p for p in parts if p.lower() in KNOWN_NETS),
        next((p for p in parts if p and p[0].isupper()), parts[0]),
    )
    net = net_token.capitalize()

    return {"experiment": name, "net": net, "kind": kind}


def _collect_legacy_format(
    perf_csv: Path,
    verbose: bool,
) -> Tuple[List[dict], List[dict]]:
    """
    Extrai linhas de resultados do formato legado
    (model_performance_summary.csv dentro de final_models/).

    Ignora diretórios que já têm metadata.json — esses foram processados
    pelo modo novo.
    """
    if perf_csv.parent.name != "final_models":
        return [], []

    exp_dir = perf_csv.parent.parent
    # Se o diretório pai (ou avô para feature-extraction) já tem metadata.json,
    # pula — foi coletado pelo modo novo.
    if (exp_dir / "metadata.json").exists():
        return [], []
    # Para feature-extraction: exp_dir é o algoritmo, pai é o experimento
    if (exp_dir.parent / "metadata.json").exists():
        return [], []

    parent = exp_dir
    if parent.name.lower() in ALGO_NAMES:
        algorithm = parent.name
        exp_dir   = parent.parent
    else:
        algorithm = ""

    exp_info   = _parse_experiment_name(exp_dir.name)
    exp_info["algorithm"] = algorithm
    meta_flags = _infer_meta_from_path(exp_dir)

    try:
        general_df = pd.read_csv(perf_csv)
    except pd.errors.EmptyDataError:
        _vprint(verbose, f"[WARN] {perf_csv} vazio, pulando…")
        return [], []

    if general_df.empty:
        return [], []

    general_rows = []
    for _, g in general_df.iterrows():
        row_metrics = {k: g[k] for k in general_df.columns if k.lower() != "class"}
        general_rows.append({**exp_info, **meta_flags, **row_metrics})

    per_class_rows: List[dict] = []
    per_class_csv = perf_csv.parent / PER_CLASS_CSV_BASENAME
    if not per_class_csv.exists():
        _vprint(verbose, f"[WARN] {per_class_csv} não encontrado")
        return general_rows, []

    try:
        per_df = _normalize_per_class_df(pd.read_csv(per_class_csv))
        metric_cols = [c for c in per_df.columns if c != "class"]
        for _, r in per_df.iterrows():
            row = {**exp_info, **meta_flags, "class": str(r["class"])}
            for m in metric_cols:
                row[m] = r[m]
            per_class_rows.append(row)
    except Exception as exc:
        _vprint(verbose, f"[WARN] Falha ao ler {per_class_csv}: {exc}")

    return general_rows, per_class_rows

# ---------------------------------------------------------------------------
# COLETA PRINCIPAL
# ---------------------------------------------------------------------------

def collect(verbose: bool = False) -> None:
    general_rows: List[dict] = []
    per_rows:     List[dict] = []

    # ---- Novo formato: metadata.json ----
    new_format_paths = list(RESULTS_DIR.rglob("metadata.json"))
    _vprint(verbose, f"Encontrados {len(new_format_paths)} arquivo(s) metadata.json (novo formato).")

    for metadata_path in new_format_paths:
        g, p = _collect_new_format(metadata_path, verbose)
        general_rows.extend(g)
        per_rows.extend(p)

    # ---- Formato legado: final_models/model_performance_summary.csv ----
    legacy_paths = list(RESULTS_DIR.rglob(GENERAL_CSV_BASENAME))
    _vprint(verbose, f"Encontrados {len(legacy_paths)} CSV(s) de performance (inclui legado e novo).")

    for perf_csv in legacy_paths:
        g, p = _collect_legacy_format(perf_csv, verbose)
        general_rows.extend(g)
        per_rows.extend(p)

    # ---- Salvando ----
    if not general_rows:
        print("[ERROR] Nenhuma métrica geral encontrada!", file=sys.stderr)
        sys.exit(1)
    if not per_rows:
        print("[WARN] Nenhuma métrica por classe encontrada.", file=sys.stderr)

    OUT_GENERAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(general_rows).to_csv(OUT_GENERAL_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    if per_rows:
        pd.DataFrame(per_rows).to_csv(OUT_PER_CLASS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"CSVs gerados:")
    print(f"  {OUT_GENERAL_CSV.name}  ->  {len(general_rows)} linhas")
    if per_rows:
        print(f"  {OUT_PER_CLASS_CSV.name}  ->  {len(per_rows)} linhas")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Consolida resultados de todos os experimentos")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    collect(verbose=args.verbose)
