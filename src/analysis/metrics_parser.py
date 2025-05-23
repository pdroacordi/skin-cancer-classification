"""Utilities for extracting numbers from the multitude of text files produced
by the training pipeline.  We *never* reload the heavy models – we simply
parse the already‑saved metrics. That keeps the script *fast* and completely
self‑contained."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict

from .constants import CLASSES

# regex patterns reused by both train and test files -------------------------
_PATTERNS = {
    "accuracy": [
        r"Average.*?Accuracy:\s*([0-9]*\.?[0-9]+)",
        r"Accuracy:\s*([0-9]*\.?[0-9]+)",
        r"Test Accuracy:\s*([0-9]*\.?[0-9]+)",
    ],
    "precision": [
        r"Average.*?Precision:\s*([0-9]*\.?[0-9]+)",
        r"Precision:\s*([0-9]*\.?[0-9]+)",
    ],
    "recall": [
        r"Average.*?Recall:\s*([0-9]*\.?[0-9]+)",
        r"Recall:\s*([0-9]*\.?[0-9]+)",
    ],
    "f1_score": [
        r"Average.*?F1.*?([0-9]*\.?[0-9]+)",
        r"F1 Score:\s*([0-9]*\.?[0-9]+)",
        r"F1:\s*([0-9]*\.?[0-9]+)",
    ],
    # fallback: scikit‑learn classification report style (macro avg row)
    "_macro": [r"macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)"],
}

_NUM_OR_NAME = r"(?:[0-6]|akiec|bcc|bkl|df|mel|nv|vasc)"
_FLOAT       = r"([0-9]*\.?[0-9]+)"

_RGX_CLASS = re.compile(
    rf"^\s*{_NUM_OR_NAME}\s+{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}",
    flags=re.IGNORECASE | re.MULTILINE
)

_NUM_TO_SIGLA = {
    "0": "akiec",
    "1": "bcc",
    "2": "bkl",
    "3": "df",
    "4": "mel",
    "5": "nv",
    "6": "vasc",
}


def parse_metrics(file_path: Path) -> Dict[str, float]:
    """Return a dict with accuracy/precision/recall/f1_score from *file_path*.

    Any missing metric is filled with 0.0 so the caller never has to guard
    against KeyErrors.
    """
    metrics: Dict[str, float] = {}
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return {k: 0.0 for k in ("accuracy", "precision", "recall", "f1_score")}

    # first pass – explicit lines ------------------------------------------------
    for metric, patterns in _PATTERNS.items():
        if metric == "_macro":
            continue
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    value = float(match.group(1))
                    if 0.0 <= value <= 1.0:
                        metrics[metric] = value
                        break
                except (ValueError, IndexError):
                    continue

    # second pass – macro avg fallback ------------------------------------------
    if len(metrics) < 4:  # we are missing something
        macro_re = re.compile(_PATTERNS["_macro"][0])
        m = macro_re.search(content)
        if m:
            precision, recall, f1 = map(float, m.groups())
            metrics.setdefault("precision", precision)
            metrics.setdefault("recall", recall)
            metrics.setdefault("f1_score", f1)

    # pad any still‑missing metric with zeros -----------------------------------
    for key in ("accuracy", "precision", "recall", "f1_score"):
        metrics.setdefault(key, 0.0)

    return metrics

def parse_class_f1(content: str) -> Dict[str, float]:
    """
       Devolve {sigla: f1-score} para as sete classes.
       Reconhece linhas que começam com 0-6 **ou** com a própria sigla.
    """
    out: dict[str, float] = {}
    for m in _RGX_CLASS.finditer(content):
        label_raw = m.group(0).split()[0].lower()  # primeiro token
        sigla = _NUM_TO_SIGLA.get(label_raw, label_raw)  # converte '0'→'akiec'
        f1 = float(m.group(3))  # 3.º número capturado = f1-score
        out[sigla] = f1
    return out