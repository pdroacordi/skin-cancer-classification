"""Enhanced metrics parser that now reliably captures *accuracy* even when only
present in the `classification_report` section (no separating colon).  This
fixes the zero‑accuracy issue seen in radar charts.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Regex patterns (ordered from most‑specific to most‑generic) ---------------
# ---------------------------------------------------------------------------
_PATTERNS: Dict[str, List[str]] = {
    # Accuracy lines written explicitly by our training scripts or by
    # sklearn's classification_report (two different styles) ----------------
    "accuracy": [
        r"Average.*?Accuracy:\s*([0-9]*\.?[0-9]+)",  # e.g. cross‑val summary
        r"Accuracy:\s*([0-9]*\.?[0-9]+)",           # simple key‑value
        r"Test Accuracy:\s*([0-9]*\.?[0-9]+)",      # custom test print
        # sklearn's report:  'accuracy      0.84     1503' (no colon)
        r"^\s*accuracy\s+([0-9]*\.?[0-9]+)\b",      # fallback, start of line
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
    # When individual metrics are missing we try the `macro avg` line from the
    # classification report:  precision  recall  f1-score  support -----------
    "_macro": [r"macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",],
}


# ---------------------------------------------------------------------------
# Public helper --------------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_metrics(file_path: Path) -> Dict[str, float]:
    """Extract *accuracy*, *precision*, *recall* and *f1_score* from *file_path*.

    Any missing metric is filled with **0.0** so the caller never worries about
    KeyErrors.  Accuracy is now picked up even when present only in the
    confusion‑matrix‑style summary produced by scikit‑learn.
    """

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return {k: 0.0 for k in ("accuracy", "precision", "recall", "f1_score")}

    metrics: Dict[str, float] = {}

    # Pass 1 – direct key‑value matches --------------------------------------
    for metric, patterns in _PATTERNS.items():
        if metric == "_macro":
            continue
        for pat in patterns:
            m = re.search(pat, content, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if m:
                try:
                    val = float(m.group(1))
                    if 0.0 <= val <= 1.0:
                        metrics[metric] = val
                        break  # stop at first hit for this metric
                except (ValueError, IndexError):
                    continue

    # Pass 2 – macro‑average fallback (fills precision/recall/f1 if missing) -
    if any(k not in metrics for k in ("precision", "recall", "f1_score")):
        m = re.search(_PATTERNS["_macro"][0], content, flags=re.IGNORECASE)
        if m:
            precision, recall, f1 = map(float, m.groups())
            metrics.setdefault("precision", precision)
            metrics.setdefault("recall", recall)
            metrics.setdefault("f1_score", f1)

    # Default zeros for whatever is still missing ---------------------------
    for key in ("accuracy", "precision", "recall", "f1_score"):
        metrics.setdefault(key, 0.0)

    return metrics
