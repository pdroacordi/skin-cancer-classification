"""
RunContext — experiment execution context.

Encapsulates the run's identity (run_id), result directory, and active
configuration. It is the single object responsible for writing metadata.json,
so every experiment automatically gets a machine-readable record of the flags
that were active, the metrics obtained, and the artifacts produced.

Without this, metadata must be inferred from directory names via regex —
a fragile approach that breaks whenever a new flag is added.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunContext:
    """
    Immutable experiment context.  Create via RunContext.create(), not directly.

    Attributes:
        run_id:          Unique identifier for this execution (timestamp + config hash).
        pipeline:        Pipeline name, e.g. "cnn_classifier" or "feature_extraction".
        result_dir:      Absolute Path to the directory where all outputs are written.
        config_snapshot: Dict copy of all active config flags at the time of creation.
    """

    run_id: str
    pipeline: str
    result_dir: Path
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        pipeline: str,
        result_dir: str,
        config_snapshot: Dict[str, Any],
    ) -> "RunContext":
        """
        Create a new RunContext with a unique run_id.

        run_id format: {pipeline}_{YYYYMMDDTHHMMSS}_{6-char hash}
        The hash is derived from the config snapshot, so identical configs
        on the same second produce the same run_id (idempotent reruns).
        """
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        config_str = json.dumps(config_snapshot, sort_keys=True, default=str)
        short_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        run_id = f"{pipeline}_{timestamp}_{short_hash}"
        return cls(
            run_id=run_id,
            pipeline=pipeline,
            result_dir=Path(result_dir),
            config_snapshot=config_snapshot,
        )

    # ------------------------------------------------------------------ #
    #  Path helpers                                                        #
    # ------------------------------------------------------------------ #

    def artifact_path(self, relative_name: str) -> Path:
        """Return the absolute path for a named artifact within result_dir."""
        return self.result_dir / relative_name

    # ------------------------------------------------------------------ #
    #  Finalization                                                        #
    # ------------------------------------------------------------------ #

    def finalize(
        self,
        metrics: Dict[str, Any],
        artifacts: List[Dict[str, str]],
        status: str = "success",
        notes: str = "",
    ) -> None:
        """
        Write metadata.json to result_dir.

        This is the single place that produces the machine-readable experiment
        record consumed by analysis tools and LLMs.  Every key in the JSON
        has an explicit description so it is self-documenting.

        Args:
            metrics:   Dict of scalar metric names → values (e.g. macro_f1).
            artifacts: List of {'path': str, 'description': str} dicts.
            status:    "success" | "failed" | "partial".
            notes:     Free-text field (empty string by default).
        """
        self.result_dir.mkdir(parents=True, exist_ok=True)

        metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "pipeline": self.pipeline,
            "status": status,
            "parameters": self.config_snapshot,
            "metrics": metrics,
            "artifacts": artifacts,
            "notes": notes,
        }

        metadata_path = self.result_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
