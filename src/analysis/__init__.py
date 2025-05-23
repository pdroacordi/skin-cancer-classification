"""Top‑level package for paper‑ready plotting utilities."""

from importlib import metadata
__all__ = [
    "constants",
    "metrics_parser",
    "results_collector",
    "plotter",
]

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"