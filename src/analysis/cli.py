"""Little Click‑powered CLI wrapper so you can run everything with one command."""

from __future__ import annotations
from pathlib import Path
import click

from .constants import OUTPUT_DIR
from .results_collector import CollectorConfig, ResultsCollector
from .plotter import Plotter


@click.command(context_settings={"show_default": True})
@click.option("--results-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), default=Path("results"), help="Folder with all experiment artefacts.")
@click.option("--figures-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=OUTPUT_DIR, help="Where PNGs will be written.")
def main(results_dir: Path, figures_dir: Path) -> None:
    """Generate every plot required for the *Results & Discussion* section."""
    print("🔍  Vasculhando métricas em", results_dir)
    collector = ResultsCollector(CollectorConfig(results_dir=results_dir))
    collector.collect()
    train_df, test_df = collector.to_dataframes()
    print(f"📈  {len(train_df)} registros de treino | {len(test_df)} de teste encontrados.")

    plotter = Plotter(train_df, test_df, out_dir=figures_dir)
    plotter.make_all_figures()
    print("🎉  Todas as figuras foram geradas!")

if __name__ == "__main__":  # pragma: no cover
    main()