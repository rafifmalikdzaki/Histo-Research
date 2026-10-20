"""Unified command-line interface for histodae.

The ``histodae`` console script exposes every stage of the pipeline as a
subcommand::

    histodae train --model-name dae_kan_attention --dataset hepar
    histodae extract --model-name dae_kan_attention --phase test
    histodae cluster --base-dir auto_analysis --k 6
    histodae stats --input results/raw_clustering_metrics.csv
    histodae plots --input results/raw_clustering_metrics.csv
    histodae tables --input results/raw_clustering_metrics.csv
    histodae evaluate --model dae_kan_attention --run-tests
    histodae pipeline --models baseline dae_kan_attention --seeds 42
    histodae data hepar --data-path ./data/processed

Each subcommand delegates to the ``argparse``-based ``main()`` of the
corresponding module in this package, so the unified CLI and the individual
``histodae-*`` console scripts can never drift apart. Flags are therefore
passed through verbatim — run ``histodae <command> --help`` for the full
option list of that stage.

The legacy per-stage console scripts (``histodae-train``, ``histodae-extract``,
…) remain available and call the very same functions.
"""

import click

from histodae import __version__

# (command name, module inside this package, one-line help)
_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("train", "train", "Train a model variant."),
    ("extract", "extract", "Extract test-split embeddings + biology metadata."),
    ("pipeline", "pipeline", "Run the corrected end-to-end sweep (train -> extract -> cluster -> stats)."),
    ("cluster", "cluster", "Compute clustering metrics over extracted embeddings."),
    ("stats", "stats", "Run statistical significance tests (paired, Holm-corrected)."),
    ("plots", "plots", "Generate clustering metric plots."),
    ("tables", "tables", "Generate LaTeX tables from clustering results."),
    ("evaluate", "evaluate", "Evaluate multi-seed experiment statistics."),
)

_PASS_THROUGH = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
}


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="histodae")
def cli() -> None:
    """histodae — attention-enhanced dual autoencoder clustering for liver histopathology."""


def _make_command(name: str, module: str, help_text: str):
    """Build a click command that forwards every argument to an argparse ``main()``."""

    @click.command(
        name=name,
        help=help_text,
        add_help_option=False,  # let argparse own --help
        context_settings=_PASS_THROUGH,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def command(ctx: click.Context, args: tuple[str, ...]) -> None:
        import sys
        from importlib import import_module

        entry = import_module(f"{__name__}.{module}")
        # argparse derives its usage line from sys.argv[0]; point it at the
        # unified command name so `histodae train --help` reads naturally.
        original = sys.argv[0]
        sys.argv[0] = f"histodae {name}"
        try:
            code = entry.main(list(args)) or 0
        finally:
            sys.argv[0] = original
        ctx.exit(code)
    return command


for _name, _module, _help in _COMMANDS:
    cli.add_command(_make_command(_name, _module, _help))


@cli.group("data")
def data_group() -> None:
    """Build dataset CSV manifests (train/test splits with biology metadata)."""


@data_group.command(
    "hepar",
    help="Create leak-free HeparUnifiedPNG train/test/fold CSV files.",
    add_help_option=False,
    context_settings=_PASS_THROUGH,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def data_hepar(ctx: click.Context, args: tuple[str, ...]) -> None:
    from .make_hepar_csv import main

    ctx.exit(main(list(args)) or 0)


@data_group.command(
    "pannuke",
    help="Create PANnuke train/test CSV files.",
    add_help_option=False,
    context_settings=_PASS_THROUGH,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def data_pannuke(ctx: click.Context, args: tuple[str, ...]) -> None:
    from .make_pannuke_csv import main

    ctx.exit(main(list(args)) or 0)


if __name__ == "__main__":  # pragma: no cover
    cli()
