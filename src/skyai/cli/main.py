"""skyai CLI entry point"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

import typer

app = typer.Typer(
    name="skyai",
    help="SkyAI training/eval/sample harness",
    no_args_is_help=True,
)

@app.callback()
def main() -> None:
    """SkyAI training/eval/sample harness"""

@app.command()
def version() -> None:
    """Print the installed skyai version"""
    typer.echo(_pkg_version("skyai"))

if __name__ == "__main__":
    app()