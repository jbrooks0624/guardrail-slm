"""Command-line entry point for guardrail-slm."""

import typer
import yaml

from guardrail_slm.eval.contract import load_eval
from guardrail_slm.taxonomy import load_taxonomy

app = typer.Typer(
    no_args_is_help=True,
    help="Fine-tuned small-model guardrail classifier toolkit.",
)


@app.callback()
def main() -> None:
    """Fine-tuned small-model guardrail classifier toolkit."""


@app.command("show-config")
def show_config() -> None:
    """Print the locked taxonomy and measurement contract."""
    taxonomy = load_taxonomy()
    contract = load_eval()
    typer.echo("taxonomy:")
    typer.echo(yaml.safe_dump(taxonomy.model_dump(), sort_keys=False).rstrip())
    typer.echo("eval:")
    typer.echo(yaml.safe_dump(contract.model_dump(), sort_keys=False).rstrip())
