"""CLI maps flags onto show-config and later phase commands."""

import subprocess
import sys

from guardrail_slm.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_help_lists_show_config() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "show-config" in result.stdout


def test_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "guardrail_slm", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "show-config" in result.stdout


def test_show_config_prints_taxonomy_and_contract() -> None:
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0, result.output
    assert "benign_token: A" in result.stdout
    assert "projected_precision" in result.stdout
    assert "attack_prevalence: 0.3" in result.stdout
    assert "p95_added_ms: 100" in result.stdout
