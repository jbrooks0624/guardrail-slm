"""Dataset build entry is argparse, not a Typer command."""

import subprocess
import sys


def test_module_help_lists_allow_hub() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "guardrail_slm.data.build", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--allow-hub" in result.stdout
    assert "--allow-llm" in result.stdout
    assert "--stage" in result.stdout
    assert "synth" in result.stdout
    assert "splits" in result.stdout
    assert "leak-audit" in result.stdout
