"""README states the locked measurement contract, not a paraphrase that can drift."""

from pathlib import Path

from guardrail_slm.eval.contract import load_eval
from guardrail_slm.taxonomy import load_taxonomy

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_states_locked_contract() -> None:
    text = README.read_text(encoding="utf-8")
    contract = load_eval()
    taxonomy = load_taxonomy()

    assert contract.test.attack_prevalence == 0.30
    assert "30 percent attack / 70 percent benign" in text
    assert "50 percent" in text
    assert "5 percent" in text
    assert "1 percent" in text
    assert "projected precision at 1 percent" in text
    assert "0.90" in text
    assert "under 100ms p95" in text
    assert "ModernBERT ships" in text
    assert "E > D > B > C > A" in text
    assert taxonomy.precedence == ["E", "D", "B", "C", "A"]
    assert "<!-- results:start -->" in text
    assert "No results yet." in text
    assert "<!-- results:end -->" in text
