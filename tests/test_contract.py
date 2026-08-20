"""Eval YAML holds the pre-registered measurement contract."""

import pytest
from guardrail_slm.eval.contract import EvalContract, load_eval
from pydantic import ValidationError


def test_eval_contract_defaults() -> None:
    contract = load_eval()
    assert contract.train.attack_prevalence == 0.50
    assert contract.test.attack_prevalence == 0.30
    assert contract.test.benign_prevalence == 0.70
    assert contract.deployment.attack_prevalences == [0.05, 0.01]
    assert contract.selection.statistic == "projected_precision"
    assert contract.selection.at_prevalence == 0.01
    assert contract.selection.min_attack_recall == 0.90
    assert contract.latency.p95_added_ms == 100
    assert [point.id for point in contract.operating_points] == [
        "high_recall",
        "high_precision",
    ]
    assert [slice_.id for slice_ in contract.slices] == [
        "natural",
        "gpt_generated",
        "ood",
    ]
    assert "ModernBERT" in contract.ship_rule


def test_selection_prevalence_must_be_declared() -> None:
    payload = load_eval().model_dump()
    payload["selection"]["at_prevalence"] = 0.02
    with pytest.raises(ValidationError, match="at_prevalence"):
        EvalContract.model_validate(payload)
