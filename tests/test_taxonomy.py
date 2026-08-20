"""Taxonomy YAML and single-token label contract."""

import pytest
from guardrail_slm.taxonomy import LABEL_TOKENS, Taxonomy, load_taxonomy
from pydantic import ValidationError


def test_taxonomy_defaults() -> None:
    taxonomy = load_taxonomy()
    assert tuple(taxonomy.labels) == LABEL_TOKENS
    assert taxonomy.benign_token == "A"
    assert taxonomy.labels["A"].name == "benign"
    assert taxonomy.labels["B"].name == "roleplay"
    assert taxonomy.labels["C"].name == "instruction_override"
    assert taxonomy.labels["D"].name == "encoding_obfuscation"
    assert taxonomy.labels["E"].name == "context_injection"
    assert taxonomy.precedence == ["E", "D", "B", "C", "A"]
    assert taxonomy.attack_tokens == ("B", "C", "D", "E")
    for token in taxonomy.labels:
        assert len(token) == 1


def test_precedence_must_cover_every_label() -> None:
    with pytest.raises(ValidationError, match="precedence"):
        Taxonomy.model_validate(
            {
                "benign_token": "A",
                "labels": {
                    "A": {"name": "benign"},
                    "B": {"name": "roleplay"},
                    "C": {"name": "instruction_override"},
                    "D": {"name": "encoding_obfuscation"},
                    "E": {"name": "context_injection"},
                },
                "precedence": ["E", "D", "B", "C"],
            }
        )


def test_labels_must_be_a_through_e() -> None:
    with pytest.raises(ValidationError, match="labels"):
        Taxonomy.model_validate(
            {
                "benign_token": "A",
                "labels": {
                    "A": {"name": "benign"},
                    "B": {"name": "roleplay"},
                    "C": {"name": "instruction_override"},
                    "D": {"name": "encoding_obfuscation"},
                    "X": {"name": "other"},
                },
                "precedence": ["X", "D", "B", "C", "A"],
            }
        )
