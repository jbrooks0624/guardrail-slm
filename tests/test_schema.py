"""Source catalog YAML and SourceRecord / LabeledRecord contract."""

from pathlib import Path

import pytest
from guardrail_slm.data.config import SOURCE_IDS, load_sources_config
from guardrail_slm.data.schema import AssignedRecord, LabeledRecord, SourceRecord, write_jsonl
from pydantic import ValidationError


def _base_source(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "record_id": "abc",
        "text": "hello",
        "source_id": "deepset",
        "hub_name": "deepset/prompt-injections",
        "license": "Apache-2.0",
        "provenance": "human",
        "binary_label": "attack",
        "original_split": "train",
        "allowed_in_train": True,
    }
    payload.update(overrides)
    return payload


def test_sources_config_defaults() -> None:
    catalog = load_sources_config()
    assert tuple(catalog.sources) == SOURCE_IDS
    assert catalog.synth_min_train_eligible_e == 500
    assert catalog.splits.train_target_size == 20000
    assert catalog.splits.pair_holdout_fraction == 0.15
    assert catalog.splits.tactic_holdout_fraction == 0.20
    assert catalog.splits.leak_cosine_threshold == 0.90
    assert catalog.splits.leak_jaccard_threshold == 0.80
    assert catalog.sources["xtram1"].allowed_in_train is False
    assert catalog.sources["wildjailbreak"].gated is True
    assert catalog.sources["wildjailbreak"].hub_config == "train"
    assert catalog.sources["wildjailbreak"].hub_tsv_file == "train/train.tsv"
    assert catalog.sources["xtram1"].text_field == "text"
    assert catalog.sources["wildjailbreak"].license == "ODC-BY"
    assert catalog.sources["dolly"].license == "CC-BY-SA-3.0"
    assert catalog.sources["deepset"].label_map["0"] == "benign"
    assert catalog.sources["deepset"].label_map["1"] == "attack"


def test_synthetic_record_requires_synthetic_provenance() -> None:
    with pytest.raises(ValidationError, match="synthetic"):
        SourceRecord.model_validate(_base_source(synthetic=True, provenance="human"))


def test_synthetic_record_cannot_be_benign() -> None:
    with pytest.raises(ValidationError, match="benign class"):
        SourceRecord.model_validate(
            _base_source(synthetic=True, provenance="synthetic", binary_label="benign")
        )


def test_synthetic_record_is_not_test_eligible() -> None:
    record = SourceRecord.model_validate(_base_source(synthetic=True, provenance="synthetic"))
    assert record.eligible_for_test() is False
    assert SourceRecord.model_validate(_base_source()).eligible_for_test() is True


def test_vanilla_id_and_text_must_come_together() -> None:
    with pytest.raises(ValidationError, match="vanilla"):
        SourceRecord.model_validate(_base_source(vanilla_text="payload"))
    with pytest.raises(ValidationError, match="vanilla"):
        SourceRecord.model_validate(_base_source(vanilla_id="deadbeef"))


def test_labeled_record_requires_label_in_categories_all() -> None:
    with pytest.raises(ValidationError, match="categories_all"):
        LabeledRecord.model_validate(
            {
                **_base_source(),
                "label": "C",
                "categories_all": ["B"],
                "label_method": "rule",
            }
        )


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    record = SourceRecord.model_validate(_base_source())
    path = tmp_path / "normalized.jsonl"
    assert write_jsonl([record], path) == 1
    loaded = SourceRecord.model_validate_json(path.read_text(encoding="utf-8").strip())
    assert loaded == record


def test_assigned_record_rejects_synthetic_test() -> None:
    with pytest.raises(ValidationError, match="synthetic"):
        AssignedRecord.model_validate(
            {
                **_base_source(synthetic=True, provenance="synthetic"),
                "label": "E",
                "categories_all": ["E"],
                "label_method": "synth",
                "split": "test",
                "slice_id": "natural",
                "group_id": "g1",
            }
        )
