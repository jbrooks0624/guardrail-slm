"""Shared row schema for every dataset stage."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from guardrail_slm.taxonomy import LABEL_TOKENS

Provenance = Literal["human", "gpt_generated", "synthetic"]
BinaryLabel = Literal["benign", "attack"]
LabelToken = Literal["A", "B", "C", "D", "E"]
LabelMethod = Literal[
    "source_benign",
    "source_binary",
    "source_type",
    "rule",
    "tactic_map",
    "residual",
    "llm",
    "synth",
]
WildjailbreakType = Literal[
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
]
SplitName = Literal["train", "val", "test"]
SliceId = Literal["natural", "gpt_generated", "ood"]


class SourceRecord(BaseModel):
    """One normalized prompt from a Hub source, before five-way labeling."""

    record_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    hub_name: str = Field(min_length=1)
    license: str = Field(min_length=1)
    provenance: Provenance
    binary_label: BinaryLabel
    original_split: str = Field(min_length=1)
    allowed_in_train: bool
    wildjailbreak_data_type: WildjailbreakType | None = None
    tactics: tuple[str, ...] = ()
    vanilla_text: str | None = None
    vanilla_id: str | None = None
    scaffold: str = ""
    synthetic: bool = False

    @model_validator(mode="after")
    def synthetic_matches_provenance(self) -> SourceRecord:
        if self.synthetic and self.provenance != "synthetic":
            raise ValueError("synthetic records must have provenance='synthetic'")
        if self.synthetic and not self.allowed_in_train:
            raise ValueError("synthetic records must be allowed_in_train")
        if self.synthetic and self.binary_label != "attack":
            raise ValueError(
                "synthetic records must be attacks; synth does not manufacture the benign class"
            )
        if self.vanilla_text and not self.vanilla_id:
            raise ValueError("vanilla_text requires vanilla_id")
        if self.vanilla_id and not self.vanilla_text:
            raise ValueError("vanilla_id requires vanilla_text")
        return self

    def eligible_for_test(self) -> bool:
        """Synthetic rows are train and validation only; they never enter a test slice."""
        return not self.synthetic


class LabeledRecord(SourceRecord):
    """Source record plus the single-token label and full multi-label set."""

    label: LabelToken
    categories_all: tuple[LabelToken, ...] = Field(min_length=1)
    label_method: LabelMethod

    @model_validator(mode="after")
    def label_is_in_categories(self) -> LabeledRecord:
        if self.label not in LABEL_TOKENS:
            raise ValueError(f"label must be one of {LABEL_TOKENS}")
        if self.label not in self.categories_all:
            raise ValueError("label must appear in categories_all")
        extra = set(self.categories_all) - set(LABEL_TOKENS)
        if extra:
            raise ValueError(f"unknown tokens in categories_all: {sorted(extra)}")
        return self


class AssignedRecord(LabeledRecord):
    """Labeled row after split assignment. Train and val have no slice."""

    split: SplitName
    slice_id: SliceId | None = None
    group_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def split_matches_slice(self) -> AssignedRecord:
        if self.split == "test":
            if self.slice_id is None:
                raise ValueError("test rows require slice_id")
            if not self.eligible_for_test():
                raise ValueError("synthetic rows cannot be assigned to test")
        elif self.slice_id is not None:
            raise ValueError("train/val rows must not have a slice_id")
        return self


def write_jsonl(records: Iterable[BaseModel], path: Path) -> int:
    """Write Pydantic records as JSONL. Returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_source_records(path: Path) -> Iterator[SourceRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield SourceRecord.model_validate(json.loads(line))


def read_labeled_records(path: Path) -> Iterator[LabeledRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield LabeledRecord.model_validate(json.loads(line))


def read_assigned_records(path: Path) -> Iterator[AssignedRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield AssignedRecord.model_validate(json.loads(line))
