"""Dataset source catalog loaded from config/sources.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from guardrail_slm.data.schema import BinaryLabel, Provenance
from guardrail_slm.settings import get_settings

SOURCE_IDS = ("deepset", "jackhhao", "dolly", "wildjailbreak", "xtram1")
HUMAN_ATTACK_SOURCES = frozenset({"deepset", "jackhhao"})


class SplitConfig(BaseModel):
    seed: int = 0
    val_fraction: float = Field(default=0.10, gt=0, lt=1)
    tactic_holdout_fraction: float = Field(default=0.20, ge=0, lt=1)
    pair_holdout_fraction: float = Field(default=0.15, ge=0, lt=1)
    minhash_jaccard: float = Field(default=0.80, gt=0, le=1)
    minhash_num_perm: int = Field(default=128, ge=16)
    shingle_n: int = Field(default=5, ge=2)
    prevalence_tolerance: float = Field(default=0.02, gt=0, lt=0.5)
    train_target_size: int = Field(default=20000, ge=2)
    leak_embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    leak_cosine_threshold: float = Field(default=0.90, gt=0, le=1)
    leak_jaccard_threshold: float = Field(default=0.80, gt=0, le=1)


class SourceSpec(BaseModel):
    source_id: str = Field(min_length=1)
    hub_name: str = Field(min_length=1)
    license: str = Field(min_length=1)
    provenance: Provenance
    allowed_in_train: bool
    gated: bool = False
    hub_config: str | None = None
    text_field: str | None = None
    label_field: str | None = None
    label_map: dict[str, BinaryLabel] = Field(default_factory=dict)
    always_benign: bool = False
    instruction_field: str | None = None
    context_field: str | None = None
    data_type_field: str | None = None
    vanilla_field: str | None = None
    adversarial_field: str | None = None
    tactics_field: str | None = None
    data_type_labels: dict[str, BinaryLabel] = Field(default_factory=dict)
    expected_rows: dict[str, int] = Field(default_factory=dict)
    string_columns: tuple[str, ...] = ()
    hub_load_kwargs: dict[str, object] = Field(default_factory=dict)
    hub_tsv_file: str | None = None

    @field_validator("label_map", "data_type_labels", mode="before")
    @classmethod
    def stringify_map_keys(cls, value: object) -> object:
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        return value


class SourcesConfig(BaseModel):
    synth_min_train_eligible_e: int = Field(ge=0)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    sources: dict[str, SourceSpec]

    @field_validator("sources")
    @classmethod
    def known_source_ids(cls, value: dict[str, SourceSpec]) -> dict[str, SourceSpec]:
        if tuple(value) != SOURCE_IDS:
            raise ValueError(f"sources must be keyed {SOURCE_IDS} in that order")
        return value


def load_sources_config(path: Path | None = None) -> SourcesConfig:
    sources_path = path if path is not None else get_settings().sources_path
    with sources_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    raw_sources = payload.get("sources") or {}
    sources = {
        source_id: SourceSpec.model_validate({"source_id": source_id, **spec})
        for source_id, spec in raw_sources.items()
    }
    return SourcesConfig.model_validate(
        {
            "synth_min_train_eligible_e": payload.get("synth_min_train_eligible_e", 500),
            "splits": payload.get("splits") or {},
            "sources": sources,
        }
    )
