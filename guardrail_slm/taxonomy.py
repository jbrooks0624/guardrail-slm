"""Label taxonomy loaded from config/taxonomy.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from guardrail_slm.settings import get_settings

LABEL_TOKENS = ("A", "B", "C", "D", "E")


class LabelDef(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""


class Taxonomy(BaseModel):
    benign_token: str
    labels: dict[str, LabelDef]
    precedence: list[str]

    @model_validator(mode="after")
    def tokens_match_contract(self) -> "Taxonomy":
        if tuple(self.labels) != LABEL_TOKENS:
            raise ValueError(f"labels must be keyed {LABEL_TOKENS} in that order")
        for token in self.labels:
            if len(token) != 1:
                raise ValueError(f"label token {token!r} must be a single character")
        if self.benign_token not in self.labels:
            raise ValueError(f"benign_token {self.benign_token!r} is not a label")
        if set(self.precedence) != set(self.labels) or len(self.precedence) != len(self.labels):
            raise ValueError("precedence must list every label token exactly once")
        return self

    @property
    def attack_tokens(self) -> tuple[str, ...]:
        return tuple(token for token in self.labels if token != self.benign_token)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    taxonomy_path = path if path is not None else get_settings().taxonomy_path
    with taxonomy_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Taxonomy.model_validate(payload)
