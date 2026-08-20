"""Measurement contract loaded from config/eval.yaml."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from guardrail_slm.settings import get_settings


class Prevalence(BaseModel):
    attack_prevalence: float = Field(gt=0, lt=1)

    @property
    def benign_prevalence(self) -> float:
        return 1.0 - self.attack_prevalence


class Deployment(BaseModel):
    attack_prevalences: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def prevalences_are_probabilities(self) -> "Deployment":
        for value in self.attack_prevalences:
            if not 0 < value < 1:
                raise ValueError("deployment attack_prevalences must be in (0, 1)")
        return self


class Selection(BaseModel):
    statistic: Literal["projected_precision"]
    at_prevalence: float = Field(gt=0, lt=1)
    min_attack_recall: float = Field(gt=0, le=1)


class LatencyBudget(BaseModel):
    p95_added_ms: float = Field(gt=0)


class NamedItem(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class EvalContract(BaseModel):
    train: Prevalence
    test: Prevalence
    deployment: Deployment
    selection: Selection
    latency: LatencyBudget
    operating_points: list[NamedItem] = Field(min_length=1)
    slices: list[NamedItem] = Field(min_length=1)
    ship_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def selection_prevalence_is_declared(self) -> "EvalContract":
        if self.selection.at_prevalence not in self.deployment.attack_prevalences:
            raise ValueError(
                "selection.at_prevalence must be one of deployment.attack_prevalences"
            )
        return self


def load_eval(path: Path | None = None) -> EvalContract:
    eval_path = path if path is not None else get_settings().eval_path
    with eval_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return EvalContract.model_validate(payload)
