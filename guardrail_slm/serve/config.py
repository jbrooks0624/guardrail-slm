"""Cost inputs loaded from config/cost.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from guardrail_slm.settings import get_settings


class GpuCost(BaseModel):
    default: str = Field(min_length=1)
    hourly_usd: dict[str, float]


class CostConfig(BaseModel):
    gpu: GpuCost
    api: dict[str, dict]


def load_cost(path: Path | None = None) -> CostConfig:
    cost_path = path if path is not None else get_settings().cost_path
    with cost_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return CostConfig.model_validate(payload)
