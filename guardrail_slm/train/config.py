"""Training and sweep definitions loaded from config/train.yaml and config/sweep.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from guardrail_slm.settings import get_settings


class QuantizationConfig(BaseModel):
    load_in_4bit: bool
    bnb_4bit_quant_type: str = Field(min_length=1)
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str = Field(min_length=1)


class LoraConfig(BaseModel):
    target_modules: list[str] = Field(min_length=1)


class TrainConfig(BaseModel):
    base_model: str = Field(min_length=1)
    max_new_tokens: int = Field(ge=1)
    completion_only: bool
    quantization: QuantizationConfig
    lora: LoraConfig


class SweepConfig(BaseModel):
    rank: list[int] = Field(min_length=1)
    learning_rate: list[float] = Field(min_length=1)
    num_train_epochs: list[int] = Field(min_length=1)


def load_train(path: Path | None = None) -> TrainConfig:
    train_path = path if path is not None else get_settings().train_path
    with train_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return TrainConfig.model_validate(payload)


def load_sweep(path: Path | None = None) -> SweepConfig:
    sweep_path = path if path is not None else get_settings().sweep_path
    with sweep_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return SweepConfig.model_validate(payload)
