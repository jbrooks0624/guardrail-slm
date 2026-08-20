"""Training, sweep, and cost YAML defaults."""

from guardrail_slm.serve.config import load_cost
from guardrail_slm.train.config import load_sweep, load_train


def test_train_defaults() -> None:
    config = load_train()
    assert config.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.max_new_tokens == 1
    assert config.completion_only is True
    assert config.quantization.bnb_4bit_quant_type == "nf4"
    assert config.quantization.bnb_4bit_compute_dtype == "bfloat16"
    assert "q_proj" in config.lora.target_modules
    assert "down_proj" in config.lora.target_modules


def test_sweep_defaults() -> None:
    config = load_sweep()
    assert config.rank == [8, 16, 32]
    assert config.learning_rate == [5.0e-5, 1.0e-4, 2.0e-4]
    assert config.num_train_epochs == [2, 3]


def test_cost_defaults() -> None:
    config = load_cost()
    assert config.gpu.default == "l4"
    assert config.gpu.hourly_usd["l4"] == 0.44
    assert config.gpu.hourly_usd["a10g"] == 0.75
