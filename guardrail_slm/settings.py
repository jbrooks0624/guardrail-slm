"""Environment and filesystem settings for guardrail-slm."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "config" / "taxonomy.yaml"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "config" / "eval.yaml"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "config" / "train.yaml"
DEFAULT_SWEEP_PATH = PROJECT_ROOT / "config" / "sweep.yaml"
DEFAULT_COST_PATH = PROJECT_ROOT / "config" / "cost.yaml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


class Settings(BaseSettings):
    """Runtime settings loaded from the environment and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    hf_token: str = ""
    wandb_api_key: str = ""
    data_dir: Path = PROJECT_ROOT / "data"
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH
    eval_path: Path = DEFAULT_EVAL_PATH
    train_path: Path = DEFAULT_TRAIN_PATH
    sweep_path: Path = DEFAULT_SWEEP_PATH
    cost_path: Path = DEFAULT_COST_PATH
    results_dir: Path = DEFAULT_RESULTS_DIR

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def require_openai_api_key(self) -> str:
        """Return OPENAI_API_KEY, or raise if it is blank."""
        key = self.openai_api_key.strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key."
            )
        return key

    def require_anthropic_api_key(self) -> str:
        """Return ANTHROPIC_API_KEY, or raise if it is blank."""
        key = self.anthropic_api_key.strip()
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add a key."
            )
        return key

    def require_hf_token(self) -> str:
        """Return HF_TOKEN, or raise if it is blank."""
        token = self.hf_token.strip()
        if not token:
            raise ValueError(
                "HF_TOKEN is not set. Copy .env.example to .env and add a Hub token "
                "(needed for gated datasets such as allenai/wildjailbreak)."
            )
        return token

    def require_wandb_api_key(self) -> str:
        """Return WANDB_API_KEY, or raise if it is blank."""
        key = self.wandb_api_key.strip()
        if not key:
            raise ValueError(
                "WANDB_API_KEY is not set. Copy .env.example to .env and add a key."
            )
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
