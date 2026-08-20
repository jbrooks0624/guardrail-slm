"""Settings paths and required secrets."""

import pytest
from guardrail_slm.settings import Settings


def test_data_dirs_are_under_data_dir(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    assert settings.raw_dir == tmp_path / "raw"
    assert settings.interim_dir == tmp_path / "interim"
    assert settings.processed_dir == tmp_path / "processed"


def test_require_openai_api_key_raises_when_blank() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(openai_api_key="  ").require_openai_api_key()
