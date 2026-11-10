from __future__ import annotations

from pathlib import Path

import pytest

from synthgen.config import load_settings, read_environment
from synthgen.domain.errors import ConfigError

SETTINGS_FILE = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def test_defaults_load_with_spec_values() -> None:
    settings = load_settings(SETTINGS_FILE, {})
    assert settings.llm.provider == "stub"
    assert settings.llm.model == "openai/gpt-oss-120b"
    assert settings.llm.api_key is None
    assert settings.profiling.sample_size == 200_000
    assert settings.generation.chunk_rows == 250_000
    caps = settings.agent.caps
    assert (caps.clarify, caps.plan_rounds, caps.planner_tool_calls) == (2, 3, 8)
    assert (caps.spec_repair, caps.validation_repair, caps.llm_reprompts) == (3, 3, 3)
    assert settings.validation.ks_max == 0.10
    assert settings.paths.state_db_path == Path("data") / "state.db"


def test_prefixed_env_overrides_nested_value() -> None:
    settings = load_settings(
        SETTINGS_FILE,
        {"SYNTHGEN_PATHS__DATA_DIR": "/srv/elsewhere", "SYNTHGEN_AGENT__CAPS__SPEC_REPAIR": "5"},
    )
    assert settings.paths.data_dir == Path("/srv/elsewhere")
    assert settings.agent.caps.spec_repair == 5


def test_llm_env_names_override_llm_group() -> None:
    settings = load_settings(
        SETTINGS_FILE,
        {"LLM_PROVIDER": "openai_compatible", "LLM_API_KEY": "secret-value", "LLM_MODEL": "m"},
    )
    assert settings.llm.provider == "openai_compatible"
    assert settings.llm.model == "m"
    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_unrelated_env_is_ignored() -> None:
    assert load_settings(SETTINGS_FILE, {"HOME": "/x", "PATH": "/y"}).llm.provider == "stub"


@pytest.mark.parametrize(
    "environ",
    [
        {"LLM_PROVIDER": "anthropic"},
        {"SYNTHGEN_GENERATION__CHUNK_ROWS": "0"},
        {"SYNTHGEN_VALIDATION__KS_MAX": "not-a-number"},
        {"SYNTHGEN_PATHS__UNKNOWN_KEY": "x"},
        {"SYNTHGEN_PATHS__DATA_DIR__DEEPER": "x"},
    ],
)
def test_invalid_values_raise_config_error(environ: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        load_settings(SETTINGS_FILE, environ)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "absent.yaml", {})


def test_non_mapping_file_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path, {})


def test_env_file_is_merged_under_real_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-file\nLLM_BASE_URL=http://file\n", encoding="utf-8")
    merged = read_environment({"LLM_MODEL": "from-env"}, env_file)
    assert merged == {"LLM_MODEL": "from-env", "LLM_BASE_URL": "http://file"}


def test_missing_env_file_is_optional(tmp_path: Path) -> None:
    assert read_environment({"A": "1"}, tmp_path / ".env") == {"A": "1"}
    assert read_environment({"A": "1"}, None) == {"A": "1"}
