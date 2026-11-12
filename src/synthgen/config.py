"""Typed settings: YAML defaults from config/settings.yaml, overridden by environment variables.

The environment is passed in as a mapping so loading stays explicit and testable. Overrides use
``SYNTHGEN_<GROUP>__<KEY>`` (nested with ``__``) plus the unprefixed ``LLM_*`` names from SPEC 4.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from synthgen.domain.errors import ConfigError

ENV_PREFIX = "SYNTHGEN_"
ENV_NESTING = "__"
LLM_ENV_KEYS: dict[str, str] = {
    "LLM_PROVIDER": "provider",
    "LLM_BASE_URL": "base_url",
    "LLM_MODEL": "model",
    "LLM_API_KEY": "api_key",
    "LLM_STUB_FIXTURES": "stub_fixtures",
}

Ratio = Field(ge=0.0, le=1.0)


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LlmSettings(_Group):
    """LLM endpoint, sampling temperatures and HTTP resilience (SPEC 4.1, 4.4, 4.5)."""

    provider: Literal["stub", "openai_compatible"]
    base_url: str
    model: str
    api_key: SecretStr | None
    stub_fixtures: Path
    temperature: float = Field(ge=0.0, le=2.0)
    planner_temperature: float = Field(ge=0.0, le=2.0)
    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    backoff_base_seconds: float = Field(gt=0)
    backoff_max_seconds: float = Field(gt=0)
    backoff_jitter: float = Field(ge=0.0, le=1.0)
    output_mode: Literal["json_schema", "json_object", "prompt"]
    max_tokens: int | None = Field(ge=1)
    reasoning_effort: Literal["low", "medium", "high"] | None
    requests_per_minute: int = Field(ge=1)


class ProfilingSettings(_Group):
    """Sample sizes and statistic resolutions for the profiler (SPEC 8.2, 8.3)."""

    sample_size: int = Field(ge=1)
    max_sample_cells: int = Field(ge=1)
    max_select_columns: int = Field(ge=1, le=2000)
    sample_seed: int = Field(ge=0)
    cache_max_age_days: int = Field(ge=0)
    histogram_bins: int = Field(ge=1)
    quantile_points: int = Field(ge=2)
    percentiles: tuple[float, ...]
    max_full_frequencies: int = Field(ge=1)
    top_frequencies: int = Field(ge=1)
    rare_value_share: float = Ratio
    internal_min_count: int = Field(ge=1)
    shapiro_max_rows: int = Field(ge=3)
    fit_sample_size: int = Field(ge=10)
    max_fit_columns: int = Field(ge=0)
    kde_points: int = Field(ge=8)
    zscore_limit: float = Field(gt=0)
    modified_z_limit: float = Field(gt=0)
    iqr_fence: float = Field(gt=0)
    round_bases: tuple[float, ...]
    sentinel_dates: tuple[str, ...]
    weekend_days: tuple[int, ...]
    business_hours: tuple[int, int]
    max_series_months: int = Field(ge=1)
    min_series_days: int = Field(ge=3)
    max_series_days: int = Field(ge=3)
    max_pair_columns: int = Field(ge=2)
    pair_sample_size: int = Field(ge=10)
    max_pair_levels: int = Field(ge=2)
    strong_association: float = Field(gt=0, le=1)
    rule_min_share: float = Field(gt=0, le=1)
    max_arithmetic_columns: int = Field(ge=0)
    max_conditional_levels: int = Field(ge=1)
    max_parent_columns: int = Field(ge=0)
    parent_sample_rows: int = Field(ge=10)
    behaviour_entities: int = Field(ge=1)
    behaviour_key_batch: int = Field(ge=1)
    max_state_columns: int = Field(ge=0)
    max_label_columns: int = Field(ge=0)
    max_stratified_columns: int = Field(ge=0)
    separation_bins: int = Field(ge=2)
    card_top_values: int = Field(ge=1)
    mapper_batch_tables: int = Field(ge=1)
    mapper_batch_columns: int = Field(ge=1)
    card_max_columns: int = Field(ge=1)


class GenerationSettings(_Group):
    """Row targets, chunking and retries for the engine (SPEC 8.5, 10)."""

    default_root_rows: int = Field(ge=1)
    max_rows_per_table: int = Field(ge=1)
    chunk_rows: int = Field(ge=1)
    dry_run_max_root_entities: int = Field(ge=1)
    chunk_retries: int = Field(ge=0)
    text_pool_max_values: int = Field(ge=1)
    workers: int = Field(ge=1)
    semantic_pool_size: int = Field(ge=1)
    faker_locale: str
    birth_dates: tuple[date, date]
    max_copula_columns: int = Field(ge=2)
    copula_min_rho: float = Field(ge=0.0, le=1.0)
    max_sequence_events: int = Field(ge=1)


class LoopCaps(_Group):
    """Caps on every agent loop (SPEC 3.4). A reached cap ends the job as needs_review."""

    clarify: int = Field(ge=0)
    plan_rounds: int = Field(ge=0)
    planner_tool_calls: int = Field(ge=0)
    spec_repair: int = Field(ge=0)
    validation_repair: int = Field(ge=0)
    llm_reprompts: int = Field(ge=0)


class AgentSettings(_Group):
    """Agent loop caps and limits on tool results and pending plans."""

    caps: LoopCaps
    tool_result_max_tokens: int = Field(ge=1)
    plan_expiry_days: int = Field(ge=1)


class ValidationSettings(_Group):
    """Effect-size thresholds for the four validation families (SPEC 8.6)."""

    ks_max: float = Ratio
    wasserstein_iqr_max: float = Field(ge=0.0)
    js_category_max: float = Ratio
    null_rate_diff_max: float = Ratio
    spearman_diff_max: float = Field(ge=0.0, le=2.0)
    js_temporal_max: float = Ratio
    ks_children_max: float = Ratio
    row_count_tolerance: float = Ratio
    scenario_share_rel_tolerance: float = Ratio
    signature_pass_min: float = Ratio
    scenario_ks_min: float = Ratio
    privacy_sample_size: int = Field(ge=1)
    min_fidelity_rows: int = Field(ge=1)
    privacy_identifier_min_ratio: float = Ratio


class SeedSettings(_Group):
    """Fixed seed and size multiplier for the fake source databases (SPEC 11)."""

    random_seed: int = Field(ge=0)
    scale: float = Field(gt=0.0, le=10.0)


class PathSettings(_Group):
    """Locations of source data, state databases, metadata and output."""

    data_dir: Path
    output_dir: Path
    metadata_dir: Path
    templates_dir: Path | None
    state_db: str
    checkpoints_db: str

    @property
    def state_db_path(self) -> Path:
        """Full path of the state database (jobs, artifacts, LLM calls, profile cache)."""
        return self.data_dir / self.state_db

    @property
    def checkpoints_db_path(self) -> Path:
        """Full path of the LangGraph checkpoint database."""
        return self.data_dir / self.checkpoints_db


class Settings(_Group):
    """All settings, grouped as in SPEC 10."""

    llm: LlmSettings
    profiling: ProfilingSettings
    generation: GenerationSettings
    agent: AgentSettings
    validation: ValidationSettings
    seed: SeedSettings
    paths: PathSettings


def read_environment(environ: Mapping[str, str], env_file: Path | None) -> dict[str, str]:
    """Merge an optional .env file under the given environment; real variables win."""
    merged: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        merged.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    merged.update(environ)
    return merged


def load_settings(yaml_path: Path, environ: Mapping[str, str]) -> Settings:
    """Load YAML defaults, apply environment overrides and validate the result."""
    data = _read_yaml(yaml_path)
    for key, value in environ.items():
        if key in LLM_ENV_KEYS:
            _set_path(data, ["llm", LLM_ENV_KEYS[key]], value, key)
        elif key.startswith(ENV_PREFIX):
            path = key.removeprefix(ENV_PREFIX).lower().split(ENV_NESTING)
            _set_path(data, path, value, key)
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid settings: {exc}") from exc


def _read_yaml(yaml_path: Path) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read settings file {yaml_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"settings file {yaml_path} must contain a mapping")
    return loaded


def _set_path(data: dict[str, object], path: list[str], value: str, source: str) -> None:
    node = data
    for part in path[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"{source} overrides inside a non-group setting {part!r}")
        node = child
    node[path[-1]] = value
