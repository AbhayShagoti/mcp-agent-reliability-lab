"""Runtime configuration. Paid providers stay off unless their env vars are set."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["mock", "ollama", "openai", "anthropic", "hf", "llamacpp"]


class Settings(BaseSettings):
    """Environment-driven settings.

    The default provider is the deterministic mock, so tests and CI never need an API key.
    """

    model_config = SettingsConfigDict(env_prefix="ARL_", extra="ignore")

    llm_provider: LLMProvider = "mock"
    api_token: str = "dev-token"
    database_url: str = "sqlite:///./data/ops.db"
    redis_url: str | None = None
    otel_endpoint: str | None = None
    service_name: str = "arl-api"
    approval_threshold_cents: int = 5000
    approval_secret: str = "dev-approval-secret"
    today: str = "2026-11-01"
    orders_mcp_url: str | None = None
    policy_mcp_url: str | None = None
    notify_mcp_url: str | None = None
    rate_limit_per_minute: int = 60
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b-instruct"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"
    hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llamacpp_base_url: str = "http://127.0.0.1:8088/v1"
    llamacpp_model: str = "qwen2.5-1.5b-instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
