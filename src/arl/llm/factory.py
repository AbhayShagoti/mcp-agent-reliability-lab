"""Pick an LLM adapter from settings. The default is the mock."""

from __future__ import annotations

from arl.config import Settings
from arl.llm.http import AnthropicLLM, OllamaLLM, OpenAICompatibleLLM, OpenAILLM
from arl.llm.mock import MockLLM
from arl.llm.types import LLMClient


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLM(mode="policy")
    if provider == "ollama":
        return OllamaLLM(settings.ollama_base_url, settings.ollama_model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; refusing to start the OpenAI provider")
        return OpenAILLM(settings.openai_api_key, settings.openai_model)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; refusing to start the Anthropic provider")
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    if provider == "llamacpp":
        return OpenAICompatibleLLM(settings.llamacpp_base_url, settings.llamacpp_model)
    if provider == "hf":
        raise RuntimeError(
            "The hf provider is used by the Gradio Space (app/space), not the API server. Use mock or ollama here."
        )
    raise RuntimeError(f"unknown LLM provider: {provider}")
