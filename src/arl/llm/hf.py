"""Optional local Hugging Face generation. Weights are not downloaded in CI."""

from __future__ import annotations

from collections.abc import Callable

from arl.llm.types import ChatMessage


def render_prompt(messages: list[ChatMessage]) -> str:
    lines = []
    for message in messages:
        lines.append(f"{message.role.upper()}: {message.content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def generate(
    model_name: str,
    prompt: str,
    *,
    runner: Callable[[str, str], str] | None = None,
) -> str:
    """Generate text. Pass ``runner`` in tests to avoid loading weights."""
    if runner is not None:
        return runner(model_name, prompt)
    return _generate_with_transformers(model_name, prompt)


def _generate_with_transformers(model_name: str, prompt: str) -> str:  # pragma: no cover
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    result = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
    return str(result[0]["generated_text"])
