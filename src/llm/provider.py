"""
LLM provider abstraction.

Default: Ollama (no API key required).
To switch providers update config/config.yaml and add the appropriate key to .env.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

# ── Message types ─────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str           # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)


# ── Base ──────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Common interface for every LLM backend."""

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        ...


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """Ollama running locally (or in another container)."""

    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int):
        import ollama as _ollama  # lazy import
        self._ollama = _ollama
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        # point the SDK at our host
        self._client = _ollama.Client(host=base_url)

    def _to_sdk_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        resp = self._client.chat(
            model=self.model,
            messages=self._to_sdk_messages(messages),
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                "num_ctx": kwargs.get("num_ctx", 2048),  # limit KV-cache RAM
            },
        )
        return LLMResponse(
            content=resp["message"]["content"],
            model=self.model,
            usage={
                "prompt_tokens": resp.get("prompt_eval_count", 0),
                "completion_tokens": resp.get("eval_count", 0),
            },
        )

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        for chunk in self._client.chat(
            model=self.model,
            messages=self._to_sdk_messages(messages),
            stream=True,
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                "num_ctx": kwargs.get("num_ctx", 2048),  # limit KV-cache RAM
            },
        ):
            yield chunk["message"]["content"]


# ── OpenAI-compatible (OpenAI, Azure, Groq, Together, …) ─────────────────────

class OpenAIProvider(LLMProvider):
    """Any OpenAI-compatible API. Requires OPENAI_API_KEY (or compatible var)."""

    def __init__(
        self,
        model: str,
        base_url: str | None,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
    ):
        from openai import OpenAI  # lazy import
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url,
        )

    def _to_sdk_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._to_sdk_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=self.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        for chunk in self._client.chat.completions.create(
            model=self.model,
            messages=self._to_sdk_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Anthropic Claude. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str, temperature: float, max_tokens: int):
        import anthropic  # lazy import
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _split_system(self, messages: list[Message]):
        system = next((m.content for m in messages if m.role == "system"), "")
        rest = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        return system, rest

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        system, rest = self._split_system(messages)
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=rest,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return LLMResponse(
            content=resp.content[0].text,
            model=self.model,
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
        )

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        system, rest = self._split_system(messages)
        with self._client.messages.stream(
            model=self.model,
            system=system,
            messages=rest,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        ) as stream:
            for text in stream.text_stream:
                yield text


# ── Factory ───────────────────────────────────────────────────────────────────

def build_provider(cfg: dict) -> LLMProvider:
    """Build the correct LLMProvider from the config dict."""
    provider = cfg.get("provider", "ollama").lower()
    model = os.environ.get("DEFAULT_MODEL") or cfg.get("model", "qwen2.5:7b")
    temperature = float(cfg.get("temperature", 0.7))
    max_tokens = int(cfg.get("max_tokens", 4096))

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaProvider(model=model, base_url=base_url, temperature=temperature, max_tokens=max_tokens)

    if provider in ("openai", "azure", "groq", "together"):
        base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAIProvider(model=model, base_url=base_url, temperature=temperature, max_tokens=max_tokens, api_key=api_key)

    if provider == "anthropic":
        return AnthropicProvider(model=model, temperature=temperature, max_tokens=max_tokens)

    raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: ollama, openai, anthropic")
