# wizard_arena/llm_clients.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from a .env file if present.
load_dotenv()


@dataclass(frozen=True)
class ModelSpec:
    """Parsed representation of a model identifier like 'openai:gpt-4o'.

    provider: one of "openai", "anthropic", "gemini", "grok".
    model: the provider-specific model name, e.g. "gpt-4o" or
           "claude-sonnet-4-20250514".
    """
    provider: str
    model: str

    @classmethod
    def parse(cls, raw: str) -> "ModelSpec":
        if ":" not in raw:
            raise ValueError(
                f"Model string '{raw}' must be of the form '<provider>:<model_name>'"
            )
        provider, model = raw.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in {"openai", "anthropic", "gemini", "grok"}:
            raise ValueError(
                f"Unknown provider '{provider}'. Expected one of "
                f"'openai', 'anthropic', 'gemini', 'grok'."
            )
        if not model:
            raise ValueError(f"Model name missing in '{raw}'")
        return cls(provider=provider, model=model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def label(self) -> str:
        """Human-readable label for score tables, etc."""
        return str(self)


class LLMRouter:
    """Thin wrapper around multiple LLM providers.

    - OpenAI via the official `openai` Python SDK and the Responses API.
    - Anthropic via the `anthropic` Python SDK and the Messages API.
    - Google Gemini via the `google-genai` SDK (`google.genai`).
    - Grok via the OpenAI SDK, using the xAI base URL.

    The router exposes a single `complete()` method that returns plain text. The
    caller (e.g. an agent) is responsible for turning that text into structured
    decisions (bids, card indices, trump suit, etc.).
    """

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> None:
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None
        self._grok_client = None

        # Allow explicit API keys to override environment variables if desired.
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        # For Gemini, google-genai can pick up GEMINI_API_KEY or GOOGLE_API_KEY.
        self.gemini_api_key = (
            gemini_api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.xai_api_key = xai_api_key or os.getenv("XAI_API_KEY")

        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

    # --- Public API -----------------------------------------------------

    def complete(
        self,
        model_spec: ModelSpec,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a text completion from the given model."""
        provider = model_spec.provider
        model = model_spec.model
        max_tokens = max_output_tokens or self.max_output_tokens
        temp = self._effective_temperature(temperature)

        logger.debug(
            "LLMRouter.complete provider=%s model=%s max_output_tokens=%s temperature=%s",
            provider,
            model,
            max_tokens,
            temp,
        )

        if provider == "openai":
            return self._complete_openai(model, prompt, system_prompt, max_tokens, temp)
        if provider == "anthropic":
            return self._complete_anthropic(
                model, prompt, system_prompt, max_tokens, temp
            )
        if provider == "gemini":
            return self._complete_gemini(model, prompt, system_prompt, max_tokens, temp)
        if provider == "grok":
            return self._complete_grok(model, prompt, system_prompt, max_tokens, temp)

        raise ValueError(f"Unsupported provider: {provider}")

    # --- Provider-specific helpers -------------------------------------

    def _effective_temperature(self, override: Optional[float]) -> float:
        if override is None:
            return self.temperature
        return float(override)

    # OpenAI (Responses API)
    def _ensure_openai(self):
        if self._openai_client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required for provider 'openai'. "
                "Install with `pip install openai`."
            ) from exc

        # The `OpenAI` client picks up OPENAI_API_KEY from the environment by
        # default; we also allow overriding. :contentReference[oaicite:4]{index=4}
        self._openai_client = OpenAI(
            api_key=self.openai_api_key,
        )

    def _complete_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        self._ensure_openai()
        assert self._openai_client is not None

        # Use the Responses API with chat-style messages if a system prompt is
        # provided, otherwise send plain text input. :contentReference[oaicite:5]{index=5}
        if system_prompt:
            input_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            input_payload = prompt

        request_kwargs = {
            "model": model,
            "input": input_payload,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        try:
            response = self._openai_client.responses.create(**request_kwargs)
        except Exception as exc:  # pragma: no cover - network/API specific
            if self._is_temperature_unsupported_error(exc) and "temperature" in request_kwargs:
                logger.info(
                    "OpenAI model %s does not support temperature; retrying without it",
                    model,
                )
                request_kwargs.pop("temperature", None)
                response = self._openai_client.responses.create(**request_kwargs)
            else:
                raise
        # The official SDK exposes a convenience `output_text` property that
        # concatenates the text content from the response. :contentReference[oaicite:6]{index=6}
        text = response.output_text  # type: ignore[attr-defined]
        if not isinstance(text, str):
            text = json.dumps(text)
        return text

    def _is_temperature_unsupported_error(self, exc: Exception) -> bool:
        """Return True if an OpenAI error indicates temperature is unsupported."""
        message = getattr(exc, "message", None) or str(exc)
        lowered = message.lower()
        return "temperature" in lowered and "not supported" in lowered

    # Anthropic (Messages API)
    def _ensure_anthropic(self):
        if self._anthropic_client is not None:
            return
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "anthropic package is required for provider 'anthropic'. "
                "Install with `pip install anthropic`."
            ) from exc

        # The client uses ANTHROPIC_API_KEY from the environment. :contentReference[oaicite:7]{index=7}
        self._anthropic_client = anthropic.Anthropic(
            api_key=self.anthropic_api_key,
        )

    def _complete_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        self._ensure_anthropic()
        assert self._anthropic_client is not None

        # Messages API: messages=[{"role": "user", "content": "..."}],
        # optional system=... and max_tokens. :contentReference[oaicite:8]{index=8}
        message = self._anthropic_client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    # Google Gemini via google-genai
    def _ensure_gemini(self):
        if self._gemini_client is not None:
            return
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-genai package is required for provider 'gemini'. "
                "Install with `pip install google-genai`."
            ) from exc

        # Client can pick up GEMINI_API_KEY or GOOGLE_API_KEY, or use our override.
        # :contentReference[oaicite:9]{index=9}
        if self.gemini_api_key:
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self._gemini_client = genai.Client()

    def _complete_gemini(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        self._ensure_gemini()
        assert self._gemini_client is not None
        from google.genai import types  # type: ignore

        # Use generate_content with a GenerateContentConfig for system + options. :contentReference[oaicite:10]{index=10}
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        response = self._gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            return str(response)
        return text

    # Grok via OpenAI SDK pointed at xAI's API endpoint
    def _ensure_grok(self):
        if self._grok_client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required for provider 'grok'. "
                "Install with `pip install openai`."
            ) from exc

        if not self.xai_api_key:
            raise RuntimeError(
                "XAI_API_KEY environment variable is required to use provider 'grok'."
            )

        # xAI's docs show using either curl against /chat/completions or an SDK.
        # Here we reuse the OpenAI client with base_url="https://api.x.ai/v1". :contentReference[oaicite:11]{index=11}
        self._grok_client = OpenAI(
            api_key=self.xai_api_key,
            base_url="https://api.x.ai/v1",
        )

    def _complete_grok(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        self._ensure_grok()
        assert self._grok_client is not None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self._grok_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        choice = completion.choices[0]
        content = getattr(choice.message, "content", "")  # type: ignore[attr-defined]
        if isinstance(content, str):
            return content
        try:
            return "".join(part["text"] for part in content if "text" in part)  # type: ignore[index]
        except Exception:
            return str(content)
