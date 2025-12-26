# wizard_arena/litellm_cost_tracker.py
from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import litellm
from litellm import completion_cost, token_counter

from .paths import ensure_results_dir

logger = logging.getLogger(__name__)

# Where we persist the cumulative running totals across runs.
DEFAULT_COST_PATH = ensure_results_dir() / "llm_costs.json"


@dataclass
class TokenUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TokenUsage":
        def _grab(*keys: str) -> Optional[int]:
            for key in keys:
                if key in raw and raw[key] is not None:
                    try:
                        return int(raw[key])
                    except Exception:
                        continue
            return None

        prompt = _grab("prompt_tokens", "input_tokens")
        completion = _grab("completion_tokens", "output_tokens", "output_tokens_count")
        total = _grab("total_tokens", "total_token_count", "total_tokens_count")
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    def merge_missing(self, other: "TokenUsage") -> "TokenUsage":
        """Fill in any None fields from another usage instance."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens
            if self.prompt_tokens is not None
            else other.prompt_tokens,
            completion_tokens=self.completion_tokens
            if self.completion_tokens is not None
            else other.completion_tokens,
            total_tokens=self.total_tokens
            if self.total_tokens is not None
            else other.total_tokens,
        )

    def has_any(self) -> bool:
        return (
            self.prompt_tokens is not None
            or self.completion_tokens is not None
            or self.total_tokens is not None
        )


def _empty_totals() -> Dict[str, Any]:
    return {
        "total_cost_usd": 0.0,
        "total_tokens": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "per_model": {},
        "updated_at": None,
    }


def _empty_model_totals() -> Dict[str, Any]:
    return {
        "cost_usd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


class LiteLLMCostTracker:
    """
    Keep a running (persisted) total of LLM spend using LiteLLM's cost helpers.

    - Uses LiteLLM's cost map (`completion_cost`) whenever token usage is known.
    - Persists totals to ``wizard_arena/results/llm_costs.json``.
    - Tracks both cumulative totals and per-run deltas for reporting.
    """

    def __init__(self, persist_path: Path = DEFAULT_COST_PATH) -> None:
        self.persist_path = persist_path
        self._lock = threading.Lock()
        self._running_totals = self._load_totals()
        self._session_totals = _empty_totals()
        self._run_start_totals = deepcopy(self._running_totals)
        self._callbacks_registered = False

    # ------------------------------------------------------------------ #
    # Persistence helpers                                               #
    # ------------------------------------------------------------------ #

    def _load_totals(self) -> Dict[str, Any]:
        ensure_results_dir()
        if not self.persist_path.exists():
            return _empty_totals()
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load LLM cost totals from %s: %s; starting fresh",
                self.persist_path,
                exc,
            )
            return _empty_totals()

        # Normalize structure in case of older/partial files.
        merged = _empty_totals()
        merged.update({k: v for k, v in data.items() if k in merged})
        per_model = data.get("per_model", {}) if isinstance(data, dict) else {}
        if isinstance(per_model, Mapping):
            merged["per_model"] = dict(per_model)
        return merged

    def _save_totals(self) -> None:
        ensure_results_dir()
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(self._running_totals, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------ #
    # LiteLLM callback wiring                                           #
    # ------------------------------------------------------------------ #

    def register_litellm_callback(self) -> None:
        """Attach this tracker's success callback to LiteLLM if not already."""
        with self._lock:
            if self._callbacks_registered:
                return
            callbacks = getattr(litellm, "success_callback", [])
            callback_fn = self._litellm_success_callback
            if callback_fn not in callbacks:
                callbacks = list(callbacks) + [callback_fn]
                litellm.success_callback = callbacks
            self._callbacks_registered = True

    def _litellm_success_callback(
        self,
        kwargs: Dict[str, Any],
        completion_response: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        LiteLLM-compatible callback that logs cost/usage for any SDK call.
        """
        model = kwargs.get("model") or getattr(completion_response, "model", None)
        usage = self._extract_usage(completion_response)
        latency_seconds = None
        try:
            latency_seconds = (end_time - start_time).total_seconds()
        except Exception:
            latency_seconds = None

        if model:
            self.record_completion(
                model=model,
                messages=kwargs.get("messages") or kwargs.get("input"),
                output_text=self._extract_text(completion_response),
                raw_usage=usage.to_dict() if usage else None,
                latency_seconds=latency_seconds,
            )

    # ------------------------------------------------------------------ #
    # Public API                                                        #
    # ------------------------------------------------------------------ #

    def start_run(self) -> None:
        """Capture the running total at the start of a run and reset session."""
        with self._lock:
            self._run_start_totals = deepcopy(self._running_totals)
            self._session_totals = _empty_totals()

    def record_completion(
        self,
        *,
        model: str,
        model_label: Optional[str] = None,
        messages: Optional[Iterable[Mapping[str, Any]]] = None,
        output_text: Optional[str] = None,
        raw_usage: Optional[Any] = None,
        latency_seconds: Optional[float] = None,
    ) -> None:
        """Record a single LLM call into cumulative and session totals."""
        usage = self._normalize_usage(
            model=model,
            raw_usage=raw_usage,
            messages=messages,
            output_text=output_text,
        )

        cost_usd: Optional[float] = None
        completion_response: Dict[str, Any] = {"model": model}
        if usage and usage.has_any():
            completion_response["usage"] = usage.to_dict()
            try:
                cost_usd = float(completion_cost(completion_response=completion_response))
            except Exception:
                logger.debug("LiteLLM cost calculation failed for %s", model, exc_info=True)

        model_key = model_label or model

        with self._lock:
            self._apply_update(self._running_totals, model_key, usage, cost_usd)
            self._apply_update(self._session_totals, model_key, usage, cost_usd)
            self._running_totals["updated_at"] = datetime.utcnow().isoformat()
            if latency_seconds is not None:
                # Latency isn't persisted, but logging it here keeps the interface open.
                logger.debug(
                    "LLM call latency model=%s latency=%.3fs cost=%s usage=%s",
                    model,
                    latency_seconds,
                    f"{cost_usd:.6f}" if cost_usd is not None else "<unknown>",
                    usage.to_dict() if usage else "<unknown>",
                )

    def summarize_run(self) -> Dict[str, Any]:
        """Return before/after totals plus the per-run delta."""
        with self._lock:
            return {
                "before": deepcopy(self._run_start_totals),
                "after": deepcopy(self._running_totals),
                "delta": deepcopy(self._session_totals),
            }

    def persist(self) -> None:
        with self._lock:
            self._save_totals()

    def log_run_summary(self) -> None:
        """Emit a human-readable summary for the current run."""
        summary = self.summarize_run()
        before = summary["before"]
        after = summary["after"]
        delta = summary["delta"]

        logger.info(
            "LLM spend summary: running total before $%.6f (prompt_tokens=%d, completion_tokens=%d, total_tokens=%d)",
            before["total_cost_usd"],
            before["total_tokens"]["prompt_tokens"],
            before["total_tokens"]["completion_tokens"],
            before["total_tokens"]["total_tokens"],
        )
        if delta["per_model"]:
            logger.info("This run spend by model:")
            for model, stats in sorted(delta["per_model"].items()):
                logger.info(
                    "  %s -> cost $%.6f, prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
                    model,
                    stats.get("cost_usd", 0.0),
                    stats.get("prompt_tokens", 0),
                    stats.get("completion_tokens", 0),
                    stats.get("total_tokens", 0),
                )
        else:
            logger.info("No LLM spend recorded for this run.")

        logger.info(
            "Running total after $%.6f (prompt_tokens=%d, completion_tokens=%d, total_tokens=%d)",
            after["total_cost_usd"],
            after["total_tokens"]["prompt_tokens"],
            after["total_tokens"]["completion_tokens"],
            after["total_tokens"]["total_tokens"],
        )

    # ------------------------------------------------------------------ #
    # Internals                                                         #
    # ------------------------------------------------------------------ #

    def _apply_update(
        self,
        totals: MutableMapping[str, Any],
        model: str,
        usage: Optional[TokenUsage],
        cost_usd: Optional[float],
    ) -> None:
        per_model = totals.setdefault("per_model", {})
        model_totals = per_model.get(model) or _empty_model_totals()
        per_model[model] = model_totals

        if cost_usd is not None:
            totals["total_cost_usd"] = float(totals.get("total_cost_usd", 0.0)) + cost_usd
            model_totals["cost_usd"] = float(model_totals.get("cost_usd", 0.0)) + cost_usd

        if usage:
            prompt = usage.prompt_tokens
            completion = usage.completion_tokens
            total = usage.total_tokens
            if total is None and prompt is not None and completion is not None:
                total = prompt + completion
            if prompt is not None:
                totals["total_tokens"]["prompt_tokens"] += prompt
                model_totals["prompt_tokens"] = model_totals.get("prompt_tokens", 0) + prompt
            if completion is not None:
                totals["total_tokens"]["completion_tokens"] += completion
                model_totals["completion_tokens"] = model_totals.get(
                    "completion_tokens", 0
                ) + completion
            if total is not None and isinstance(total, int):
                totals["total_tokens"]["total_tokens"] += total
                model_totals["total_tokens"] = model_totals.get("total_tokens", 0) + total

    def _normalize_usage(
        self,
        *,
        model: str,
        raw_usage: Optional[Any],
        messages: Optional[Iterable[Mapping[str, Any]]],
        output_text: Optional[str],
    ) -> Optional[TokenUsage]:
        usage = None
        if raw_usage:
            if isinstance(raw_usage, Mapping):
                usage = TokenUsage.from_mapping(raw_usage)
            else:
                usage = self._extract_usage(raw_usage)

        # If provider usage is missing fields, try to fill with token_counter estimates.
        estimated = self._estimate_usage(model, messages, output_text)
        if usage and estimated:
            usage = usage.merge_missing(estimated)
        elif estimated:
            usage = estimated

        return usage

    def _estimate_usage(
        self,
        model: str,
        messages: Optional[Iterable[Mapping[str, Any]]],
        output_text: Optional[str],
    ) -> Optional[TokenUsage]:
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        if messages:
            try:
                normalized_messages = list(messages)
                if all(isinstance(item, Mapping) for item in normalized_messages):
                    prompt_tokens = int(
                        token_counter(model=model, messages=normalized_messages)
                    )
            except Exception:
                logger.debug(
                    "LiteLLM prompt token estimation failed for %s",
                    model,
                    exc_info=True,
                )

        if output_text:
            try:
                completion_tokens = int(
                    token_counter(model=model, text=str(output_text))
                )
            except Exception:
                logger.debug(
                    "LiteLLM completion token estimation failed for %s",
                    model,
                    exc_info=True,
                )

        if prompt_tokens is None and completion_tokens is None:
            return None

        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _extract_text(self, completion_response: Any) -> Optional[str]:
        text = getattr(completion_response, "output_text", None)
        if isinstance(text, str):
            return text
        if hasattr(completion_response, "content"):
            try:
                content = completion_response.content
                if isinstance(content, str):
                    return content
            except Exception:
                pass
        if isinstance(completion_response, Mapping):
            # OpenAI/LLM-like responses.
            choices = completion_response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                msg_content = message.get("content")
                if isinstance(msg_content, str):
                    return msg_content
        return None

    def _extract_usage(self, response: Any) -> Optional[TokenUsage]:
        if response is None:
            return None

        if isinstance(response, Mapping):
            usage = response.get("usage") or response.get("usage_metadata")
            if isinstance(usage, Mapping):
                return TokenUsage.from_mapping(usage)

        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            if isinstance(usage_obj, Mapping):
                return TokenUsage.from_mapping(usage_obj)
            attr_map = {}
            for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens", "total_token_count"):
                if hasattr(usage_obj, key):
                    try:
                        attr_map[key] = getattr(usage_obj, key)
                    except Exception:
                        continue
            if attr_map:
                return TokenUsage.from_mapping(attr_map)

        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            if isinstance(usage_meta, Mapping):
                return TokenUsage.from_mapping(usage_meta)
            attr_map = {}
            for key in ("prompt_token_count", "input_tokens", "input_token_count"):
                if hasattr(usage_meta, key):
                    attr_map["prompt_tokens"] = getattr(usage_meta, key)
            for key in ("candidates_token_count", "completion_tokens", "output_tokens", "output_token_count"):
                if hasattr(usage_meta, key):
                    attr_map["completion_tokens"] = getattr(usage_meta, key)
            for key in ("total_token_count", "total_tokens"):
                if hasattr(usage_meta, key):
                    attr_map["total_tokens"] = getattr(usage_meta, key)
            if attr_map:
                return TokenUsage.from_mapping(attr_map)

        return None


# Global tracker instance used throughout the app.
running_cost_tracker = LiteLLMCostTracker()
running_cost_tracker.register_litellm_callback()
