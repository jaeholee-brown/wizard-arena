from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents.llm_agents import (
    LLMCallFailed,
    ParsedModelResponse,
    LLMWizardAgent,
    WIZARD_RULES_SUMMARY,
)
from .engine import GameEngine
from .llm_clients import LLMRouter, ModelSpec


class InteractionLogger:
    """Lightweight text logger for capturing raw LLM outputs."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._chunks: List[str] = []

    def log(
        self,
        *,
        agent_label: str,
        purpose: str,
        attempt: int,
        max_attempts: int,
        temperature: float,
        max_output_tokens: int,
        system_prompt: Optional[str],
        prompt: str,
        observation_json: str,
        raw_output: Optional[str],
        parsed_json: Optional[Dict[str, Any]] = None,
        rationale: Optional[str] = None,
        final_json_text: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        lines = [
            f"=== {agent_label} | {purpose} ===",
            f"Call: attempt {attempt}/{max_attempts} | "
            f"temperature={temperature} | max_output_tokens={max_output_tokens}",
            f"System prompt: {system_prompt or '<none>'}",
            "",
            "Prompt sent to LLM:",
            prompt.strip(),
            "",
            "Observation JSON (embedded in prompt):",
            observation_json.strip(),
            "",
            "Raw output:",
            (raw_output or "").strip(),
        ]
        if parsed_json is not None:
            lines.extend(
                [
                    "",
                    "Parsed JSON:",
                    repr(parsed_json),
                ]
            )
        if rationale:
            lines.extend(
                [
                    "",
                    "Model rationale before FINAL_JSON:",
                    rationale.strip(),
                ]
            )
        if final_json_text:
            lines.extend(
                [
                    "",
                    "FINAL_JSON block:",
                    final_json_text.strip(),
                ]
            )
        if error is not None:
            lines.extend(
                [
                    "",
                    f"Error: {error}",
                ]
            )

        self._chunks.append("\n".join(lines).strip())

    def flush(self) -> None:
        self.output_path.write_text("\n\n".join(self._chunks), encoding="utf-8")


class LoggingLLMWizardAgent(LLMWizardAgent):
    """LLM agent that records raw model outputs via an InteractionLogger."""

    def __init__(
        self,
        *args: Any,
        interaction_logger: InteractionLogger,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._interaction_logger = interaction_logger

    def _safe_complete(
        self,
        *,
        purpose: str,
        observation: Dict[str, Any],
        system_prompt: str,
        instruction_suffix: str,
    ) -> Optional[Dict[str, Any]]:
        prompt, obs_json = self._build_prompt(
            observation=observation,
            instruction_suffix=instruction_suffix,
        )

        raw_output: Optional[str] = None
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_api_retries + 1):
            try:
                raw_output = self.router.complete(
                    self.model_spec,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._interaction_logger.log(
                    agent_label=self.label,
                    purpose=purpose,
                    attempt=attempt,
                    max_attempts=self._max_api_retries,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    observation_json=obs_json,
                    raw_output=raw_output,
                    parsed_json=None,
                    rationale=None,
                    final_json_text=None,
                    error=str(exc),
                )
                logging.getLogger(__name__).warning(
                    "LLM %s failed for %s (attempt %d/%d): %s",
                    self.label,
                    purpose,
                    attempt,
                    self._max_api_retries,
                    exc,
                )
                if attempt < self._max_api_retries:
                    time.sleep(self._retry_delay_seconds)
                else:
                    raise LLMCallFailed(
                        label=self.label,
                        purpose=purpose,
                        attempts=self._max_api_retries,
                        error=last_error,
                    )
                continue

            try:
                parsed = self._parse_model_response(raw_output)
            except Exception as exc:  # noqa: BLE001
                self._interaction_logger.log(
                    agent_label=self.label,
                    purpose=purpose,
                    attempt=attempt,
                    max_attempts=self._max_api_retries,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    observation_json=obs_json,
                    raw_output=raw_output,
                    parsed_json=None,
                    rationale=None,
                    final_json_text=None,
                    error=str(exc),
                )
                logging.getLogger(__name__).warning(
                    "LLM %s returned unparsable output for %s: %s",
                    self.label,
                    purpose,
                    exc,
                )
                return None

            self._interaction_logger.log(
                agent_label=self.label,
                purpose=purpose,
                attempt=attempt,
                max_attempts=self._max_api_retries,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                system_prompt=system_prompt,
                prompt=prompt,
                observation_json=obs_json,
                raw_output=raw_output,
                parsed_json=parsed.data if parsed else None,
                rationale=parsed.rationale if isinstance(parsed, ParsedModelResponse) else None,
                final_json_text=parsed.final_json_text if isinstance(parsed, ParsedModelResponse) else None,
                error=None,
            )
            return parsed.data if parsed else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single Wizard round and save the raw outputs from each "
            "LLM agent to a text file."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Exactly three models of the form '<provider>:<model_name>'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_round_outputs.txt"),
        help="Where to write the raw LLM outputs (default: debug_round_outputs.txt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment and fallback agents.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM agents.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max output tokens requested from each LLM call (default: 256).",
    )
    parser.add_argument(
        "--cards-per-player",
        type=int,
        default=1,
        help="Number of cards to deal each player (default: 1, i.e., round 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    models = args.models
    if len(models) != 3:
        raise SystemExit(
            f"Expected exactly three models for this debug run; got {len(models)}."
        )

    logger = InteractionLogger(args.output)
    router = LLMRouter(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    agents = [
        LoggingLLMWizardAgent(
            ModelSpec.parse(model),
            router=router,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed + idx,
            interaction_logger=logger,
        )
        for idx, model in enumerate(models)
    ]

    engine = GameEngine(
        agents=agents,
        player_names=[agent.label for agent in agents],
        rng_seed=args.seed,
    )

    round_state = engine._start_round(  # noqa: SLF001
        round_index=0,
        cards_per_player=args.cards_per_player,
        dealer_id=0,
        is_last_round=False,
    )
    engine.game_state.current_round_index = 0
    engine.game_state.rounds.append(round_state)

    engine._bidding_phase(round_state)  # noqa: SLF001
    engine._trick_phase(round_state)  # noqa: SLF001
    engine._score_round(round_state)  # noqa: SLF001

    logger.flush()
    logging.info("Wrote raw outputs to %s", args.output)


if __name__ == "__main__":
    main()

'''
python3 -m wizard_arena.debug_round \
  --models openai:gpt-4o-mini anthropic:claude-3-5-haiku-latest gemini:gemini-2.0-flash-lite \
  --output debugging_unparse.txt \
  --seed 0 \
  --cards-per-player 2
'''
