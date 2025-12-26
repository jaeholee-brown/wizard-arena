# wizard_arena/agents/llm_agents.py
from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ..cards import Suit
from .base import WizardAgent
from .random_agent import RandomWizardAgent
from ..llm_clients import LLMRouter, ModelSpec
from ..verbose_logger import FailureLogger, VerboseGameLogger

ESSENTIAL_RULES = """
Essentials:
- 60-card deck: suits RED/BLUE/GREEN/YELLOW (number cards 1–13), plus 4 WIZARD and 4 JESTER cards (no suit).
- Trump: if the face-up card is a NUMBER, its suit is trump; if it is a JESTER, there is NO TRUMP; if it is a WIZARD, the dealer chooses any trump suit; in the last round there is NO TRUMP.
- Trick lead: the first NUMBER card played in a trick sets the led suit, unless the trick is led with a WIZARD (no led suit is set).
- If you have any NUMBER cards in the led suit you must play one of them, except that you may always play a WIZARD or JESTER instead.
- If there is no led suit yet (for example because the trick was led with a WIZARD or JESTER and no NUMBER has been played), any card is legal.
- Trick winner: if any WIZARD is played, the earliest WIZARD wins; otherwise, if there is a trump suit, the highest NUMBER in the trump suit wins; otherwise, the highest NUMBER in the led suit wins; if all cards are JESTERs, the earliest JESTER wins.
- Card indices in your hand are 0-indexed; you may only choose an index from `legal_move_indices` when playing a card.
- Objective: by the end of the round, win exactly the number of tricks you bid (no more and no less).
- Scoring: if tricks_won == bid then your score change is +20 + 10 * bid; otherwise your score change is -10 * abs(tricks_won - bid). Total score is cumulative across rounds.
- Information: you see only your own hand plus public data (trump, bids, tricks, scores). The environment has already computed `legal_move_indices` for valid plays.
""".strip()

logger = logging.getLogger(__name__)


@dataclass
class ParsedModelResponse:
    data: Dict[str, Any]
    rationale: str
    final_json_text: str


class LLMCallFailed(RuntimeError):
    """Raised when an LLM call exhausts all retry attempts."""

    def __init__(self, *, label: str, purpose: str, attempts: int, error: Exception):
        message = (
            f"LLM {label} failed for {purpose} after {attempts} attempts: {error}"
        )
        super().__init__(message)
        self.label = label
        self.purpose = purpose
        self.attempts = attempts
        self.error = error


class LLMWizardAgent(WizardAgent):
    """Wizard agent that delegates its decisions to an LLM.

    It uses an :class:`LLMRouter` to talk to different providers (OpenAI,
    Anthropic, Gemini, Grok) based on a ``ModelSpec`` such as
    ``openai:gpt-4o`` or ``anthropic:claude-opus-4.5``.

    The agent always *tries* to follow the JSON contract described in the
    prompts. If the model output can't be parsed or is out-of-bounds, it falls
    back to a simple :class:`RandomWizardAgent` so the game can continue. If an
    API call fails, the agent retries up to 5 times before raising
    :class:`LLMCallFailed` to halt the run.
    """

    def __init__(
        self,
        model: Union[str, ModelSpec],
        *,
        router: Optional[LLMRouter] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        seed: Optional[int] = None,
        verbose_logger: Optional[VerboseGameLogger] = None,
        failure_logger: Optional[FailureLogger] = None,
    ) -> None:
        if isinstance(model, ModelSpec):
            self.model_spec = model
        else:
            self.model_spec = ModelSpec.parse(model)

        self.router = router or LLMRouter(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

        # Fallback agent in case JSON parsing fails.
        self._rng = random.Random(seed)
        self._fallback = RandomWizardAgent(rng=self._rng)

        self._current_game_id: Optional[str] = None
        self._rules_prompt_sent = False
        self._verbose_logger = verbose_logger
        self._failure_logger = failure_logger

        # Retry behavior for provider/API failures.
        self._max_api_retries = 5
        self._retry_delay_seconds = 2.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        """Human-friendly label for logs and tables."""
        return self.model_spec.label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_observation(self, observation: Dict[str, Any]) -> str:
        """Turn a game observation into JSON for use in prompts."""
        try:
            return json.dumps(
                observation,
                default=str,
                sort_keys=True,
                indent=2,
            )
        except TypeError:
            return json.dumps(
                {"error": "failed to serialize observation"},
                indent=2,
            )

    def _describe_card(self, card: Dict[str, Any]) -> str:
        """Return a human-friendly string for a card dict."""
        ctype = (card or {}).get("type")
        if ctype == "number":
            suit = (card.get("suit") or "").upper()
            rank = card.get("rank")
            return f"{suit} {rank}"
        if ctype == "wizard":
            return "WIZARD"
        if ctype == "jester":
            return "JESTER"
        return "UNKNOWN"

    def _format_hand_indices(self, hand: list[Dict[str, Any]]) -> str:
        """Produce '0: RED 5, 1: JESTER' style mapping for the hand."""
        if not hand:
            return "<empty hand>"
        parts = [f"{i}: {self._describe_card(card)}" for i, card in enumerate(hand)]
        return ", ".join(parts)

    def _format_legal_options(
        self,
        hand: list[Dict[str, Any]],
        legal_indices: list[int],
    ) -> str:
        """Map legal indices to their card descriptions."""
        if not legal_indices:
            return "<none provided>"
        parts = []
        for idx in legal_indices:
            card = hand[idx] if 0 <= idx < len(hand) else None
            parts.append(f"{idx}: {self._describe_card(card or {})}")
        return ", ".join(parts)

    def _state_digest(self, observation: Dict[str, Any]) -> str:
        """Construct a compact digest for quick reference."""
        player = observation.get("player") or {}
        pid = player.get("id")

        trump = (observation.get("trump") or {}).get("suit") or "NONE"
        leader = observation.get("leader_id")
        bids = observation.get("bids") or observation.get("bids_so_far") or {}
        your_bid = bids.get(pid)
        tricks_taken = observation.get("tricks_taken_so_far") or {}
        tricks_won = tricks_taken.get(pid, 0)
        legal_indices = observation.get("legal_move_indices") or []
        hand = observation.get("hand") or []

        parts = [
            f"Player {pid} bid: {your_bid if your_bid is not None else '<none>'}; tricks_won: {tricks_won}",
            f"Trump: {trump}",
        ]
        if leader is not None:
            parts.append(f"Current trick leader: {leader}")
        parts.append(f"Cards in hand: {len(hand)}")
        if legal_indices:
            parts.append(
                f"Legal moves: {legal_indices} -> {self._format_legal_options(hand, legal_indices)}"
            )

        return "; ".join(parts)

    def _build_prompt(
        self,
        *,
        observation: Dict[str, Any],
        instruction_suffix: str,
    ) -> tuple[str, str]:
        """Construct the user prompt and return it with the obs JSON string."""
        game_id = (observation.get("game") or {}).get("game_id")
        if game_id != self._current_game_id:
            self._current_game_id = game_id
            self._rules_prompt_sent = False

        obs_json = self._sanitize_observation(observation)
        prefix_lines = [
            "You are playing the trick-taking card game Wizard. "
            "You must respect the game rules at all times.",
            ESSENTIAL_RULES,
        ]
        prefix_lines.append(
            "Follow the Wizard rules above exactly. They are fixed for this entire game and do not change between turns."
        )
        self._rules_prompt_sent = True

        digest = self._state_digest(observation)
        if digest:
            prefix_lines.append(
                "Quick state digest (for fast reference; use the JSON below for full details):"
            )
            prefix_lines.append(digest)

        prefix_lines.append(instruction_suffix)
        prefix_lines.append(
            "Here is a JSON description of the current game state from your perspective:"
        )
        prefix_lines.append(obs_json)
        prefix_lines.append(
            "If you include reasoning, place it before `FINAL_JSON:`. "
            "End with `FINAL_JSON:` followed by ONLY the JSON object; after the "
            "`FINAL_JSON:` label, no other text or punctuation should appear after the closing brace."
        )

        prompt = "\n\n".join(prefix_lines)
        return prompt, obs_json

    def _decode_json_prefix(self, text: str) -> tuple[Dict[str, Any], str]:
        """Decode the first JSON object found at the start of text."""
        decoder = json.JSONDecoder()
        start_brace = text.find("{")
        if start_brace == -1:
            raise ValueError("No JSON object found in model output")
        trimmed = text[start_brace:].lstrip()
        try:
            obj, end_idx = decoder.raw_decode(trimmed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode JSON: {exc}") from exc

        if isinstance(obj, list):
            if not obj:
                raise ValueError("Empty JSON list from model output")
            obj = obj[0]

        if not isinstance(obj, dict):
            raise ValueError(f"Expected a JSON object, got {type(obj)}")

        json_text = trimmed[:end_idx]
        return obj, json_text

    def _parse_model_response(self, text: str) -> ParsedModelResponse:
        """Extract rationale and JSON object using the FINAL_JSON delimiter."""
        cleaned = text.strip()
        # Regex keeps us resilient to stray punctuation or spacing around the delimiter.
        match = re.search(r"FINAL_JSON\s*:?", cleaned, flags=re.IGNORECASE)
        if match:
            rationale = cleaned[: match.start()].strip()
            candidate = cleaned[match.end() :].strip()
        else:
            rationale = ""
            candidate = cleaned

        obj, json_text = self._decode_json_prefix(candidate)
        return ParsedModelResponse(
            data=obj,
            rationale=rationale,
            final_json_text=json_text.strip(),
        )

    def _log_failure(
        self,
        *,
        purpose: str,
        observation: Dict[str, Any],
        prompt: Optional[str],
        observation_json: Optional[str],
        raw_output: Optional[str],
        parsed_json: Optional[Dict[str, Any]],
        error: str,
    ) -> None:
        if not self._failure_logger:
            return

        game_info = observation.get("game") or {}
        self._failure_logger.log_failure(
            agent_label=self.label,
            purpose=purpose,
            game_id=game_info.get("game_id"),
            round_index=game_info.get("round_index"),
            phase=observation.get("phase"),
            prompt=prompt,
            observation_json=observation_json,
            raw_output=raw_output,
            parsed_json=parsed_json,
            error=error,
        )

    def _safe_complete(
        self,
        *,
        purpose: str,
        observation: Dict[str, Any],
        system_prompt: str,
        instruction_suffix: str,
    ) -> Optional[Dict[str, Any]]:
        """Call the router and return parsed JSON.

        Returns ``None`` for unparsable responses and raises ``LLMCallFailed``
        after exhausting API retries.
        """
        prompt, obs_json = self._build_prompt(
            observation=observation,
            instruction_suffix=instruction_suffix,
        )
        last_error: Optional[Exception] = None
        parsed: Optional[ParsedModelResponse] = None
        raw_output: Optional[str] = None
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
                logger.warning(
                    "LLM %s failed for %s (attempt %d/%d): %s",
                    self.label,
                    purpose,
                    attempt,
                    self._max_api_retries,
                    exc,
                )
                if attempt < self._max_api_retries:
                    time.sleep(self._retry_delay_seconds)
                continue

            logger.debug(
                "LLM %s (%s) raw output: %s", self.label, purpose, raw_output
            )

            try:
                parsed = self._parse_model_response(raw_output)
            except Exception as exc:  # noqa: BLE001
                self._log_failure(
                    purpose=purpose,
                    observation=observation,
                    prompt=prompt,
                    observation_json=obs_json,
                    raw_output=raw_output,
                    parsed_json=None,
                    error=str(exc),
                )
                logger.warning(
                    "LLM %s returned unparsable output for %s: %s",
                    self.label,
                    purpose,
                    exc,
                )
                if self._verbose_logger:
                    self._verbose_logger.log_interaction(
                        agent_label=self.label,
                        purpose=purpose,
                        game_id=(observation.get("game") or {}).get("game_id"),
                        round_index=(observation.get("game") or {}).get(
                            "round_index"
                        ),
                        phase=observation.get("phase"),
                        prompt=prompt,
                        observation_json=obs_json,
                        raw_output=raw_output,
                        rationale_text=None,
                        final_json_text=None,
                        parsed_json=None,
                        error=str(exc),
                    )
                return None

            if self._verbose_logger and parsed is not None:
                self._verbose_logger.log_interaction(
                    agent_label=self.label,
                    purpose=purpose,
                    game_id=(observation.get("game") or {}).get("game_id"),
                    round_index=(observation.get("game") or {}).get(
                        "round_index"
                    ),
                    phase=observation.get("phase"),
                    prompt=prompt,
                    observation_json=obs_json,
                    raw_output=raw_output,
                    rationale_text=parsed.rationale,
                    final_json_text=parsed.final_json_text,
                    parsed_json=parsed.data,
                    error=None,
                )
            return parsed.data

        assert last_error is not None
        self._log_failure(
            purpose=purpose,
            observation=observation,
            prompt=prompt,
            observation_json=obs_json,
            raw_output=raw_output,
            parsed_json=None,
            error=f"LLM API failure after {self._max_api_retries} attempts: {last_error}",
        )
        raise LLMCallFailed(
            label=self.label,
            purpose=purpose,
            attempts=self._max_api_retries,
            error=last_error,
        )

    # ------------------------------------------------------------------
    # WizardAgent interface
    # ------------------------------------------------------------------

    def choose_bid(self, observation: Dict[str, Any]) -> int:
        """Choose how many tricks to bid for the current round."""
        game_info = observation.get("game", {})
        cards_per_player = int(game_info.get("cards_per_player", 0))

        system_prompt = (
            "You are a strong Wizard player focusing on accurate bidding. "
            "You want to maximize your final score, not just win individual rounds. "
            "Always output valid JSON."
        )
        instruction_suffix = (
            "Decide how many tricks you expect to win this round.\n"
            f"Legal bids are integers from 0 to {cards_per_player} inclusive.\n"
            "Your goal is to finish the round with tricks_won matching your bid, "
            "not to take every trick.\n"
            f"Your hand (index -> card): {self._format_hand_indices(observation.get('hand') or [])}\n"
            "Scoring reminder: if tricks_won == bid your score goes up by 20 + 10 * bid; "
            "otherwise your score decreases by 10 * abs(tricks_won - bid).\n"
            "Plan across the full round, and consider the number of players and tricks available to be won.\n"
            "You may include a rationale, then end with "
            'FINAL_JSON: {"bid": <integer>} with nothing after the closing brace.'
        )

        result = self._safe_complete(
            purpose="choose_bid",
            observation=observation,
            system_prompt=system_prompt,
            instruction_suffix=instruction_suffix,
        )

        if result is None or "bid" not in result:
            self._log_failure(
                purpose="choose_bid",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error="Missing bid in model response; using fallback bid.",
            )
            return self._fallback.choose_bid(observation)

        try:
            bid = int(result["bid"])
        except Exception:  # noqa: BLE001
            self._log_failure(
                purpose="choose_bid",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error=f"Non-integer bid produced: {result.get('bid')!r}; using fallback bid.",
            )
            logger.warning(
                "LLM %s produced non-integer bid: %r",
                self.label,
                result.get("bid"),
            )
            return self._fallback.choose_bid(observation)

        bid = max(0, min(cards_per_player, bid))
        return bid

    def choose_trump(self, observation: Dict[str, Any]) -> Suit:
        """Choose a trump suit when a Wizard is turned up as the trump card.

        Your engine calls this only in that situation, and expects a Suit enum.
        """
        # If the observation includes allowed trump suits, respect that.
        allowed = observation.get("allowed_trump_suits") or [
            "RED",
            "BLUE",
            "GREEN",
            "YELLOW",
        ]

        system_prompt = (
            "You are choosing a trump suit for the Wizard card game. "
            "Pick the suit that gives you the best strategic advantage "
            "based on your hand and the current round. "
            "Always output valid JSON."
        )
        allowed_str = ", ".join(allowed)
        top_card_desc = self._describe_card(observation.get("top_card") or {})
        game_info = observation.get("game") or {}
        round_number = int(game_info.get("round_index", 0)) + 1
        total_rounds = observation.get("rounds_total")
        tricks_this_round = observation.get("tricks_this_round")
        round_blurb = f"Round {round_number}" + (
            f" of {total_rounds}" if total_rounds is not None else ""
        )
        instruction_suffix = (
            "Choose one trump suit from the allowed options.\n"
            f"Allowed trump_suit values: {allowed_str}.\n"
            f"Top card was a Wizard ({top_card_desc}), so you may choose any trump suit.\n"
            f"{round_blurb}; tricks this round: {tricks_this_round}.\n"
            "Pick the suit that maximizes your expected total score by the end of the game.\n"
            "You may include a rationale, then end with "
            'FINAL_JSON: {"trump_suit": <string>} with nothing after the closing brace.'
        )

        result = self._safe_complete(
            purpose="choose_trump",
            observation=observation,
            system_prompt=system_prompt,
            instruction_suffix=instruction_suffix,
        )

        if result is None or "trump_suit" not in result:
            self._log_failure(
                purpose="choose_trump",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error="Missing trump_suit in model response; using fallback trump choice.",
            )
            return self._fallback.choose_trump(observation)

        raw_choice = result.get("trump_suit")
        if not isinstance(raw_choice, str):
            self._log_failure(
                purpose="choose_trump",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error=f"Non-string trump_suit produced: {raw_choice!r}; using fallback trump choice.",
            )
            logger.warning(
                "LLM %s produced non-string trump_suit: %r",
                self.label,
                raw_choice,
            )
            return self._fallback.choose_trump(observation)

        choice_upper = raw_choice.strip().upper()

        if choice_upper not in allowed:
            self._log_failure(
                purpose="choose_trump",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error=f"trump_suit {choice_upper!r} not in allowed set {allowed}; using fallback trump choice.",
            )
            logger.warning(
                "LLM %s chose trump_suit %r not in allowed set %s; falling back.",
                self.label,
                choice_upper,
                allowed,
            )
            return self._fallback.choose_trump(observation)

        try:
            return Suit[choice_upper]
        except KeyError:
            self._log_failure(
                purpose="choose_trump",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error=f"trump_suit {choice_upper!r} did not map to Suit enum; using fallback trump choice.",
            )
            logger.warning(
                "LLM %s chose trump_suit %r which doesn't map to Suit enum; "
                "falling back.",
                self.label,
                choice_upper,
            )
            return self._fallback.choose_trump(observation)

    def choose_card(self, observation: Dict[str, Any]) -> int:
        """Choose which card index to play for the current trick.

        The observation from your engine is expected to contain:

          - ``hand``: a JSON-serializable view of the hand (used in prompts).
          - ``legal_move_indices``: indices into the *current* hand that are legal.
          - additional context about the current trick and round.
        """
        legal_indices = observation.get("legal_move_indices")

        if not isinstance(legal_indices, list) or not legal_indices:
            self._log_failure(
                purpose="choose_card",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=None,
                error="Observation missing legal_move_indices; using fallback card.",
            )
            logger.warning(
                "Observation missing legal_move_indices; using fallback agent."
            )
            return self._fallback.choose_card(observation)

        system_prompt = (
            "You are choosing which card to play for this trick in Wizard. "
            "The environment has already computed `legal_move_indices`, which "
            "are the indices in your hand that you are allowed to play. "
            "You must pick one of those indices. "
            "Always output valid JSON."
        )
        instruction_suffix = (
            "Pick exactly one integer index from the `legal_move_indices` list "
            "in the observation. Card indices are 0-indexed within your current hand.\n"
            f"Hand with indices: {self._format_hand_indices(observation.get('hand') or [])}\n"
            f"Legal move options (index -> card): {self._format_legal_options(observation.get('hand') or [], legal_indices)}\n"
            "Scoring reminder: if tricks_won == bid your score goes up by 20 + 10 * bid; "
            "otherwise your score decreases by 10 * abs(tricks_won - bid).\n"
            "You may include a rationale, then end with "
            'FINAL_JSON: {"card_index": <integer>} with nothing after the closing brace.'
        )

        result = self._safe_complete(
            purpose="choose_card",
            observation=observation,
            system_prompt=system_prompt,
            instruction_suffix=instruction_suffix,
        )

        if result is None or "card_index" not in result:
            self._log_failure(
                purpose="choose_card",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error="Missing card_index in model response; using fallback card.",
            )
            return self._fallback.choose_card(observation)

        try:
            idx = int(result["card_index"])
        except Exception:  # noqa: BLE001
            self._log_failure(
                purpose="choose_card",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error=f"Non-integer card_index produced: {result.get('card_index')!r}; using fallback card.",
            )
            logger.warning(
                "LLM %s produced non-integer card_index: %r",
                self.label,
                result.get("card_index"),
            )
            return self._fallback.choose_card(observation)

        if idx not in legal_indices:
            self._log_failure(
                purpose="choose_card",
                observation=observation,
                prompt=None,
                observation_json=self._sanitize_observation(observation),
                raw_output=None,
                parsed_json=result,
                error=f"Illegal card_index {idx}; legal indices: {legal_indices}. Using fallback card.",
            )
            logger.warning(
                "LLM %s chose illegal card_index %s; legal indices: %s. Using fallback.",
                self.label,
                idx,
                legal_indices,
            )
            return self._fallback.choose_card(observation)

        return idx
