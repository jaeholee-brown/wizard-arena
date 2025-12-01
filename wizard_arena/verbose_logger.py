# wizard_arena/verbose_logger.py
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


class VerboseGameLogger:
    """Accumulates detailed, turn-by-turn logs for Wizard games."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: List[str] = []
        self._lock = Lock()

    def log_interaction(
        self,
        *,
        agent_label: str,
        purpose: str,
        game_id: Optional[str],
        round_index: Optional[int],
        phase: Optional[str],
        prompt: str,
        observation_json: str,
        raw_output: Optional[str],
        rationale_text: Optional[str],
        final_json_text: Optional[str],
        parsed_json: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        header_parts = [
            f"Agent: {agent_label}",
            f"Purpose: {purpose}",
        ]
        if game_id is not None:
            header_parts.append(f"Game: {game_id}")
        if round_index is not None:
            header_parts.append(f"Round: {round_index}")
        if phase is not None:
            header_parts.append(f"Phase: {phase}")
        header = " | ".join(header_parts)

        lines = [
            f"=== {header} ===",
            "Prompt:",
            prompt.strip(),
            "",
            "Observation JSON:",
            observation_json.strip(),
            "",
            "Raw output:",
            (raw_output or "").strip(),
        ]
        if rationale_text:
            lines.extend(
                [
                    "",
                    "Rationale before FINAL_JSON:",
                    rationale_text.strip(),
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
        if parsed_json is not None:
            lines.extend(
                [
                    "",
                    "Parsed JSON object:",
                    repr(parsed_json),
                ]
            )
        if error:
            lines.extend(
                [
                    "",
                    f"Error: {error}",
                ]
            )

        entry = "\n".join(lines).strip()
        with self._lock:
            self._entries.append(entry)

    def flush(self) -> None:
        with self._lock:
            if not self._entries:
                return
            to_write = "\n\n".join(self._entries)
            self._entries.clear()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(to_write, encoding="utf-8")


class FailureLogger:
    """Captures only failure cases so they are recorded separately."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: List[str] = []
        self._lock = Lock()

    def log_failure(
        self,
        *,
        agent_label: str,
        purpose: str,
        game_id: Optional[str],
        round_index: Optional[int],
        phase: Optional[str],
        error: str,
        prompt: Optional[str] = None,
        observation_json: Optional[str] = None,
        raw_output: Optional[str] = None,
        parsed_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        header_parts = [
            f"Agent: {agent_label}",
            f"Purpose: {purpose}",
        ]
        if game_id is not None:
            header_parts.append(f"Game: {game_id}")
        if round_index is not None:
            header_parts.append(f"Round: {round_index}")
        if phase is not None:
            header_parts.append(f"Phase: {phase}")
        header = " | ".join(header_parts)

        lines = [
            f"=== {header} ===",
            f"Error: {error}",
        ]
        if prompt:
            lines.extend(["", "Prompt:", prompt.strip()])
        if observation_json:
            lines.extend(["", "Observation JSON:", observation_json.strip()])
        if raw_output:
            lines.extend(["", "Raw output:", raw_output.strip()])
        if parsed_json is not None:
            lines.extend(["", "Parsed JSON object:", repr(parsed_json)])

        entry = "\n".join(lines).strip()
        with self._lock:
            self._entries.append(entry)

    def flush(self) -> None:
        with self._lock:
            if not self._entries:
                return
            to_write = "\n\n".join(self._entries)
            self._entries.clear()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(to_write, encoding="utf-8")
