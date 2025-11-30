# wizard_arena/game_log.py
from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

from .rules import score_round
from .state import GameState, PlayerState, RoundState

FIELDNAMES = [
    "game_id",
    "round_index",
    "cards_per_player",
    "dealer_id",
    "player_id",
    "player_name",
    "bid",
    "tricks_won",
    "round_delta",
    "total_score",
    "trump_suit",
]


def _compute_tricks_won(
    round_state: RoundState,
    players: List[PlayerState],
) -> Dict[int, int]:
    """Count tricks won per player from the round's trick history."""
    counts: Dict[int, int] = {p.id: 0 for p in players}
    for trick in round_state.tricks:
        if trick.winner_id is None:
            raise ValueError("Trick without winner in RoundState")
        if trick.winner_id not in counts:
            raise ValueError(f"Unknown winner_id {trick.winner_id}")
        counts[trick.winner_id] += 1
    return counts


def _is_round_complete(round_state: RoundState, num_players: int) -> bool:
    """Return True if the round contains full bids and tricks for all players."""

    if len(round_state.bids) != num_players:
        return False

    if len(round_state.tricks) != round_state.cards_per_player:
        return False

    return all(trick.winner_id is not None for trick in round_state.tricks)


def build_round_score_rows(
    game_state: GameState,
    game_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build a list of rows summarizing per-round scores for CSV export.

    Each row corresponds to (round, player) and has keys in FIELDNAMES. Any
    round missing full bids or tricks is skipped so partially played games can
    still be logged.
    """
    players = game_state.players
    running_scores: Dict[int, int] = {p.id: 0 for p in players}
    rows: List[Dict[str, Any]] = []
    num_players = len(players)

    for round_state in game_state.rounds:
        if not _is_round_complete(round_state, num_players):
            continue
        tricks_won = _compute_tricks_won(round_state, players)
        deltas = score_round(round_state, players)

        for p in players:
            pid = p.id
            running_scores[pid] += deltas[pid]

            row: Dict[str, Any] = {
                "game_id": game_id,
                "round_index": round_state.round_index,
                "cards_per_player": round_state.cards_per_player,
                "dealer_id": round_state.dealer_id,
                "player_id": pid,
                "player_name": p.name,
                "bid": round_state.bids.get(pid, 0),
                "tricks_won": tricks_won[pid],
                "round_delta": deltas[pid],
                "total_score": running_scores[pid],
                "trump_suit": (
                    round_state.trump_suit.name
                    if round_state.trump_suit is not None
                    else None
                ),
            }
            rows.append(row)

    return rows


def write_round_scores_csv(
    game_state: GameState,
    path,
    game_id: Optional[str] = None,
) -> None:
    """
    Write per-round scores to a CSV file.

    `path` can be a string or any path-like object accepted by `open`.
    """
    rows = build_round_score_rows(game_state, game_id=game_id)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDNAMES})
