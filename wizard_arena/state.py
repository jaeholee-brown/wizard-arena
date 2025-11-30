# wizard_arena/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .cards import Card, Suit


@dataclass
class PlayerState:
    id: int
    name: str
    model_spec: Optional[str] = None  # e.g. "openai:gpt-5.1" (not used in step 1)
    score: int = 0
    current_bid: Optional[int] = None
    tricks_won_this_round: int = 0


@dataclass
class Trick:
    # (player_id, card) pairs in play order
    plays: List[tuple[int, Card]] = field(default_factory=list)
    # suit established for this trick, if any
    led_suit: Optional[Suit] = None
    winner_id: Optional[int] = None


@dataclass
class RoundState:
    round_index: int
    cards_per_player: int
    dealer_id: int
    trump_suit: Optional[Suit]
    trump_card: Optional[Card]
    hands: Dict[int, List[Card]]
    bids: Dict[int, int] = field(default_factory=dict)
    tricks: List[Trick] = field(default_factory=list)


@dataclass
class GameState:
    players: List[PlayerState]
    rounds: List[RoundState] = field(default_factory=list)
    current_round_index: int = 0

    @property
    def num_players(self) -> int:
        return len(self.players)
