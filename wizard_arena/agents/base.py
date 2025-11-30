# wizard_arena/agents/base.py
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from ..cards import Suit


@runtime_checkable
class WizardAgent(Protocol):
    """
    Interface that all Wizard players must implement.

    `observation` is a JSON-like dict containing:
      - game-level info (round, cards_per_player, etc.)
      - player info
      - trump info
      - phase-specific info (hand, legal moves, bids, trick history, etc.)
    """

    def choose_bid(self, observation: Dict[str, Any]) -> int:
        """Return the bid (0..cards_per_player)."""

        raise NotImplementedError

    def choose_card(self, observation: Dict[str, Any]) -> int:
        """
        Return the index into the player's current hand of the card to play.

        The observation will include:
          - "hand": list[card_dict]
          - "legal_move_indices": list[int]
        """
        raise NotImplementedError

    def choose_trump(self, observation: Dict[str, Any]) -> Suit:
        """
        When a Wizard is turned up as trump, the dealer chooses the trump suit.

        Observation includes the dealer's hand and round/game context.
        """
        raise NotImplementedError
