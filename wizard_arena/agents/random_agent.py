# wizard_arena/agents/random_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import random

from ..cards import Card, CardType, Suit
from .base import WizardAgent


@dataclass
class RandomWizardAgent(WizardAgent):
    """
    A simple baseline agent with a bit of structure:

    - choose_bid: heuristic based on number of non-Jester cards, then random jitter.
    - choose_card: pick uniformly among legal moves.
    - choose_trump: pick the suit you have most of among number cards, tie-break randomly.
    """

    rng: random.Random

    def choose_bid(self, observation: Dict[str, Any]) -> int:
        hand = observation["hand"]  # list of dicts from card_to_dict
        hand_size = len(hand)

        # Count cards that can reasonably win tricks (everything except Jesters).
        num_trick_potential = sum(
            1 for c in hand if c["type"] != CardType.JESTER.value
        )
        max_bid = hand_size

        # Base expectation ~ half of potential trick winners.
        expected = min(max_bid, max(0, num_trick_potential // 2))

        # Add small random jitter.
        low = max(0, expected - 1)
        high = min(max_bid, expected + 1)
        return self.rng.randint(low, high)

    def choose_card(self, observation: Dict[str, Any]) -> int:
        legal_indices = observation["legal_move_indices"]
        return self.rng.choice(legal_indices)

    def choose_trump(self, observation: Dict[str, Any]) -> Suit:
        """
        Choose the suit with the most number cards in our hand; break ties randomly.
        """
        hand_cards: list[Card] = observation["hand_cards"]
        counts = {suit: 0 for suit in Suit}
        for c in hand_cards:
            if c.type == CardType.NUMBER:
                counts[c.suit] += 1

        max_count = max(counts.values())
        candidates = [s for s, cnt in counts.items() if cnt == max_count]
        return self.rng.choice(candidates)
