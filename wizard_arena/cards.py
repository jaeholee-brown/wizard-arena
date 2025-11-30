# wizard_arena/cards.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import enum
import random


class Suit(enum.Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"


class CardType(enum.Enum):
    NUMBER = "number"
    WIZARD = "wizard"
    JESTER = "jester"


@dataclass(frozen=True)
class Card:
    """
    Representation of a Wizard card.

    - NUMBER cards: type=NUMBER, suit in Suit, rank 1–13.
    - WIZARD / JESTER: type=WIZARD/JESTER, suit=None, rank=None.
    """
    type: CardType
    suit: Optional[Suit] = None
    rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.type == CardType.NUMBER:
            if self.suit is None or self.rank is None:
                raise ValueError("Number cards must have suit and rank")
            if not (1 <= self.rank <= 13):
                raise ValueError("Number card rank must be between 1 and 13")
        else:
            if self.suit is not None or self.rank is not None:
                raise ValueError("Wizard/Jester cards must not have suit or rank")

    def __str__(self) -> str:
        if self.type == CardType.NUMBER:
            return f"{self.suit.name.title()} {self.rank}"
        return self.type.name.title()


def card_to_dict(card: Card) -> Dict[str, Any]:
    """Convert a Card to a JSON-serializable dict."""
    return {
        "type": card.type.value,
        "suit": card.suit.name if card.suit is not None else None,
        "rank": card.rank,
    }


def dict_to_card(data: Dict[str, Any]) -> Card:
    """Convert a dict back into a Card."""
    ctype = CardType(data["type"])
    if ctype == CardType.NUMBER:
        suit = Suit[data["suit"]]
        rank = int(data["rank"])
        return Card(type=ctype, suit=suit, rank=rank)
    return Card(type=ctype)


class Deck:
    """
    A standard Wizard deck:
    - 52 number cards: 4 suits × ranks 1–13
    - 4 Wizards
    - 4 Jesters
    """

    def __init__(self) -> None:
        self.cards: List[Card] = []
        # Number cards
        for suit in Suit:
            for rank in range(1, 14):
                self.cards.append(Card(CardType.NUMBER, suit, rank))
        # 4 Wizards and 4 Jesters
        for _ in range(4):
            self.cards.append(Card(CardType.WIZARD))
            self.cards.append(Card(CardType.JESTER))

        if len(self.cards) != 60:
            raise RuntimeError("Deck must contain exactly 60 cards")

    def shuffle(self, rng: Optional[random.Random] = None) -> None:
        """Shuffle the deck in place. Uses provided RNG if given."""
        if rng is None:
            random.shuffle(self.cards)
        else:
            rng.shuffle(self.cards)

    def deal(
        self,
        num_players: int,
        cards_per_player: int,
    ) -> Tuple[List[List[Card]], List[Card]]:
        """
        Deal cards to players.

        Returns (hands, remaining_cards), where:
        - hands: list of length num_players, each a list[Card] of length cards_per_player
        - remaining_cards: the undealt cards (for trump determination, etc.)
        """
        total_needed = num_players * cards_per_player
        if total_needed > len(self.cards):
            raise ValueError("Not enough cards in deck to deal")

        hands: List[List[Card]] = [[] for _ in range(num_players)]
        idx = 0
        for _ in range(cards_per_player):
            for p in range(num_players):
                hands[p].append(self.cards[idx])
                idx += 1

        remaining = self.cards[idx:]
        return hands, remaining
