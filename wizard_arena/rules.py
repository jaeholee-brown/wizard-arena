# wizard_arena/rules.py
from __future__ import annotations

from typing import Dict, List, Optional

from .cards import Card, CardType, Suit
from .state import PlayerState, RoundState, Trick


def determine_trump(
    top_card: Optional[Card],
    dealer_choice: Optional[Suit],
    is_last_round: bool,
) -> Optional[Suit]:
    """
    Determine trump suit for the round.

    - Last round: always no trump.
    - No top card: no trump.
    - Jester as top card: no trump.
    - Wizard as top card: dealer chooses trump (dealer_choice required).
    - Number card: suit of top card.
    """
    if is_last_round:
        return None
    if top_card is None:
        return None
    if top_card.type == CardType.JESTER:
        return None
    if top_card.type == CardType.WIZARD:
        if dealer_choice is None:
            raise ValueError("dealer_choice required when top card is Wizard")
        return dealer_choice
    # Number card
    return top_card.suit


def legal_moves(hand: List[Card], led_suit: Optional[Suit]) -> List[int]:
    """
    Return indices into `hand` that are legally playable given the led suit.

    Rules implemented:
    - If no led suit yet (first card of trick), any card is legal.
    - If player has NUMBER cards of the led suit, they must follow suit,
      but Wizards and Jesters are always allowed as alternatives.
    - If player cannot follow led suit (no NUMBER cards of that suit),
      they may play any card.
    """
    if led_suit is None:
        return list(range(len(hand)))

    wizard_or_jester = [
        i for i, c in enumerate(hand)
        if c.type in (CardType.WIZARD, CardType.JESTER)
    ]
    follow_suit = [
        i for i, c in enumerate(hand)
        if c.type == CardType.NUMBER and c.suit == led_suit
    ]

    if follow_suit:
        # Must follow suit with number cards, but Wizard/Jester are always allowed.
        return follow_suit + wizard_or_jester

    # Can't follow suit; anything goes.
    return list(range(len(hand)))


def winner_of_trick(trick: Trick, trump_suit: Optional[Suit]) -> int:
    """
    Determine the winner of a completed trick.

    Priority:
    1. First Wizard in the trick.
    2. Highest NUMBER card of trump suit (if any).
    3. Highest NUMBER card of led suit (if any).
    4. If all Jesters (or no trump/led suit candidates), first Jester wins.
    """
    if not trick.plays:
        raise ValueError("Cannot determine winner of an empty trick")

    # 1. First Wizard wins
    for player_id, card in trick.plays:
        if card.type == CardType.WIZARD:
            return player_id

    # 2. Highest trump number card, if any
    if trump_suit is not None:
        best_player: Optional[int] = None
        best_rank = -1
        for player_id, card in trick.plays:
            if card.type == CardType.NUMBER and card.suit == trump_suit:
                if card.rank is not None and card.rank > best_rank:
                    best_rank = card.rank
                    best_player = player_id
        if best_player is not None:
            return best_player

    # 3. Highest card of led suit, if any
    led_suit = trick.led_suit
    if led_suit is not None:
        best_player = None
        best_rank = -1
        for player_id, card in trick.plays:
            if card.type == CardType.NUMBER and card.suit == led_suit:
                if card.rank is not None and card.rank > best_rank:
                    best_rank = card.rank
                    best_player = player_id
        if best_player is not None:
            return best_player

    # 4. All Jesters / no candidates ⇒ first Jester wins
    for player_id, card in trick.plays:
        if card.type == CardType.JESTER:
            return player_id

    # Should be unreachable in Wizard
    raise RuntimeError("Failed to determine trick winner")


def score_round(round_state: RoundState, players: List[PlayerState]) -> Dict[int, int]:
    """
    Score a round according to Wizard scoring:

    For each player:
    - If tricks_won == bid: 20 + 10 * tricks_won
    - Else: −10 * abs(tricks_won − bid)
    """
    tricks_won: Dict[int, int] = {p.id: 0 for p in players}
    for trick in round_state.tricks:
        if trick.winner_id is None:
            raise ValueError("Trick has no winner set")
        tricks_won[trick.winner_id] += 1

    deltas: Dict[int, int] = {}
    for p in players:
        bid = round_state.bids.get(p.id, 0)
        won = tricks_won[p.id]
        if won == bid:
            delta = 20 + 10 * won
        else:
            delta = -10 * abs(won - bid)
        deltas[p.id] = delta

    return deltas
