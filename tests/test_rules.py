# tests/test_rules.py
import pytest

from wizard_arena.cards import Card, CardType, Suit
from wizard_arena.rules import (
    determine_trump,
    legal_moves,
    score_round,
    winner_of_trick,
)
from wizard_arena.state import PlayerState, RoundState, Trick


def test_determine_trump_basic_cases():
    top = Card(CardType.NUMBER, Suit.RED, 5)

    # Last round: no trump regardless of top card
    assert (
        determine_trump(top_card=top, dealer_choice=None, is_last_round=True)
        is None
    )

    # Jester as top card: no trump
    jester = Card(CardType.JESTER)
    assert (
        determine_trump(
            top_card=jester, dealer_choice=None, is_last_round=False
        )
        is None
    )

    # Number card: trump is suit of card
    res = determine_trump(
        top_card=top, dealer_choice=None, is_last_round=False
    )
    assert res == Suit.RED

    # Wizard requires dealer choice
    wiz = Card(CardType.WIZARD)
    with pytest.raises(ValueError):
        determine_trump(
            top_card=wiz, dealer_choice=None, is_last_round=False
        )
    res = determine_trump(
        top_card=wiz,
        dealer_choice=Suit.GREEN,
        is_last_round=False,
    )
    assert res == Suit.GREEN


def test_legal_moves_follow_suit_and_specials():
    hand = [
        Card(CardType.NUMBER, Suit.RED, 3),  # idx 0: follows
        Card(CardType.NUMBER, Suit.BLUE, 9),   # idx 1: other suit
        Card(CardType.WIZARD),                  # idx 2: wizard
        Card(CardType.JESTER),                  # idx 3: jester
        Card(CardType.NUMBER, Suit.RED, 10), # idx 4: follows
    ]
    idxs = legal_moves(hand, led_suit=Suit.RED)
    # Must follow suit with number cards, but Wizards/Jesters always allowed
    assert set(idxs) == {0, 2, 3, 4}


def test_legal_moves_when_cannot_follow_suit():
    hand = [
        Card(CardType.NUMBER, Suit.BLUE, 2),
        Card(CardType.WIZARD),
        Card(CardType.JESTER),
    ]
    idxs = legal_moves(hand, led_suit=Suit.RED)
    # No RED number cards -> any card is legal
    assert set(idxs) == {0, 1, 2}


def test_legal_moves_no_led_suit():
    hand = [Card(CardType.JESTER), Card(CardType.WIZARD)]
    idxs = legal_moves(hand, led_suit=None)
    assert set(idxs) == {0, 1}


def test_winner_of_trick_wizard_priority():
    trick = Trick()
    trick.led_suit = Suit.RED
    trick.plays = [
        (0, Card(CardType.NUMBER, Suit.RED, 13)),
        (1, Card(CardType.WIZARD)),
        (2, Card(CardType.NUMBER, Suit.RED, 1)),
    ]

    winner = winner_of_trick(trick, trump_suit=Suit.BLUE)
    assert winner == 1


def test_winner_of_trick_trump_over_led():
    trick = Trick()
    trick.led_suit = Suit.RED
    trick.plays = [
        (0, Card(CardType.NUMBER, Suit.RED, 13)),
        (1, Card(CardType.NUMBER, Suit.BLUE, 5)),
        (2, Card(CardType.NUMBER, Suit.BLUE, 11)),
    ]

    winner = winner_of_trick(trick, trump_suit=Suit.BLUE)
    assert winner == 2


def test_winner_of_trick_led_suit_when_no_trump():
    trick = Trick()
    trick.led_suit = Suit.GREEN
    trick.plays = [
        (0, Card(CardType.NUMBER, Suit.GREEN, 4)),
        (1, Card(CardType.NUMBER, Suit.BLUE, 12)),
        (2, Card(CardType.NUMBER, Suit.GREEN, 10)),
    ]

    winner = winner_of_trick(trick, trump_suit=None)
    assert winner == 2


def test_winner_of_trick_all_jesters():
    trick = Trick()
    trick.led_suit = None
    trick.plays = [
        (0, Card(CardType.JESTER)),
        (1, Card(CardType.JESTER)),
    ]

    winner = winner_of_trick(trick, trump_suit=None)
    assert winner == 0


def _make_round_state_for_scoring(
    trick_winners,
    bids,
) -> RoundState:
    tricks: list[Trick] = []
    for w in trick_winners:
        t = Trick()
        t.winner_id = w
        tricks.append(t)

    round_state = RoundState(
        round_index=0,
        cards_per_player=len(trick_winners),
        dealer_id=0,
        trump_suit=None,
        trump_card=None,
        hands={},
        bids=bids,
        tricks=tricks,
    )
    return round_state


def test_score_round_perfect_bids():
    players = [
        PlayerState(id=0, name="A"),
        PlayerState(id=1, name="B"),
    ]
    # Each wins exactly one trick, each bids 1
    round_state = _make_round_state_for_scoring(
        trick_winners=[0, 1],
        bids={0: 1, 1: 1},
    )

    deltas = score_round(round_state, players)
    # 20 + 10 * tricks == 20 + 10 * 1 = 30
    assert deltas[0] == 30
    assert deltas[1] == 30


def test_score_round_over_and_under_bids():
    players = [
        PlayerState(id=0, name="A"),
        PlayerState(id=1, name="B"),
    ]
    # Player 0 wins 2 but bid 1 -> -10
    # Player 1 wins 0 and bid 0 -> 20
    round_state = _make_round_state_for_scoring(
        trick_winners=[0, 0],
        bids={0: 1, 1: 0},
    )

    deltas = score_round(round_state, players)
    assert deltas[0] == -10
    assert deltas[1] == 20
