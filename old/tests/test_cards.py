# tests/test_cards.py
import random

from wizard_arena.cards import (
    Card,
    CardType,
    Deck,
    Suit,
    card_to_dict,
    dict_to_card,
)


def test_deck_composition():
    deck = Deck()
    assert len(deck.cards) == 60

    wizards = [c for c in deck.cards if c.type == CardType.WIZARD]
    jesters = [c for c in deck.cards if c.type == CardType.JESTER]
    assert len(wizards) == 4
    assert len(jesters) == 4

    for suit in Suit:
        nums = [
            c
            for c in deck.cards
            if c.type == CardType.NUMBER and c.suit == suit
        ]
        assert len(nums) == 13


def test_deal_basic():
    deck = Deck()
    deck.shuffle(random.Random(123))
    hands, remaining = deck.deal(4, 5)

    assert len(hands) == 4
    for hand in hands:
        assert len(hand) == 5

    assert len(remaining) == 60 - 4 * 5


def test_shuffle_determinism_with_seed():
    rng1 = random.Random(42)
    rng2 = random.Random(42)

    d1 = Deck()
    d2 = Deck()
    d1.shuffle(rng1)
    d2.shuffle(rng2)

    assert [str(c) for c in d1.cards] == [str(c) for c in d2.cards]


def test_card_roundtrip_dict():
    num_card = Card(CardType.NUMBER, Suit.RED, 7)
    wizard = Card(CardType.WIZARD)
    jester = Card(CardType.JESTER)

    for card in (num_card, wizard, jester):
        data = card_to_dict(card)
        card2 = dict_to_card(data)
        assert card == card2
