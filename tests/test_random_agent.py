# tests/test_random_agent.py
import random

from wizard_arena.agents.random_agent import RandomWizardAgent
from wizard_arena.cards import Card, CardType, Suit, card_to_dict


def test_random_agent_choose_bid_range():
    rng = random.Random(123)
    agent = RandomWizardAgent(rng=rng)

    hand_cards = [
        Card(CardType.NUMBER, Suit.RED, 5),
        Card(CardType.JESTER),
        Card(CardType.WIZARD),
    ]
    obs = {
        "phase": "bidding",
        "hand": [card_to_dict(c) for c in hand_cards],
        "bids_so_far": {},
        "game": {
            "round_index": 0,
            "cards_per_player": len(hand_cards),
            "num_players": 3,
        },
        "player": {"id": 0, "name": "A", "score": 0},
        "trump": {"suit": None, "card": None},
        "scores": {0: 0, 1: 0, 2: 0},
    }

    bid = agent.choose_bid(obs)
    assert isinstance(bid, int)
    assert 0 <= bid <= len(hand_cards)


def test_random_agent_choose_card_from_legal():
    rng = random.Random(0)
    agent = RandomWizardAgent(rng=rng)

    obs = {
        "phase": "play",
        "hand": [
            {"type": CardType.WIZARD.value, "suit": None, "rank": None},
            {"type": CardType.JESTER.value, "suit": None, "rank": None},
            {
                "type": CardType.NUMBER.value,
                "suit": Suit.RED.name,
                "rank": 7,
            },
        ],
        "legal_move_indices": [0, 2],
    }

    idx = agent.choose_card(obs)
    assert idx in [0, 2]


def test_random_agent_choose_trump_uses_majority_suit():
    rng = random.Random(456)
    agent = RandomWizardAgent(rng=rng)

    hand_cards = [
        Card(CardType.NUMBER, Suit.RED, 1),
        Card(CardType.NUMBER, Suit.RED, 2),
        Card(CardType.NUMBER, Suit.RED, 3),
        Card(CardType.NUMBER, Suit.GREEN, 5),
    ]
    obs = {
        "phase": "choose_trump",
        "hand_cards": hand_cards,
        "hand": [card_to_dict(c) for c in hand_cards],
    }

    chosen = agent.choose_trump(obs)
    # RED is the majority suit in the hand
    assert chosen == Suit.RED
