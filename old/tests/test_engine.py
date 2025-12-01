# tests/test_engine.py
import random

from wizard_arena.agents.random_agent import RandomWizardAgent
from wizard_arena.engine import GameEngine
from wizard_arena.rules import score_round


def _make_engine(num_players: int = 3) -> GameEngine:
    agents = [
        RandomWizardAgent(rng=random.Random(100 + i))
        for i in range(num_players)
    ]
    names = [f"P{i}" for i in range(num_players)]
    engine = GameEngine(agents=agents, player_names=names, rng_seed=999)
    return engine


def test_full_game_basic_invariants():
    engine = _make_engine(3)
    game_state = engine.play_game()

    assert len(game_state.players) == 3
    assert engine.max_rounds == 60 // 3
    assert len(game_state.rounds) == engine.max_rounds

    num_players = game_state.num_players

    for r in game_state.rounds:
        # One trick per card in hand
        assert len(r.tricks) == r.cards_per_player
        # A bid from each player
        assert len(r.bids) == num_players

        for pid in range(num_players):
            # Hands should be empty at end of round
            assert len(r.hands[pid]) == 0
            bid = r.bids[pid]
            assert 0 <= bid <= r.cards_per_player

        for t in r.tricks:
            # Exactly one play from each player
            assert len(t.plays) == num_players
            assert t.winner_id is not None

    # Final scores should equal sum of per-round deltas
    totals = {p.id: 0 for p in game_state.players}
    for r in game_state.rounds:
        deltas = score_round(r, game_state.players)
        for pid in totals:
            totals[pid] += deltas[pid]

    for p in game_state.players:
        assert p.score == totals[p.id]
