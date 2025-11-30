# tests/test_game_log.py
import random

from wizard_arena.agents.random_agent import RandomWizardAgent
from wizard_arena.engine import GameEngine
from wizard_arena.game_log import (
    FIELDNAMES,
    build_round_score_rows,
    write_round_scores_csv,
)
from wizard_arena.state import RoundState


def _make_sample_game(num_players: int = 3):
    agents = [
        RandomWizardAgent(rng=random.Random(200 + i))
        for i in range(num_players)
    ]
    names = [f"P{i}" for i in range(num_players)]
    engine = GameEngine(agents=agents, player_names=names, rng_seed=321)
    return engine.play_game()


def test_build_round_score_rows_basic():
    game_state = _make_sample_game()
    rows = build_round_score_rows(game_state, game_id="test-game")

    num_rounds = len(game_state.rounds)
    num_players = len(game_state.players)
    assert len(rows) == num_rounds * num_players

    # Row structure looks sane
    sample = rows[0]
    for field in FIELDNAMES:
        assert field in sample

    # total_score in last round for each player should match final scores
    totals_from_rows = {p.id: 0 for p in game_state.players}
    # Iterate in order; last row per player contains final total_score
    for row in rows:
        pid = row["player_id"]
        totals_from_rows[pid] = row["total_score"]

    for p in game_state.players:
        assert totals_from_rows[p.id] == p.score


def test_write_round_scores_csv(tmp_path):
    game_state = _make_sample_game()
    path = tmp_path / "scores.csv"

    write_round_scores_csv(game_state, path, game_id="csv-game")

    contents = path.read_text(encoding="utf-8").strip().splitlines()
    # Header + at least one data row
    assert len(contents) > 1

    header = contents[0].split(",")
    assert "round_index" in header
    assert "player_id" in header
    assert "total_score" in header


def test_build_round_score_rows_skips_incomplete_round():
    game_state = _make_sample_game()
    completed_rows = build_round_score_rows(game_state, game_id="complete")

    # Add an incomplete round (no bids/tricks) to ensure it is ignored.
    empty_round = RoundState(
        round_index=99,
        cards_per_player=1,
        dealer_id=0,
        trump_suit=None,
        trump_card=None,
        hands={pid: [] for pid in range(len(game_state.players))},
    )
    game_state.rounds.append(empty_round)

    rows_with_incomplete = build_round_score_rows(game_state, game_id="mixed")
    assert len(rows_with_incomplete) == len(completed_rows)
