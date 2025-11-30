# wizard_arena/cli.py
from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import List

from .agents import LLMCallFailed, LLMWizardAgent
from .engine import GameEngine
from .game_log import FIELDNAMES, build_round_score_rows
from .llm_clients import LLMRouter
from .verbose_logger import FailureLogger, VerboseGameLogger


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Wizard games between multiple LLMs and log per-round scores "
            "to a CSV file."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "List of models of the form '<provider>:<model_name>'. "
            "For example: "
            "openai:gpt-4o anthropic:claude-opus-4.5 "
            "gemini:gemini-2.5-flash grok:grok-4.1"
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of full games to play (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for all LLM agents (default: 0.0).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max output tokens requested from each LLM call (default: 256).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="wizard_llm_scores.csv",
        help="Path to the output CSV file (default: wizard_llm_scores.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for the environment and fallback agents.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...). Default: INFO.",
    )
    parser.add_argument(
        "--verbose-log",
        type=str,
        default=None,
        help="Optional path for a detailed turn-by-turn log file.",
    )
    parser.add_argument(
        "--failure-log",
        type=str,
        default=None,
        help="Optional path to capture only failed LLM calls/parse issues.",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    models = args.models
    num_players = len(models)

    if num_players < 3 or num_players > 6:
        raise SystemExit(
            f"Wizard requires between 3 and 6 players; got {num_players} models."
        )

    logging.info("Models: %s", ", ".join(models))
    logging.info("Games to play: %d", args.games)
    logging.info("Output CSV: %s", args.csv)
    if args.verbose_log:
        logging.info("Verbose log: %s", args.verbose_log)

    # Shared router instance for all agents.
    router = LLMRouter(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    verbose_logger = (
        VerboseGameLogger(Path(args.verbose_log))
        if args.verbose_log
        else None
    )
    failure_logger = (
        FailureLogger(Path(args.failure_log))
        if args.failure_log
        else None
    )

    all_rows = []
    games_played = 0

    early_stop = False

    for game_index in range(args.games):
        logging.info("Starting game %d/%d", game_index + 1, args.games)
        game_id = f"game-{game_index}"

        game_models = list(models)
        random.Random(args.seed + game_index).shuffle(game_models)
        logging.info("Seating order for %s: %s", game_id, ", ".join(game_models))

        agents: List[LLMWizardAgent] = []
        for i, model_str in enumerate(game_models):
            agent_seed = args.seed + game_index * 1000 + i
            agent = LLMWizardAgent(
                model=model_str,
                router=router,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                seed=agent_seed,
                verbose_logger=verbose_logger,
                failure_logger=failure_logger,
            )
            agents.append(agent)

        player_names = [agent.label for agent in agents]

        engine_seed = args.seed + game_index
        engine = GameEngine(
            agents=agents,
            player_names=player_names,
            rng_seed=engine_seed,
            game_label=game_id,
        )

        try:
            game_state = engine.play_game()
            logging.info(
                "Finished game %d/%d (%s)",
                game_index + 1,
                args.games,
                game_id,
            )
        except LLMCallFailed as exc:
            logging.error(
                "Halting after repeated LLM failures in %s: %s",
                game_id,
                exc,
            )
            game_state = engine.game_state
            early_stop = True

        rows = build_round_score_rows(game_state, game_id=game_id)
        all_rows.extend(rows)

        games_played += 1

        if early_stop:
            break

    # Write all rows to a single CSV.
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    if early_stop:
        logging.info(
            "Paused run after %d/%d games; wrote %d rows to %s",
            games_played,
            args.games,
            len(all_rows),
            args.csv,
        )
    else:
        logging.info(
            "Finished %d games; wrote %d rows to %s",
            games_played,
            len(all_rows),
            args.csv,
        )

    if verbose_logger:
        verbose_logger.flush()
    if failure_logger:
        failure_logger.flush()


if __name__ == "__main__":
    main()

'''
python3 -m wizard_arena.cli \
  --models \
    openai:gpt-4o-mini \
    anthropic:claude-3-5-haiku-latest \
    gemini:gemini-2.0-flash \
  --games 1 \
  --csv wizard_results_test_3.csv \
  --temperature 0.0 \
  --max-output-tokens 64
'''

'''
python3 -m wizard_arena.cli \
  --models \
    openai:gpt-4o-mini \
    anthropic:claude-3-5-haiku-latest \
    gemini:gemini-2.0-flash \
  --games 40 \
  --csv wizard_results_40_games.csv \
  --temperature 0.0 \
  --max-output-tokens 64 \
  --seed 1
'''
