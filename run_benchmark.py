from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime
import logging
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from wizard_arena.agents import LLMCallFailed, LLMWizardAgent
from wizard_arena.agents.benchmark_agents import (
    MediumAgent,
    RandomAgent,
    StrongAgent,
)
from wizard_arena.engine import GameEngine
from wizard_arena.game_log import FIELDNAMES, build_round_score_rows
from wizard_arena.llm_clients import LLMRouter
from wizard_arena.litellm_cost_tracker import running_cost_tracker
from wizard_arena.paths import ensure_results_dir
from wizard_arena.verbose_logger import FailureLogger, VerboseGameLogger

NUM_PLAYERS = 4
OPPONENT_LABELS = ["RandomAgent", "MediumAgent", "StrongAgent"]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a target LLM Wizard agent against a fixed pool of opponents."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Target model of the form '<provider>:<model_name>', e.g. openai:gpt-4o."
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
        help="Sampling temperature for the target LLM (default: 0.0).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max output tokens for LLM calls (default: 256).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="wizard_benchmark_scores.csv",
        help="Path to output CSV file (default: wizard_benchmark_scores.csv).",
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
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=1,
        help="Max number of games to play concurrently (default: 1).",
    )
    parser.add_argument(
        "--no-seat-rotation",
        action="store_true",
        help="Disable seat rotation (target stays in seat 0 every game).",
    )

    return parser.parse_args(argv)


def _build_opponents(
    game_index: int,
    *,
    args: argparse.Namespace,
) -> Tuple[List[Any], List[str]]:
    seeds = [args.seed + game_index * 1000 + i + 1 for i in range(3)]

    agents = [
        RandomAgent(seed=seeds[0]),
        MediumAgent(seed=seeds[1]),
        StrongAgent(seed=seeds[2]),
    ]
    return agents, OPPONENT_LABELS


def _seat_players(
    agents: List[Any],
    names: List[str],
    *,
    game_index: int,
    rotate: bool,
) -> Tuple[List[Any], List[str], int]:
    if not rotate:
        return agents, names, 0

    shift = game_index % len(agents)
    if shift == 0:
        return agents, names, shift

    return (
        agents[shift:] + agents[:shift],
        names[shift:] + names[:shift],
        shift,
    )


def _play_single_game(
    game_index: int,
    *,
    args: argparse.Namespace,
    verbose_logger: Optional[VerboseGameLogger],
    failure_logger: Optional[FailureLogger],
) -> Tuple[List[Dict[str, Any]], bool, str]:
    game_id = f"game-{game_index}"

    router = LLMRouter(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    target_seed = args.seed + game_index * 1000
    target_agent = LLMWizardAgent(
        model=args.model,
        router=router,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=target_seed,
        verbose_logger=verbose_logger,
        failure_logger=failure_logger,
    )

    opponent_agents, opponent_names = _build_opponents(
        game_index,
        args=args,
    )

    agents = [target_agent] + opponent_agents
    names = [target_agent.label] + opponent_names

    if len(agents) != NUM_PLAYERS:
        raise RuntimeError(
            f"Expected {NUM_PLAYERS} players, got {len(agents)}."
        )

    agents, names, shift = _seat_players(
        agents,
        names,
        game_index=game_index,
        rotate=not args.no_seat_rotation,
    )
    if shift:
        logging.info("Rotated seating by %d for %s", shift, game_id)

    logging.info("Seating order for %s: %s", game_id, ", ".join(names))

    engine_seed = args.seed + game_index
    engine = GameEngine(
        agents=agents,
        player_names=names,
        rng_seed=engine_seed,
        game_label=game_id,
    )

    early_stop = False
    try:
        game_state = engine.play_game()
        logging.info("Finished %s", game_id)
    except LLMCallFailed as exc:
        logging.error(
            "Halting %s after repeated LLM failures: %s",
            game_id,
            exc,
        )
        game_state = engine.game_state
        early_stop = True
    except NotImplementedError as exc:
        logging.error(
            "Opponent agent not implemented (%s). Implement agents in "
            "wizard_arena/agents/benchmark_agents.py or run with "
            "--use-random-opponents.",
            exc,
        )
        game_state = engine.game_state
        early_stop = True

    rows = build_round_score_rows(game_state, game_id=game_id)
    return rows, early_stop, game_id


def _slugify_model(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model.strip())
    return slug.strip("-") or "model"


async def _play_single_game_async(
    game_index: int,
    *,
    args: argparse.Namespace,
    verbose_logger: Optional[VerboseGameLogger],
    failure_logger: Optional[FailureLogger],
) -> Tuple[List[Dict[str, Any]], bool, str]:
    return await asyncio.to_thread(
        _play_single_game,
        game_index,
        args=args,
        verbose_logger=verbose_logger,
        failure_logger=failure_logger,
    )


async def async_main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    if args.games < 1:
        raise SystemExit("--games must be >= 1")

    if args.parallel_games < 1:
        raise SystemExit("--parallel-games must be >= 1")

    results_root = ensure_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = _slugify_model(args.model)
    run_dir = results_root / f"{timestamp}_g{args.games}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _name_in_run_dir(path_like: Optional[str], default_name: str) -> Path:
        name = Path(path_like).name if path_like else default_name
        return run_dir / name

    csv_path = _name_in_run_dir(args.csv, "wizard_benchmark_scores.csv")
    verbose_path = _name_in_run_dir(
        args.verbose_log, "wizard_benchmark_verbose.log"
    )
    failure_path = _name_in_run_dir(
        args.failure_log, "wizard_benchmark_failures.log"
    )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logging.info("Target model: %s", args.model)
    logging.info("Games to play: %d", args.games)
    logging.info("Output folder: %s", run_dir)
    logging.info("Output CSV: %s", csv_path)
    logging.info("Verbose log: %s", verbose_path)
    logging.info("Failure log: %s", failure_path)

    logging.info(
        "Opponents: fixed pool (RandomAgent, MediumAgent, StrongAgent)"
    )
    running_cost_tracker.start_run()

    verbose_logger = VerboseGameLogger(verbose_path)
    failure_logger = FailureLogger(failure_path)

    all_rows: List[Dict[str, Any]] = []
    games_played = 0
    early_stop = False

    parallel_games = min(args.parallel_games, args.games)

    for batch_start in range(0, args.games, parallel_games):
        batch_end = min(batch_start + parallel_games, args.games)
        batch_indices = list(range(batch_start, batch_end))
        logging.info(
            "Starting games %s",
            ", ".join(str(i + 1) for i in batch_indices),
        )

        tasks = [
            asyncio.create_task(
                _play_single_game_async(
                    game_index=game_index,
                    args=args,
                    verbose_logger=verbose_logger,
                    failure_logger=failure_logger,
                )
            )
            for game_index in batch_indices
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.error("Game task failed: %s", result)
                early_stop = True
                continue

            rows, stopped, game_id = result
            all_rows.extend(rows)
            games_played += 1
            if stopped:
                early_stop = True
                logging.error("Halting after errors in %s", game_id)

        if early_stop:
            break

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
    running_cost_tracker.persist()
    running_cost_tracker.log_run_summary()


def main(argv: List[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
