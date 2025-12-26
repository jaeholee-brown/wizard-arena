# wizard_arena/cli.py
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .agents import LLMCallFailed, LLMWizardAgent
from .engine import GameEngine
from .game_log import FIELDNAMES, build_round_score_rows
from .llm_clients import LLMRouter, ModelSpec
from .litellm_cost_tracker import running_cost_tracker
from .paths import ensure_results_dir, resolve_results_path
from .verbose_logger import FailureLogger, VerboseGameLogger

# Conservative provider RPM budgets to estimate safe parallelism. These can be
# overridden per run if needed.
DEFAULT_RPM_BUDGETS: Dict[str, int] = {
    "openai": 500,
    "anthropic": 1000,
    "gemini": 4000,
    "grok": 500,
}
DEFAULT_AVG_RESPONSE_SECONDS = 1.2
RPM_SAFETY_FRACTION = 0.85

# --------------------------------------------------------------------------- #
# Helpers for estimating safe parallelism                                    #
# --------------------------------------------------------------------------- #


def _estimate_calls_per_minute_per_game(
    avg_response_seconds: float = DEFAULT_AVG_RESPONSE_SECONDS,
) -> float:
    """Upper-bound on per-game request rate given an average LLM latency.

    Calls within a single game are serialized, so the best-case rate is roughly
    60 / latency seconds per call.
    """
    return 60.0 / max(avg_response_seconds, 0.2)


def _recommended_parallel_games(
    models: List[str],
    *,
    rpm_budgets: Dict[str, int],
    avg_response_seconds: float,
) -> Tuple[int, float, Dict[str, int]]:
    """Return (recommended_parallel, per_game_rpm, per_provider_caps)."""
    calls_per_minute = _estimate_calls_per_minute_per_game(avg_response_seconds)
    provider_counts = Counter(ModelSpec.parse(m).provider for m in models)
    effective_budgets = {
        provider: int(budget * RPM_SAFETY_FRACTION)
        for provider, budget in rpm_budgets.items()
    }
    fallback_budget = min(effective_budgets.values()) if effective_budgets else None

    caps: Dict[str, int] = {}
    for provider, count in provider_counts.items():
        budget = effective_budgets.get(provider, fallback_budget)
        if budget is None:
            continue
        provider_rate = calls_per_minute * (count / len(models))
        cap = (
            math.floor(budget / provider_rate)
            if provider_rate > 0
            else 1
        )
        caps[provider] = max(cap, 1)

    if caps:
        recommended = max(1, min(caps.values()))
    else:
        # Unknown provider: stay conservative.
        recommended = 1
    return recommended, calls_per_minute, caps

# Conservative provider RPM budgets to estimate safe parallelism. These can be
# overridden per run if needed.
DEFAULT_RPM_BUDGETS: Dict[str, int] = {
    "openai": 500,
    "anthropic": 1000,
    "gemini": 4000,
    "grok": 500,
}
DEFAULT_AVG_RESPONSE_SECONDS = 1.2
RPM_SAFETY_FRACTION = 0.85


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
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=None,
        help=(
            "Max number of games to play concurrently. "
            "If omitted, a safe value is auto-computed from provider RPM budgets."
        ),
    )
    parser.add_argument(
        "--avg-response-seconds",
        type=float,
        default=DEFAULT_AVG_RESPONSE_SECONDS,
        help=(
            "Assumed average LLM response latency used to estimate safe "
            "parallelism (default: %(default)s seconds)."
        ),
    )

    return parser.parse_args(argv)


def _play_single_game(
    game_index: int,
    *,
    models: List[str],
    args: argparse.Namespace,
    verbose_logger: Optional[VerboseGameLogger],
    failure_logger: Optional[FailureLogger],
) -> Tuple[List[Dict[str, Any]], bool, str]:
    """Run one game synchronously (meant for thread execution)."""
    game_id = f"game-{game_index}"

    game_models = list(models)
    random.Random(args.seed + game_index).shuffle(game_models)
    logging.info("Seating order for %s: %s", game_id, ", ".join(game_models))

    router = LLMRouter(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

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

    rows = build_round_score_rows(game_state, game_id=game_id)
    return rows, early_stop, game_id


async def _play_single_game_async(
    game_index: int,
    *,
    models: List[str],
    args: argparse.Namespace,
    verbose_logger: Optional[VerboseGameLogger],
    failure_logger: Optional[FailureLogger],
) -> Tuple[List[Dict[str, Any]], bool, str]:
    return await asyncio.to_thread(
        _play_single_game,
        game_index,
        models=models,
        args=args,
        verbose_logger=verbose_logger,
        failure_logger=failure_logger,
    )


async def async_main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_results_dir()

    csv_path = resolve_results_path(args.csv)
    verbose_path = (
        resolve_results_path(args.verbose_log) if args.verbose_log else None
    )
    failure_path = (
        resolve_results_path(args.failure_log) if args.failure_log else None
    )

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
    logging.info("Output CSV: %s", csv_path)
    if verbose_path:
        logging.info("Verbose log: %s", verbose_path)
    if failure_path:
        logging.info("Failure log: %s", failure_path)

    rpm_budgets = dict(DEFAULT_RPM_BUDGETS)
    recommended_parallel, calls_per_minute, caps = _recommended_parallel_games(
        models,
        rpm_budgets=rpm_budgets,
        avg_response_seconds=args.avg_response_seconds,
    )
    parallel_games = args.parallel_games or recommended_parallel
    if parallel_games < 1:
        parallel_games = 1
    parallel_games = min(parallel_games, args.games)

    logging.info(
        "Estimated per-game request rate: %.1f rpm "
        "(avg response %.2fs, safety %.0f%%)",
        calls_per_minute,
        args.avg_response_seconds,
        RPM_SAFETY_FRACTION * 100,
    )
    if caps:
        logging.info(
            "Provider parallel caps (after safety buffer): %s",
            ", ".join(f"{provider}:{cap}" for provider, cap in caps.items()),
        )
    if args.parallel_games and args.parallel_games > recommended_parallel:
        logging.warning(
            "Requested %d parallel games exceeds recommended %d; "
            "watch provider rate limits.",
            args.parallel_games,
            recommended_parallel,
        )
    logging.info("Running up to %d game(s) concurrently", parallel_games)
    running_cost_tracker.start_run()

    verbose_logger = (
        VerboseGameLogger(verbose_path)
        if verbose_path
        else None
    )
    failure_logger = (
        FailureLogger(failure_path)
        if failure_path
        else None
    )

    all_rows: List[Dict[str, Any]] = []
    games_played = 0
    early_stop = False

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
                    models=models,
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

    # Write all rows to a single CSV.
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

'''
python3 -m wizard_arena.cli \
  --models \
    openai:gpt-4o-mini \
    anthropic:claude-3-5-haiku-latest \
    gemini:gemini-2.0-flash \
  --games 60 \
  --parallel-games 12 \
  --csv wizard_results_60_games.csv \
  --verbose-log wizard_results_60_games_verbose.log \
  --failure-log wizard_results_60_games_failures.log \
  --temperature 0.0 \
  --max-output-tokens 640 \
  --seed 1
  '''
