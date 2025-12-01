# wizard_arena/engine.py
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from .agents.base import WizardAgent
from .agents.random_agent import RandomWizardAgent  # optional, just handy
from .cards import Card, CardType, Deck, Suit, card_to_dict
from .rules import determine_trump, legal_moves, score_round, winner_of_trick
from .state import GameState, PlayerState, RoundState, Trick

logger = logging.getLogger(__name__)


class GameEngine:
    """
    Orchestrates a full Wizard game using pluggable agents.

    This module is *pure* game logic: no HTTP, no LLM calls. Agents just
    implement the WizardAgent protocol.
    """

    def __init__(
        self,
        agents: List[WizardAgent],
        player_names: Optional[List[str]] = None,
        rng_seed: Optional[int] = None,
        game_label: Optional[str] = None,
    ) -> None:
        if not 3 <= len(agents) <= 6:
            raise ValueError("Wizard supports 3 to 6 players")

        self.agents: List[WizardAgent] = agents

        if player_names is None:
            player_names = [f"Player {i}" for i in range(len(agents))]
        if len(player_names) != len(agents):
            raise ValueError("player_names must match number of agents")

        self.rng = random.Random(rng_seed)
        self.game_label = game_label

        self.game_state = GameState(
            players=[
                PlayerState(id=i, name=name)
                for i, name in enumerate(player_names)
            ]
        )
        # Number of rounds determined by deck size and player count.
        self.max_rounds = 60 // self.game_state.num_players

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def play_game(self) -> GameState:
        """Play a full game from scratch and return the final GameState."""
        dealer_id = 0
        for round_index in range(self.max_rounds):
            cards_per_player = round_index + 1
            is_last_round = round_index == self.max_rounds - 1

            round_state = self._start_round(
                round_index=round_index,
                cards_per_player=cards_per_player,
                dealer_id=dealer_id,
                is_last_round=is_last_round,
            )
            self.game_state.rounds.append(round_state)
            self.game_state.current_round_index = round_index

            self._bidding_phase(round_state)
            self._trick_phase(round_state)
            self._score_round(round_state)
            logger.info(
                "Finished round %d/%d%s",
                round_index + 1,
                self.max_rounds,
                f" for {self.game_label}" if self.game_label else "",
            )

            dealer_id = (dealer_id + 1) % self.game_state.num_players

        logger.info(
            "Finished game%s",
            f" {self.game_label}" if self.game_label else "",
        )
        return self.game_state

    # -------------------------------------------------------------------------
    # Round lifecycle
    # -------------------------------------------------------------------------

    def _start_round(
        self,
        round_index: int,
        cards_per_player: int,
        dealer_id: int,
        is_last_round: bool,
    ) -> RoundState:
        deck = Deck()
        deck.shuffle(self.rng)

        hands_list, remaining = deck.deal(
            self.game_state.num_players, cards_per_player
        )
        hands = {pid: hand for pid, hand in enumerate(hands_list)}

        trump_card: Optional[Card] = None
        trump_suit: Optional[Suit] = None
        dealer_choice: Optional[Suit] = None

        if not is_last_round:
            if not remaining:
                raise RuntimeError("No remaining cards to determine trump")
            trump_card = remaining[0]

            if trump_card.type == CardType.WIZARD:
                # Dealer chooses trump suit.
                dealer_agent = self.agents[dealer_id]
                obs = self._build_trump_observation(
                    dealer_id,
                    round_index,
                    cards_per_player,
                    hands[dealer_id],
                    trump_card,
                )
                dealer_choice = dealer_agent.choose_trump(obs)

            trump_suit = determine_trump(
                top_card=trump_card,
                dealer_choice=dealer_choice,
                is_last_round=is_last_round,
            )
        else:
            trump_suit = None

        # Reset per-round stats.
        for p in self.game_state.players:
            p.tricks_won_this_round = 0
            p.current_bid = None

        return RoundState(
            round_index=round_index,
            cards_per_player=cards_per_player,
            dealer_id=dealer_id,
            trump_suit=trump_suit,
            trump_card=trump_card,
            hands=hands,
        )

    def _bidding_phase(self, round_state: RoundState) -> None:
        """Ask each agent for their bid and store it in the round state."""
        num_players = self.game_state.num_players
        # Bidding starts left of the dealer and proceeds clockwise so dealer bids last.
        bidding_order = [
            (round_state.dealer_id + offset) % num_players
            for offset in range(1, num_players + 1)
        ]

        for bidder_position, pid in enumerate(bidding_order):
            agent = self.agents[pid]
            hand = round_state.hands[pid]

            obs = self._build_common_observation_base(pid, round_state)
            obs.update(
                {
                    "phase": "bidding",
                    "hand": [card_to_dict(c) for c in hand],
                    "bids_so_far": dict(round_state.bids),
                    "bidding_order": bidding_order,
                    "current_bidder_seat_index": bidder_position,
                }
            )

            bid = agent.choose_bid(obs)
            if not isinstance(bid, int):
                raise ValueError("Agent returned non-int bid")
            if bid < 0 or bid > round_state.cards_per_player:
                # Clamp to valid range; we don't want to crash entire game here.
                bid = max(0, min(round_state.cards_per_player, bid))

            round_state.bids[pid] = bid
            self.game_state.players[pid].current_bid = bid

    def _trick_phase(self, round_state: RoundState) -> None:
        """Play all tricks for the round."""
        leader_id = (round_state.dealer_id + 1) % self.game_state.num_players

        for _ in range(round_state.cards_per_player):
            trick = Trick()
            current_player = leader_id

            for _turn in range(self.game_state.num_players):
                hand = round_state.hands[current_player]
                legal_indices = legal_moves(hand, trick.led_suit)
                trick_history = [
                    {
                        "plays": [
                            {"player_id": pid, "card": card_to_dict(card)}
                            for pid, card in t.plays
                        ],
                        "led_suit": t.led_suit.name if t.led_suit else None,
                        "winner_id": t.winner_id,
                    }
                    for t in round_state.tricks
                ]

                obs = self._build_common_observation_base(
                    current_player, round_state
                )
                obs.update(
                    {
                        "phase": "play",
                        "hand": [card_to_dict(c) for c in hand],
                        "legal_move_indices": legal_indices,
                        "current_trick": {
                            "plays": [
                                {"player_id": pid, "card": card_to_dict(card)}
                                for pid, card in trick.plays
                            ],
                            "led_suit": trick.led_suit.name
                            if trick.led_suit
                            else None,
                        },
                        "trick_index": len(round_state.tricks),
                        "trick_history": trick_history,
                        "leader_id": leader_id,
                        "tricks_taken_so_far": {
                            p.id: p.tricks_won_this_round
                            for p in self.game_state.players
                        },
                        "bids": dict(round_state.bids),
                        "hand_sizes": {
                            pid: len(round_state.hands[pid])
                            for pid in range(self.game_state.num_players)
                        },
                    }
                )

                agent = self.agents[current_player]
                move_index = agent.choose_card(obs)
                if move_index not in legal_indices:
                    # If agent chooses illegal index, auto-correct to first legal.
                    move_index = legal_indices[0]

                card = hand.pop(move_index)
                trick.plays.append((current_player, card))

                # Establish led suit the first time a NUMBER card is played,
                # unless the trick was led with a Wizard (no led suit).
                if (
                    trick.led_suit is None
                    and card.type == CardType.NUMBER
                    and trick.plays[0][1].type != CardType.WIZARD
                ):
                    trick.led_suit = card.suit

                # Next player clockwise.
                current_player = (current_player + 1) % self.game_state.num_players

            # Determine trick winner.
            winner_id = winner_of_trick(trick, round_state.trump_suit)
            trick.winner_id = winner_id
            round_state.tricks.append(trick)
            self.game_state.players[winner_id].tricks_won_this_round += 1

            # Winner leads next trick.
            leader_id = winner_id

    def _score_round(self, round_state: RoundState) -> None:
        deltas = score_round(round_state, self.game_state.players)
        for p in self.game_state.players:
            p.score += deltas[p.id]

    # -------------------------------------------------------------------------
    # Observation builders
    # -------------------------------------------------------------------------

    def _build_common_observation_base(
        self, player_id: int, round_state: RoundState
    ) -> Dict[str, Any]:
        player = self.game_state.players[player_id]
        scores = {p.id: p.score for p in self.game_state.players}
        return {
            "game": {
                "game_id": self.game_label,
                "round_index": round_state.round_index,
                "cards_per_player": round_state.cards_per_player,
                "num_players": self.game_state.num_players,
                "dealer_id": round_state.dealer_id,
            },
            "player": {
                "id": player.id,
                "name": player.name,
                "score": player.score,
            },
            "trump": {
                "suit": round_state.trump_suit.name
                if round_state.trump_suit
                else None,
                "card": card_to_dict(round_state.trump_card)
                if round_state.trump_card
                else None,
            },
            "scores": scores,
            "seating_order": [p.id for p in self.game_state.players],
            "player_names": {p.id: p.name for p in self.game_state.players},
        }

    def _build_trump_observation(
        self,
        dealer_id: int,
        round_index: int,
        cards_per_player: int,
        hand: List[Card],
        top_card: Card,
    ) -> Dict[str, Any]:
        # Lightweight synthetic RoundState for context.
        tmp_round = RoundState(
            round_index=round_index,
            cards_per_player=cards_per_player,
            dealer_id=dealer_id,
            trump_suit=None,
            trump_card=None,
            hands={},
        )
        obs = self._build_common_observation_base(dealer_id, tmp_round)
        obs["phase"] = "choose_trump"
        obs["hand_cards"] = hand[:]  # actual Card objects
        obs["hand"] = [card_to_dict(c) for c in hand]
        obs["top_card"] = card_to_dict(top_card)
        obs["rounds_total"] = self.max_rounds
        obs["rounds_remaining"] = self.max_rounds - round_index
        obs["tricks_this_round"] = cards_per_player
        return obs
