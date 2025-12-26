from __future__ import annotations

import random
from typing import Any, Dict, Optional

from ..cards import Card, CardType, Deck, Suit, card_to_dict, dict_to_card
from ..rules import legal_moves, winner_of_trick
from ..state import Trick
from .base import WizardAgent

_RANK_ACE = 13
_RANK_KING = 12
_RANK_QUEEN = 11
_RANK_JACK = 10
_RANK_MID_TRUMP_MIN = 7
_RANK_MID_TRUMP_MAX = 9


def _split_suits(hand: list[Dict[str, Any]]) -> Dict[str, list[int]]:
    suits = {suit.name: [] for suit in Suit}
    for card in hand:
        if card["type"] == CardType.NUMBER.value:
            suits[card["suit"]].append(int(card["rank"]))
    return suits


def _estimate_tricks(
    hand: list[Dict[str, Any]], trump_suit: Optional[str]
) -> tuple[float, int, float, float]:
    num_wizards = sum(
        1 for card in hand if card["type"] == CardType.WIZARD.value
    )
    num_jesters = sum(
        1 for card in hand if card["type"] == CardType.JESTER.value
    )

    suits = _split_suits(hand)
    trump_cards = suits.get(trump_suit, []) if trump_suit else []
    trump_count = len(trump_cards)

    wizards_points = 0.90 * num_wizards
    trump_points = 0.0
    high_trump_points = 0.0
    for rank in trump_cards:
        if rank == _RANK_ACE:
            trump_points += 0.75
            high_trump_points += 0.75
        elif rank == _RANK_KING:
            trump_points += 0.60
            high_trump_points += 0.60
        elif rank == _RANK_QUEEN:
            trump_points += 0.45
            high_trump_points += 0.45
        elif rank == _RANK_JACK:
            trump_points += 0.35
            high_trump_points += 0.35
        elif _RANK_MID_TRUMP_MIN <= rank <= _RANK_MID_TRUMP_MAX:
            trump_points += 0.25
        else:
            trump_points += 0.15 if trump_count >= 3 else 0.05

    side_points = 0.0
    side_fragile_points = 0.0
    voids = 0
    for suit_name, ranks in suits.items():
        if trump_suit and suit_name == trump_suit:
            continue
        length = len(ranks)
        if length == 0:
            voids += 1
            continue

        top_rank = max(ranks)
        if top_rank == _RANK_ACE:
            if length >= 2:
                side_points += 0.45
            else:
                side_points += 0.25
                side_fragile_points += 0.25
        elif top_rank == _RANK_KING:
            if length >= 3:
                side_points += 0.25
            else:
                side_points += 0.10
                side_fragile_points += 0.10
        elif top_rank == _RANK_QUEEN:
            if length >= 4:
                side_points += 0.15
            else:
                side_points += 0.05
                side_fragile_points += 0.05

        if length >= 5:
            side_points += 0.05
        elif length >= 4:
            side_points += 0.00

    void_bonus = 0.0
    if trump_count >= 1:
        per_void = 0.20 if trump_count >= 3 else 0.10
        void_bonus = per_void * voids

    expected = wizards_points + trump_points + side_points + void_bonus
    core_points = wizards_points + high_trump_points
    return expected, num_jesters, core_points, side_fragile_points


def _round_bid(
    expected: float,
    hand_size: int,
    num_jesters: int,
    core_points: float,
    fragile_points: float,
    rng: random.Random,
) -> int:
    base = int(expected)
    frac = expected - base
    up_prob = frac

    if num_jesters > 0:
        up_prob -= 0.10 * min(num_jesters, 2)

    if expected > 0.0:
        if num_jesters == 0:
            core_ratio = core_points / expected
            if core_ratio >= 0.60:
                up_prob += 0.05
        fragile_ratio = fragile_points / expected
        if fragile_ratio >= 0.35:
            up_prob -= 0.10

    up_prob = max(0.0, min(1.0, up_prob))
    bid = base + (1 if rng.random() < up_prob else 0)
    return max(0, min(hand_size, bid))


def _infer_led_suit(
    plays: list[tuple[int, Any]], existing_led_suit: Optional[Suit]
) -> Optional[Suit]:
    if existing_led_suit is not None:
        return existing_led_suit
    if not plays:
        return None
    if plays[0][1].type == CardType.WIZARD:
        return None
    for _pid, card in plays:
        if card.type == CardType.NUMBER:
            return card.suit
    return None


def _card_power(
    card: Any,
    trump_suit: Optional[Suit],
    led_suit: Optional[Suit],
    wizard_already_played: bool,
) -> int:
    if card.type == CardType.WIZARD:
        return -1 if wizard_already_played else 100
    if card.type == CardType.JESTER:
        return 0
    rank = card.rank or 0
    if trump_suit is not None and card.suit == trump_suit:
        return 80 + rank
    if led_suit is not None and card.suit == led_suit:
        return 60 + rank
    return 40 + rank


def _pick_by_power(
    rng: random.Random,
    indices: list[int],
    power_map: Dict[int, int],
    *,
    pick_max: bool,
) -> int:
    best_value = max(power_map[idx] for idx in indices) if pick_max else min(
        power_map[idx] for idx in indices
    )
    candidates = [idx for idx in indices if power_map[idx] == best_value]
    return rng.choice(candidates)


def _build_full_deck() -> list[Card]:
    return Deck().cards[:]


def _remove_cards(deck: list[Card], cards: list[Card]) -> list[Card]:
    remaining = deck[:]
    for card in cards:
        for i, existing in enumerate(remaining):
            if existing == card:
                remaining.pop(i)
                break
    return remaining


def _collect_played_cards(
    trick_history: list[Dict[str, Any]],
    current_trick: Dict[str, Any],
) -> list[Card]:
    played: list[Card] = []
    for trick in trick_history:
        for play in trick.get("plays", []):
            played.append(dict_to_card(play["card"]))
    for play in current_trick.get("plays", []):
        played.append(dict_to_card(play["card"]))
    return played


def _build_remaining_deck(
    observation: Dict[str, Any], hand_cards: list[Card]
) -> list[Card]:
    deck = _build_full_deck()
    cards_to_remove = list(hand_cards)

    trump_info = observation.get("trump") or {}
    trump_card = trump_info.get("card")
    if trump_card is not None:
        cards_to_remove.append(dict_to_card(trump_card))
    else:
        top_card = observation.get("top_card")
        if top_card is not None:
            cards_to_remove.append(dict_to_card(top_card))

    if "trick_history" in observation or "current_trick" in observation:
        trick_history = observation.get("trick_history", [])
        current_trick = observation.get("current_trick", {"plays": []})
        cards_to_remove.extend(
            _collect_played_cards(trick_history, current_trick)
        )

    return _remove_cards(deck, cards_to_remove)


def _infer_void_suits(
    trick_history: list[Dict[str, Any]],
    current_trick: Dict[str, Any],
    *,
    our_id: int,
    num_players: int,
) -> Dict[int, set[Suit]]:
    voids: Dict[int, set[Suit]] = {
        pid: set() for pid in range(num_players) if pid != our_id
    }
    all_tricks = trick_history + [current_trick]
    for trick in all_tricks:
        led_name = trick.get("led_suit")
        if led_name is None:
            continue
        led_suit = Suit[led_name]
        for play in trick.get("plays", []):
            pid = play["player_id"]
            if pid == our_id:
                continue
            card = dict_to_card(play["card"])
            if card.type == CardType.NUMBER and card.suit != led_suit:
                voids[pid].add(led_suit)
    return voids


def _card_allowed(card: Card, voids: set[Suit]) -> bool:
    return not (card.type == CardType.NUMBER and card.suit in voids)


def _sample_hidden_hands(
    remaining_cards: list[Card],
    *,
    player_ids: list[int],
    hand_sizes: Dict[int, int],
    voids: Dict[int, set[Suit]],
    rng: random.Random,
    max_attempts: int = 200,
) -> Dict[int, list[Card]]:
    if not player_ids:
        return {}

    ordered_players = sorted(
        player_ids, key=lambda pid: len(voids.get(pid, set())), reverse=True
    )
    for _ in range(max_attempts):
        deck = remaining_cards[:]
        rng.shuffle(deck)
        hands: Dict[int, list[Card]] = {pid: [] for pid in player_ids}
        success = True
        for pid in ordered_players:
            needed = hand_sizes[pid]
            allowed = [
                card
                for card in deck
                if _card_allowed(card, voids.get(pid, set()))
            ]
            if len(allowed) < needed:
                success = False
                break
            chosen = rng.sample(allowed, needed)
            hands[pid] = chosen
            for card in chosen:
                deck.remove(card)
        if success:
            return hands

    deck = remaining_cards[:]
    rng.shuffle(deck)
    hands = {}
    idx = 0
    for pid in player_ids:
        needed = hand_sizes[pid]
        hands[pid] = deck[idx : idx + needed]
        idx += needed
    return hands


def _enumerate_hidden_hands(
    remaining_cards: list[Card],
    *,
    player_ids: list[int],
    hand_sizes: Dict[int, int],
    voids: Dict[int, set[Suit]],
    limit: int = 5000,
) -> list[Dict[int, list[Card]]]:
    """
    Exhaustively enumerate hidden hand allocations for small endgames.
    Stops after `limit` allocations to avoid explosion.
    """
    results: list[Dict[int, list[Card]]] = []
    n_players = len(player_ids)
    if not player_ids or not remaining_cards:
        return results

    def backtrack(idx: int, deck: list[Card], acc: Dict[int, list[Card]]) -> None:
        if len(results) >= limit:
            return
        if idx == n_players:
            results.append({pid: acc[pid][:] for pid in player_ids})
            return

        pid = player_ids[idx]
        needed = hand_sizes[pid]
        allowed = [
            card for card in deck if _card_allowed(card, voids.get(pid, set()))
        ]
        if len(allowed) < needed:
            return

        from itertools import combinations

        for combo in combinations(range(len(allowed)), needed):
            if len(results) >= limit:
                break
            chosen_cards = [allowed[i] for i in combo]
            next_deck = deck[:]
            for card in chosen_cards:
                next_deck.remove(card)
            acc[pid] = chosen_cards
            backtrack(idx + 1, next_deck, acc)
            acc[pid] = []

    backtrack(0, remaining_cards[:], {pid: [] for pid in player_ids})
    return results


def _simulation_bid_from_cards(
    hand_cards: list[Card],
    trump_suit: Optional[Suit],
    rng: random.Random,
) -> int:
    """
    Estimate a bid for Monte Carlo simulation purposes.
    Self-contained heuristic used by Strong agent during rollouts.
    Independent of any other agent's bidding logic.
    """
    hand_dicts = [card_to_dict(card) for card in hand_cards]
    trump_name = trump_suit.name if trump_suit else None
    # Use the original _estimate_tricks for simulation (stable baseline)
    expected, num_jesters, core_points, fragile_points = _estimate_tricks(
        hand_dicts, trump_name
    )
    return _round_bid(
        expected,
        len(hand_cards),
        num_jesters,
        core_points,
        fragile_points,
        rng,
    )


def _choose_card_target_bid(
    hand: list[Card],
    legal_indices: list[int],
    *,
    player_id: int,
    bid: int,
    tricks_won: int,
    current_plays: list[tuple[int, Card]],
    led_suit: Optional[Suit],
    trump_suit: Optional[Suit],
    num_players: int,
    rng: random.Random,
) -> int:
    wizard_already_played = any(
        card.type == CardType.WIZARD for _pid, card in current_plays
    )
    is_last_player = len(current_plays) == num_players - 1
    power_map = {
        idx: _card_power(hand[idx], trump_suit, led_suit, wizard_already_played)
        for idx in legal_indices
    }

    def would_win(card: Card) -> bool:
        if is_last_player:
            plays = current_plays + [(player_id, card)]
            eval_led = _infer_led_suit(plays, led_suit)
            trick = Trick(plays=plays, led_suit=eval_led)
            return winner_of_trick(trick, trump_suit) == player_id

        if wizard_already_played:
            return False
        if card.type == CardType.WIZARD:
            return True
        if card.type == CardType.JESTER:
            return False

        plays = current_plays + [(player_id, card)]
        eval_led = _infer_led_suit(plays, led_suit)
        trick = Trick(plays=plays, led_suit=eval_led)
        return winner_of_trick(trick, trump_suit) == player_id

    if tricks_won < bid:
        if not current_plays:
            trump_indices = [
                idx
                for idx in legal_indices
                if hand[idx].type == CardType.NUMBER
                and trump_suit is not None
                and hand[idx].suit == trump_suit
            ]
            if trump_indices:
                min_rank = min(hand[idx].rank for idx in trump_indices)
                candidates = [
                    idx
                    for idx in trump_indices
                    if hand[idx].rank == min_rank
                ]
                return rng.choice(candidates)

            wizard_indices = [
                idx
                for idx in legal_indices
                if hand[idx].type == CardType.WIZARD
            ]
            if wizard_indices:
                return rng.choice(wizard_indices)

            number_indices = [
                idx
                for idx in legal_indices
                if hand[idx].type == CardType.NUMBER
            ]
            if number_indices:
                max_rank = max(hand[idx].rank for idx in number_indices)
                candidates = [
                    idx
                    for idx in number_indices
                    if hand[idx].rank == max_rank
                ]
                return rng.choice(candidates)

            return rng.choice(legal_indices)

        winning_indices = [
            idx for idx in legal_indices if would_win(hand[idx])
        ]
        if winning_indices:
            return _pick_by_power(
                rng, winning_indices, power_map, pick_max=False
            )
        return _pick_by_power(rng, legal_indices, power_map, pick_max=True)

    safe_indices = [
        idx for idx in legal_indices if not would_win(hand[idx])
    ]
    candidates = safe_indices if safe_indices else legal_indices
    return _pick_by_power(rng, candidates, power_map, pick_max=False)


def _round_score_delta(tricks_won: int, bid: int) -> int:
    if tricks_won == bid:
        return 20 + 10 * tricks_won
    return -10 * abs(tricks_won - bid)


def _simulate_round(
    hands: Dict[int, list[Card]],
    bids: Dict[int, int],
    tricks_won: Dict[int, int],
    leader_id: int,
    trump_suit: Optional[Suit],
    current_plays: list[tuple[int, Card]],
    led_suit: Optional[Suit],
    *,
    rng: random.Random,
) -> Dict[int, int]:
    num_players = len(hands)
    plays = current_plays[:]
    cur_led = _infer_led_suit(plays, led_suit)
    current_player = (
        (plays[-1][0] + 1) % num_players if plays else leader_id
    )

    while True:
        total_cards = sum(len(hand) for hand in hands.values())
        if total_cards == 0 and not plays:
            break
        if len(plays) == num_players:
            eval_led = _infer_led_suit(plays, cur_led)
            winner = winner_of_trick(
                Trick(plays=plays, led_suit=eval_led), trump_suit
            )
            tricks_won[winner] += 1
            leader_id = winner
            plays = []
            cur_led = None
            current_player = leader_id
            continue

        hand = hands[current_player]
        legal_indices = legal_moves(hand, cur_led)
        idx = _choose_card_target_bid(
            hand,
            legal_indices,
            player_id=current_player,
            bid=bids.get(current_player, 0),
            tricks_won=tricks_won.get(current_player, 0),
            current_plays=plays,
            led_suit=cur_led,
            trump_suit=trump_suit,
            num_players=num_players,
            rng=rng,
        )
        card = hand.pop(idx)
        plays.append((current_player, card))
        if (
            cur_led is None
            and card.type == CardType.NUMBER
            and plays[0][1].type != CardType.WIZARD
        ):
            cur_led = card.suit
        current_player = (current_player + 1) % num_players

    return tricks_won


# =============================================================================
# Strong Agent Enhancements
# =============================================================================


def _optimal_single_card_bid(
    card: Card,
    trump_suit: Optional[Suit],
    num_players: int,
) -> int:
    """
    Return optimal bid for 1-card round using statistical analysis.

    Math: Bid 1 if win probability > 42.86%
    - E[bid 1] = 30p - 10(1-p) = 40p - 10
    - E[bid 0] = 20(1-p) - 10p = 20 - 30p
    - Bid 1 when 40p - 10 > 20 - 30p, i.e., p > 30/70 ≈ 42.86%

    Thresholds derived from Wizard strategy forums and statistical analysis.
    """
    if card.type == CardType.WIZARD:
        return 1
    if card.type == CardType.JESTER:
        return 0

    # Trump card - use player-count-specific thresholds
    if trump_suit and card.type == CardType.NUMBER and card.suit == trump_suit:
        rank = card.rank or 0
        if num_players == 3:
            # 3 players: almost any trump wins often enough
            return 1 if rank >= 2 else 0
        elif num_players == 4:
            # 4 players: need 3+ of trump (threshold ~43% win rate)
            return 1 if rank >= 3 else 0
        elif num_players == 5:
            # 5 players: need 9+ of trump
            return 1 if rank >= 9 else 0
        else:  # 6 players
            # 6 players: need 10+ of trump
            return 1 if rank >= 10 else 0

    # Off-suit - rarely wins in multiplayer (< 25% typically)
    return 0


def _is_overbid_round(bids: Dict[int, int], cards_per_player: int) -> tuple[bool, int]:
    """
    Determine if round is overbid or underbid.

    Returns:
        (is_overbid, difference): True if total bids > tricks available
        difference is positive for overbid, negative for underbid
    """
    total_bids = sum(bids.values())
    diff = total_bids - cards_per_player
    return diff > 0, diff


def _identify_quick_tricks(
    hand: list[Card],
    trump_suit: Optional[Suit],
    trick_history: list[Dict[str, Any]],
    current_trick: Dict[str, Any],
) -> tuple[list[int], list[int]]:
    """
    Identify indices of guaranteed winners and losers in hand.

    Returns:
        (quick_winner_indices, quick_loser_indices)
    """
    # Count Wizards played so far
    wizards_played = 0
    for trick in trick_history:
        for play in trick.get("plays", []):
            card_dict = play.get("card", {})
            if card_dict.get("type") == "WIZARD":
                wizards_played += 1
    for play in current_trick.get("plays", []):
        card_dict = play.get("card", {})
        if card_dict.get("type") == "WIZARD":
            wizards_played += 1

    # Count Wizards in our hand
    wizards_in_hand = sum(1 for c in hand if c.type == CardType.WIZARD)

    quick_winners = []
    quick_losers = []

    for idx, card in enumerate(hand):
        if card.type == CardType.WIZARD:
            quick_winners.append(idx)
        elif card.type == CardType.JESTER:
            quick_losers.append(idx)

    # Trump Ace is guaranteed winner if all Wizards are accounted for
    if wizards_played + wizards_in_hand == 4:
        for idx, card in enumerate(hand):
            if (card.type == CardType.NUMBER
                    and trump_suit is not None
                    and card.suit == trump_suit
                    and card.rank == 13):  # Ace
                if idx not in quick_winners:
                    quick_winners.append(idx)

    return quick_winners, quick_losers


def _analyze_remaining_cards(
    observation: Dict[str, Any],
    hand_cards: list[Card],
    trump_suit: Optional[Suit],
) -> Dict[str, Any]:
    """Summaries of unseen cards for card-counting heuristics."""
    remaining = _build_remaining_deck(observation, hand_cards)

    highest_by_suit: Dict[Suit, int] = {}
    highest_trump: Optional[int] = None
    trump_count = 0
    wizards_left = 0

    for card in remaining:
        if card.type == CardType.WIZARD:
            wizards_left += 1
            continue
        if card.type != CardType.NUMBER:
            continue

        rank = card.rank or 0
        if trump_suit is not None and card.suit == trump_suit:
            trump_count += 1
            if highest_trump is None or rank > highest_trump:
                highest_trump = rank

        current = highest_by_suit.get(card.suit, -1)
        if rank > current:
            highest_by_suit[card.suit] = rank

    return {
        "highest_by_suit": highest_by_suit,
        "highest_trump": highest_trump,
        "trump_count": trump_count,
        "wizards_left": wizards_left,
    }


def _card_counting_quick_winners(
    hand: list[Card],
    trump_suit: Optional[Suit],
    remaining_info: Dict[str, Any],
) -> list[int]:
    """
    Extra quick winners inferred from remaining-deck analysis.

    - If we hold the top remaining trump, treat it as a quick winner.
    - If no trumps or wizards remain unseen, top cards of each suit become strong.
    """
    winners: list[int] = []
    highest_trump = remaining_info.get("highest_trump")
    trump_count = remaining_info.get("trump_count", 0)
    wizards_left = remaining_info.get("wizards_left", 0)
    highest_by_suit: Dict[Suit, int] = remaining_info.get(
        "highest_by_suit", {}
    )

    for idx, card in enumerate(hand):
        if card.type != CardType.NUMBER:
            continue
        rank = card.rank or 0

        if trump_suit is not None and card.suit == trump_suit:
            if highest_trump is None or rank > highest_trump:
                winners.append(idx)

        if trump_count == 0 and wizards_left == 0:
            remaining_top = highest_by_suit.get(card.suit, -1)
            if rank > remaining_top:
                winners.append(idx)

    return winners


def _last_player_optimal_choice(
    hand: list[Card],
    legal_indices: list[int],
    current_plays: list[tuple[int, Card]],
    led_suit: Optional[Suit],
    trump_suit: Optional[Suit],
    bid: int,
    tricks_won: int,
    player_id: int,
) -> Optional[int]:
    """
    Deterministic optimal play when we're last in the trick.

    Returns the optimal card index, or None if MC should be used instead.
    """
    need_tricks = bid - tricks_won

    # Classify cards by whether they would win this trick
    winning_cards = []  # (idx, card, power)
    losing_cards = []   # (idx, card, power)

    # Check if Wizard already played in this trick
    wizard_played = any(c.type == CardType.WIZARD for _, c in current_plays)

    for idx in legal_indices:
        card = hand[idx]
        plays = current_plays + [(player_id, card)]

        # Determine led suit from first non-Wizard play
        eval_led = led_suit
        if eval_led is None:
            for _, c in plays:
                if c.type == CardType.NUMBER:
                    eval_led = c.suit
                    break
                elif c.type == CardType.WIZARD:
                    break  # Wizard leads, no suit

        trick = Trick(plays=plays, led_suit=eval_led)
        winner = winner_of_trick(trick, trump_suit)

        power = _card_power(card, trump_suit, eval_led, wizard_played)

        if winner == player_id:
            winning_cards.append((idx, card, power))
        else:
            losing_cards.append((idx, card, power))

    if need_tricks > 0:
        # Need to win: pick cheapest winner, or if none, preserve best cards
        if winning_cards:
            # Pick lowest-power winning card (save strong cards for later)
            return min(winning_cards, key=lambda x: x[2])[0]
        else:
            # Can't win - discard lowest power card
            return min(losing_cards, key=lambda x: x[2])[0]
    else:
        # Don't need to win: discard high-value loser, or use cheapest winner
        if losing_cards:
            # Discard highest power card (we don't need it)
            return max(losing_cards, key=lambda x: x[2])[0]
        else:
            # Must win - use cheapest winner
            return min(winning_cards, key=lambda x: x[2])[0]


class RandomAgent(WizardAgent):
    """
    Simple random baseline.

    - Bids a random integer in [0, hand_size]
    - Plays a random legal card
    - Chooses a random trump suit when asked
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def choose_bid(self, observation: Dict[str, Any]) -> int:
        hand = observation.get("hand", [])
        hand_size = len(hand)
        return self._rng.randint(0, hand_size)

    def choose_card(self, observation: Dict[str, Any]) -> int:
        legal = observation.get("legal_move_indices", [])
        if not legal:
            return 0
        return self._rng.choice(legal)

    def choose_trump(self, observation: Dict[str, Any]) -> Suit:
        return self._rng.choice(list(Suit))


class MediumAgent(WizardAgent):
    """
    Medium-difficulty agent using Light Monte Carlo simulation.

    Uses the same MC approach as StrongAgent but with
    fewer rollouts for faster execution and weaker play:
    - Bid rollouts: 4 (vs Strong's 80)
    - Play rollouts: 6 (vs Strong's 120)
    - Trump rollouts: 2 (vs Strong's 40)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        # Very Light MC rollouts (relative to Strong)
        self._bid_rollouts = 4
        self._play_rollouts = 6
        self._trump_rollouts = 2

    def choose_bid(self, observation: Dict[str, Any]) -> int:
        hand_cards = [dict_to_card(card) for card in observation["hand"]]
        num_players = observation["game"]["num_players"]
        dealer_id = observation["game"]["dealer_id"]
        leader_id = (dealer_id + 1) % num_players
        trump_name = observation["trump"]["suit"]
        trump_suit = Suit[trump_name] if trump_name else None

        bids_so_far = dict(observation.get("bids_so_far", {}))
        bidding_order = observation.get("bidding_order", [])
        current_index = observation.get("current_bidder_seat_index", 0)

        player_id = observation["player"]["id"]
        hand_size = len(hand_cards)
        total_bids = hand_size + 1
        per_bid = max(1, self._bid_rollouts // total_bids)

        best_bid = 0
        best_score = None
        bid_order = list(range(total_bids))
        self._rng.shuffle(bid_order)

        for bid in bid_order:
            total_score = 0.0
            for _ in range(per_bid):
                remaining = _build_remaining_deck(observation, hand_cards)
                hand_sizes = {
                    pid: observation["game"]["cards_per_player"]
                    for pid in range(num_players)
                }
                other_ids = [
                    pid for pid in range(num_players) if pid != player_id
                ]
                sampled = _sample_hidden_hands(
                    remaining,
                    player_ids=other_ids,
                    hand_sizes=hand_sizes,
                    voids={pid: set() for pid in other_ids},
                    rng=self._rng,
                )
                hands = {pid: sampled[pid] for pid in other_ids}
                hands[player_id] = hand_cards[:]

                bids = dict(bids_so_far)
                bids[player_id] = bid

                if bidding_order:
                    for pid in bidding_order[current_index + 1 :]:
                        if pid in bids:
                            continue
                        bids[pid] = _simulation_bid_from_cards(
                            hands[pid], trump_suit, self._rng
                        )
                    for pid in bidding_order[:current_index]:
                        if pid in bids:
                            continue
                        bids[pid] = _simulation_bid_from_cards(
                            hands[pid], trump_suit, self._rng
                        )
                else:
                    for pid in other_ids:
                        if pid not in bids:
                            bids[pid] = _simulation_bid_from_cards(
                                hands[pid], trump_suit, self._rng
                            )

                tricks_won = {pid: 0 for pid in range(num_players)}
                final_tricks = _simulate_round(
                    hands,
                    bids,
                    tricks_won,
                    leader_id,
                    trump_suit,
                    [],
                    None,
                    rng=self._rng,
                )
                total_score += _round_score_delta(final_tricks[player_id], bid)

            avg_score = total_score / per_bid
            # Small downward bias to counter over-optimistic bidding
            avg_score -= 0.3 * bid
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_bid = bid

        return best_bid

    def choose_card(self, observation: Dict[str, Any]) -> int:
        player_id = observation["player"]["id"]
        num_players = observation["game"]["num_players"]
        hand_cards = [dict_to_card(card) for card in observation["hand"]]
        legal_indices = observation["legal_move_indices"]
        trick_history = observation.get("trick_history", [])
        current_trick = observation.get("current_trick", {"plays": []})

        trump_name = observation["trump"]["suit"]
        trump_suit = Suit[trump_name] if trump_name else None
        led_name = current_trick.get("led_suit")
        led_suit = Suit[led_name] if led_name else None
        leader_id = observation["leader_id"]

        bids = dict(observation["bids"])
        tricks_won = dict(observation["tricks_taken_so_far"])
        hand_sizes = dict(observation["hand_sizes"])

        total_rollouts = max(1, self._play_rollouts)
        per_move = max(1, total_rollouts // len(legal_indices))
        extra = total_rollouts - per_move * len(legal_indices)

        move_order = legal_indices[:]
        self._rng.shuffle(move_order)
        best_move = move_order[0]
        best_score = None

        voids = _infer_void_suits(
            trick_history,
            current_trick,
            our_id=player_id,
            num_players=num_players,
        )

        for idx in move_order:
            rollouts = per_move + (1 if extra > 0 else 0)
            if extra > 0:
                extra -= 1
            total_score = 0.0
            for _ in range(rollouts):
                remaining = _build_remaining_deck(observation, hand_cards)
                other_ids = [
                    pid for pid in range(num_players) if pid != player_id
                ]
                sampled = _sample_hidden_hands(
                    remaining,
                    player_ids=other_ids,
                    hand_sizes=hand_sizes,
                    voids=voids,
                    rng=self._rng,
                )
                hands = {pid: sampled[pid] for pid in other_ids}
                our_hand = hand_cards[:]
                chosen_card = our_hand.pop(idx)
                hands[player_id] = our_hand

                plays = [
                    (play["player_id"], dict_to_card(play["card"]))
                    for play in current_trick.get("plays", [])
                ]
                plays.append((player_id, chosen_card))

                final_tricks = _simulate_round(
                    hands,
                    bids,
                    dict(tricks_won),
                    leader_id,
                    trump_suit,
                    plays,
                    led_suit,
                    rng=self._rng,
                )
                total_score += _round_score_delta(
                    final_tricks[player_id], bids[player_id]
                )

            avg_score = total_score / rollouts
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_move = idx

        return best_move

    def choose_trump(self, observation: Dict[str, Any]) -> Suit:
        hand_cards = observation.get("hand_cards", [])
        num_players = observation["game"]["num_players"]
        dealer_id = observation["game"]["dealer_id"]
        leader_id = (dealer_id + 1) % num_players
        player_id = observation["player"]["id"]

        candidates = list(Suit)
        self._rng.shuffle(candidates)
        per_suit = max(1, self._trump_rollouts // len(candidates))
        best_suit = candidates[0]
        best_score = None

        for suit in candidates:
            total_score = 0.0
            for _ in range(per_suit):
                remaining = _build_remaining_deck(
                    observation, list(hand_cards)
                )
                hand_sizes = {
                    pid: observation["game"]["cards_per_player"]
                    for pid in range(num_players)
                }
                other_ids = [
                    pid for pid in range(num_players) if pid != player_id
                ]
                sampled = _sample_hidden_hands(
                    remaining,
                    player_ids=other_ids,
                    hand_sizes=hand_sizes,
                    voids={pid: set() for pid in other_ids},
                    rng=self._rng,
                )
                hands = {pid: sampled[pid] for pid in other_ids}
                hands[player_id] = list(hand_cards)

                bids = {}
                for pid in range(num_players):
                    bids[pid] = _simulation_bid_from_cards(
                        hands[pid], suit, self._rng
                    )

                tricks_won = {pid: 0 for pid in range(num_players)}
                final_tricks = _simulate_round(
                    hands,
                    bids,
                    tricks_won,
                    leader_id,
                    suit,
                    [],
                    None,
                    rng=self._rng,
                )
                total_score += _round_score_delta(
                    final_tricks[player_id], bids[player_id]
                )

            avg_score = total_score / per_suit
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_suit = suit

        return best_suit


class StrongAgent(WizardAgent):
    """
    Strong Monte Carlo agent with advanced enhancements:
    1. Optimal bidding for 1-card rounds (skip MC, use math)
    2. Bid-informed opponent hand sampling
    3. Full card counting for accurate late-game sampling
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._bid_rollouts = 80
        self._play_rollouts = 120
        self._trump_rollouts = 40

    def choose_bid(self, observation: Dict[str, Any]) -> int:
        hand_cards = [dict_to_card(card) for card in observation["hand"]]
        num_players = observation["game"]["num_players"]
        dealer_id = observation["game"]["dealer_id"]
        leader_id = (dealer_id + 1) % num_players
        trump_name = observation["trump"]["suit"]
        trump_suit = Suit[trump_name] if trump_name else None

        bids_so_far = dict(observation.get("bids_so_far", {}))
        bidding_order = observation.get("bidding_order", [])
        current_index = observation.get("current_bidder_seat_index", 0)

        player_id = observation["player"]["id"]
        hand_size = len(hand_cards)

        # Enhancement: Use optimal bid for 1-card rounds (skip MC)
        if hand_size == 1:
            return _optimal_single_card_bid(hand_cards[0], trump_suit, num_players)

        total_bids = hand_size + 1
        per_bid = max(1, self._bid_rollouts // total_bids)

        best_bid = 0
        best_score = None
        bid_order = list(range(total_bids))
        self._rng.shuffle(bid_order)

        for bid in bid_order:
            total_score = 0.0
            for _ in range(per_bid):
                remaining = _build_remaining_deck(observation, hand_cards)
                hand_sizes_map = {
                    pid: observation["game"]["cards_per_player"]
                    for pid in range(num_players)
                }
                other_ids = [
                    pid for pid in range(num_players) if pid != player_id
                ]
                sampled = _sample_hidden_hands(
                    remaining,
                    player_ids=other_ids,
                    hand_sizes=hand_sizes_map,
                    voids={pid: set() for pid in other_ids},
                    rng=self._rng,
                )
                hands = {pid: sampled[pid] for pid in other_ids}
                hands[player_id] = hand_cards[:]

                bids = dict(bids_so_far)
                bids[player_id] = bid

                if bidding_order:
                    for pid in bidding_order[current_index + 1 :]:
                        if pid in bids:
                            continue
                        bids[pid] = _simulation_bid_from_cards(
                            hands[pid], trump_suit, self._rng
                        )
                    for pid in bidding_order[:current_index]:
                        if pid in bids:
                            continue
                        bids[pid] = _simulation_bid_from_cards(
                            hands[pid], trump_suit, self._rng
                        )
                else:
                    for pid in other_ids:
                        if pid not in bids:
                            bids[pid] = _simulation_bid_from_cards(
                                hands[pid], trump_suit, self._rng
                            )

                tricks_won = {pid: 0 for pid in range(num_players)}
                final_tricks = _simulate_round(
                    hands,
                    bids,
                    tricks_won,
                    leader_id,
                    trump_suit,
                    [],
                    None,
                    rng=self._rng,
                )
                total_score += _round_score_delta(final_tricks[player_id], bid)

            avg_score = total_score / per_bid
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_bid = bid

        return best_bid

    def choose_card(self, observation: Dict[str, Any]) -> int:
        """
        Choose card with three enhancements:
        1. Last-player deterministic optimization
        2. Quick tricks heuristic
        3. Overbid/underbid round awareness
        """
        player_id = observation["player"]["id"]
        num_players = observation["game"]["num_players"]
        hand_cards = [dict_to_card(card) for card in observation["hand"]]
        legal_indices = observation["legal_move_indices"]
        trick_history = observation.get("trick_history", [])
        current_trick = observation.get("current_trick", {"plays": []})

        trump_name = observation["trump"]["suit"]
        trump_suit = Suit[trump_name] if trump_name else None
        led_name = current_trick.get("led_suit")
        led_suit = Suit[led_name] if led_name else None
        leader_id = observation["leader_id"]

        bids = dict(observation["bids"])
        tricks_won = dict(observation["tricks_taken_so_far"])
        hand_sizes = dict(observation["hand_sizes"])
        cards_per_player = observation["game"]["cards_per_player"]

        my_bid = bids[player_id]
        my_tricks = tricks_won.get(player_id, 0)
        need_tricks = my_bid - my_tricks
        remaining_info = _analyze_remaining_cards(
            observation, hand_cards, trump_suit
        )

        # Build current plays for analysis
        current_plays = [
            (play["player_id"], dict_to_card(play["card"]))
            for play in current_trick.get("plays", [])
        ]

        # =================================================================
        # Enhancement 3: Last player optimization (deterministic)
        # When we're last to play, we have perfect information for this trick
        # =================================================================
        if len(current_plays) == num_players - 1:
            optimal = _last_player_optimal_choice(
                hand_cards,
                legal_indices,
                current_plays,
                led_suit,
                trump_suit,
                my_bid,
                my_tricks,
                player_id,
            )
            if optimal is not None:
                return optimal

        # =================================================================
        # Enhancement 2: Quick tricks heuristic
        # Skip MC for obvious plays (guaranteed winners/losers)
        # =================================================================
        quick_winners, quick_losers = _identify_quick_tricks(
            hand_cards, trump_suit, trick_history, current_trick
        )
        # Card-counting quick winners (top trump, or top-of-suit when no trumps/wizards remain)
        cc_winners = _card_counting_quick_winners(
            hand_cards, trump_suit, remaining_info
        )
        if cc_winners:
            quick_winners = list(set(quick_winners + cc_winners))

        # Filter to legal moves only
        legal_winners = [i for i in quick_winners if i in legal_indices]
        legal_losers = [i for i in quick_losers if i in legal_indices]

        # If we need exactly N more tricks and have exactly N quick winners, use them
        if need_tricks > 0 and len(legal_winners) == need_tricks:
            if legal_winners:
                return legal_winners[0]

        # If we've met our bid and only have quick losers available, use any
        if need_tricks <= 0 and legal_losers and len(legal_losers) == len(legal_indices):
            return self._rng.choice(legal_losers)

        # =================================================================
        # Enhancement 1: Overbid/underbid round awareness
        # Adjust MC scoring based on round dynamics
        # =================================================================
        is_overbid, bid_diff = _is_overbid_round(bids, cards_per_player)

        # MC rollouts with adjusted scoring
        total_rollouts = max(1, self._play_rollouts)
        per_move = max(1, total_rollouts // len(legal_indices))
        extra = total_rollouts - per_move * len(legal_indices)

        move_order = legal_indices[:]
        self._rng.shuffle(move_order)
        best_move = move_order[0]
        best_score = None

        voids = _infer_void_suits(
            trick_history,
            current_trick,
            our_id=player_id,
            num_players=num_players,
        )

        remaining = _build_remaining_deck(observation, hand_cards)
        other_ids = [pid for pid in range(num_players) if pid != player_id]

        # Precompute scenarios (shared across moves) to reduce variance
        precomputed: list[Dict[int, list[Card]]] = []
        enumerated: list[Dict[int, list[Card]]] = []
        if len(remaining) <= 7:
            enumerated = _enumerate_hidden_hands(
                remaining,
                player_ids=other_ids,
                hand_sizes=hand_sizes,
                voids=voids,
                limit=2000,
            )
        if enumerated:
            precomputed = enumerated
        else:
            precomputed = [
                _sample_hidden_hands(
                    remaining,
                    player_ids=other_ids,
                    hand_sizes=hand_sizes,
                    voids=voids,
                    rng=self._rng,
                )
                for _ in range(total_rollouts)
            ]

        for idx in move_order:
            rollouts = per_move + (1 if extra > 0 else 0)
            if extra > 0:
                extra -= 1
            total_score = 0.0
            scenarios = precomputed if precomputed else []
            if precomputed and len(precomputed) > rollouts:
                scenarios = precomputed[:rollouts]

            for sampled in scenarios:
                hands = {pid: list(sampled[pid]) for pid in other_ids}
                our_hand = hand_cards[:]
                chosen_card = our_hand.pop(idx)
                hands[player_id] = our_hand

                plays = [
                    (play["player_id"], dict_to_card(play["card"]))
                    for play in current_trick.get("plays", [])
                ]
                plays.append((player_id, chosen_card))

                final_tricks = _simulate_round(
                    hands,
                    bids,
                    dict(tricks_won),
                    leader_id,
                    trump_suit,
                    plays,
                    led_suit,
                    rng=self._rng,
                )

                base_score = _round_score_delta(
                    final_tricks[player_id], bids[player_id]
                )

                if is_overbid and base_score > 0:
                    base_score += 0.5 * abs(bid_diff)
                elif not is_overbid and base_score > 0:
                    my_final_tricks = final_tricks[player_id]
                    if my_final_tricks > my_bid:
                        base_score -= 0.3 * abs(bid_diff)

                total_score += base_score

            avg_score = total_score / rollouts
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_move = idx

        return best_move

    def choose_trump(self, observation: Dict[str, Any]) -> Suit:
        hand_cards = observation.get("hand_cards", [])
        num_players = observation["game"]["num_players"]
        dealer_id = observation["game"]["dealer_id"]
        leader_id = (dealer_id + 1) % num_players
        player_id = observation["player"]["id"]

        candidates = list(Suit)
        self._rng.shuffle(candidates)
        per_suit = max(1, self._trump_rollouts // len(candidates))
        best_suit = candidates[0]
        best_score = None

        for suit in candidates:
            total_score = 0.0
            for _ in range(per_suit):
                remaining = _build_remaining_deck(
                    observation, list(hand_cards)
                )
                hand_sizes = {
                    pid: observation["game"]["cards_per_player"]
                    for pid in range(num_players)
                }
                other_ids = [
                    pid for pid in range(num_players) if pid != player_id
                ]
                sampled = _sample_hidden_hands(
                    remaining,
                    player_ids=other_ids,
                    hand_sizes=hand_sizes,
                    voids={pid: set() for pid in other_ids},
                    rng=self._rng,
                )
                hands = {pid: sampled[pid] for pid in other_ids}
                hands[player_id] = list(hand_cards)

                bids = {}
                for pid in range(num_players):
                    bids[pid] = _simulation_bid_from_cards(
                        hands[pid], suit, self._rng
                    )

                tricks_won = {pid: 0 for pid in range(num_players)}
                final_tricks = _simulate_round(
                    hands,
                    bids,
                    tricks_won,
                    leader_id,
                    suit,
                    [],
                    None,
                    rng=self._rng,
                )
                total_score += _round_score_delta(
                    final_tricks[player_id], bids[player_id]
                )

            avg_score = total_score / per_suit
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_suit = suit

        return best_suit
