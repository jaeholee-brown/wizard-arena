import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Load data ----
csv_path = "wizard_arena/results/wizard_results_60_games.csv"   # <- change to your actual path
df = pd.read_csv(csv_path)

# ---- 2. Filter to games with exactly 60 rows ----
counts = df['game_id'].value_counts()
valid_games = counts[counts == 60].index
df = df[df['game_id'].isin(valid_games)].copy()

# ---- 3. Compute per-round over/under ----
# For each round: sum bids across all players,
# compare to actual number of tricks in that round = cards_per_player
per_round = (
    df.groupby(['game_id', 'round_index', 'cards_per_player'])
      .agg(total_bid=('bid', 'sum'))
      .reset_index()
)

# round_miss < 0: underbid (total bids too low)
# round_miss > 0: overbid (total bids too high)
per_round['round_miss'] = per_round['total_bid'] - per_round['cards_per_player']

# ---- 4. Histogram of round over/under ----
values = per_round['round_miss']

# nice integer-centered bins
miss_min = values.min()
miss_max = values.max()
bins = np.arange(np.floor(miss_min) - 0.5, np.ceil(miss_max) + 1.5, 1.0)

plt.figure(figsize=(8, 5))
plt.hist(values, bins=bins, rwidth=0.8)
plt.axvline(0, linestyle='--')  # exact-total-bid line

plt.xlabel("Round over/under (sum(bid) - cards_per_player)")
plt.ylabel("Number of rounds")
plt.title("Histogram of round-level over/under bidding\n(negative = underbid, positive = overbid)")
plt.grid(True, axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
