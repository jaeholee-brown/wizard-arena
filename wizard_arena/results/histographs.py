import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Load data & filter games with exactly 60 rows ----
csv_path = "wizard_arena/results/wizard_benchmark_scores.csv"   # <- change to your path
df = pd.read_csv(csv_path)

### Count rows per game_id
counts = df['game_id'].value_counts()

### Keep only game_ids that appear in exactly 60 rows
valid_games = counts[counts == 60].index
df = df[df['game_id'].isin(valid_games)].copy()

# Expecting: 'player_name', 'bid', 'tricks_won'

# ---- 2. Compute bid miss ----
# negative -> undertrick; positive -> overtrick
df['miss'] = df['tricks_won'] - df['bid']

# ---- 3. Set up per-model histograms ----
models = sorted(df['player_name'].unique())
n_models = len(models)

# common bin edges across all models so histos are comparable
miss_min = df['miss'].min()
miss_max = df['miss'].max()
bins = np.arange(np.floor(miss_min) - 0.5, np.ceil(miss_max) + 1.5, 1.0)

fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)

# axes might be a single Axes if only one model
axes = np.atleast_1d(axes)

for ax, model in zip(axes, models):
    subset = df[df['player_name'] == model]['miss']
    ax.hist(subset, bins=bins, rwidth=0.8)
    ax.axvline(0, linestyle='--')  # exact-bid line
    ax.set_title(model)
    ax.set_xlabel("miss (tricks_won - bid)")
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)

axes[0].set_ylabel("Count")

plt.suptitle("Histogram of bid miss by model\n(negative = under, positive = over)", y=1.03)
plt.tight_layout()
plt.show()
