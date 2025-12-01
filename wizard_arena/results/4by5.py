import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Load data ----
csv_path = "wizard_arena/results/wizard_results_60_games.csv"  # <- change this to your path
df = pd.read_csv(csv_path)

# ---- 2. Filter to games with exactly 60 rows ----
counts = df['game_id'].value_counts()
valid_games = counts[counts == 60].index
df = df[df['game_id'].isin(valid_games)].copy()

# ---- 3. Compute bid miss per row ----
# negative -> underbid (not enough tricks)
# positive -> overbid (too many tricks)
df['miss'] = df['tricks_won'] - df['bid']

models = sorted(df['player_name'].unique())

# ---- 4. For each model, make a 4x5 grid of histograms (one per round) ----
for model in models:
    sub = df[df['player_name'] == model]

    # Common bins for this model so rounds are comparable
    miss_min = sub['miss'].min()
    miss_max = sub['miss'].max()
    bins = np.arange(np.floor(miss_min) - 0.5, np.ceil(miss_max) + 1.5, 1.0)

    fig, axes = plt.subplots(4, 5, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for r, ax in enumerate(axes):
        # Assuming round_index 0..19 (20 rounds)
        round_data = sub[sub['round_index'] == r]['miss']

        if not round_data.empty:
            ax.hist(round_data, bins=bins, rwidth=0.8)
        ax.axvline(0, linestyle='--')  # exact-bid line
        ax.set_title(f"Round {r}", fontsize=9)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    # Common labels
    fig.suptitle(f"Bid miss distribution by round\n{model}", y=0.95)
    fig.text(0.5, 0.04, "miss = tricks_won - bid", ha='center')
    fig.text(0.04, 0.5, "Count", va='center', rotation='vertical')

    plt.tight_layout(rect=[0.06, 0.06, 1, 0.93])

    # Optional: save instead of / in addition to showing
    # fig.savefig(f"bid_miss_rounds_{model.replace(':', '_')}.png", dpi=200)

    plt.show()
