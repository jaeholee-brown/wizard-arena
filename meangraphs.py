import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Load data ----
# Change this to your actual path
csv_path = "wizard_results_seed1_50games.csv"
df = pd.read_csv(csv_path)

# Expecting at least these columns:
# 'game_id', 'round_index', 'player_name', 'total_score'

# ---- 2. Compute per-round stats (mean, std, count) ----
round_stats = (
    df.groupby(['player_name', 'round_index'])['total_score']
      .agg(['mean', 'std', 'count'])
      .reset_index()
)

# 95% confidence interval: mean ± 1.96 * (std / sqrt(n))
round_stats['se'] = round_stats['std'] / np.sqrt(round_stats['count'])
round_stats['ci95'] = 1.96 * round_stats['se']

# ---- 3. Plot: per-round mean total score with 95% CI ----
models = sorted(df['player_name'].unique())

plt.figure(figsize=(10, 6))
for model in models:
    sub = round_stats[round_stats['player_name'] == model].sort_values('round_index')
    plt.errorbar(
        sub['round_index'],
        sub['mean'],
        yerr=sub['ci95'],
        marker='o',
        capsize=3,
        label=model
    )

plt.xlabel("Round index")
plt.ylabel("Mean total score across games")
plt.title("Per-round mean total score with 95% CI (averaged over games)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
