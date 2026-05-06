"""
Manual evaluation plots.

1. plot_heatmap()     – Win-rate heatmap from a manually entered NxN matrix.
2. plot_bar_baselines() – Grouped bar chart: win rate of each agent vs every baseline.

Run directly:
    python utils/plot_evaluation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ===========================================================================
# 1. HEATMAP – trained vs trained
# ===========================================================================
# Row i, column j = win rate of agent i (Team A) against agent j (Team B).
# Use None for the diagonal or any unplayed matchup.

HEATMAP_AGENT_NAMES = [
    "ffa_classic_50", # 79
    "ffa_gumbel_50", # 78
    "ffa_classic_inf", # 73
    "ffa_gumbel_inf", # 72
]

HEATMAP_MATRIX = [
    #  
    [None,  0.183,  0.002, 0.002],  # 79
    [0.339,  None,  0.004, 0.0],  # 78
    [0.879,  0.852,  None, 0.237],  # 73
    [0.872,  0.770, 0.075,  None],  # 72
]#   79        78    73       72




# ===========================================================================
# 2. BALKENDIAGRAMM – trained vs baselines
# ===========================================================================
# BAR_AGENTS   : agent short names (one bar group per agent)
# BAR_BASELINES: baseline names (one bar colour per baseline)
# BAR_MATRIX[i][j] = win rate of agent i against baseline j

# BAR_AGENTS = [
#     "ffa_stochastic_25", #35
#     "ffa_stochastic_mix", # 37
#     "ffa_stochastic_inf", #33
# ]

BAR_AGENTS = [
    "ffa_classic_50", #
    "ffa_gumbel_50", # 
    "ffa_classic_inf", #
    "ffa_gumbel_inf", #
]

BAR_BASELINES = [
    "random",
    "rule_based",
    "untrained MuZero",
    "untrained Gumbel MuZero",
]

BAR_MATRIX = [
    #  random  rule_based  untrained MuZero
    [0.375,    0.555,       0.592,       0.513],   # 79
    [0.468,    0.676,       0.544,       0.503],   # 78 
    [0.938,    0.964,       0.914,       0.909],   # 73
    [0.946,    0.934,       0.940,       0.956]    # 72
]  


   # e.g. "utils/bar_baselines.png", or None to only show


# ===========================================================================
# Plot functions
# ===========================================================================

def plot_heatmap(agent_names, matrix, output_file=None, title="Win Rate Heatmap"):
    n = len(agent_names)
    mat = np.array(
        [[np.nan if v is None else v for v in row] for row in matrix],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))
    masked = np.ma.masked_invalid(mat)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=1.0)
    im = ax.imshow(masked, norm=norm, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Win rate")

    # Gray diagonal patches
    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   color="#aaaaaa", zorder=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agent_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(agent_names, fontsize=9)
    ax.set_xlabel("Opponents")
    ax.set_ylabel("Agent")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="white", zorder=3)
            elif not np.isnan(mat[i, j]):
                text_color = "white" if abs(norm(mat[i, j]) - 0.5) > 0.3 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold", zorder=3)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {output_file}")
    plt.show()


def plot_bar_baselines(agent_names, baseline_names, matrix, output_file=None,
                       title="Win Rate vs Baselines"):
    """
    Grouped bar chart with a zoomed y-axis.
    The y-axis starts just below the minimum value so small differences become visible.
    """
    mat = np.array(matrix, dtype=float)   # shape (n_agents, n_baselines)
    n_agents = len(agent_names)
    n_baselines = len(baseline_names)

    x = np.arange(n_agents)
    total_width = 0.85
    bar_width = total_width / n_baselines
    offsets = np.linspace(-total_width / 2 + bar_width / 2,
                           total_width / 2 - bar_width / 2,
                           n_baselines)

    # Zoom y-axis: start 10 % of the value range below the minimum
    valid = mat[~np.isnan(mat)]
    margin = max((valid.max() - valid.min()) * 0.5, 0.02)
    y_min = max(0.0, valid.min() - margin)
    y_max = min(1.0, valid.max() + margin * 0.5)

    fig, ax = plt.subplots(figsize=(max(8, n_agents * 1.5), 5))

    colors = plt.cm.Set2(np.linspace(0, 1, n_baselines))
    for k, (baseline, offset, color) in enumerate(zip(baseline_names, offsets, colors)):
        bars = ax.bar(x + offset, mat[:, k], width=bar_width, label=baseline, color=color,
                      edgecolor="white", bottom=0)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (y_max - y_min) * 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Win rate")
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
    ax.axhline(0.25, color="grey", linewidth=0.8, linestyle="--", label="25 %") # FFA expected is 0.25
    ax.set_title(title)
    ax.legend(title="Baseline", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    if output_file:
        out = output_file.replace(".png", "_bar.png") if output_file else None
        plt.savefig(out or output_file, dpi=150, bbox_inches="tight")
        print(f"Bar chart saved to {out or output_file}")
    plt.show()


def plot_lollipop_baselines(agent_names, baseline_names, matrix, output_file=None,
                            title="Win Rate vs Baselines"):
    """
    Lollipop (dot + stem) chart — better than bars when values are clustered near 1.
    Each subplot row = one baseline; dots = agents.
    """
    mat = np.array(matrix, dtype=float)   # shape (n_agents, n_baselines)
    n_agents = len(agent_names)
    n_baselines = len(baseline_names)

    valid = mat[~np.isnan(mat)]
    margin = max((valid.max() - valid.min()) * 0.5, 0.02)
    y_min = max(0.0, valid.min() - margin)
    y_max = min(1.0, valid.max() + margin * 0.5)

    colors = plt.cm.Set2(np.linspace(0, 1, n_baselines))
    fig, axes = plt.subplots(n_baselines, 1,
                             figsize=(max(8, n_agents * 1.2), 3 * n_baselines),
                             sharex=True)
    if n_baselines == 1:
        axes = [axes]

    x = np.arange(n_agents)
    for k, (baseline, ax, color) in enumerate(zip(baseline_names, axes, colors)):
        vals = mat[:, k]
        ax.hlines(vals, x - 0.4, x + 0.4, colors=color, linewidth=2.5, alpha=0.7)
        ax.scatter(x, vals, color=color, s=80, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (y_max - y_min) * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.axhline(0.25, color="grey", linewidth=0.8, linestyle="--", label="25 %") # FFA expected is 0.25
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)
        ax.set_ylabel(f"Win rate\nvs {baseline}", fontsize=9)
        ax.set_yticks(np.round(np.linspace(y_min, y_max, 5), 3))

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(agent_names, rotation=15, ha="right", fontsize=9)
    fig.suptitle(title, fontsize=11, y=1.01)

    plt.tight_layout()
    if output_file:
        out = output_file.replace(".png", "_lollipop.png") if output_file else None
        plt.savefig(out or output_file, dpi=150, bbox_inches="tight")
        print(f"Lollipop chart saved to {out or output_file}")
    plt.show()


# ===========================================================================
if __name__ == "__main__":
    HEATMAP_OUTPUT = "images/dMADNheatmap_ffa.png"   # e.g. "utils/heatmap.png", or None to only show
    BAR_OUTPUT = "images/dMADNbar_baselines_ffa.png"
    plot_heatmap(HEATMAP_AGENT_NAMES, HEATMAP_MATRIX, output_file=HEATMAP_OUTPUT)
    plot_bar_baselines(BAR_AGENTS, BAR_BASELINES, BAR_MATRIX, output_file=BAR_OUTPUT)
    #plot_lollipop_baselines(BAR_AGENTS, BAR_BASELINES, BAR_MATRIX, output_file=BAR_OUTPUT)
