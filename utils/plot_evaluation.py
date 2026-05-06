"""
Manual evaluation plots.

1. plot_heatmap()     – Win-rate heatmap from a manually entered NxN matrix.
2. plot_bar_baselines() – Grouped bar chart: win rate of each agent vs every baseline.

Run directly:
    python utils/plot_evaluation.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================
# 1. HEATMAP – trained vs trained
# ===========================================================================
# Row i, column j = win rate of agent i (Team A) against agent j (Team B).
# Use None for the diagonal or any unplayed matchup.

HEATMAP_AGENT_NAMES = [
    "team_stochastic_25", #30
    "team_stochastic_mix", #36
    "team_stochastic_inf", #34
]

HEATMAP_MATRIX = [
    #  
    [None,  0.605,  0.378],  # 
    [0.395,  None,  0.284],  # 
    [0.622,  0.716,  None],  # 
]




# ===========================================================================
# 2. BALKENDIAGRAMM – trained vs baselines
# ===========================================================================
# BAR_AGENTS   : agent short names (one bar group per agent)
# BAR_BASELINES: baseline names (one bar colour per baseline)
# BAR_MATRIX[i][j] = win rate of agent i against baseline j

BAR_AGENTS = [
    "team_stochastic_25",
    "team_stochastic_mix",
    "team_stochastic_inf",
]

BAR_BASELINES = [
    "random",
    "rule_based",
    "untrained MuZero",
]

BAR_MATRIX = [
    #  random  rule_based  untrained MuZero
    [0.739,    0.536,       0.818],   
    [0.596,    0.367,       0.740],   
    [0.811,    0.615,       0.853],    
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
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Win rate (Team A)")

    # Gray diagonal patches
    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   color="#aaaaaa", zorder=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agent_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(agent_names, fontsize=9)
    ax.set_xlabel("Opponent (Team B)")
    ax.set_ylabel("Agent (Team A)")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="white", zorder=3)
            elif not np.isnan(mat[i, j]):
                text_color = "white" if abs(mat[i, j] - 0.5) > 0.3 else "black"
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
    ax.set_ylabel("Win rate (Team A)")
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="50 %")
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
        ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
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
    HEATMAP_OUTPUT = "images/MADNheatmap.png"   # e.g. "utils/heatmap.png", or None to only show
    BAR_OUTPUT = "images/MADNbar_baselines.png"
    plot_heatmap(HEATMAP_AGENT_NAMES, HEATMAP_MATRIX, output_file=HEATMAP_OUTPUT)
    plot_bar_baselines(BAR_AGENTS, BAR_BASELINES, BAR_MATRIX, output_file=BAR_OUTPUT)
    #plot_lollipop_baselines(BAR_AGENTS, BAR_BASELINES, BAR_MATRIX, output_file=BAR_OUTPUT)
