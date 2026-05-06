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
    "classic_nobelief_50", 
    "classic_nobelief_inf", 
    "sample_nobelief_50", 
    "sample_nobelief_inf", 
    "session_nobelief_50", 
    "session_nobelief_inf", 
]

# HEATMAP_MATRIX = [
#     #  
#     [None,  0.593,  0.127,  0.025,  0.004,  0.004],  # 36
#     [0.407,  None,  0.104,  0.014,  0.,  0.002],  # 28
#     [0.873,  0.896,  None,  0.209,  0.071,  0.069],  # 33
#     [0.975,  0.986,  0.791,  None,  0.210,  0.209],  # 34
#     [0.996,  1.,  0.929,  0.790,  None,  0.522],  # 25
#     [0.996,  0.998,  0.931,  0.791,  0.478,   None], # 27
# ]
HEATMAP_MATRIX = [
    #  
    [None,  0.609,  0.047,  0.07,  0.008,  0.001],  # 41
    [0.390,  None,  0.023,  0.001,  0.001,  0.001],  # 40
    [0.953,  0.997,  None,  0.137,  0.157,  0.070],  # 22
    [0.993,  0.999,  0.863,  None,  0.489,  0.282],  # 39
    [0.992,  0.999,  0.843,  0.511,  None,  0.317],  # 23
    [0.999,  0.999,  0.930,  0.718,  0.683,   None], # 24
]




# ===========================================================================
# 2. BALKENDIAGRAMM – trained vs baselines
# ===========================================================================
# BAR_AGENTS   : agent short names (one bar group per agent)
# BAR_BASELINES: baseline names (one bar colour per baseline)
# BAR_MATRIX[i][j] = win rate of agent i against baseline j

BAR_AGENTS = [
    "classic_belief_50", 
    "classic_belief_inf", 
    "sample_belief_50", 
    "sample_belief_inf", 
    "session_belief_50", 
    "session_belief_inf", 
]

BAR_BASELINES = [
    "random",
    "rule_based",
    "untrained MuZero",
]

BAR_MATRIX = [
    #  random  rule_based  untrained MuZero
    [0.126,    0.002,       0.147],   
    [0.088,    0.0,       0.126],   
    [0.621,    0.073,       0.705],    
    [0.835,    0.225,       0.882], 
    [0.944,    0.480,       0.967],
    [0.958,    0.501,       0.960], 
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


def plot_belief_duel(matchups, output_file=None,
                     title="Belief vs No-Belief: Direct Duels"):
    """
    Diverging horizontal bar chart for belief vs no-belief matchups.

    Each matchup has two bars meeting at the center:
      - Left bar  (red)  : no-belief win rate, extends LEFT  from center (0)
      - Right bar (green): belief win rate,    extends RIGHT from center (0)
    Together they always sum to 100%.

    Parameters
    ----------
    matchups : list of dicts, each with:
        - 'label': str, display name for the matchup
        - 'belief_wins': int, number of wins for the belief agent
        - 'total': int, total games played
        - 'group': str, variant group name (e.g. "Session", "Classic", "Sample")
    output_file : str or None
    title : str
    """
    n = len(matchups)
    belief_rates   = np.array([m['belief_wins'] / m['total'] for m in matchups])
    nobelief_rates = 1.0 - belief_rates  # always sums to 1 with belief_rates

    # Group separators
    groups = [m['group'] for m in matchups]
    unique_groups = list(dict.fromkeys(groups))  # preserve insertion order

    # Color pairs per group: belief = green spectrum, no-belief = complementary
    # Complement = hue + 180° on the color wheel
    GROUP_COLOR_PAIRS = {
        'Classic': ('#2e8b57', '#8b2e65'),   # sea green       ↔ deep pink-magenta
        'Sample':  ('#7cb832', "#b132b8"),   # yellow-green    ↔ blue-violet
        'Session': ("#3aaf88", '#a93a40'),   # teal-green      ↔ brick red
    }
    # Fallback greens + complements for any extra groups
    fallback_pairs = [
        ('#52b788', '#b85278'), ('#b5c935', '#7535c9'), ('#1a9850', '#981a62'),
    ]
    for i, g in enumerate(unique_groups):
        if g not in GROUP_COLOR_PAIRS:
            GROUP_COLOR_PAIRS[g] = fallback_pairs[i % len(fallback_pairs)]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 1.1 + 1.2)))

    y_pos = np.arange(n)[::-1]  # top to bottom
    height = 0.55

    for i, (y, br, nr, m) in enumerate(zip(y_pos, belief_rates, nobelief_rates, matchups)):
        g = m['group']
        belief_color, nobelief_color = GROUP_COLOR_PAIRS[g]

        # Right bar: belief win rate (starts at 0, goes right)
        ax.barh(y, br,  height=height, left=0,   color=belief_color,   edgecolor='white', linewidth=0.6)
        # Left bar:  no-belief win rate (starts at 0, goes left → negative width)
        ax.barh(y, -nr, height=height, left=0,   color=nobelief_color, edgecolor='white', linewidth=0.6)

        # Annotations: inside each bar, near the tip
        ax.text( br - 0.015, y, f"{br:.1%}",  ha='right', va='center', fontsize=8.5,
                fontweight='bold', color='white')
        ax.text(-nr + 0.015, y, f"{nr:.1%}", ha='left',  va='center', fontsize=8.5,
                fontweight='bold', color='white')

    # Center separator line
    ax.axvline(0, color='black', linewidth=1.5, zorder=4)

    # Y-axis labels
    labels = [m['label'] for m in matchups]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    # X-axis: 0 % on left, 100 % on right, center = 0
    ax.set_xlim(-1.0, 1.0)
    tick_vals = np.linspace(-1.0, 1.0, 11)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{abs(v)*100:.0f}%" for v in tick_vals], fontsize=8)
    ax.set_xlabel("Win Rate", fontsize=10)

    # Side header labels below x-axis (axes-fraction coordinates)
    ax.text(0.25, -0.12, "← No Belief", ha='center', va='top',
            fontsize=10, color='#888888', fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.75, -0.12, "Belief →",    ha='center', va='top',
            fontsize=10, color='#333333', fontweight='bold',
            transform=ax.transAxes)

    # Group separators (dashed horizontal lines between variant blocks)
    prev_group = groups[0]
    for i in range(1, n):
        if groups[i] != prev_group:
            sep_y = (y_pos[i] + y_pos[i - 1]) / 2
            ax.axhline(sep_y, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
            prev_group = groups[i]

    ax.legend([], [], frameon=False)  # no legend

    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(axis='x', alpha=0.25, linestyle=':')
    ax.set_axisbelow(True)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Belief duel chart saved to {output_file}")
    plt.show()


# ===========================================================================
# Cross-group evaluation data: Belief vs No-Belief
# ===========================================================================
BELIEF_DUEL_MATCHUPS = [
    {'label': 'Classic TD-50 target', 'belief_wins': 686, 'total': 1000, 'group': 'Classic'},
    {'label': 'Classic MC-target', 'belief_wins': 653, 'total': 995, 'group': 'Classic'},
    {'label': 'Sample TD-50 target',  'belief_wins': 523, 'total': 1000, 'group': 'Sample'},
    {'label': 'Sample MC-target',  'belief_wins': 421, 'total': 1000, 'group': 'Sample'},
    {'label': 'Session TD-50 target', 'belief_wins': 729, 'total': 1000, 'group': 'Session'},
    {'label': 'Session MC-target', 'belief_wins': 560, 'total': 1000, 'group': 'Session'},
]


# ===========================================================================
if __name__ == "__main__":
    HEATMAP_OUTPUT = "images/DOGheatmap_nobelief.png"   # e.g. "utils/heatmap.png", or None to only show
    # BAR_OUTPUT = "images/DOG_belief_bar_baselines.png"
    # BELIEF_OUTPUT = "images/DOG_belief_duel.png"
    plot_heatmap(HEATMAP_AGENT_NAMES, HEATMAP_MATRIX, output_file=HEATMAP_OUTPUT)
    # plot_bar_baselines(BAR_AGENTS, BAR_BASELINES, BAR_MATRIX, output_file=BAR_OUTPUT)
    # plot_belief_duel(BELIEF_DUEL_MATCHUPS, output_file=BELIEF_OUTPUT)
    #plot_lollipop_baselines(BAR_AGENTS, BAR_BASELINES, BAR_MATRIX, output_file=BAR_OUTPUT)
