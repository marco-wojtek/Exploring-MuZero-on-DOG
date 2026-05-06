"""
chance_accuracy.py

Evaluates how well the chance head predicts the dice distribution
for the next state, comparing predicted vs. ground-truth distributions.

Two distribution types exist:
  NORMAL:  uniform [1/6]*6
  LOCKED:  OUT_ON_ONE_AND_SIX [76/216, 16/216, ..., 76/216]

Metrics reported per scenario group:
  - Mean KL divergence (predicted || ground truth)
  - Type classification accuracy (did the model predict the right mode?)
  - Per-die-face MAE
"""
import sys, os
import jax
import jax.numpy as jnp
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from MADN.classic_madn import (
    env_reset, encode_board, valid_action, set_pins_on_board,
    env_step, dice_probabilities, is_soft_locked,
    NORMAL_DICE_DISTRIBUTION, OUT_ON_ONE_AND_SIX_DICE_DISTRIBUTION,
)
from MuZero_Classic_MADN.muzero_classic_madn import (
    repr_net, pred_net, dynamics_net,
    load_params_from_file, init_muzero_params,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
filename = "TEAMstochastic_muzero_madn_params_lr0.005_g1500_it100_seed30"
PARAM_FILE = f"MuZero_Classic_MADN/models/params/{filename}.pkl"
# PARAM_FILE = None  # ← untrained params

RULES = {
    'enable_teams': True,
    'enable_initial_free_pin': True,
    'enable_circular_board': False,
    'enable_friendly_fire': False,
    'enable_start_blocking': False,
    'enable_jump_in_goal_area': True,
    'enable_start_on_1': True,
    'enable_bonus_turn_on_6': True,
    'must_traverse_start': False,
    'enable_dice_rethrow': True,
}

# ── Load params ───────────────────────────────────────────────────────────────
input_shape = (11, 56)
if PARAM_FILE:
    params = load_params_from_file(PARAM_FILE)
    print(f"Params loaded: {PARAM_FILE}")
else:
    params = init_muzero_params(jax.random.PRNGKey(0), input_shape)
    print("Using fresh (untrained) params")

# ── Base environment ──────────────────────────────────────────────────────────
env_base = env_reset(
    0, num_players=4,
    layout=jnp.array([True, True, True, True]),
    distance=10, starting_player=0, seed=1,
    **RULES,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
NORMAL_DIST  = np.array(NORMAL_DICE_DISTRIBUTION)
LOCKED_DIST  = np.array(OUT_ON_ONE_AND_SIX_DICE_DISTRIBUTION)

def softmax(x):
    x = np.array(x, dtype=float)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()

def kl_divergence(p, q):
    """KL(p || q)"""
    p = np.array(p, dtype=float) + 1e-9
    q = np.array(q, dtype=float) + 1e-9
    return float(np.sum(p * np.log(p / q)))

def dist_type(dist):
    """Classify a distribution as NORMAL or LOCKED based on KL to each template."""
    kl_n = kl_divergence(dist, NORMAL_DIST)
    kl_l = kl_divergence(dist, LOCKED_DIST)
    return "LOCKED" if kl_l < kl_n else "NORMAL"

def predict_chance(latent, action_idx):
    action = jnp.array([action_idx])
    _, _, chance_logits, _ = dynamics_net.apply(
        params['dynamics'], latent, action, method=dynamics_net.action_dynamics
    )
    return softmax(np.array(chance_logits).flatten())

def build_env(pins, current_player, dice_val):
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    return env.replace(die=jnp.int8(dice_val))

# ── Test scenarios ────────────────────────────────────────────────────────────
# Each entry: (label, pins, current_player, dice_val, expected_next_dist_type)
#
# Board layout: distance=10, num_players=4, board_size=56
#   Main field: 0–39
#   P0 goal: [40-43]  P1 goal: [44-47]  P2 goal: [48-51]  P3 goal: [52-55]

scenarios = [
    # ── NORMAL next player ────────────────────────────────────────────────────
    ("Normal mid-game, dice=3 → P1 next (NORMAL)", "NORMAL",
     jnp.array([[10, 20, 30, -1], [15, 25, -1, -1], [ 5, 35, -1, -1], [ 8, 18, -1, -1]], dtype=jnp.int32),
     0, 3),
    ("Normal mid-game, dice=2 → P1 next (NORMAL)", "NORMAL",
     jnp.array([[10, 20, 30, -1], [15, 25, -1, -1], [ 5, 35, -1, -1], [ 8, 18, -1, -1]], dtype=jnp.int32),
     0, 2),
    ("Normal mid-game, dice=4 → P1 next (NORMAL)", "NORMAL",
     jnp.array([[ 3, 17, 29, -1], [11, 22, -1, -1], [14, 36, -1, -1], [ 6, 19, -1, -1]], dtype=jnp.int32),
     0, 4),
    ("Normal mid-game, dice=1 → P1 next (NORMAL)", "NORMAL",
     jnp.array([[ 7, 16, 27, -1], [13, 24, -1, -1], [2, 33, -1, -1], [ 9, 21, -1, -1]], dtype=jnp.int32),
     0, 1),

    # ── Bonus turn (dice=6, same team next) → next player is P0 again (NORMAL)
    ("Normal, dice=6 → P0 bonus turn (NORMAL)", "NORMAL",
     jnp.array([[10, 20, 30, -1], [15, 25, -1, -1], [ 5, 35, -1, -1], [ 8, 18, -1, -1]], dtype=jnp.int32),
     0, 6),
    ("Near-win P0, dice=6 → P0 bonus turn (NORMAL)", "NORMAL",
     jnp.array([[34, 41, 42, 43], [ 5, 15,  7, 12], [48, 49, 50, 51], [25, 28, 33, 30]], dtype=jnp.int32),
     0, 6),

    # ── LOCKED next player ────────────────────────────────────────────────────
    # P1 soft-locked: pins at last 2 goal positions (indices 46,47 = goal[2,3])
    ("P1 soft-locked, dice=3 → P1 next (LOCKED)", "LOCKED",
     jnp.array([[10, 20, 30, -1], [-1, -1, 46, 47], [ 5, 35, -1, -1], [ 8, 18, -1, -1]], dtype=jnp.int32),
     0, 3),
    ("P1 soft-locked, dice=2 → P1 next (LOCKED)", "LOCKED",
     jnp.array([[ 3, 17, 29, -1], [-1, -1, 46, 47], [14, 36, -1, -1], [ 6, 19, -1, -1]], dtype=jnp.int32),
     0, 2),
    ("P1 soft-locked, dice=1 → P1 next (LOCKED)", "LOCKED",
     jnp.array([[ 7, 16, 27, -1], [-1, -1, 46, 47], [2, 33, -1, -1], [ 9, 21, -1, -1]], dtype=jnp.int32),
     0, 1),
    ("P1 soft-locked 3-pins, dice=4 → P1 next (LOCKED)", "LOCKED",
     jnp.array([[10, 20, 30, -1], [-1, 45, 46, 47], [ 5, 35, -1, -1], [ 8, 18, -1, -1]], dtype=jnp.int32),
     0, 4),
    ("P1 fully locked, dice=5 → P1 next (LOCKED)", "LOCKED",
     jnp.array([[10, 20, 30, -1], [44, 45, 46, 47], [ 5, 35, -1, -1], [ 8, 18, -1, -1]], dtype=jnp.int32),
     0, 5),

    # ── LOCKED dice=6 → P0 bonus (NORMAL, not locked)
    ("P1 soft-locked, dice=6 → P0 bonus turn (NORMAL)", "NORMAL",
     jnp.array([[10, 20, 30, -1], [-1, -1, 46, 47], [ 5, 35, -1, -1], [ 8, 18, -1, -1]], dtype=jnp.int32),
     0, 6),

    # ── P3 soft-locked, P0 is P2 (dice≠6 → P3 next) ─────────────────────────
    ("P3 soft-locked, dice=2 from P2 → P3 next (LOCKED)", "LOCKED",
     jnp.array([[10, 20, 30, -1], [15, 25, -1, -1], [ 5, 35, -1, -1], [-1, -1, 54, 55]], dtype=jnp.int32),
     2, 2),
    ("P3 soft-locked, dice=5 from P2 → P3 next (LOCKED)", "LOCKED",
     jnp.array([[ 3, 17, 29, -1], [11, 22, -1, -1], [14, 36, -1, -1], [-1, -1, 54, 55]], dtype=jnp.int32),
     2, 5),
]

# ── Run evaluation ────────────────────────────────────────────────────────────
DIE_FACES = ["1", "2", "3", "4", "5", "6"]
COL = 8

def bar(value, width=20):
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)

print(f"\n{'=' * 80}")
print(" CHANCE HEAD ACCURACY – Dice Distribution Prediction")
print(f"{'=' * 80}")
print(f" Model: {PARAM_FILE or 'untrained'}")
print(f" Scenarios: {len(scenarios)}  |  Distributions: NORMAL vs. LOCKED\n")

all_kl_normal, all_kl_locked = [], []
all_type_correct, all_type_total = 0, 0
all_mae_normal, all_mae_locked = [], []

print(f"{'Scenario':<52} {'Next':>7} {'Type-GT':>8} {'Type-Pred':>10} {'KL↓':>7}  {'Match':>5}")
print("-" * 95)

detail_rows = []

for label, expected_type, pins, cp, dice_val in scenarios:
    env = build_env(pins, cp, dice_val)
    valid_mask = np.array(valid_action(env).flatten())
    obs = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    for a in range(4):
        if not valid_mask[a]:
            continue

        next_env, _, done = env_step(env, jnp.int32(a))
        true_dist = np.array(dice_probabilities(next_env))
        pred_dist = predict_chance(latent, a)

        gt_type  = dist_type(true_dist)
        pd_type  = dist_type(pred_dist)
        kl       = kl_divergence(true_dist, pred_dist)
        mae      = float(np.mean(np.abs(pred_dist - true_dist)))
        type_ok  = gt_type == pd_type

        all_type_correct += int(type_ok)
        all_type_total   += 1

        if gt_type == "NORMAL":
            all_kl_normal.append(kl)
            all_mae_normal.append(mae)
        else:
            all_kl_locked.append(kl)
            all_mae_locked.append(mae)

        detail_rows.append((label, f"Pin {a}", next_env.current_player, gt_type,
                            pd_type, kl, mae, type_ok, true_dist, pred_dist))

        next_player = int(next_env.current_player)
        short_label = label[:50]
        print(f"{short_label:<52} {'P'+str(next_player):>7} {gt_type:>8} {pd_type:>10} "
              f"{kl:>7.4f}  {'✓' if type_ok else '✗':>5}")
    # blank line between scenario groups
    print()

# ── Per-distribution-type summary ────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(" SUMMARY")
print(f"{'=' * 80}\n")

type_acc = 100 * all_type_correct / max(all_type_total, 1)
print(f"  Type Classification Accuracy : {all_type_correct}/{all_type_total}  ({type_acc:.1f}%)")
print(f"  (NORMAL vs. LOCKED – argmin KL to template)\n")

def dist_summary(name, kl_list, mae_list):
    if not kl_list:
        print(f"  {name:8s}: no samples")
        return
    print(f"  {name:8s}  samples={len(kl_list):3d}"
          f"  KL mean={np.mean(kl_list):.4f}  KL max={np.max(kl_list):.4f}"
          f"  MAE mean={np.mean(mae_list):.4f}")

dist_summary("NORMAL",  all_kl_normal,  all_mae_normal)
dist_summary("LOCKED",  all_kl_locked,  all_mae_locked)

overall_kl = all_kl_normal + all_kl_locked
print(f"\n  Overall   samples={len(overall_kl):3d}"
      f"  KL mean={np.mean(overall_kl):.4f}  KL max={np.max(overall_kl):.4f}\n")

print("  KL guide: 0.00 = perfect  |  < 0.05 = excellent  |  < 0.20 = good  |  > 0.50 = poor\n")

# ── Per-face comparison for a NORMAL and a LOCKED example ────────────────────
print(f"{'=' * 80}")
print(" DISTRIBUTION DETAIL  (first NORMAL and first LOCKED sample)")
print(f"{'=' * 80}\n")

shown = {"NORMAL": False, "LOCKED": False}
for label, pin_lbl, next_p, gt_type, pd_type, kl, mae, ok, true_dist, pred_dist in detail_rows:
    if shown[gt_type]:
        continue
    shown[gt_type] = True

    print(f"  [{gt_type}]  {label}  ({pin_lbl} → P{next_p})")
    print(f"  {'Face':>5}  {'Ground Truth':>13}  {'Predicted':>13}  {'|Δ|':>7}  Visual (GT ░ / Pred █)")
    print(f"  {'-' * 72}")
    for i, face in enumerate(DIE_FACES):
        gt_v  = true_dist[i]
        pr_v  = pred_dist[i]
        delta = abs(gt_v - pr_v)
        gt_bar   = "░" * int(round(float(gt_v) * 30))
        pred_bar = "█" * int(round(float(pr_v) * 30))
        print(f"  {face:>5}  {gt_v:>13.4f}  {pr_v:>13.4f}  {delta:>7.4f}  {pred_bar:<30} {gt_bar}")
    print(f"\n  KL={kl:.4f}  MAE={mae:.4f}  Type: GT={gt_type} Pred={pd_type} {'✓' if ok else '✗'}\n")

    if all(shown.values()):
        break

# ── Plots ─────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Collect all predicted distributions grouped by ground-truth type
preds_by_type  = {"NORMAL": [], "LOCKED": []}
gt_by_type     = {"NORMAL": [], "LOCKED": []}
kl_by_type     = {"NORMAL": [], "LOCKED": []}
labels_by_type = {"NORMAL": [], "LOCKED": []}

for label, pin_lbl, next_p, gt_type, pd_type, kl, mae, ok, true_dist, pred_dist in detail_rows:
    preds_by_type[gt_type].append(pred_dist)
    gt_by_type[gt_type].append(true_dist)
    kl_by_type[gt_type].append(kl)
    labels_by_type[gt_type].append(f"{label[:30]}…\n({pin_lbl})")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"Chance Head – Predicted vs. Ground-Truth Dice Distribution\n",
    fontsize=11, y=1.01,
)

x = np.arange(6)
face_labels = ["1", "2", "3", "4", "5", "6"]
bar_width = 0.25

COLORS = {
    "gt":   "#4C72B0",
    "mean": "#DD8452",
    "ind":  "#91bfdb",
}

for ax, dist_type_name, title in [
    (axes[0], "NORMAL", "NORMAL distribution  (uniform [1/6]×6)"),
    (axes[1], "LOCKED", "LOCKED distribution  (out-on-1-and-6)"),
]:
    preds  = np.array(preds_by_type[dist_type_name])   # (N, 6)
    gt_all = np.array(gt_by_type[dist_type_name])       # (N, 6)
    kls    = np.array(kl_by_type[dist_type_name])

    gt_template = gt_all[0] if len(gt_all) > 0 else np.ones(6) / 6
    pred_mean   = preds.mean(axis=0)
    pred_std    = preds.std(axis=0)

    # ── individual predictions (thin bars in background)
    for pred in preds:
        ax.bar(x + bar_width / 2, pred, width=bar_width * 0.9,
               color=COLORS["ind"], alpha=0.25, zorder=1)

    # ── ground-truth template
    ax.bar(x - bar_width / 2, gt_template, width=bar_width,
           color=COLORS["gt"], alpha=0.85, label="Ground truth", zorder=2)

    # ── mean prediction ± std
    ax.bar(x + bar_width / 2, pred_mean, width=bar_width,
           color=COLORS["mean"], alpha=0.85, label="Pred mean", zorder=3)
    ax.errorbar(x + bar_width / 2, pred_mean, yerr=pred_std,
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(face_labels)
    ax.set_xlabel("Die face")
    ax.set_ylabel("Probability")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(gt_template.max(), pred_mean.max() + pred_std.max()) * 1.25)

    n = len(preds)
    kl_m = kls.mean() if len(kls) else float("nan")
    kl_mx = kls.max()  if len(kls) else float("nan")
    stats_text = (f"n={n} predictions\n"
                  f"KL mean={kl_m:.4f}\n"
                  f"KL max={kl_mx:.4f}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ind_patch = mpatches.Patch(color=COLORS["ind"], alpha=0.5, label="Individual preds")
    ax.legend(handles=[
        mpatches.Patch(color=COLORS["gt"],   alpha=0.85, label="Ground truth"),
        mpatches.Patch(color=COLORS["mean"], alpha=0.85, label="Pred mean ± std"),
        ind_patch,
    ], fontsize=8, loc="upper left")

plt.tight_layout()
plot_path = f"MuZero_Classic_MADN/evaluation/{os.path.basename(PARAM_FILE or 'untrained')}_chance_accuracy.png"
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {plot_path}")
plt.show()
