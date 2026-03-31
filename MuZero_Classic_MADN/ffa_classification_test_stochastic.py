"""
classification_test_ffa_stochastic.py

FFA Multi-Value Head & Depth-Delta Diagnose — Stochastic MADN
==============================================================
Testet die FFA-Änderungen für stochastisches MuZero:
  1. Multi-Value Head:   pred_net → values (B, 4) — ein Value pro Spieler-Perspektive
  2. Depth-Delta Head:   chance_dynamics lernt Spielerwechsel (1) vs. 6er-Bonus (0)
                         Kommt aus chance_dynamics(afterstate, dice_outcome) — nur würfelbasiert!
  3. Binary Discount:    action_dynamics → 2 Klassen {0=Terminal, 1=Non-Terminal}
  4. Reward Head:        action_dynamics → 3 Klassen {-1, 0, +1}
  5. Chance Head:         action_dynamics → Würfelverteilung vorhersagen

Architektur-Ablauf pro MCTS-Schritt:
  Decision Node:
    action_dynamics(latent, action) → (afterstate, reward_logits[3], chance_logits[6], discount_logits[2])
  Chance Node:
    chance_dynamics(afterstate, dice) → (next_state, depth_delta_logit[1])
    depth = (depth + sigmoid(depth_delta_logit)) % 4
    root_idx = (4 - round(depth)) % 4
    value = pred_net(next_state)[:, root_idx]

Board-Layout (distance=10, num_players=4):
  Hauptfeld: 0–39
  P0-Ziel: [40,41,42,43]   P1-Ziel: [44,45,46,47]
  P2-Ziel: [48,49,50,51]   P3-Ziel: [52,53,54,55]
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
    load_params_from_file, init_muzero_params, run_stochastic_muzero_mcts,
)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
filename = "stochastic_muzero_madn_params_lr0.005_g1500_it50_seed22"
PARAM_FILE = f"MuZero_Classic_MADN/models/params/{filename}.pkl"
# PARAM_FILE = None  # ← für untrainierte Params

output_file = f"MuZero_Classic_MADN/evaluation/{filename}_ffa_stochastic_tests.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

# ═══════════════════════════════════════════════════════════════
#  SUPPORT VEKTOREN
# ═══════════════════════════════════════════════════════════════
SUPPORT_REWARD   = jnp.array([-1.0, 0.0, 1.0])
SUPPORT_DISCOUNT = jnp.array([0.0, 1.0])   # Binary: {Terminal, Non-Terminal}


def reward_logits_to_scalar(logits):
    probs = jax.nn.softmax(logits, axis=-1)
    return float(jnp.sum(probs * SUPPORT_REWARD, axis=-1).squeeze())


def reward_logits_to_probs(logits):
    return np.array(jax.nn.softmax(logits, axis=-1).squeeze())


def discount_logits_to_scalar(logits):
    """Binary: 0=Terminal, 1=Non-Terminal"""
    probs = jax.nn.softmax(logits, axis=-1)
    return float(jnp.sum(probs * SUPPORT_DISCOUNT, axis=-1).squeeze())


def discount_logits_to_probs(logits):
    return np.array(jax.nn.softmax(logits, axis=-1).squeeze())


def depth_delta_to_scalar(logit):
    """sigmoid → ≈0 für dice=6 (gleicher Spieler), ≈1 für Spielerwechsel"""
    return float(jax.nn.sigmoid(logit).squeeze())


def kl_divergence(p, q):
    p = np.array(p, dtype=float) + 1e-9
    q = np.array(q, dtype=float) + 1e-9
    return float(np.sum(p * np.log(p / q)))


def print_header(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# ═══════════════════════════════════════════════════════════════
#  PARAMS LADEN
# ═══════════════════════════════════════════════════════════════
input_shape = (11, 56)
if PARAM_FILE:
    params = load_params_from_file(PARAM_FILE)
    print(f"Params geladen: {PARAM_FILE}")
else:
    params = init_muzero_params(jax.random.PRNGKey(0), input_shape)
    print("Frische (untrainierte) Params initialisiert")

print("Architektur: FFA Multi-Value Head (4 Werte) + Binary Discount + Depth-Delta (aus chance_dynamics)\n")

# ═══════════════════════════════════════════════════════════════
#  BASIS-ENVIRONMENT  —  FFA (enable_teams=False)
# ═══════════════════════════════════════════════════════════════
RULES = {
    'enable_teams': False,           # FFA!
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

env_base = env_reset(
    0, num_players=4,
    layout=jnp.array([True, True, True, True]),
    distance=10, starting_player=0, seed=1,
    enable_teams=RULES['enable_teams'],
    enable_initial_free_pin=RULES['enable_initial_free_pin'],
    enable_circular_board=RULES['enable_circular_board'],
    enable_friendly_fire=RULES['enable_friendly_fire'],
    enable_start_blocking=RULES['enable_start_blocking'],
    enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
    enable_start_on_1=RULES['enable_start_on_1'],
    enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
    must_traverse_start=RULES['must_traverse_start'],
    enable_dice_rethrow=RULES['enable_dice_rethrow'],
)

# ── Test-States ────────────────────────────────────────────────
# P0: 1 Zug vor Sieg mit dice=5 (Pin 0 → 35+5=40 = goal[0])
pins_pre_win = jnp.array([
    [35, 41, 42, 43],
    [ 5, 15,  7, 12],
    [48, 49,  3, -1],   # FFA: P2 NICHT im Ziel
    [25, 28, 33, 30],
], dtype=jnp.int32)

# P0: 1 Zug vor Sieg mit dice=6 (Pin 0 → 34+6=40 = goal[0])
pins_pre_win_6 = jnp.array([
    [34, 41, 42, 43],
    [ 5, 15,  7, 12],
    [48, 49,  3, -1],
    [25, 28, 33, 30],
], dtype=jnp.int32)

pins_normal = jnp.array([
    [10, 20, 30, -1],
    [15, 25, -1, -1],
    [ 5, 35, -1, -1],
    [ 8, 18, -1, -1],
], dtype=jnp.int32)

# P1 soft-locked nach P0s nicht-6 Zug
pins_p1_softlocked = jnp.array([
    [10, 20, 30, -1],
    [-1, -1, 46, 47],
    [ 5, 35, -1, -1],
    [ 8, 18, -1, -1],
], dtype=jnp.int32)

test_scenarios = [
    ("PRE-WIN  (dice=5, Pin0 gewinnt)", pins_pre_win,   0, 5,  0),
    ("PRE-WIN6 (dice=6, Pin0 gewinnt)", pins_pre_win_6, 0, 6,  0),
    ("NORMAL   (dice=3)",               pins_normal,    0, 3,  None),
    ("NORMAL   (dice=6, Bonuszug)",     pins_normal,    0, 6,  None),
]


# ════════════════════════════════════════════════════════════════
#  TEST 1: MULTI-VALUE HEAD — pred_net → values (B, 4)
#  Erwartet: PRE-WIN →  values[:,0] ≈ +1  (P0 gewinnt)
#                       values[:,1..3] ≈ -1  (P1-P3 verlieren)
# ════════════════════════════════════════════════════════════════
print_header("TEST 1: MULTI-VALUE HEAD  —  pred_net → values (B, 4)")
print("  Erwartet für PRE-WIN:")
print("    values[0] ≈ +1.0  (Root-Spieler P0 gewinnt)")
print("    values[1..3] ≈ -1.0  (Gegner verlieren — FFA, kein Team-Bonus)")
print()

for name, pins, current_player, dice_val, _ in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    env   = env.replace(die=jnp.int8(dice_val))
    obs   = encode_board(env)[None, ...]

    latent = repr_net.apply(params['representation'], obs)
    _, values = pred_net.apply(params['prediction'], latent)
    v = values[0]   # (4,)

    print(f"  {name}")
    print(f"    V[0]={float(v[0]):+.4f}  V[1]={float(v[1]):+.4f}  "
          f"V[2]={float(v[2]):+.4f}  V[3]={float(v[3]):+.4f}")
    v0 = float(v[0])
    interpret = ("ROOT gewinnt ✓" if v0 > 0.3 else
                 "ROOT verliert ✓" if v0 < -0.3 else "neutral")
    print(f"    Root-Value V[0]: {interpret}")
    print()


# ════════════════════════════════════════════════════════════════
#  TEST 2: DEPTH-DELTA HEAD — korrekte Kausalität
#
#  Ablauf bei t=k:
#    obs_k  = encode_board(env_after_dice)  ← enthält die[k]
#    latent_k     = repr_net(obs_k)
#    afterstate_k = action_dynamics(latent_k, action_k)  ← enkodiert die[k]
#    depth_delta  = Dense(afterstate_normed)              ← liest die[k]!
#
#  Erwartet:
#    die[k]=6  → Bonus-Zug → gleicher Spieler → depth_delta ≈ 0.0
#    die[k]≠6  → Spielerwechsel              → depth_delta ≈ 1.0
#
#  Die[k+1] (chance_outcome) ist IRRELEVANT — depth_delta darf sich
#  bei konstantem die[k] NICHT ändern wenn die[k+1] variiert.
# ════════════════════════════════════════════════════════════════
dummy_action = jnp.array([0])  # Für Tests die einen Afterstate brauchen

print_header("TEST 2: DEPTH-DELTA HEAD  —  Kausalität: die[k] → afterstate → depth_delta")
print("  Ablauf: Spieler würfelt die[k], führt Aktion aus → afterstate enkodiert die[k]")
print("  depth_delta = Dense(afterstate_normed) liest die[k] aus afterstate")
print("  die[k+1] (chance_outcome) ist irrelevant für depth_delta!")
print()
print("  Erwartet:  die[k]=6  → depth_delta ≈ 0.0  (Bonus-Zug, gleicher Spieler)")
print("             die[k]≠6  → depth_delta ≈ 1.0  (Spielerwechsel)")
print()

# Teil A: depth_delta abhängig von die[k]
# Baue für jedes die[k] einen Afterstate und fixiere die[k+1]=3 (beliebig)
FIXED_NEXT_DICE = jnp.array([2])   # die[k+1]=3, Index 2 — fix für alle Tests

print("  --- Teil A: depth_delta in Abhängigkeit von die[k] (die[k+1]=3 fix) ---")
print(f"  {'die[k]':>8} {'dep_del':>9} {'Expected':>9} {'Logit':>8} {'Match':>6}")
print(f"  {'-'*48}")

correct_dd = 0
total_dd   = 0

for dice_k in range(1, 7):
    # obs_k enthält die[k]
    board_k = set_pins_on_board(env_base.board, pins_normal)
    env_k   = env_base.replace(board=board_k, pins=pins_normal, current_player=0)
    env_k   = env_k.replace(die=jnp.int8(dice_k))
    obs_k   = encode_board(env_k)[None, ...]
    latent_k = repr_net.apply(params['representation'], obs_k)

    # afterstate enkodiert die[k]
    afterstate_k, _, _, _ = dynamics_net.apply(
        params['dynamics'], latent_k, dummy_action, method=dynamics_net.action_dynamics
    )

    # chance_dynamics mit fixem die[k+1]=3
    _, depth_delta_logit = dynamics_net.apply(
        params['dynamics'], afterstate_k, FIXED_NEXT_DICE, method=dynamics_net.chance_dynamics
    )
    dd_sig   = depth_delta_to_scalar(depth_delta_logit)
    dd_raw   = float(depth_delta_logit.squeeze())
    expected = 0.0 if dice_k == 6 else 1.0
    match    = abs(dd_sig - expected) < 0.5
    correct_dd += int(match)
    total_dd   += 1
    marker = "  ← 6ER BONUS" if dice_k == 6 else ""
    print(f"  die[k]={dice_k:>1}  {dd_sig:>9.4f} {expected:>9.1f} "
          f"{dd_raw:>8.3f} {'✓' if match else '✗':>6}{marker}")

acc_pct = 100 * correct_dd / max(total_dd, 1)
print(f"\n  Depth-Delta Accuracy: {correct_dd}/{total_dd} ({acc_pct:.0f}%)")
if acc_pct < 50:
    print("  ⚠ Modell untrainiert bezüglich Depth-Delta")

# Teil B: Invarianz gegenüber die[k+1]
# Für festes die[k]=3 und die[k]=6: depth_delta darf sich bei wechselndem die[k+1] NICHT ändern
print()
print("  --- Teil B: Invarianz gegenüber die[k+1] (depth_delta darf sich nicht ändern) ---")
print(f"  {'die[k]':>8} {'die[k+1]':>9} {'dep_del':>9} {'Std_über_k+1':>14} {'Status'}")
print(f"  {'-'*56}")

for dice_k in [3, 6]:   # repräsentativ: einmal Wechsel, einmal Bonus
    board_k = set_pins_on_board(env_base.board, pins_normal)
    env_k   = env_base.replace(board=board_k, pins=pins_normal, current_player=0)
    env_k   = env_k.replace(die=jnp.int8(dice_k))
    obs_k   = encode_board(env_k)[None, ...]
    latent_k = repr_net.apply(params['representation'], obs_k)
    afterstate_k, _, _, _ = dynamics_net.apply(
        params['dynamics'], latent_k, dummy_action, method=dynamics_net.action_dynamics
    )

    vals_over_next_dice = []
    for dice_k1 in range(1, 7):
        dice_k1_idx = jnp.array([dice_k1 - 1])
        _, ddl = dynamics_net.apply(
            params['dynamics'], afterstate_k, dice_k1_idx, method=dynamics_net.chance_dynamics
        )
        vals_over_next_dice.append(depth_delta_to_scalar(ddl))

    std_val = float(np.std(vals_over_next_dice))
    for i, v in enumerate(vals_over_next_dice):
        dice_k1 = i + 1
        if i == 0:
            warn = "  ⚠ ABHÄNGIG von die[k+1]!" if std_val > 0.05 else "  ✓ invariant"
            print(f"  die[k]={dice_k:>1}   die[k+1]={dice_k1:>1}  {v:>9.4f} {std_val:>14.4f}{warn}")
        else:
            print(f"  {'':>8}   die[k+1]={dice_k1:>1}  {v:>9.4f}")
    print()


# ════════════════════════════════════════════════════════════════
#  TEST 3: DEPTH-DELTA KONSISTENZ über verschiedene States und Actions
#  depth_delta hängt von die[k] (im Afterstate) ab.
#  Für gleiche die[k] aber verschiedene Board-States / Actions:
#  → depth_delta sollte konsistent sein (Std klein)
# ════════════════════════════════════════════════════════════════
print_header("TEST 3: DEPTH-DELTA KONSISTENZ  —  gleiches die[k], verschiedene States/Actions")
print("  Für festes die[k] sollte depth_delta über Boards und Aktionen konsistent sein.")
print("  Std > 0.1 deutet darauf hin dass das Netz noch nicht gut die[k] aus afterstate liest.")
print()

all_states = [
    ("PRE-WIN",  pins_pre_win),
    ("PRE-WIN6", pins_pre_win_6),
    ("NORMAL",   pins_normal),
]
all_actions_idx = [0, 1, 2, 3]

print(f"  {'die[k]':>8} {'Expected':>9} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'Status'}")
print(f"  {'-'*60}")

for dice_k in range(1, 7):
    vals = []
    expected = 0.0 if dice_k == 6 else 1.0
    for state_name, pins in all_states:
        board_k = set_pins_on_board(env_base.board, pins)
        env_k   = env_base.replace(board=board_k, pins=pins, current_player=0)
        env_k   = env_k.replace(die=jnp.int8(dice_k))
        obs_k   = encode_board(env_k)[None, ...]
        latent_k = repr_net.apply(params['representation'], obs_k)
        for a in all_actions_idx:
            action_k = jnp.array([a])
            afterstate_k, _, _, _ = dynamics_net.apply(
                params['dynamics'], latent_k, action_k, method=dynamics_net.action_dynamics
            )
            # Fixe die[k+1]=3 (irrelevant für depth_delta)
            _, ddl = dynamics_net.apply(
                params['dynamics'], afterstate_k, FIXED_NEXT_DICE, method=dynamics_net.chance_dynamics
            )
            vals.append(depth_delta_to_scalar(ddl))

    vals = np.array(vals)
    std  = float(np.std(vals))
    mean = float(np.mean(vals))
    ok   = abs(mean - expected) < 0.5
    warn = "  ⚠ inkonsistent" if std > 0.1 else ""
    status = "✓ OK" if ok else "✗ falsch"
    print(f"  die[k]={dice_k:>1}  {expected:>9.1f} {mean:>7.4f} {std:>7.4f} "
          f"{float(vals.min()):>7.4f} {float(vals.max()):>7.4f}  {status}{warn}")


# ════════════════════════════════════════════════════════════════
#  TEST 4: BINARY DISCOUNT — action_dynamics → {0=Terminal, 1=Non-Terminal}
#  Erwartet: Winning Action mit Gewinn-Würfel → P(Terminal)≈1 → disc_val ≈ 0.0
#            Normale Actions                  → P(Non-Terminal)≈1 → disc_val ≈ 1.0
# ════════════════════════════════════════════════════════════════
print_header("TEST 4: BINARY DISCOUNT  —  action_dynamics → {0=Terminal, 1=Non-Terminal}")
print("  Erwartet: Winning Action (Pin 0, dice=5) → disc ≈ 0.0  (Terminal)")
print("            Normale Actions                → disc ≈ 1.0  (Non-Terminal)")
print()

disc_scenarios = [
    ("PRE-WIN  (dice=5, Pin0 → TERMINAL)", pins_pre_win,   0, 5),
    ("PRE-WIN6 (dice=6, Pin0 → TERMINAL)", pins_pre_win_6, 0, 6),
    ("NORMAL   (dice=6, Bonuszug)",         pins_normal,    0, 6),
    ("NORMAL   (dice=3)",                   pins_normal,    0, 3),
]

for name, pins, current_player, dice_val in disc_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    env   = env.replace(die=jnp.int8(dice_val))
    valid_mask = valid_action(env).flatten()
    obs   = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    print(f"  --- {name} ---")
    print(f"  {'Pin':>4} {'Valid':>5} {'disc_val':>9} {'P(Term)':>8} {'P(NonT)':>8}  {'GT':>6}  {'Match':>6}")
    print(f"  {'-'*58}")

    for a in range(4):
        if not valid_mask[a]:
            continue
        next_env, gt_r, done = env_step(env, jnp.int32(a))
        gt_disc = 0.0 if bool(done) else 1.0

        action = jnp.array([a])
        _, _, _, disc_logits = dynamics_net.apply(
            params['dynamics'], latent, action, method=dynamics_net.action_dynamics
        )
        disc_val  = discount_logits_to_scalar(disc_logits)
        disc_probs = discount_logits_to_probs(disc_logits)
        match = abs(disc_val - gt_disc) < 0.5
        print(f"  Pin{a:>1} {'O':>5} {disc_val:>9.4f} {disc_probs[0]:>8.4f} {disc_probs[1]:>8.4f}"
              f"  {gt_disc:>6.1f}  {'✓' if match else '✗':>6}")
    print()


# ════════════════════════════════════════════════════════════════
#  TEST 5: REWARD HEAD — 3 Klassen {-1, 0, +1}
#  Erwartet: Winning Action → P(+1) hoch, E[R] ≈ +1.0
# ════════════════════════════════════════════════════════════════
print_header("TEST 5: REWARD HEAD  —  3 Klassen {-1, 0, +1}")
print("  Erwartet: PRE-WIN Pin0 → E[R] ≈ +1.0,  normale Züge → E[R] ≈ 0.0")
print()

for name, pins, current_player, dice_val, winning_pin in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    env   = env.replace(die=jnp.int8(dice_val))
    valid_mask = valid_action(env).flatten()
    obs   = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    print(f"  --- {name} ---")
    print(f"  {'Pin':>4} {'Valid':>5} {'E[R]':>7} {'P(-1)':>7} {'P(0)':>7} {'P(+1)':>7}  {'GT':>6}  {'Match':>6}")
    print(f"  {'-'*62}")

    for a in range(4):
        if not valid_mask[a]:
            continue
        next_env, gt_r_raw, done = env_step(env, jnp.int32(a))
        if bool(done) and float(gt_r_raw) > 0:
            gt_reward = +1.0
        elif bool(done) and float(gt_r_raw) < 0:
            gt_reward = -1.0
        else:
            gt_reward = 0.0

        action = jnp.array([a])
        _, rew_logits, _, _ = dynamics_net.apply(
            params['dynamics'], latent, action, method=dynamics_net.action_dynamics
        )
        r_val   = reward_logits_to_scalar(rew_logits)
        r_probs = reward_logits_to_probs(rew_logits)
        match   = abs(r_val - gt_reward) < 0.5
        marker  = "  ← WIN" if a == winning_pin else ""
        print(f"  Pin{a:>1} {'O':>5} {r_val:>+7.4f} "
              f"{r_probs[0]:>7.4f} {r_probs[1]:>7.4f} {r_probs[2]:>7.4f}"
              f"  {gt_reward:>+6.1f}  {'✓' if match else '✗':>6}{marker}")
    print()


# ════════════════════════════════════════════════════════════════
#  TEST 6: CHANCE HEAD — Würfelverteilung vorhersagen
# ════════════════════════════════════════════════════════════════
print_header("TEST 6: CHANCE HEAD  —  Vorhersage der Würfelverteilung")
print("  Erwartet: NORMAL+dice≠6   → P1 nächster (uniform)    KL ≈ 0")
print("            P1-LOCKED+dice≠6 → P1 nächster (soft-lock)  KL ≈ 0")
print("            BELIEBIG+dice=6  → P0 Bonuszug (uniform)    KL ≈ 0")
norm_d = np.array(NORMAL_DICE_DISTRIBUTION)
lock_d = np.array(OUT_ON_ONE_AND_SIX_DICE_DISTRIBUTION)
print(f"\n  NORMAL:       [{', '.join(f'{v:.4f}' for v in norm_d)}]")
print(f"  OUT_1_AND_6:  [{', '.join(f'{v:.4f}' for v in lock_d)}]")
print()

chance_scenarios = [
    ("NORMAL    + dice=3  (P1, nicht locked)", pins_normal,       0, 3),
    ("NORMAL    + dice=6  (P0 Bonuszug)",      pins_normal,       0, 6),
    ("P1-LOCKED + dice=3  (P1, locked)",       pins_p1_softlocked, 0, 3),
    ("P1-LOCKED + dice=6  (P0 Bonuszug)",      pins_p1_softlocked, 0, 6),
    ("PRE-WIN   + dice=3  (P1, nicht locked)", pins_pre_win,      0, 3),
]

for name, pins, current_player, dice_val in chance_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    env   = env.replace(die=jnp.int8(dice_val))
    valid_mask = valid_action(env).flatten()
    obs   = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    print(f"  --- {name} ---")
    col_h = f"  {'Pin':>4}   {'P(1)':>6} {'P(2)':>6} {'P(3)':>6} {'P(4)':>6} {'P(5)':>6} {'P(6)':>6}  {'KL↓':>7}  {'GT'}"
    print(col_h)
    print(f"  {'-'*(len(col_h)-2)}")

    kl_vals = []
    for a in range(4):
        if not valid_mask[a]:
            continue
        next_env, _, done = env_step(env, jnp.int32(a))
        true_dist = np.array(dice_probabilities(next_env))
        gt_label  = "LOCKED" if bool(is_soft_locked(next_env)) else "normal"

        action = jnp.array([a])
        _, _, chance_logits, _ = dynamics_net.apply(
            params['dynamics'], latent, action, method=dynamics_net.action_dynamics
        )
        pred_dist = np.array(jax.nn.softmax(chance_logits).squeeze())
        kl = kl_divergence(true_dist, pred_dist)
        kl_vals.append(kl)

        print(f"  Pin{a:>1}   "
              f"{pred_dist[0]:>6.3f} {pred_dist[1]:>6.3f} {pred_dist[2]:>6.3f} "
              f"{pred_dist[3]:>6.3f} {pred_dist[4]:>6.3f} {pred_dist[5]:>6.3f}  "
              f"{kl:>7.4f}  {gt_label}")
        print(f"  {'GT':>4}   "
              f"{true_dist[0]:>6.3f} {true_dist[1]:>6.3f} {true_dist[2]:>6.3f} "
              f"{true_dist[3]:>6.3f} {true_dist[4]:>6.3f} {true_dist[5]:>6.3f}")
    if kl_vals:
        print(f"\n  → Mittleres KL: {np.mean(kl_vals):.4f}  (0=perfekt, <0.1=gut, >1.0=schlecht)\n")
    else:
        print("  → Keine validen Aktionen!\n")


# ════════════════════════════════════════════════════════════════
#  TEST 7: ROOT_IDX LOGIK — Depth → Root-Spieler-Index
#  depth kommt aus chance_dynamics (nur nach Chance-Node aktualisiert!)
#  depth=0 →  root_idx=0  (Root-Spieler selbst am Zug)
#  depth=1 →  root_idx=3
#  depth=2 →  root_idx=2
#  depth=3 →  root_idx=1
# ════════════════════════════════════════════════════════════════
print_header("TEST 7: ROOT_IDX LOGIK  —  Depth → Root-Spieler-Index")
print("  Formel: root_idx = (4 - round(depth)) % 4")
print("  Depth aktualisiert nach JEDEM Chance-Node (Würfelergebnis)")
print()

print(f"  {'depth':>8} {'root_idx':>9} {'Bedeutung'}")
print(f"  {'-'*55}")
for d in range(4):
    ri = (4 - d) % 4
    bedeutungen = {
        0: "Root-Spieler ist am Zug       → values[:,0]",
        1: "root ist 3 Chance-Nodes zurück → values[:,3]",
        2: "root ist 2 Chance-Nodes zurück → values[:,2]",
        3: "root ist 1 Chance-Node zurück  → values[:,1]",
    }
    print(f"  {d:>8}  →  {ri:>5}        {bedeutungen[d]}")
print()

# Simuliere Decision+Chance-Ablauf für 3 Schritte
board  = set_pins_on_board(env_base.board, pins_normal)
env    = env_base.replace(board=board, pins=pins_normal, current_player=0)
obs    = encode_board(env)[None, ...]
latent = repr_net.apply(params['representation'], obs)

print("  Simuliere Schritt-für-Schritt (Action → Chance) Depth-Tracking:")
print(f"  {'Step':>4} {'Action':>6} {'Dice':>5} {'dd_sig':>8} {'depth_before':>13} {'depth_after':>12} {'root_idx':>9}")
print(f"  {'-'*70}")

depth_val  = jnp.array([[0.0]])
cur_latent = latent
# Abwechselnd: normale Aktion, 6er-Bonus, normale Aktion
steps = [(0, 3), (0, 6), (1, 2)]   # (action, dice)
for step_i, (a_idx, dice_val) in enumerate(steps):
    action    = jnp.array([a_idx])
    dice_idx  = jnp.array([dice_val - 1])  # 1-based → 0-based

    afterstate, _, _, _ = dynamics_net.apply(
        params['dynamics'], cur_latent, action, method=dynamics_net.action_dynamics
    )
    next_state, depth_delta_logit = dynamics_net.apply(
        params['dynamics'], afterstate, dice_idx, method=dynamics_net.chance_dynamics
    )
    dd_sig     = float(jax.nn.sigmoid(depth_delta_logit).squeeze())
    old_depth  = float(depth_val.squeeze())
    next_depth = (depth_val + jax.nn.sigmoid(depth_delta_logit)) % 4.0
    nd_int     = int(round(float(next_depth.squeeze()))) % 4
    root_idx   = (4 - nd_int) % 4
    print(f"  {step_i:>4} Pin{a_idx:>2}  dice={dice_val} {dd_sig:>8.4f} "
          f"{old_depth:>13.4f} {float(next_depth.squeeze()):>12.4f}  {root_idx:>9}")
    depth_val  = next_depth
    cur_latent = next_state


# ════════════════════════════════════════════════════════════════
#  TEST 8: MCTS END-TO-END — findet MCTS die Winning Action?
# ════════════════════════════════════════════════════════════════
print_header("TEST 8: MCTS END-TO-END  —  findet MCTS die Winning Action?")
print("  Erwartet: PRE-WIN dice=5 → Pin 0 hat höchsten Q-Value / Action-Weight")
print()

for name, pins, current_player, dice_val, winning_pin in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    env   = env.replace(die=jnp.int8(dice_val))
    valid_mask      = valid_action(env).flatten()
    invalid_actions = (~valid_mask)[None, :]
    obs = encode_board(env)[None, ...]

    policy_out, mcts_value = run_stochastic_muzero_mcts(
        params, jax.random.PRNGKey(np.random.randint(0, 10000)), obs, invalid_actions,
        num_simulations=100, max_depth=50, temperature=0.25
    )

    q_values     = policy_out.search_tree.summary().qvalues[0]
    visit_counts = policy_out.search_tree.summary().visit_counts[0]
    weights      = policy_out.action_weights[0]

    print(f"  --- {name} ---")
    print(f"    MCTS Root-Value (Root-Spieler P{current_player}): {float(mcts_value[0]):+.4f}")
    if winning_pin is not None:
        w_q = float(q_values[winning_pin])
        w_v = int(visit_counts[winning_pin])
        w_w = float(weights[winning_pin])
        print(f"    Winning Pin {winning_pin}: Q={w_q:+.4f}, Visits={w_v}, Weight={w_w:.4f}")

    print(f"    {'Pin':>6} {'Valid':>5} {'Q-Value':>9} {'Visits':>7} {'Weight':>8}")
    print(f"    {'-'*40}")
    for a in range(4):
        if not valid_mask[a]:
            continue
        marker = "  ← WINNING" if a == winning_pin else ""
        print(f"    Pin{a:>2} {'O':>5} {float(q_values[a]):>+9.4f} "
              f"{int(visit_counts[a]):>7} {float(weights[a]):>8.4f}{marker}")
    print()


# ════════════════════════════════════════════════════════════════
#  ZUSAMMENFASSUNG: Shapes + Architektur-Check
# ════════════════════════════════════════════════════════════════
print_header("ZUSAMMENFASSUNG: FFA-Architektur Check (Stochastic)")
print(f"\n  {'Komponente':<38} {'Status'}")
print(f"  {'-'*60}")

board  = set_pins_on_board(env_base.board, pins_normal)
env    = env_base.replace(board=board, pins=pins_normal, current_player=0)
env    = env.replace(die=jnp.int8(3))
obs    = encode_board(env)[None, ...]
latent = repr_net.apply(params['representation'], obs)
action = jnp.array([0])
dice_idx = jnp.array([2])  # dice=3 → index 2

# Check 1: Multi-Value Head
_, values = pred_net.apply(params['prediction'], latent)
shape_ok = values.shape == (1, 4)
print(f"  {'pred_net values shape (1,4)':<38} "
      f"{'✓ OK (' + str(values.shape) + ')' if shape_ok else '✗ FEHLER: ' + str(values.shape)}")

# Check 2: action_dynamics returns 4 outputs
act_out = dynamics_net.apply(params['dynamics'], latent, action, method=dynamics_net.action_dynamics)
n_act = len(act_out)
print(f"  {'action_dynamics outputs (erwartet 4)':<38} "
      f"{'✓ OK (' + str(n_act) + ')' if n_act == 4 else '✗ FEHLER: ' + str(n_act)}")

afterstate_check, rew_l, chance_l, disc_l = act_out

# Check 3: Discount binary (2 classes)
disc_ok = disc_l.shape[-1] == 2
print(f"  {'discount_logits shape (2 Klassen)':<38} "
      f"{'✓ OK (' + str(disc_l.shape) + ')' if disc_ok else '✗ FEHLER: ' + str(disc_l.shape)}")

# Check 4: Reward 3 classes
rew_ok = rew_l.shape[-1] == 3
print(f"  {'reward_logits shape (3 Klassen)':<38} "
      f"{'✓ OK (' + str(rew_l.shape) + ')' if rew_ok else '✗ FEHLER: ' + str(rew_l.shape)}")

# Check 5: Chance logits 6 classes
chance_ok = chance_l.shape[-1] == 6
print(f"  {'chance_logits shape (6 Klassen)':<38} "
      f"{'✓ OK (' + str(chance_l.shape) + ')' if chance_ok else '✗ FEHLER: ' + str(chance_l.shape)}")

# Check 6: chance_dynamics returns 2 outputs (next_state, depth_delta_logit)
chance_out = dynamics_net.apply(params['dynamics'], afterstate_check, dice_idx, method=dynamics_net.chance_dynamics)
n_chance = len(chance_out)
print(f"  {'chance_dynamics outputs (erwartet 2)':<38} "
      f"{'✓ OK (' + str(n_chance) + ')' if n_chance == 2 else '✗ FEHLER: ' + str(n_chance)}")

# Check 7: depth_delta_logit shape (1, 1)
if n_chance == 2:
    ddl = chance_out[1]
    dd_ok = ddl.shape == (1, 1)
    print(f"  {'depth_delta_logit shape (1,1)':<38} "
          f"{'✓ OK (' + str(ddl.shape) + ')' if dd_ok else '✗ FEHLER: ' + str(ddl.shape)}")
else:
    dd_ok = False
    print(f"  {'depth_delta_logit shape (1,1)':<38} ✗ NICHT PRÜFBAR")

# Check 8: Depth-Delta accuracy
print(f"  {'Depth-Delta Accuracy (dice 1-6)':<38} {correct_dd}/{total_dd} ({acc_pct:.0f}%)")

all_ok = all([shape_ok, n_act == 4, disc_ok, rew_ok, chance_ok, n_chance == 2, dd_ok])
print(f"\n  Architektur-Outputs korrekt: {'JA ✓' if all_ok else 'NEIN ✗ — siehe Fehler oben'}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Tests abgeschlossen → {output_file}")
