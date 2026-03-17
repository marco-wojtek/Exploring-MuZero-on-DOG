"""
classification_test_stochastic.py

Tests für stochastisches MuZero auf klassischem MADN.
Kein winning-action-Konzept – alles ist würfelbasiert.

Board-Layout (distance=10, num_players=4, board_size=56):
  Hauptfeld: 0–39
  P0-Ziel: [40,41,42,43]   P1-Ziel: [44,45,46,47]
  P2-Ziel: [48,49,50,51]   P3-Ziel: [52,53,54,55]

Tests:
  TEST 1: REWARD HEAD   – Terminal (+1) vs. Normal (0)
  TEST 2: DISCOUNT HEAD – 6er-Bonuszug (+1) vs. Gegnerzug (-1) vs. Terminal (0)
  TEST 3: CHANCE HEAD   – Würfelverteilung: uniform vs. soft-locked
  TEST 4: MCTS DYNAMICS – Stochastischer MCTS auf relevanten States
"""
import sys, os
import functools
from jax import numpy as jnp
import jax
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
# ── CONFIG ────────────────────────────────────────────────────────────────────
filename = "stochastic_muzero_madn_params_lr0.005_g1500_it100_seed3"
PARAM_FILE = f"MuZero_Classic_MADN/models/params/{filename}.pkl"
# PARAM_FILE = None  # ← für untrainierte Params

sys.stdout = open(f"MuZero_Classic_MADN/evaluation/{filename}_stochastic_tests.txt", "w")

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
SUPPORT = jnp.array([-1.0, 0.0, 1.0])

def logits_to_scalar(logits):
    probs = jax.nn.softmax(logits, axis=-1)
    return float(jnp.sum(probs * SUPPORT, axis=-1).squeeze())

def logits_to_probs(logits):
    return np.array(jax.nn.softmax(logits, axis=-1).squeeze())

def kl_divergence(p, q):
    """KL(p || q) – misst Abstand Vorhersage q von Wahrheit p"""
    p = np.array(p, dtype=float) + 1e-9
    q = np.array(q, dtype=float) + 1e-9
    return float(np.sum(p * np.log(p / q)))

def print_header(title):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")

# ── Params laden ──────────────────────────────────────────────────────────────
input_shape = (11, 56)
if PARAM_FILE:
    params = load_params_from_file(PARAM_FILE)
    print(f"Params geladen: {PARAM_FILE}")
else:
    params = init_muzero_params(jax.random.PRNGKey(0), input_shape)
    print("Frische (untrainierte) Params initialisiert")

print("Modus: KLASSIFIKATION (3 Klassen: {-1, 0, +1})\n")

# ── Basis-Environment ─────────────────────────────────────────────────────────
# Muss exakt mit den Trainingsregeln übereinstimmen!
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

def pin_str(a_idx):
    return f"Pin {a_idx}"


# ════════════════════════════════════════════════════════════
#  TEST-ZUSTÄNDE
#  Board-Layout (distance=10, num_players=4, board_size=56):
#    Hauptfeld:   0–39
#    P0-Ziel: [40,41,42,43]   P1-Ziel: [44,45,46,47]
#    P2-Ziel: [48,49,50,51]   P3-Ziel: [52,53,54,55]
# ════════════════════════════════════════════════════════════

# P0 einen Zug vor Sieg:
#   Pin 0 bei 35, Würfel=5 → 35+5=40 = goal[0] → P0 gewinnt → reward +1
#   (Pins 1-3 bereits im Ziel: 41,42,43)
pins_pre_win = jnp.array([
    [35, 41, 42, 43],   # P0: Pin 0 bei 35, Pins 1-3 im Ziel
    [ 5, 15,  7, 12],   # P1: Mittelspiel
    [48, 49, 50, 51],   # P2: alle im Ziel (Team-Partner P0)
    [25, 28, 33, 30],   # P3: Mittelspiel
], dtype=jnp.int32)

# P0 gewinnt mit Würfel=6:
#   Pin 0 bei 34, Würfel=6 → 34+6=40 = goal[0] → P0 gewinnt (Terminal, kein Bonuszug)
pins_pre_win_6 = jnp.array([
    [34, 41, 42, 43],
    [ 5, 15,  7, 12],
    [48, 49, 50, 51],
    [25, 28, 33, 30],
], dtype=jnp.int32)

# Normales Mittelspiel (kein Gewinn in Sicht)
pins_normal = jnp.array([
    [10, 20, 30, -1],   # P0: 3 Pins auf Feld
    [15, 25, -1, -1],   # P1: 2 Pins auf Feld
    [ 5, 35, -1, -1],   # P2: 2 Pins auf Feld
    [ 8, 18, -1, -1],   # P3: 2 Pins auf Feld
], dtype=jnp.int32)

# Nach P0s Zug (nicht-6) ist P1 dran – P1 ist SOFT-LOCKED:
#   P1-Ziel: [44,45,46,47], P1-Pins: [-1,-1,46,47]
#   → 2 Pins in letzten 2 Zielpositionen → is_soft_locked(next_env) = True
#   → dice_probabilities(next_env) = OUT_ON_ONE_AND_SIX [76,16,16,16,16,76]/216
pins_p1_softlocked = jnp.array([
    [10, 20, 30, -1],   # P0: macht nicht-6 Zug → P1 kommt dran
    [-1, -1, 46, 47],   # P1: soft-locked (2 Pins in letzten 2 Zielpositionen)
    [ 5, 35, -1, -1],   # P2: Mittelspiel
    [ 8, 18, -1, -1],   # P3: Mittelspiel
], dtype=jnp.int32)


# ════════════════════════════════════════════════════════════
#  TEST 1: REWARD HEAD
#  Erwartet:
#    PRE-WIN  dice=5  Pin 0 → reward ≈ +1  (P0 gewinnt, Ziel erreicht)
#    PRE-WIN-6 dice=6 Pin 0 → reward ≈ +1  (P0 gewinnt via Würfel=6)
#    Normaler Zug            → reward ≈  0
# ════════════════════════════════════════════════════════════
print_header("TEST 1: REWARD HEAD – Terminal (+1) vs. Normal (0)")
print("Erwartet:")
print("  PRE-WIN  dice=5  Pin 0 → reward ≈ +1  (P0 gewinnt)")
print("  PRE-WIN-6 dice=6 Pin 0 → reward ≈ +1  (P0 gewinnt via Würfel=6)")
print("  Alle anderen Züge      → reward ≈  0")
print("  Klassen: P(-1)=Verlust, P(0)=Normal, P(+1)=Sieg")
print()

reward_scenarios = [
    ("PRE-WIN   (Würfel=5, Pin 0 gewinnt)", pins_pre_win,   0, 5),
    ("PRE-WIN   (Würfel=3, kein Gewinn)",   pins_pre_win,   0, 3),
    ("PRE-WIN-6 (Würfel=6, Pin 0 gewinnt)", pins_pre_win_6, 0, 6),
    ("NORMAL    (Würfel=3)",                pins_normal,    0, 3),
    ("NORMAL    (Würfel=6)",                pins_normal,    0, 6),
]

for scenario_name, pins, current_player, dice_val in reward_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    env = env.replace(die=jnp.int8(dice_val))
    valid_mask = valid_action(env).flatten()

    obs = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    print(f"--- {scenario_name} ---")
    header = f"  {'Pin':>4} {'Valid':>5} {'GT-R':>6} {'Pred-R':>7} {'P(-1)':>6} {'P(0)':>6} {'P(+1)':>6} {'Match':>5}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    correct, total = 0, 0
    for a in range(4):
        if not valid_mask[a]:
            continue

        # Ground Truth: env_step ausführen, Reward ermitteln
        next_env, gt_r_raw, done = env_step(env, jnp.int32(a))
        if done and float(gt_r_raw) > 0:
            gt_reward = +1.0
        elif done and float(gt_r_raw) < 0:
            gt_reward = -1.0
        else:
            gt_reward = 0.0

        action = jnp.array([a])
        _, reward_logits, _, _ = dynamics_net.apply(
            params['dynamics'], latent, action, method=dynamics_net.action_dynamics
        )
        pred_r = logits_to_scalar(reward_logits)
        rp = logits_to_probs(reward_logits)
        match = abs(pred_r - gt_reward) < 0.5
        correct += int(match)
        total += 1

        print(f"  {pin_str(a):>4} {'O':>5} {gt_reward:>+6.1f} {pred_r:>+7.4f} "
              f"{rp[0]:>6.3f} {rp[1]:>6.3f} {rp[2]:>6.3f} {'O' if match else 'X':>5}")

    if total > 0:
        print(f"\n  → Accuracy: {correct}/{total} ({100*correct/total:.0f}%)\n")
    else:
        print("  → Keine validen Aktionen!\n")


# ════════════════════════════════════════════════════════════
#  TEST 2: DISCOUNT HEAD
#  Erwartet:
#    dice=6 + kein Terminal → discount ≈ +1  (P0 Bonuszug, selbes Team)
#    dice≠6 + kein Terminal → discount ≈ -1  (nächstes Team)
#    Terminal (Spiel vorbei) → discount ≈  0
# ════════════════════════════════════════════════════════════
print_header("TEST 2: DISCOUNT HEAD – 6er-Bonuszug (+1) vs. Gegnerzug (-1) vs. Terminal (0)")
print("Erwartet:")
print("  dice=6, kein Terminal → discount ≈ +1  (selbes Team spielt nochmal)")
print("  dice≠6, kein Terminal → discount ≈ -1  (gegnerisches Team)")
print("  Terminal (Sieg/Niederlage) → discount ≈  0")
print()

discount_scenarios = [
    ("PRE-WIN   (dice=5, Pin 0 → TERMINAL)", pins_pre_win,   0, 5),
    ("PRE-WIN-6 (dice=6, Pin 0 → TERMINAL)", pins_pre_win_6, 0, 6),
    ("NORMAL    (dice=6, P0 Bonuszug)",       pins_normal,    0, 6),
    ("NORMAL    (dice=3, P1 nächster)",        pins_normal,    0, 3),
    ("NORMAL    (dice=1, P1 nächster)",        pins_normal,    0, 1),
]

for scenario_name, pins, current_player, dice_val in discount_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    env = env.replace(die=jnp.int8(dice_val))
    valid_mask = valid_action(env).flatten()

    obs = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    print(f"--- {scenario_name} ---")
    header = f"  {'Pin':>4} {'Valid':>5} {'GT-D':>6} {'Pred-D':>7} {'P(-1)':>6} {'P(0)':>6} {'P(+1)':>6} {'Match':>5}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    correct, total = 0, 0
    for a in range(4):
        if not valid_mask[a]:
            continue

        # Ground Truth: Discount aus Spiellogik berechnen
        next_env, gt_r_raw, done = env_step(env, jnp.int32(a))
        if done:
            gt_disc = 0.0   # Terminal → discount 0
        else:
            # Teams: P0+P2 = Team 0, P1+P3 = Team 1
            current_team = int(current_player) % 2
            next_team = int(next_env.current_player) % 2
            gt_disc = +1.0 if current_team == next_team else -1.0

        action = jnp.array([a])
        _, _, _, discount_logits = dynamics_net.apply(
            params['dynamics'], latent, action, method=dynamics_net.action_dynamics
        )
        pred_d = logits_to_scalar(discount_logits)
        dp = logits_to_probs(discount_logits)
        match = abs(pred_d - gt_disc) < 0.5
        correct += int(match)
        total += 1

        print(f"  {pin_str(a):>4} {'O':>5} {gt_disc:>+6.1f} {pred_d:>+7.4f} "
              f"{dp[0]:>6.3f} {dp[1]:>6.3f} {dp[2]:>6.3f} {'O' if match else 'X':>5}")

    if total > 0:
        print(f"\n  → Accuracy: {correct}/{total} ({100*correct/total:.0f}%)\n")
    else:
        print("  → Keine validen Aktionen!\n")


# ════════════════════════════════════════════════════════════
#  TEST 3: CHANCE HEAD – Würfelverteilung vorhersagen
#  Das Netz soll dice_probabilities(next_env) vorhersagen.
#
#  Würfelverteilungen bei enable_dice_rethrow=True, enable_start_on_1=True:
#    NORMAL:            [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
#    OUT_ON_1_AND_6:    [76, 16, 16, 16, 16, 76] / 216  (soft-locked)
#
#  Testfälle:
#    NORMAL + dice≠6 → P1 nächster, nicht locked → uniform
#    P1-SOFTLOCKED + dice≠6 → P1 nächster, locked → OUT_ON_1_AND_6
#    BELIEBIG + dice=6 → P0 Bonuszug, nicht locked → uniform
# ════════════════════════════════════════════════════════════
print_header("TEST 3: CHANCE HEAD – Vorhersage der Würfelverteilung")
print("Das Netz soll dice_probabilities(next_env) nach dem Zug vorhersagen.")
print("Erwartet:")
print("  NORMAL + dice≠6          → P1 nächster (normal)     → uniform [1/6]*6")
print("  P1-SOFTLOCKED + dice≠6   → P1 nächster (soft-lock)  → [76/216,...,76/216]")
print("  BELIEBIG + dice=6        → P0 Bonuszug (nicht lock)  → uniform [1/6]*6")
print()
normal_dist = np.array(NORMAL_DICE_DISTRIBUTION)
locked_dist = np.array(OUT_ON_ONE_AND_SIX_DICE_DISTRIBUTION)
print(f"  NORMAL_DIST:         [{', '.join(f'{v:.4f}' for v in normal_dist)}]")
print(f"  OUT_ON_1_AND_6_DIST: [{', '.join(f'{v:.4f}' for v in locked_dist)}]")
print()

chance_scenarios = [
    ("NORMAL + dice=3  (P1 nächster, nicht locked)",  pins_normal,       0, 3),
    ("NORMAL + dice=6  (P0 Bonuszug, nicht locked)",   pins_normal,       0, 6),
    ("P1-LOCKED + dice=3  (P1 nächster, locked)",      pins_p1_softlocked, 0, 3),
    ("P1-LOCKED + dice=6  (P0 Bonuszug, nicht locked)", pins_p1_softlocked, 0, 6),
    ("PRE-WIN + dice=3  (P1 nächster, nicht locked)",  pins_pre_win,      0, 3),
]

for scenario_name, pins, current_player, dice_val in chance_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    env = env.replace(die=jnp.int8(dice_val))
    valid_mask = valid_action(env).flatten()

    obs = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    print(f"--- {scenario_name} ---")
    col_h = f"  {'Pin':>4}   {'P(1)':>6} {'P(2)':>6} {'P(3)':>6} {'P(4)':>6} {'P(5)':>6} {'P(6)':>6}  {'KL↓':>7}  {'GT-Dist':>8}"
    print(col_h)
    print(f"  {'-' * (len(col_h) - 2)}")

    kl_values = []
    for a in range(4):
        if not valid_mask[a]:
            continue

        # Ground Truth nach env_step
        next_env, _, done = env_step(env, jnp.int32(a))
        true_dist = np.array(dice_probabilities(next_env))
        gt_soft = bool(is_soft_locked(next_env))

        action = jnp.array([a])
        _, _, chance_logits, _ = dynamics_net.apply(
            params['dynamics'], latent, action, method=dynamics_net.action_dynamics
        )
        pred_dist = logits_to_probs(chance_logits)
        kl = kl_divergence(true_dist, pred_dist)
        kl_values.append(kl)

        gt_label = "LOCKED" if gt_soft else "NORMAL"
        print(f"  {pin_str(a):>4}   "
              f"{pred_dist[0]:>6.3f} {pred_dist[1]:>6.3f} {pred_dist[2]:>6.3f} "
              f"{pred_dist[3]:>6.3f} {pred_dist[4]:>6.3f} {pred_dist[5]:>6.3f}  "
              f"{kl:>7.4f}  {gt_label:>8}")
        print(f"  {'GT':>4}   "
              f"{true_dist[0]:>6.3f} {true_dist[1]:>6.3f} {true_dist[2]:>6.3f} "
              f"{true_dist[3]:>6.3f} {true_dist[4]:>6.3f} {true_dist[5]:>6.3f}")

    if kl_values:
        print(f"\n  → Mittleres KL: {np.mean(kl_values):.4f}  "
              f"(0 = perfekt, < 0.1 = gut, > 1.0 = schlecht)\n")
    else:
        print("  → Keine validen Aktionen!\n")


# ════════════════════════════════════════════════════════════
#  TEST 4: MCTS DYNAMICS – Stochastischer MCTS
#  Testet ob der stochastische MCTS mit korrekten Netzwerkausgaben
#  sinnvolle Züge findet (Q-Values, Besuchszahlen, Gewichtung).
#
#  PRE-WIN dice=5: MCTS sollte Pin 0 (einziger Gewinnzug) bevorzugen
#  NORMAL:         MCTS verteilt Besuche sinnvoll
# ════════════════════════════════════════════════════════════
print_header("TEST 4: MCTS DYNAMICS – Stochastischer MCTS")
print("Erwartet:")
print("  PRE-WIN dice=5: MCTS bevorzugt Pin 0 (einziger Gewinnzug)")
print("  NORMAL:         MCTS verteilt Besuche/Q-Values sinnvoll")
print()

mcts_scenarios = [
    ("PRE-WIN   dice=5 (Pin 0 = Gewinnzug)", pins_pre_win,       0, 5,  0),
    ("PRE-WIN   dice=3 (kein Gewinn)",        pins_pre_win,       0, 3,  None),
    ("NORMAL    dice=5",                      pins_normal,        0, 5,  None),
    ("P1-LOCKED dice=3",                      pins_p1_softlocked, 0, 3,  None),
]

for scenario_name, pins, current_player, dice_val, expected_pin in mcts_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    env = env.replace(die=jnp.int8(dice_val))
    valid_mask = valid_action(env).flatten()

    obs = encode_board(env)[None, ...]
    invalid_actions = (~valid_mask)[None, :]

    policy_out, mcts_value = run_stochastic_muzero_mcts(
        params, jax.random.PRNGKey(42), obs, invalid_actions,
        num_simulations=50, max_depth=20, temperature=0.25
    )

    q_values = policy_out.search_tree.summary().qvalues[0]
    visit_counts = policy_out.search_tree.summary().visit_counts[0]
    weights = policy_out.action_weights[0]
    chosen = int(policy_out.action[0])

    print(f"--- {scenario_name} ---")
    print(f"  MCTS-Value: {float(mcts_value[0]):+.4f}  |  Gewählter Zug: {pin_str(chosen)}", end="")
    if expected_pin is not None:
        match = (chosen == expected_pin)
        print(f"  (Erwartet: {pin_str(expected_pin)}) → {'O KORREKT' if match else 'X FALSCH'}", end="")
    print()
    print()

    header = f"  {'Pin':>4} {'Valid':>5} {'Q-Value':>8} {'Besuche':>8} {'Gewicht':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for a in range(4):
        if not valid_mask[a]:
            continue
        q = float(q_values[a])
        v = int(visit_counts[a])
        w = float(weights[a])
        mark = " ← GEWÄHLT" if a == chosen else (" ← ERWARTET" if a == expected_pin else "")
        print(f"  {pin_str(a):>4} {'O':>5} {q:>+8.4f} {v:>8d} {w:>8.3f}{mark}")
    print()

# ════════════════════════════════════════════════════════════
#  Test 5: Auswahlverfahren
# Gegeben ein Zustand wo Pins 0-2 mit Würfel 5 bewegt werden können.
# Pin 0 führt ins Ziel, Pin 1 schlägt einen gegnerischen Pin, Pin 2 ist neutral.
# Analysiere welcher Pin raw bevorzugt wird und welcher nach MCTS als bester Zug herauskommt.
# ════════════════════════════════════════════════════════════

print_header("TEST 5: Auswahlverfahren – Raw vs. MCTS")
print("Gegeben ein Zustand wo Pins 0-2 mit Würfel 5 bewegt werden können.")
print("Pin 0 führt ins Ziel, Pin 1 schlägt einen gegnerischen Pin, Pin 2 ist neutral.")
print()

test5_state = jnp.array([
    [35, 10, 1, 43],   # P0: Pin 0 bei 35, Pins 1-3 im Ziel
    [ 5, 15,  7, 12],   # P1: Mittelspiel
    [48, 49, 50, 51],   # P2: alle im Ziel (Team-Partner P0)
    [25, 28, 33, 30],   # P3: Mittelspiel
], dtype=jnp.int32)

board = set_pins_on_board(env_base.board, test5_state)
env = env_base.replace(board=board, pins=test5_state, current_player=0)
env = env.replace(die=jnp.int8(5))
valid_mask = valid_action(env).flatten()

obs = encode_board(env)[None, ...]
latent = repr_net.apply(params['representation'], obs)

print("RAW Netzwerkausgaben für Pins 0-2:")
header = f"  {'Pin':>4} {'Valid':>5} {'Reward':>7} {'Discount':>8} {'Chance':>8}"
print(header)
print(f"  {'-' * (len(header) - 2)}")
for a in range(3):
    if not valid_mask[a]:
        continue
    action = jnp.array([a])
    _, reward_logits, chance_logits, discount_logits = dynamics_net.apply(
        params['dynamics'], latent, action, method=dynamics_net.action_dynamics
    )
    pred_r = logits_to_scalar(reward_logits)
    pred_d = logits_to_scalar(discount_logits)
    pred_c = logits_to_probs(chance_logits)
    print(f"  {pin_str(a):>4} {'O':>5} {pred_r:>+7.4f} {pred_d:>+8.4f} [{', '.join(f'{v:.3f}' for v in pred_c)}]")

print("\nMCTS-Auswahl für diesen Zustand:")
invalid_actions = (~valid_mask)[None, :]
policy_out, mcts_value = run_stochastic_muzero_mcts(
    params, jax.random.PRNGKey(123), obs, invalid_actions,
    num_simulations=75, max_depth=50, temperature=0.0
)
q_values = policy_out.search_tree.summary().qvalues[0]
visit_counts = policy_out.search_tree.summary().visit_counts[0]
weights = policy_out.action_weights[0]
chosen = int(policy_out.action[0])

header = f"  {'Pin':>4} {'Valid':>5} {'Q-Value':>8} {'Besuche':>8} {'Gewicht':>8}"
print(header)
print(f"  {'-' * (len(header) - 2)}")
for a in range(3):
    if not valid_mask[a]:
        continue
    q = float(q_values[a])
    v = int(visit_counts[a])
    w = float(weights[a])
    mark = " ← GEWÄHLT" if a == chosen else ""
    print(f"  {pin_str(a):>4} {'O':>5} {q:>+8.4f} {v:>8d} {w:>8.3f}{mark}")
print()
print(f"  MCTS-Value: {float(mcts_value[0]):+.4f}  |  Gewählter Zug: {pin_str(chosen)}")

# ════════════════════════════════════════════════════════════
#  ZUSAMMENFASSUNG
# ════════════════════════════════════════════════════════════
print_header("ZUSAMMENFASSUNG")
print("""
TEST 1 – REWARD HEAD:
  Vorhersage: +1 bei Sieg (Terminal), 0 bei normalem Zug.
  Key: PRE-WIN dice=5 Pin 0 → reward muss ≈ +1 sein.

TEST 2 – DISCOUNT HEAD:
  Vorhersage: +1 bei Würfel=6 (Bonuszug, selbes Team),
              -1 bei normalem Zug (gegnerisches Team),
               0 bei Terminal (Spiel vorbei).

TEST 3 – CHANCE HEAD:
  Vorhersage der Würfelverteilung für den nächsten Spieler:
  - Normal:     [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]  (KL ≈ 0)
  - Soft-locked: [76/216, 16/216, .., 76/216]     (KL ≈ 0)
  Schlechte KL (> 0.5) deutet darauf hin, dass die
  soft-locked Zustände nicht gelernt wurden.

TEST 4 – MCTS DYNAMICS:
  Der stochastische MCTS muss Reward, Discount und Chance
  gemeinsam nutzen, um sinnvolle Q-Values zu berechnen.
  PRE-WIN dice=5: Pin 0 sollte höchste Besuche/Q-Values haben.
""")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Tests abgeschlossen.")