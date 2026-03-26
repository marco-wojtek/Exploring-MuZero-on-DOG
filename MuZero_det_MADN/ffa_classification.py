"""
Test-Skript: FFA Multi-Value Head & Depth-Delta Diagnose
=========================================================
Testet die neuen FFA (Free-for-All) Änderungen:
  1. Multi-Value Head:   pred_net gibt values (B,4) zurück — ein Value pro Spieler-Perspektive
  2. Depth-Delta Head:   dynamics_net lernt ob Spieler wechselt (1) oder 6er-Bonus (0)
  3. Binary Discount:    2 Klassen {0=Terminal, 1=Non-Terminal} — kein Vorzeichen-Flip mehr
  4. Reward Head:        3 Klassen {-1, 0, +1} aus Root-Spieler-Perspektive

Architektur-Zusammenfassung:
  dynamics_net → (next_latent, reward_logits[3], discount_logits[2], depth_delta_logit[1])
  pred_net     → (policy_logits[24], values[4])
"""
import sys, os

from jax import numpy as jnp
import jax
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from MADN.deterministic_madn import (
    env_reset, encode_board, valid_action, set_pins_on_board,
    env_step, map_action, get_winner, winning_action
)
from MuZero_det_MADN.muzero_deterministic_madn import (
    repr_net, pred_net, dynamics_net, load_params_from_file, init_muzero_params,
    run_muzero_mcts
)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
filename = "Experiment_53_100"
PARAM_FILE = f"MuZero_det_MADN/models/params/{filename}.pkl"
# PARAM_FILE = None  # Für frische (untrainierte) Params

output_file = f"MuZero_det_MADN/evaluation/{filename}_ffa_test.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

# ═══════════════════════════════════════════════════════════════
#  SUPPORT VEKTOREN
# ═══════════════════════════════════════════════════════════════
SUPPORT_REWARD   = jnp.array([-1.0, 0.0, 1.0])   # 3 Klassen
SUPPORT_DISCOUNT = jnp.array([0.0, 1.0])           # 2 Klassen: {Terminal, Non-Terminal}


def reward_logits_to_scalar(logits):
    probs = jax.nn.softmax(logits, axis=-1)
    return float(jnp.sum(probs * SUPPORT_REWARD, axis=-1).squeeze())


def reward_logits_to_probs(logits):
    return jax.nn.softmax(logits, axis=-1).squeeze()


def discount_logits_to_scalar(logits):
    """Binary discount: 0=Terminal, 1=Non-Terminal"""
    probs = jax.nn.softmax(logits, axis=-1)
    return float(jnp.sum(probs * SUPPORT_DISCOUNT, axis=-1).squeeze())


def discount_logits_to_probs(logits):
    return jax.nn.softmax(logits, axis=-1).squeeze()


def depth_delta_logit_to_scalar(logit):
    """Sigmoid → ≈0 für 6er-Bonus-Zug (gleicher Spieler), ≈1 für Spielerwechsel"""
    return float(jax.nn.sigmoid(logit).squeeze())


def action_str(idx):
    pin  = idx // 6
    move = idx % 6 + 1
    six  = " (6er!)" if move == 6 else ""
    return f"Pin {pin}, Move {move}{six}"


def print_header(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# ═══════════════════════════════════════════════════════════════
#  PARAMS LADEN
# ═══════════════════════════════════════════════════════════════
input_shape = (34, 56)
if PARAM_FILE:
    params = load_params_from_file(PARAM_FILE)
    print(f"Params geladen: {PARAM_FILE}")
else:
    params = init_muzero_params(jax.random.PRNGKey(0), input_shape)
    print("Frische (untrainierte) Params initialisiert")

print("Architektur: FFA Multi-Value Head (4 Werte) + Binary Discount + Depth-Delta\n")

# ═══════════════════════════════════════════════════════════════
#  BASIS-ENVIRONMENT
# ═══════════════════════════════════════════════════════════════
env_base = env_reset(
    0, num_players=4,
    layout=jnp.array([True, True, True, True]),
    distance=10, starting_player=0, seed=1,
    enable_teams=False,          # FFA!
    enable_initial_free_pin=True,
    enable_circular_board=False
)

# ── Test-States ────────────────────────────────────────────────
# P0: 1 Zug vor Sieg mit Move 5 (action 4)
pins_pre_win = jnp.array([
    [35, 41, 42, 43],
    [ 5, 15,  7, 12],
    [48, 49,  3, -1],
    [25, 28, 33, 30],
], dtype=jnp.int32)

# P0: 1 Zug vor Sieg mit Move 6 / 6er-Bonus (action 5)
pins_pre_win_6 = jnp.array([
    [34, 41, 42, 43],
    [ 5, 15,  7, 12],
    [48, 49,  3, -1],
    [25, 28, 33, 30],
], dtype=jnp.int32)

# P0 verliert bald: alle anderen Spieler kurz vor Ziel
pins_pre_lose = jnp.array([
    [-1, -1, -1,  2],
    [ 5, 44, 45, 46],
    [ 1,  3, 20, 21],
    [-1, 53,  0, 55],
], dtype=jnp.int32)

pins_normal = jnp.array([
    [10, 20, 30, -1],
    [15, 25, -1, -1],
    [ 5, 35, -1, -1],
    [ 8, 18, -1, -1],
], dtype=jnp.int32)

test_scenarios = [
    ("PRE-WIN  (Move 5 gewinnt)", pins_pre_win,   0, 4),
    ("PRE-WIN6 (Move 6 gewinnt)", pins_pre_win_6, 0, 5),
    ("PRE-LOSE (P0 verliert)",    pins_pre_lose,  0, None),
    ("NORMAL   (Mittelspiel)",    pins_normal,    0, None),
]


# ════════════════════════════════════════════════════════════════
#  TEST 1: MULTI-VALUE HEAD — values (B, 4)
#  Erwartet: PRE-WIN → values[:,0] ≈ +1 (P0 gewinnt)
#                     values[:,1..3] ≈ -1 (P1..P3 verlieren)
# ════════════════════════════════════════════════════════════════
print_header("TEST 1: MULTI-VALUE HEAD  —  pred_net → values (B, 4)")
print("  Erwartet für PRE-WIN:")
print("    values[0] ≈ +1.0  (Root-Spieler P0 gewinnt)")
print("    values[1..3] ≈ -1.0  (Gegner verlieren)")
print()

for name, pins, current_player, _ in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs  = encode_board(env)[None, ...]

    latent = repr_net.apply(params['representation'], obs)
    policy_logits, values = pred_net.apply(params['prediction'], latent)
    # values shape: (1, 4)
    v = values[0]   # (4,)

    print(f"  {name}")
    print(f"    V[0]={float(v[0]):+.4f}  V[1]={float(v[1]):+.4f}  "
          f"V[2]={float(v[2]):+.4f}  V[3]={float(v[3]):+.4f}")
    v_rel_0 = float(v[0])
    interpret = ("ROOT gewinnt  ✓" if v_rel_0 > 0.3 else
                 "ROOT verliert ✓" if v_rel_0 < -0.3 else "neutral")
    print(f"    Root-Value V[0]: {interpret}")
    print()


# ════════════════════════════════════════════════════════════════
#  TEST 2: DEPTH-DELTA HEAD  —  6er vs Spielerwechsel
#  Erwartet: Move 6 (Actions 5,11,17,23) → depth_delta ≈ 0.0 (gleicher Spieler)
#            Andere Moves                 → depth_delta ≈ 1.0 (Spielerwechsel)
# ════════════════════════════════════════════════════════════════
print_header("TEST 2: DEPTH-DELTA HEAD  —  dynamics_net → depth_delta_logit")
print("  Target: depth_delta_sigmoid ≈ 0.0  für Action 6 (6er Bonus-Zug)")
print("          depth_delta_sigmoid ≈ 1.0  für alle anderen Actions")
print()

board = set_pins_on_board(env_base.board, pins_normal)
env   = env_base.replace(board=board, pins=pins_normal, current_player=0)
obs   = encode_board(env)[None, ...]
valid_mask = valid_action(env).flatten()
latent = repr_net.apply(params['representation'], obs)

correct_dd = 0
total_dd   = 0

print(f"  {'Action':<24} {'Valid':>5} {'DepthDelta':>11} {'Expected':>9} {'Logit':>8} {'Match':>6}")
print(f"  {'-'*63}")

for a_idx in range(24):
    if not valid_mask[a_idx]:
        continue
    action = jnp.array([a_idx])
    _, _, _, depth_delta_logit = dynamics_net.apply(params['dynamics'], latent, action)
    dd_sig  = depth_delta_logit_to_scalar(depth_delta_logit)
    dd_raw  = float(depth_delta_logit.squeeze())
    move    = a_idx % 6 + 1
    is_six  = (move == 6)
    expected = 0.0 if is_six else 1.0
    match    = abs(dd_sig - expected) < 0.5
    correct_dd += int(match)
    total_dd   += 1
    marker = "  ← 6ER" if is_six else ""
    print(f"  {action_str(a_idx):<24} {'O':>5} {dd_sig:>11.4f} {expected:>9.1f} "
          f"{dd_raw:>8.3f} {'✓' if match else '✗':>6}{marker}")

acc_pct = 100 * correct_dd / max(total_dd, 1)
print(f"\n  Depth-Delta Accuracy: {correct_dd}/{total_dd} ({acc_pct:.0f}%)")
if acc_pct < 50:
    print("  ⚠ Modell untrainiert bezüglich Depth-Delta — erwartet bei frischen Params")


# ════════════════════════════════════════════════════════════════
#  TEST 3: BINARY DISCOUNT  —  {0=Terminal, 1=Non-Terminal}
#  Erwartet: Winning Action → P(Terminal)≈1 → discount_val ≈ 0.0
#            Normale Actions  → P(Non-Terminal)≈1 → discount_val ≈ 1.0
# ════════════════════════════════════════════════════════════════
print_header("TEST 3: BINARY DISCOUNT HEAD  —  {0=Terminal, 1=Non-Terminal}")
print("  Erwartet: Winning Action  → discount_val ≈ 0.0  (Terminal)")
print("            Normale Actions → discount_val ≈ 1.0  (Non-Terminal)")
print()

for name, pins, current_player, winning_idx in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs   = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    latent = repr_net.apply(params['representation'], obs)

    print(f"  --- {name} ---")
    print(f"  {'Action':<24} {'Val':>7} {'P(Term)':>8} {'P(NonT)':>8} {'Interpret'}")
    print(f"  {'-'*58}")

    for a_idx in range(24):
        if not valid_mask[a_idx]:
            continue
        action = jnp.array([a_idx])
        _, _, disc_logits, _ = dynamics_net.apply(params['dynamics'], latent, action)
        disc_val  = discount_logits_to_scalar(disc_logits)
        disc_probs = discount_logits_to_probs(disc_logits)
        p_term    = float(disc_probs[0])
        p_nonterm = float(disc_probs[1])
        interpret = "TERMINAL ✓" if p_term > 0.5 else "non-term"
        marker    = "  ← WINNING" if a_idx == winning_idx else ""
        print(f"  {action_str(a_idx):<24} {disc_val:>7.4f} {p_term:>8.4f} {p_nonterm:>8.4f}  "
              f"{interpret}{marker}")
    print()


# ════════════════════════════════════════════════════════════════
#  TEST 4: REWARD HEAD  —  Root-Perspektive nach FFA-Logik
#  Erwartet: Winning Action (old_depth=0) → reward ≈ +1
#            Normal Actions              → reward ≈  0
# ════════════════════════════════════════════════════════════════
print_header("TEST 4: REWARD HEAD  —  3 Klassen {-1, 0, +1} — FFA Root-Perspektive")
print("  Erwartet bei PRE-WIN: Winning Action → P(+1) hoch, E[R] ≈ +1.0")
print("  (Reward wird intern geflipt wenn Gegner am Zug; hier testet pred_net direkt)")
print()

for name, pins, current_player, winning_idx in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs   = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    win_mask   = winning_action(env).flatten()
    latent = repr_net.apply(params['representation'], obs)

    print(f"  --- {name} ---")
    print(f"  {'Action':<24} {'E[R]':>7} {'P(-1)':>7} {'P(0)':>7} {'P(+1)':>7} {'Disc':>7}")
    print(f"  {'-'*58}")

    for a_idx in range(24):
        if not valid_mask[a_idx]:
            continue
        action = jnp.array([a_idx])
        _, rew_logits, disc_logits, _ = dynamics_net.apply(params['dynamics'], latent, action)
        r_val  = reward_logits_to_scalar(rew_logits)
        r_probs = reward_logits_to_probs(rew_logits)
        d_val  = discount_logits_to_scalar(disc_logits)
        wins   = "WIN" if win_mask[a_idx] else ""
        marker = "  <<<" if a_idx == winning_idx else ""
        print(f"  {action_str(a_idx):<24} {r_val:>+7.4f} "
              f"{float(r_probs[0]):>7.4f} {float(r_probs[1]):>7.4f} {float(r_probs[2]):>7.4f} "
              f"{d_val:>7.4f}  {wins}{marker}")
    print()


# ════════════════════════════════════════════════════════════════
#  TEST 5: DEPTH-DELTA KONSISTENZ über verschiedene States
#  Depth-Delta hängt nur von der Action ab (via action_one_hot)
#  → sollte über verschiedene Spielzustände gleich sein
# ════════════════════════════════════════════════════════════════
print_header("TEST 5: DEPTH-DELTA KONSISTENZ über verschiedene States")
print("  depth_delta_logit kommt aus Dense(action_one_hot) → state-unabhängig")
print("  Std > 0.05 deutet auf Fehler hin")
print()

all_states = [
    ("PRE-WIN",  pins_pre_win),
    ("PRE-LOSE", pins_pre_lose),
    ("NORMAL",   pins_normal),
]

dd_per_action = {a: [] for a in range(24)}

for state_name, pins in all_states:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=0)
    obs   = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)
    for a_idx in range(24):
        action = jnp.array([a_idx])
        _, _, _, depth_delta_logit = dynamics_net.apply(params['dynamics'], latent, action)
        dd_per_action[a_idx].append(depth_delta_logit_to_scalar(depth_delta_logit))

print(f"  {'Action':<24} {'PRE-WIN':>9} {'PRE-LOSE':>9} {'NORMAL':>9} {'Std':>7} {'Status'}")
print(f"  {'-'*70}")

for a_idx in range(24):
    vals = dd_per_action[a_idx]
    std  = float(np.std(vals))
    move = a_idx % 6 + 1
    expected = 0.0 if move == 6 else 1.0
    ok = all(abs(v - expected) < 0.5 for v in vals)
    warn = "  ⚠ STATE-ABHÄNGIG" if std > 0.05 else ""
    status = "✓ OK" if ok else "✗ falsch"
    print(f"  {action_str(a_idx):<24} {vals[0]:>9.4f} {vals[1]:>9.4f} {vals[2]:>9.4f} "
          f"{std:>7.4f}  {status}{warn}")


# ════════════════════════════════════════════════════════════════
#  TEST 6: RECURRENT INFERENCE — depth-tracking durch MCTS
#  Testet ob der root_idx korrekt berechnet wird:
#    depth=0 → root_idx=0  (Root-Spieler selbst)
#    depth=1 → root_idx=3
#    depth=2 → root_idx=2
#    depth=3 → root_idx=1
# ════════════════════════════════════════════════════════════════
print_header("TEST 6: ROOT_IDX LOGIK  —  Depth → Root-Spieler-Index")
print("  Formel: root_idx = (4 - next_depth_int) % 4")
print()
header_row = f"  {'next_depth':>12} {'root_idx':>9} {'Bedeutung'}"
print(header_row)
print(f"  {'-'*50}")
for d in range(4):
    ri = (4 - d) % 4
    if d == 0:
        bedeutung = "Root-Spieler ist am Zug → values[:,0]"
    elif d == 1:
        bedeutung = "root ist 3 Züge zurück → values[:,3]"
    elif d == 2:
        bedeutung = "root ist 2 Züge zurück → values[:,2]"
    else:
        bedeutung = "root ist 1 Zug zurück  → values[:,1]"
    print(f"  {d:>12}  {'→':>2}  {ri:>5}        {bedeutung}")
print()

# Teste numerisch mit der recurrent_fn Logik
board  = set_pins_on_board(env_base.board, pins_normal)
env    = env_base.replace(board=board, pins=pins_normal, current_player=0)
obs    = encode_board(env)[None, ...]
valid_mask = valid_action(env).flatten()
latent = repr_net.apply(params['representation'], obs)

print("  Simuliere Schritt-für-Schritt Depth-Tracking (depth_delta über 3 Schritte):")
print(f"  {'Step':>4} {'Action':>24} {'dd_sig':>8} {'depth_before':>13} {'depth_after':>12} {'root_idx':>9}")
print(f"  {'-'*75}")

depth_val = jnp.array([[0.0]])  # Start: Root
cur_latent = latent
for step_i, a_idx in enumerate([0, 5, 6]):   # Move1, 6er-Bonus, Move1_wieder
    action = jnp.array([a_idx])
    next_latent, _, _, depth_delta_logit = dynamics_net.apply(params['dynamics'], cur_latent, action)
    dd_sig     = float(jax.nn.sigmoid(depth_delta_logit).squeeze())
    old_depth  = float(depth_val.squeeze())
    next_depth = (depth_val + jax.nn.sigmoid(depth_delta_logit)) % 4.0
    nd_int     = int(round(float(next_depth.squeeze()))) % 4
    root_idx   = (4 - nd_int) % 4
    print(f"  {step_i:>4} {action_str(a_idx):>24} {dd_sig:>8.4f} {old_depth:>13.4f} "
          f"{float(next_depth.squeeze()):>12.4f}  {root_idx:>9}")
    depth_val  = next_depth
    cur_latent = next_latent


# ════════════════════════════════════════════════════════════════
#  TEST 7: MCTS END-TO-END — erkennt MCTS die Winning Action?
# ════════════════════════════════════════════════════════════════
print_header("TEST 7: MCTS END-TO-END  —  findet MCTS die Winning Action?")
print("  Erwartet: Winning Action hat höchsten Q-Value / Action-Weight")
print()

for name, pins, current_player, winning_idx in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env   = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs   = encode_board(env)[None, ...]
    invalid_actions = (~valid_action(env).flatten())[None, :]

    policy_out, mcts_value = run_muzero_mcts(
        params, jax.random.PRNGKey(np.random.randint(0, 10000)), obs, invalid_actions,
        num_simulations=100, max_depth=50, temperature=0.25
    )

    q_values     = policy_out.search_tree.summary().qvalues[0]
    visit_counts = policy_out.search_tree.summary().visit_counts[0]
    weights      = policy_out.action_weights[0]
    top5         = jnp.argsort(-weights)[:5]

    print(f"  --- {name} ---")
    print(f"    MCTS Root-Value (Root-Spieler P{current_player}): {float(mcts_value.squeeze()):+.4f}")

    if winning_idx is not None:
        w_q = float(q_values[winning_idx])
        w_v = int(visit_counts[winning_idx])
        w_w = float(weights[winning_idx])
        print(f"    Winning Action {winning_idx} ({action_str(winning_idx)}): "
              f"Q={w_q:+.4f}, Visits={w_v}, Weight={w_w:.4f}")

    print(f"    Top 5 nach Weight:")
    for rank, a in enumerate(top5):
        a = int(a)
        marker = "  ← WINNING" if a == winning_idx else ""
        print(f"      #{rank+1}: {action_str(a):<24} Q={float(q_values[a]):+.4f}"
              f"  Visits={int(visit_counts[a]):>4}  Weight={float(weights[a]):.4f}{marker}")
    print()


# ════════════════════════════════════════════════════════════════
#  ZUSAMMENFASSUNG
# ════════════════════════════════════════════════════════════════
print_header("ZUSAMMENFASSUNG: FFA-Architektur Check")

print(f"\n  {'Komponente':<30} {'Status'}")
print(f"  {'-'*50}")

# Check 1: Multi-Value Head shape
board  = set_pins_on_board(env_base.board, pins_pre_win)
env    = env_base.replace(board=board, pins=pins_pre_win, current_player=0)
obs    = encode_board(env)[None, ...]
latent = repr_net.apply(params['representation'], obs)
_, values = pred_net.apply(params['prediction'], latent)
shape_ok = values.shape == (1, 4)
print(f"  {'Multi-Value Head shape (1,4)':<30} {'✓ OK (' + str(values.shape) + ')' if shape_ok else '✗ FEHLER: ' + str(values.shape)}")

# Check 2: Dynamics returns 4 outputs
action = jnp.array([0])
dyn_out = dynamics_net.apply(params['dynamics'], latent, action)
n_out = len(dyn_out)
print(f"  {'DynamicsNet Outputs (erwartet 4)':<30} {'✓ OK (' + str(n_out) + ')' if n_out == 4 else '✗ FEHLER: ' + str(n_out) + ' Outputs'}")

# Check 3: depth_delta_logit shape
depth_delta_logit = dyn_out[3]
dd_shape_ok = depth_delta_logit.shape == (1, 1)
print(f"  {'depth_delta_logit shape (1,1)':<30} {'✓ OK (' + str(depth_delta_logit.shape) + ')' if dd_shape_ok else '✗ FEHLER: ' + str(depth_delta_logit.shape)}")

# Check 4: discount logits shape (binary = 2 classes)
disc_logits = dyn_out[2]
disc_shape_ok = disc_logits.shape[-1] == 2
print(f"  {'Discount Logits (2 Klassen)':<30} {'✓ OK (' + str(disc_logits.shape) + ')' if disc_shape_ok else '✗ FEHLER: ' + str(disc_logits.shape)}")

# Check 5: reward logits shape (3 classes)
rew_logits = dyn_out[1]
rew_shape_ok = rew_logits.shape[-1] == 3
print(f"  {'Reward Logits (3 Klassen)':<30} {'✓ OK (' + str(rew_logits.shape) + ')' if rew_shape_ok else '✗ FEHLER: ' + str(rew_logits.shape)}")

# Check 6: Depth-Delta accuracy auf NORMAL state
print(f"  {'Depth-Delta Accuracy':<30} {correct_dd}/{total_dd} ({acc_pct:.0f}%)")

print(f"\n  Architektur-Outputs korrekt: "
      f"{'JA ✓' if all([shape_ok, n_out==4, dd_shape_ok, disc_shape_ok, rew_shape_ok]) else 'NEIN ✗ — siehe Fehler oben'}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Tests abgeschlossen → {output_file}")