"""
Test-Skript: Reward- und Discount-Head Diagnose
=================================================
Testet ob DynNet korrekte Rewards (terminal vs normal) und
Discounts (eigener Zug vs Gegnerzug, 6er-Regel) vorhersagt.

Reward-Test:  Manuell konstruierte Zustände 1 Aktion vor Sieg/Niederlage.
Discount-Test: Prüft ob 6er-Aktionen discount≈+1 und andere discount≈-1 ergeben.
"""
import sys

from jax import numpy as jnp
import jax
import numpy as np
from MADN.deterministic_madn import (
    env_reset, encode_board, valid_action, set_pins_on_board,
    env_step, map_action, get_winner, winning_action
)
from MuZero.muzero_deterministic_madn import (
    repr_net, pred_net, dynamics_net, load_params_from_file, init_muzero_params
)
filename = "Expermiment100"
sys.stdout = open(f"{filename}_reward_discount.txt", "w")
# --- Config ---------------------------------------------------------------------------
PARAM_FILE = f"models/params/{filename}.pkl"  # ← Anpassen!
# PARAM_FILE = None  # ← Uncomment für frische (untrainierte) Params

input_shape = (34, 56)
if PARAM_FILE:
    params = load_params_from_file(PARAM_FILE)
    print(f"Params geladen: {PARAM_FILE}")
else:
    params = init_muzero_params(jax.random.PRNGKey(0), input_shape)
    print("Frische (untrainierte) Params initialisiert")

# --- Basis-Environment ---------------------------------------------------------─
# distance=10 -> board_size=40, total=56
# Player 0: start=0,  target=39, goal=[40,41,42,43]
# Player 1: start=10, target=9,  goal=[44,45,46,47]
# Player 2: start=20, target=19, goal=[48,49,50,51]
# Player 3: start=30, target=29, goal=[52,53,54,55]
env_base = env_reset(
    0, num_players=4,
    layout=jnp.array([True, True, True, True]),
    distance=10, starting_player=0, seed=1,
    enable_teams=True, enable_initial_free_pin=True,
    enable_circular_board=False
)

def action_str(idx):
    """Action Index -> 'Pin X, Move Y' String"""
    pin = idx // 6
    move = idx % 6 + 1
    is_six = " (6er!)" if move == 6 else ""
    return f"Pin {pin}, Move {move}{is_six}"

def print_header(title):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


# ════════════════════════════════════════════════════════════
#  TEST-ZUSTÄNDE: 1 Aktion vor Sieg / Niederlage
# ════════════════════════════════════════════════════════════

# --- PRE-WIN: Player 0 (Team A) gewinnt mit EINER Aktion ---
# Player 0: Pin 0 bei Pos 35 -> Move 5 -> x=35+5-39=1 -> goal[0]=40 O
#           Pin 1,2,3 bereits im Ziel [41,42,43]
# Player 2: Alle 4 Pins im Ziel (Teammate muss auch fertig sein!)
# Player 1,3: irgendwo auf dem Brett (ungefährlich)
pins_pre_win = jnp.array([
    [35, 41, 42, 43],    # P0: Pin 0 bei 35, Rest im Ziel
    [ 5, 15,  7, 12],    # P1: verstreut
    [48, 49, 50, 51],    # P2: alle im Ziel (Team A komplett)
    [25, 28, 33, 30],    # P3: verstreut
], dtype=jnp.int32)
# Winning Action: action_index = 0*6 + (5-1) = 4 -> Pin 0, Move 5

# --- PRE-WIN-6: Sieg durch eine 6er-Aktion ---
# Player 0: Pin 0 bei Pos 34 -> Move 6 -> x=34+6-39=1 -> goal[0]=40 O
# (Interessant weil die 6er-Regel auch einen Extra-Zug gibt)
pins_pre_win_6 = jnp.array([
    [34, 41, 42, 43],    # P0: Pin 0 bei 34, Rest im Ziel
    [ 5, 15,  7, 12],    # P1: verstreut
    [48, 49, 50, 51],    # P2: alle im Ziel
    [25, 28, 33, 30],    # P3: verstreut
], dtype=jnp.int32)
# Winning Action: action_index = 0*6 + (6-1) = 5 -> Pin 0, Move 6

# --- PRE-LOSE: Gegner (Player 1, Team B) gewinnt mit einer Aktion ---
# Wir testen aus Sicht von Player 0.
# Player 1 am Zug -> nach diesem Zug verliert Team A.
# Aber: DynNet sieht nur den latent state von Player 0's Perspektive
# -> Wir testen stattdessen: State wo Player 0 NICHTS tun kann und
#   der Gegner im nächsten Zug gewinnt. Hier zeigen wir den State
#   aus P0-Sicht und prüfen ob Value ≈ -1.
# Alternativ: Normal-Zustand zum Vergleich.
pins_pre_lose = jnp.array([
    [-1, -1, -1,  2],    # P0: 3 Pins im Haus, 1 auf Feld 2
    [ 5, 44, 45, 46],    # P1: Pin 0 bei 5, Rest im Ziel -> Move 5 gewinnt
    [ 1,  3, 20, 21],    # P2: verstreut (Teammate von P0)
    [52, 53, 54, 55],    # P3: alle im Ziel (Team B komplett)
], dtype=jnp.int32)

# --- NORMAL: Mittlerer Spielzustand (Reward sollte ≈ 0 sein) ---
pins_normal = jnp.array([
    [10, 20, 30, -1],    # P0: verstreut
    [15, 25, -1, -1],    # P1: verstreut
    [ 5, 35, -1, -1],    # P2: verstreut
    [ 8, 18, -1, -1],    # P3: verstreut
], dtype=jnp.int32)


# ════════════════════════════════════════════════════════════
#  TEST 1: REWARD HEAD - Terminal vs Normal
# ════════════════════════════════════════════════════════════
print_header("TEST 1: REWARD HEAD - Terminal vs Normal")
print("Erwartet: Winning Action -> reward ≈ +1, Normal Actions -> reward ≈ 0")
print()

test_scenarios = [
    ("PRE-WIN (Move 5 gewinnt)",   pins_pre_win,   0, 4),   # action 4 = Pin 0, Move 5
    ("PRE-WIN-6 (Move 6 gewinnt)", pins_pre_win_6, 0, 5),   # action 5 = Pin 0, Move 6
    ("PRE-LOSE (P0 Perspektive)",  pins_pre_lose,  0, None), # kein spezifischer Gewinn-Zug
    ("NORMAL (Mittelspiel)",       pins_normal,    0, None),
]

for scenario_name, pins, current_player, winning_action_idx in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()

    # Ground Truth: Welche Aktionen gewinnen wirklich?
    win_mask = winning_action(env).flatten()  # (24,) bool

    # RepNet -> Latent
    latent = repr_net.apply(params['representation'], obs)
    _, direct_value = pred_net.apply(params['prediction'], latent)

    print(f"--- {scenario_name} ---")
    print(f"   Direct Value: {float(direct_value.squeeze()):.4f}")
    print(f"   Valid actions: {int(valid_mask.sum())}")
    print(f"   Ground-truth winning actions: {[int(i) for i in jnp.where(win_mask)[0]]}")
    print()
    print(f"   {'Action':<22} {'Valid':>5} {'Wins':>5} {'Reward':>8} {'Discount':>9} {'NextVal':>8}")
    print(f"   {'-'*62}")

    for a_idx in range(24):
        if not valid_mask[a_idx]:
            continue

        action = jnp.array([a_idx])
        next_latent, reward_logits, discount_logits = dynamics_net.apply(
            params['dynamics'], latent, action
        )
        reward = float(jnp.tanh(reward_logits).squeeze())
        discount = float(jnp.tanh(discount_logits).squeeze())
        _, next_value = pred_net.apply(params['prediction'], next_latent)
        next_val = float(next_value.squeeze())

        wins = "O WIN" if win_mask[a_idx] else ""
        marker = " <<<" if a_idx == winning_action_idx else ""

        print(f"   {action_str(a_idx):<22} {'O':>5} {wins:>5} {reward:>+8.4f} {discount:>+9.4f} {next_val:>+8.4f}{marker}")

    print()


# ════════════════════════════════════════════════════════════
#  TEST 2: DISCOUNT HEAD - 6er-Regel
# ════════════════════════════════════════════════════════════
print_header("TEST 2: DISCOUNT HEAD - 6er-Regel")
print("Erwartet: Move 6 (Action 5,11,17,23) -> discount ≈ +1 (eigener Zug)")
print("          Andere Moves              -> discount ≈ -1 (Gegnerzug)")
print()

# Verwende den NORMAL State -> keine besonderen Terminal-Effekte
board = set_pins_on_board(env_base.board, pins_normal)
env = env_base.replace(board=board, pins=pins_normal, current_player=0)
obs = encode_board(env)[None, ...]
valid_mask = valid_action(env).flatten()
latent = repr_net.apply(params['representation'], obs)

print(f"   {'Action':<22} {'Valid':>5} {'Discount':>9} {'Expected':>9} {'Match':>6}")
print(f"   {'-'*55}")

correct = 0
total = 0
for a_idx in range(24):
    if not valid_mask[a_idx]:
        continue

    action = jnp.array([a_idx])
    _, _, discount_logits = dynamics_net.apply(params['dynamics'], latent, action)
    discount = float(jnp.tanh(discount_logits).squeeze())

    move = a_idx % 6 + 1
    expected = +1.0 if move == 6 else -1.0
    match = abs(discount - expected) < 0.5
    correct += int(match)
    total += 1

    print(f"   {action_str(a_idx):<22} {'O':>5} {discount:>+9.4f} {expected:>+9.1f} {'O' if match else 'X':>6}")

print(f"\n   Discount Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")


# ════════════════════════════════════════════════════════════
#  TEST 3: DISCOUNT über verschiedene States
# ════════════════════════════════════════════════════════════
print_header("TEST 3: DISCOUNT - Konsistenz über verschiedene States")
print("Discount hängt NUR von action_embed ab -> sollte über States gleich sein")
print()

all_states = [
    ("PRE-WIN",  pins_pre_win),
    ("PRE-LOSE", pins_pre_lose),
    ("NORMAL",   pins_normal),
]

# Sammle Discounts für jede Action über alle States
discounts_per_action = {a: [] for a in range(24)}

for name, pins in all_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    for a_idx in range(24):
        action = jnp.array([a_idx])
        _, _, discount_logits = dynamics_net.apply(params['dynamics'], latent, action)
        discount = float(jnp.tanh(discount_logits).squeeze())
        discounts_per_action[a_idx].append(discount)

print(f"   {'Action':<22} {'PRE-WIN':>9} {'PRE-LOSE':>9} {'NORMAL':>9} {'Std':>7}")
print(f"   {'-'*58}")

for a_idx in range(24):
    vals = discounts_per_action[a_idx]
    std = np.std(vals)
    marker = " ← WARNUNG: state-abhängig!" if std > 0.1 else ""
    print(f"   {action_str(a_idx):<22} {vals[0]:>+9.4f} {vals[1]:>+9.4f} {vals[2]:>+9.4f} {std:>7.4f}{marker}")


# ════════════════════════════════════════════════════════════
#  TEST 4: ENV GROUND TRUTH - Verifiziere States mit env_step
# ════════════════════════════════════════════════════════════
print_header("TEST 4: GROUND TRUTH - env_step Verifikation")
print("Prüft ob die manuellen States wirklich 1 Aktion vor Sieg/Niederlage sind")
print()

for scenario_name, pins, current_player, expected_win_action in [
    ("PRE-WIN",   pins_pre_win,   0, 4),
    ("PRE-WIN-6", pins_pre_win_6, 0, 5),
]:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    valid_mask = valid_action(env).flatten()
    win_mask = winning_action(env).flatten()

    print(f"--- {scenario_name} ---")
    print(f"   Current Player: {current_player}")
    print(f"   Pins P0: {[int(p) for p in pins[0]]}")
    print(f"   Goal P0: {[int(g) for g in env.goal[0]]}")
    print(f"   Target P0: {int(env.target[0])}")
    print(f"   Valid: {[int(i) for i in jnp.where(valid_mask)[0]]}")
    print(f"   Winning: {[int(i) for i in jnp.where(win_mask)[0]]}")

    if expected_win_action is not None:
        action = map_action(jnp.array(expected_win_action))
        env_after, reward, done = env_step(env, action)
        print(f"   -> Action {expected_win_action} ({action_str(expected_win_action)}): "
              f"reward={int(reward)}, done={bool(done)}")
        print(f"   -> Pins P0 nach Zug: {[int(p) for p in env_after.pins[0]]}")

    print()


# ════════════════════════════════════════════════════════════
#  TEST 5: MCTS Q-VALUES für PRE-WIN State
# ════════════════════════════════════════════════════════════
print_header("TEST 5: MCTS Q-VALUES - Erkennt MCTS den Winning Move?")
print("Erwartet: Winning Action hat höchsten Q-Value / höchstes Gewicht")
print()

from MuZero.muzero_deterministic_madn import run_muzero_mcts

for scenario_name, pins, current_player, winning_action_idx in [
    ("PRE-WIN",   pins_pre_win,   0, 4),
    ("PRE-WIN-6", pins_pre_win_6, 0, 5),
    ("PRE-LOSE",  pins_pre_lose,  0, None),
    ("NORMAL",    pins_normal,    0, None),
]:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs = encode_board(env)[None, ...]
    invalid_actions = (~valid_action(env).flatten())[None, :]

    policy_out, mcts_value = run_muzero_mcts(
        params, jax.random.PRNGKey(42), obs, invalid_actions,
        num_simulations=200, max_depth=50, temperature=0.25
    )

    # Q-Values aus dem Search Tree
    q_values = policy_out.search_tree.summary().qvalues[0]  # (24,)
    visit_counts = policy_out.search_tree.summary().visit_counts[0]  # (24,)
    weights = policy_out.action_weights[0]  # (24,)

    top5 = jnp.argsort(-weights)[:5]

    print(f"--- {scenario_name} ---")
    print(f"   MCTS Value: {float(mcts_value.squeeze()):.4f}")
    if winning_action_idx is not None:
        w_q = float(q_values[winning_action_idx])
        w_visits = int(visit_counts[winning_action_idx])
        w_weight = float(weights[winning_action_idx])
        print(f"   Winning Action {winning_action_idx} ({action_str(winning_action_idx)}): "
              f"Q={w_q:+.4f}, Visits={w_visits}, Weight={w_weight:.3f}")

    print(f"   Top 5 by Weight:")
    for rank, a in enumerate(top5):
        a = int(a)
        marker = " < WINNING" if a == winning_action_idx else ""
        print(f"     #{rank+1}: {action_str(a):<22} Q={float(q_values[a]):+.4f}, "
              f"Visits={int(visit_counts[a]):>3}, Weight={float(weights[a]):.3f}{marker}")
    print()


# ════════════════════════════════════════════════════════════
#  ZUSAMMENFASSUNG
# ════════════════════════════════════════════════════════════
print_header("ZUSAMMENFASSUNG")

for scenario_name, pins, current_player, winning_action_idx in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    latent = repr_net.apply(params['representation'], obs)

    # Sammle Rewards für gültige Actions
    rewards = []
    for a_idx in range(24):
        if not valid_mask[a_idx]:
            continue
        action = jnp.array([a_idx])
        _, reward_logits, _ = dynamics_net.apply(params['dynamics'], latent, action)
        rewards.append(float(jnp.tanh(reward_logits).squeeze()))

    rewards = np.array(rewards)
    win_reward = None
    if winning_action_idx is not None:
        action = jnp.array([winning_action_idx])
        _, reward_logits, _ = dynamics_net.apply(params['dynamics'], latent, action)
        win_reward = float(jnp.tanh(reward_logits).squeeze())

    print(f"\n  {scenario_name}:")
    print(f"    Reward range:  [{rewards.min():+.4f}, {rewards.max():+.4f}]")
    print(f"    Reward mean:   {rewards.mean():+.4f}")
    print(f"    Reward std:    {rewards.std():.4f}")
    if win_reward is not None:
        print(f"    Winning reward: {win_reward:+.4f}  {'O GOOD' if win_reward > 0.3 else 'X ZU NIEDRIG - Reward Head lernt nicht!'}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Tests abgeschlossen.")
