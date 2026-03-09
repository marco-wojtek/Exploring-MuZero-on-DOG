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

# ═══════════════════════════════════════════════════════════════
#  CONFIG — hier anpassen!
# ═══════════════════════════════════════════════════════════════
filename = "Experiment_33_100"
PARAM_FILE = f"models/params/{filename}.pkl"  # ← Anpassen!
# PARAM_FILE = None  # ← Uncomment für frische (untrainierte) Params

# True  = Kategorisches System (3 Klassen: {-1, 0, +1}, softmax → Erwartungswert)
# False = Klassisches System  (1 Logit, tanh → [-1, +1])
USE_CLASSIFICATION = True

sys.stdout = open(f"{filename}_reward_discount.txt", "w")

# --- Logit → Scalar Konvertierung -------------------------------------------------
SUPPORT = jnp.array([-1.0, 0.0, 1.0])

def logits_to_scalar(logits):
    """Konvertiert Reward/Discount-Logits zu Skalar, je nach Modus."""
    if USE_CLASSIFICATION:
        probs = jax.nn.softmax(logits, axis=-1)
        return float(jnp.sum(probs * SUPPORT, axis=-1).squeeze())
    else:
        return float(jnp.tanh(logits).squeeze())

def logits_to_probs(logits):
    """Gibt Klassen-Wahrscheinlichkeiten zurück (nur Klassifikation)."""
    if USE_CLASSIFICATION:
        probs = jax.nn.softmax(logits, axis=-1)
        return probs.squeeze()
    return None

# --- Params laden ------------------------------------------------------------------
input_shape = (34, 56)
if PARAM_FILE:
    params = load_params_from_file(PARAM_FILE)
    print(f"Params geladen: {PARAM_FILE}")
else:
    params = init_muzero_params(jax.random.PRNGKey(0), input_shape)
    print("Frische (untrainierte) Params initialisiert")

mode_str = "KLASSIFIKATION (3 Klassen: {-1, 0, +1})" if USE_CLASSIFICATION else "KLASSISCH (tanh)"
print(f"Modus: {mode_str}\n")

# --- Basis-Environment ---------------------------------------------------------─
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

pins_pre_win = jnp.array([
    [35, 41, 42, 43],
    [ 5, 15,  7, 12],
    [48, 49, 50, 51],
    [25, 28, 33, 30],
], dtype=jnp.int32)

pins_pre_win_6 = jnp.array([
    [34, 41, 42, 43],
    [ 5, 15,  7, 12],
    [48, 49, 50, 51],
    [25, 28, 33, 30],
], dtype=jnp.int32)

pins_pre_lose = jnp.array([
    [-1, -1, -1,  2],
    [ 5, 44, 45, 46],
    [ 1,  3, 20, 21],
    [52, 53, 54, 55],
], dtype=jnp.int32)

pins_normal = jnp.array([
    [10, 20, 30, -1],
    [15, 25, -1, -1],
    [ 5, 35, -1, -1],
    [ 8, 18, -1, -1],
], dtype=jnp.int32)


# ════════════════════════════════════════════════════════════
#  TEST 1: REWARD HEAD - Terminal vs Normal
# ════════════════════════════════════════════════════════════
print_header("TEST 1: REWARD HEAD - Terminal vs Normal")
print("Erwartet: Winning Action -> reward ≈ +1, Normal Actions -> reward ≈ 0")
if USE_CLASSIFICATION:
    print("Modus: Klassifikation — Klasse 0=-1, Klasse 1=0, Klasse 2=+1")
print()

test_scenarios = [
    ("PRE-WIN (Move 5 gewinnt)",   pins_pre_win,   0, 4),
    ("PRE-WIN-6 (Move 6 gewinnt)", pins_pre_win_6, 0, 5),
    ("PRE-LOSE (P0 Perspektive)",  pins_pre_lose,  0, None),
    ("NORMAL (Mittelspiel)",       pins_normal,    0, None),
]

for scenario_name, pins, current_player, winning_action_idx in test_scenarios:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    win_mask = winning_action(env).flatten()

    latent = repr_net.apply(params['representation'], obs)
    _, direct_value = pred_net.apply(params['prediction'], latent)

    print(f"--- {scenario_name} ---")
    print(f"   Direct Value: {float(direct_value.squeeze()):.4f}")
    print(f"   Valid actions: {int(valid_mask.sum())}")
    print(f"   Ground-truth winning actions: {[int(i) for i in jnp.where(win_mask)[0]]}")
    print()

    if USE_CLASSIFICATION:
        header = f"   {'Action':<22} {'Valid':>5} {'Wins':>5} {'Reward':>8} {'P(-1)':>6} {'P(0)':>6} {'P(+1)':>6} {'Disc':>8} {'NextVal':>8}"
    else:
        header = f"   {'Action':<22} {'Valid':>5} {'Wins':>5} {'Reward':>8} {'Discount':>9} {'NextVal':>8}"
    print(header)
    print(f"   {'-' * (len(header) - 3)}")

    for a_idx in range(24):
        if not valid_mask[a_idx]:
            continue

        action = jnp.array([a_idx])
        next_latent, reward_logits, discount_logits = dynamics_net.apply(
            params['dynamics'], latent, action
        )
        reward = logits_to_scalar(reward_logits)
        discount = logits_to_scalar(discount_logits)
        _, next_value = pred_net.apply(params['prediction'], next_latent)
        next_val = float(next_value.squeeze())

        wins = "O WIN" if win_mask[a_idx] else ""
        marker = " <<<" if a_idx == winning_action_idx else ""

        if USE_CLASSIFICATION:
            rp = logits_to_probs(reward_logits)
            print(f"   {action_str(a_idx):<22} {'O':>5} {wins:>5} {reward:>+8.4f} "
                  f"{float(rp[0]):>6.3f} {float(rp[1]):>6.3f} {float(rp[2]):>6.3f} "
                  f"{discount:>+8.4f} {next_val:>+8.4f}{marker}")
        else:
            print(f"   {action_str(a_idx):<22} {'O':>5} {wins:>5} {reward:>+8.4f} "
                  f"{discount:>+9.4f} {next_val:>+8.4f}{marker}")

    print()


# ════════════════════════════════════════════════════════════
#  TEST 2: DISCOUNT HEAD - 6er-Regel + Terminal
# ════════════════════════════════════════════════════════════
print_header("TEST 2: DISCOUNT HEAD - 6er-Regel" + (" + Terminal" if USE_CLASSIFICATION else ""))
print("Erwartet: Move 6 (Action 5,11,17,23) -> discount ≈ +1 (eigener Zug)")
print("          Andere Moves              -> discount ≈ -1 (Gegnerzug)")
if USE_CLASSIFICATION:
    print("          Terminal (Winning Move)   -> discount ≈  0 (Spiel vorbei)")
print()

board = set_pins_on_board(env_base.board, pins_normal)
env = env_base.replace(board=board, pins=pins_normal, current_player=0)
obs = encode_board(env)[None, ...]
valid_mask = valid_action(env).flatten()
latent = repr_net.apply(params['representation'], obs)

if USE_CLASSIFICATION:
    header = f"   {'Action':<22} {'Valid':>5} {'Disc':>8} {'P(-1)':>6} {'P(0)':>6} {'P(+1)':>6} {'Exp':>6} {'Match':>6}"
else:
    header = f"   {'Action':<22} {'Valid':>5} {'Discount':>9} {'Expected':>9} {'Match':>6}"
print(header)
print(f"   {'-' * (len(header) - 3)}")

correct = 0
total = 0
for a_idx in range(24):
    if not valid_mask[a_idx]:
        continue

    action = jnp.array([a_idx])
    _, _, discount_logits = dynamics_net.apply(params['dynamics'], latent, action)
    discount = logits_to_scalar(discount_logits)

    move = a_idx % 6 + 1
    expected = +1.0 if move == 6 else -1.0
    match = abs(discount - expected) < 0.5
    correct += int(match)
    total += 1

    if USE_CLASSIFICATION:
        dp = logits_to_probs(discount_logits)
        print(f"   {action_str(a_idx):<22} {'O':>5} {discount:>+8.4f} "
              f"{float(dp[0]):>6.3f} {float(dp[1]):>6.3f} {float(dp[2]):>6.3f} "
              f"{expected:>+6.1f} {'O' if match else 'X':>6}")
    else:
        print(f"   {action_str(a_idx):<22} {'O':>5} {discount:>+9.4f} "
              f"{expected:>+9.1f} {'O' if match else 'X':>6}")

print(f"\n   Discount Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")


# ════════════════════════════════════════════════════════════
#  TEST 3: DISCOUNT über verschiedene States
# ════════════════════════════════════════════════════════════
print_header("TEST 3: DISCOUNT - Konsistenz über verschiedene States")
print("Discount hängt NUR von action_embed ab -> sollte über States gleich sein")
if USE_CLASSIFICATION:
    print("(Bei Klassifikation: Terminal-States könnten abweichen → discount≈0)")
print()

all_states = [
    ("PRE-WIN",  pins_pre_win),
    ("PRE-LOSE", pins_pre_lose),
    ("NORMAL",   pins_normal),
]

discounts_per_action = {a: [] for a in range(24)}

for name, pins in all_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    latent = repr_net.apply(params['representation'], obs)

    for a_idx in range(24):
        action = jnp.array([a_idx])
        _, _, discount_logits = dynamics_net.apply(params['dynamics'], latent, action)
        discount = logits_to_scalar(discount_logits)
        discounts_per_action[a_idx].append(discount)

print(f"   {'Action':<22} {'PRE-WIN':>9} {'PRE-LOSE':>9} {'NORMAL':>9} {'Std':>7}")
print(f"   {'-'*58}")

for a_idx in range(24):
    vals = discounts_per_action[a_idx]
    std = np.std(vals)
    marker = " ← WARNUNG: state-abhängig!" if std > 0.1 else ""
    print(f"   {action_str(a_idx):<22} {vals[0]:>+9.4f} {vals[1]:>+9.4f} {vals[2]:>+9.4f} {std:>7.4f}{marker}")


# ════════════════════════════════════════════════════════════
#  TEST 3b: REWARD KLASSEN-VERTEILUNG (nur Klassifikation)
# ════════════════════════════════════════════════════════════
if USE_CLASSIFICATION:
    print_header("TEST 3b: REWARD KLASSEN-VERTEILUNG über States")
    print("Zeigt P(-1), P(0), P(+1) für jede Action in verschiedenen States")
    print("Erwartet: PRE-WIN Winning Action → P(+1) hoch")
    print()

    for scenario_name, pins, current_player, winning_action_idx in test_scenarios:
        board = set_pins_on_board(env_base.board, pins)
        env = env_base.replace(board=board, pins=pins, current_player=current_player)
        obs = encode_board(env)[None, ...]
        valid_mask = valid_action(env).flatten()
        win_mask = winning_action(env).flatten()
        latent = repr_net.apply(params['representation'], obs)

        print(f"--- {scenario_name} ---")
        print(f"   {'Action':<22} {'Wins':>5} {'E[R]':>7} {'P(-1)':>7} {'P(0)':>7} {'P(+1)':>7}")
        print(f"   {'-'*57}")

        for a_idx in range(24):
            if not valid_mask[a_idx]:
                continue
            action = jnp.array([a_idx])
            _, reward_logits, _ = dynamics_net.apply(params['dynamics'], latent, action)
            reward = logits_to_scalar(reward_logits)
            rp = logits_to_probs(reward_logits)
            wins = "O WIN" if win_mask[a_idx] else ""
            marker = " <<<" if a_idx == winning_action_idx else ""
            print(f"   {action_str(a_idx):<22} {wins:>5} {reward:>+7.4f} "
                  f"{float(rp[0]):>7.4f} {float(rp[1]):>7.4f} {float(rp[2]):>7.4f}{marker}")
        print()


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
#  TEST 5: DIREKTE POLICY PREDICTION (ohne MCTS)
# ════════════════════════════════════════════════════════════
print_header("TEST 5: DIREKTE POLICY - Prior Logits vs MCTS")
print("Zeigt die rohe Policy-Ausgabe des PredictionNetwork (ohne MCTS)")
print("Vergleich: Wo legt das Netz von sich aus Gewicht hin?")
print()

for scenario_name, pins, current_player, winning_action_idx in [
    ("PRE-WIN",   pins_pre_win,   0, 4),
    ("PRE-WIN-6", pins_pre_win_6, 0, 5),
    ("PRE-LOSE",  pins_pre_lose,  0, None),
    ("NORMAL",    pins_normal,    0, None),
]:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=current_player)
    obs = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    latent = repr_net.apply(params['representation'], obs)
    prior_logits, value = pred_net.apply(params['prediction'], latent)
    prior_logits = prior_logits[0]
    value = float(value.squeeze())

    # Maskiere invalide Actions für Softmax
    masked_logits = jnp.where(valid_mask, prior_logits, -1e9)
    prior_probs = jax.nn.softmax(masked_logits)

    top5 = jnp.argsort(-prior_probs)[:5]

    print(f"--- {scenario_name} ---")
    print(f"   Direct Value: {value:.4f}")
    print(f"   Top 5 by Prior Probability:")
    for rank, a in enumerate(top5):
        a = int(a)
        marker = " < WINNING" if a == winning_action_idx else ""
        print(f"     #{rank+1}: {action_str(a):<22} P={float(prior_probs[a]):.4f}, "
              f"Logit={float(prior_logits[a]):+.2f}{marker}")
    if winning_action_idx is not None:
        win_rank = int(jnp.sum(prior_probs > prior_probs[winning_action_idx])) + 1
        print(f"   Winning Action Rank: #{win_rank} "
              f"(P={float(prior_probs[winning_action_idx]):.4f}, "
              f"Logit={float(prior_logits[winning_action_idx]):+.2f})")
    print()


# ════════════════════════════════════════════════════════════
#  TEST 6: MCTS Q-VALUES für PRE-WIN State
# ════════════════════════════════════════════════════════════
print_header("TEST 6: MCTS Q-VALUES - Erkennt MCTS den Winning Move?")
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
        params, jax.random.PRNGKey(np.random.randint(0, 10000)), obs, invalid_actions,
        num_simulations=100, max_depth=50, temperature=0.25
    )

    q_values = policy_out.search_tree.summary().qvalues[0]
    visit_counts = policy_out.search_tree.summary().visit_counts[0]
    weights = policy_out.action_weights[0]

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

    rewards = []
    for a_idx in range(24):
        if not valid_mask[a_idx]:
            continue
        action = jnp.array([a_idx])
        _, reward_logits, _ = dynamics_net.apply(params['dynamics'], latent, action)
        rewards.append(logits_to_scalar(reward_logits))

    rewards = np.array(rewards)
    win_reward = None
    if winning_action_idx is not None:
        action = jnp.array([winning_action_idx])
        _, reward_logits, _ = dynamics_net.apply(params['dynamics'], latent, action)
        win_reward = logits_to_scalar(reward_logits)

    print(f"\n  {scenario_name}:")
    print(f"    Reward range:  [{rewards.min():+.4f}, {rewards.max():+.4f}]")
    print(f"    Reward mean:   {rewards.mean():+.4f}")
    print(f"    Reward std:    {rewards.std():.4f}")
    if win_reward is not None:
        print(f"    Winning reward: {win_reward:+.4f}  "
              f"{'O GOOD' if win_reward > 0.3 else 'X ZU NIEDRIG - Reward Head lernt nicht!'}")

        if USE_CLASSIFICATION:
            action = jnp.array([winning_action_idx])
            _, reward_logits, discount_logits = dynamics_net.apply(
                params['dynamics'], latent, action
            )
            rp = logits_to_probs(reward_logits)
            dp = logits_to_probs(discount_logits)
            print(f"    Winning R-Probs: P(-1)={float(rp[0]):.4f}, P(0)={float(rp[1]):.4f}, P(+1)={float(rp[2]):.4f}")
            print(f"    Winning D-Probs: P(-1)={float(dp[0]):.4f}, P(0)={float(dp[1]):.4f}, P(+1)={float(dp[2]):.4f}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Tests abgeschlossen.")