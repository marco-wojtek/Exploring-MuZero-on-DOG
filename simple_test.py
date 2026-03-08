from jax import numpy as jnp
import jax
import numpy as np
import os, sys
from MADN.deterministic_madn import env_reset, encode_board, valid_action, set_pins_on_board
from MuZero.muzero_deterministic_madn import (
    run_muzero_mcts, load_params_from_file, repr_net, pred_net, 
    dynamics_net, init_muzero_params
)
sys.stdout = open("simple_test_output.txt", "w")
input_shape = (34, 56)
params = load_params_from_file("models/params/TEST6_80.pkl")
# params = init_muzero_params(jax.random.PRNGKey(0), input_shape)

env_base = env_reset(0, num_players=4, layout=jnp.array([True, True, True, True]),
    distance=10, starting_player=0, seed=1, enable_teams=True,
    enable_initial_free_pin=True, enable_circular_board=False)

# ============================================================
# TEST STATES
# ============================================================
pins_winning = jnp.array([
    [38, 41, 42, 40],
    [1, 44, 12, 0],
    [48, 49, 50, 51],
    [52, 55, 54, 53],
])

pins_losing = jnp.array([
    [-1, -1, -1, 11],
    [39, 44, 45, 46],
    [1, 2, 3, 4],
    [52, 55, 54, 53],
])

pins_mid = jnp.array([
    [10, 20, 30, -1],
    [15, 25, -1, -1],
    [5, 35, -1, -1],
    [8, 18, -1, -1],
])

test_states = [
    ("WINNING", pins_winning),
    ("LOSING", pins_losing),
    ("MID", pins_mid),
]

# ============================================================
# TEST 1: Direct Value (RepNet + PredNet)
# ============================================================
print("=" * 70)
print("TEST 1: DIRECT VALUES (RepNet → PredNet)")
print("=" * 70)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    
    latent = repr_net.apply(params['representation'], obs)
    policy_logits, direct_value = pred_net.apply(params['prediction'], latent)
    
    # Policy Analyse
    valid_mask = valid_action(env).flatten()
    masked_logits = jnp.where(valid_mask, policy_logits[0], -1e9)
    policy_probs = jax.nn.softmax(masked_logits)
    top_actions = jnp.argsort(-policy_probs)[:5]
    
    print(f"\n{name}:")
    print(f"  Direct Value:    {float(direct_value.squeeze()):.4f}")
    print(f"  Latent mean:     {float(latent.mean()):.4f}")
    print(f"  Latent std:      {float(latent.std()):.4f}")
    print(f"  Latent min:      {float(latent.min()):.4f}")
    print(f"  Latent max:      {float(latent.max()):.4f}")
    print(f"  Valid actions:   {int(valid_mask.sum())}")
    print(f"  Top 5 actions:   {[int(a) for a in top_actions]}")
    print(f"  Top 5 probs:     {[f'{float(policy_probs[a]):.3f}' for a in top_actions]}")
    print(f"  Policy entropy:  {float(-jnp.sum(policy_probs * jnp.log(policy_probs + 1e-8))):.4f}")

# ============================================================
# TEST 2: DynNet Latent Consistency
# Prüft ob DynNet-Latents ähnlich wie RepNet-Latents aussehen
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: DYNAMICS NETWORK CONSISTENCY")
print("=" * 70)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    
    latent_rep = repr_net.apply(params['representation'], obs)
    _, value_rep = pred_net.apply(params['prediction'], latent_rep)
    
    # Wende DynNet mit verschiedenen Actions an
    valid_actions = jnp.where(valid_mask, size=24, fill_value=-1)[0]
    valid_actions = valid_actions[valid_actions >= 0]
    
    print(f"\n{name} (RepNet Value: {float(value_rep.squeeze()):.4f}):")
    print(f"  RepNet Latent:  mean={float(latent_rep.mean()):.4f}, std={float(latent_rep.std()):.4f}")
    
    dyn_values = []
    dyn_means = []
    dyn_stds = []
    
    for action_idx in range(24):
        if not valid_mask[action_idx]:
            continue
            
        action = jnp.array([action_idx])
        latent_dyn, _, _ = dynamics_net.apply(
            params['dynamics'], latent_rep, action
        )
        _, value_dyn = pred_net.apply(params['prediction'], latent_dyn)
        
        dyn_values.append(float(value_dyn.squeeze()))
        dyn_means.append(float(latent_dyn.mean()))
        dyn_stds.append(float(latent_dyn.std()))
    
    dyn_values = np.array(dyn_values)
    dyn_means = np.array(dyn_means)
    dyn_stds = np.array(dyn_stds)
    
    print(f"  DynNet Latent:  mean={dyn_means.mean():.4f}±{dyn_means.std():.4f}, "
          f"std={dyn_stds.mean():.4f}±{dyn_stds.std():.4f}")
    print(f"  DynNet Values:  mean={dyn_values.mean():.4f}, "
          f"min={dyn_values.min():.4f}, max={dyn_values.max():.4f}")
    print(f"  Value Shift:    {float(value_rep.squeeze()):.4f} → {dyn_values.mean():.4f} "
          f"(Δ={dyn_values.mean() - float(value_rep.squeeze()):.4f})")

# ============================================================
# TEST 3: Multi-Step DynNet (Latent Drift über mehrere Schritte)
# Prüft ob Latent States über Unroll-Steps stabil bleiben
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: MULTI-STEP DYNAMICS (Latent Drift)")
print("=" * 70)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    
    # Erste gültige Action für wiederholte Anwendung
    first_valid = int(jnp.argmax(valid_mask))
    action = jnp.array([first_valid])
    
    latent = repr_net.apply(params['representation'], obs)
    _, value_0 = pred_net.apply(params['prediction'], latent)
    
    print(f"\n{name} (Action={first_valid} repeated):")
    print(f"  Step 0 (RepNet): Value={float(value_0.squeeze()):.4f}, "
          f"mean={float(latent.mean()):.4f}, std={float(latent.std()):.4f}, "
          f"min={float(latent.min()):.4f}, max={float(latent.max()):.4f}")
    
    for step in range(1, 11):
        latent, _, _ = dynamics_net.apply(params['dynamics'], latent, action)
        _, value_k = pred_net.apply(params['prediction'], latent)
        
        print(f"  Step {step:2d} (DynNet): Value={float(value_k.squeeze()):.4f}, "
              f"mean={float(latent.mean()):.4f}, std={float(latent.std()):.4f}, "
              f"min={float(latent.min()):.4f}, max={float(latent.max()):.4f}")

# ============================================================
# TEST 4: MCTS Analyse
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: MCTS VALUES")
print("=" * 70)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    invalid_actions = (~valid_action(env).flatten())[None, :]
    
    latent = repr_net.apply(params['representation'], obs)
    _, direct_value = pred_net.apply(params['prediction'], latent)
    
    # MCTS mit verschiedenen Simulationsbudgets
    print(f"\n{name} (Direct Value: {float(direct_value.squeeze()):.4f}):")
    
    policy_out, mcts_value = run_muzero_mcts(
        params, jax.random.PRNGKey(42), obs, invalid_actions,
        num_simulations=100, max_depth=50, temperature=0.25
    )
    
    top3 = jnp.argsort(-policy_out.action_weights[0])[:3]
    top3_weights = [float(policy_out.action_weights[0, a]) for a in top3]
    
    print(f"  Sims={100:3d}: MCTS Value={float(mcts_value.squeeze()):.4f}, "
            f"Top3={[int(a) for a in top3]} weights={[f'{w:.3f}' for w in top3_weights]}")

# ============================================================
# TEST 5: Seed Robustheit (gleicher State, verschiedene MCTS Seeds)
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: MCTS SEED ROBUSTNESS")
print("=" * 70)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    invalid_actions = (~valid_action(env).flatten())[None, :]
    
    mcts_values = []
    for seed in range(10):
        policy_out, mcts_value = run_muzero_mcts(
            params, jax.random.PRNGKey(seed), obs, invalid_actions,
            num_simulations=100, max_depth=10, temperature=0.25
        )
        mcts_values.append(float(mcts_value.squeeze()))
    
    mcts_values = np.array(mcts_values)
    print(f"\n{name}:")
    print(f"  MCTS Values: mean={mcts_values.mean():.4f}, "
          f"std={mcts_values.std():.4f}, "
          f"min={mcts_values.min():.4f}, max={mcts_values.max():.4f}")

# ============================================================
# TEST 6: Cosine Similarity RepNet vs DynNet Latents
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: LATENT SPACE SIMILARITY (RepNet vs DynNet)")
print("=" * 70)

def cosine_sim(a, b):
    return float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8))

def l2_dist(a, b):
    return float(jnp.linalg.norm(a - b))

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    valid_mask = valid_action(env).flatten()
    
    latent_rep = repr_net.apply(params['representation'], obs)
    
    first_valid = int(jnp.argmax(valid_mask))
    action = jnp.array([first_valid])
    
    print(f"\n{name} (Action={first_valid}):")
    
    latent = latent_rep
    for step in range(1, 6):
        latent, _, _ = dynamics_net.apply(params['dynamics'], latent, action)
        
        cos = cosine_sim(latent_rep[0], latent[0])
        l2 = l2_dist(latent_rep[0], latent[0])
        
        print(f"  Step {step}: Cosine={cos:.4f}, L2={l2:.4f}")

# ============================================================
# TEST 7: MCTS Simulations Analyse
# ============================================================
print("\n" + "=" * 70)
print("TEST 7: MCTS VALUES")
print("=" * 70)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    invalid_actions = (~valid_action(env).flatten())[None, :]
    
    latent = repr_net.apply(params['representation'], obs)
    _, direct_value = pred_net.apply(params['prediction'], latent)
    
    # MCTS mit verschiedenen Simulationsbudgets
    print(f"\n{name} (Direct Value: {float(direct_value.squeeze()):.4f}):")
    
    for num_sims in [10, 50, 100, 200]:
        policy_out, mcts_value = run_muzero_mcts(
            params, jax.random.PRNGKey(42), obs, invalid_actions,
            num_simulations=num_sims, max_depth=10, temperature=0.25
        )
        
        top3 = jnp.argsort(-policy_out.action_weights[0])[:3]
        top3_weights = [float(policy_out.action_weights[0, a]) for a in top3]
        
        print(f"  Sims={num_sims:3d}: MCTS Value={float(mcts_value.squeeze()):.4f}, "
              f"Top3={[int(a) for a in top3]} weights={[f'{w:.3f}' for w in top3_weights]}")

# ============================================================
# TEST 8: MCTS Depth Analysis
# ============================================================
print("\n" + "=" * 70)
print("TEST 8: MCTS DEPTH IMPACT")
print("=" * 70)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    invalid_actions = (~valid_action(env).flatten())[None, :]
    
    latent = repr_net.apply(params['representation'], obs)
    _, direct_value = pred_net.apply(params['prediction'], latent)
    
    print(f"\n{name} (Direct Value: {float(direct_value.squeeze()):.4f}):")
    
    for max_depth in [1, 3, 5, 10, 20, 50]:
        policy_out, mcts_value = run_muzero_mcts(
            params, jax.random.PRNGKey(42), obs, invalid_actions,
            num_simulations=100, max_depth=max_depth, temperature=0.25
        )
        print(f"  Depth={max_depth:2d}: MCTS Value={float(mcts_value.squeeze()):.4f}")
# ============================================================
# ZUSAMMENFASSUNG
# ============================================================
print("\n" + "=" * 70)
print(" SUMMARY: KEY METRICS")
print("=" * 70)
print(f"{'State':<10} {'Direct':>8} {'MCTS(100)':>10} {'Δ(MCTS-Dir)':>12} "
      f"{'DynNet Δ':>10} {'Lat std':>8}")
print("-" * 60)

for name, pins in test_states:
    board = set_pins_on_board(env_base.board, pins)
    env = env_base.replace(board=board, pins=pins, current_player=0)
    obs = encode_board(env)[None, ...]
    invalid_actions = (~valid_action(env).flatten())[None, :]
    valid_mask = valid_action(env).flatten()
    
    latent = repr_net.apply(params['representation'], obs)
    _, direct_value = pred_net.apply(params['prediction'], latent)
    
    # DynNet 1-Step Value
    first_valid = int(jnp.argmax(valid_mask))
    action = jnp.array([first_valid])
    latent_dyn, _, _ = dynamics_net.apply(params['dynamics'], latent, action)
    _, dyn_value = pred_net.apply(params['prediction'], latent_dyn)
    
    policy_out, mcts_value = run_muzero_mcts(
        params, jax.random.PRNGKey(42), obs, invalid_actions,
        num_simulations=100, max_depth=10, temperature=0.25
    )
    
    d = float(direct_value.squeeze())
    m = float(mcts_value.squeeze())
    dv = float(dyn_value.squeeze())
    s = float(latent.std())
    
    print(f"{name:<10} {d:>8.4f} {m:>10.4f} {m-d:>12.4f} {dv-d:>10.4f} {s:>8.4f}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Output written to simple_test_output.txt")
# from jax import numpy as jnp
# import jax
# from MADN.deterministic_madn import env_reset, encode_board, valid_action, set_pins_on_board
# from MuZero.muzero_deterministic_madn import run_muzero_mcts, load_params_from_file, repr_net, pred_net, dynamics_net, init_muzero_params

# input_shape = (34, 56)
# # params = load_params_from_file("models/params/TEST5_50.pkl")
# params = init_muzero_params(jax.random.PRNGKey(0), input_shape)
# env_base = env_reset(0, num_players=4, layout=jnp.array([True, True, True, True]),
#     distance=10, starting_player=0, seed=1, enable_teams=True,
#     enable_initial_free_pin=True, enable_circular_board=False)

# # State 1: Spieler 0 kurz vor Sieg
# pins_winning = jnp.array([
#     [38, 41, 42, 40],   # Spieler 0: kurz vor Ziel ← Sollte Value ≈ +1
#     [1, 44, 12, 0],
#     [48, 49, 50, 51],
#     [52, 55, 54, 53],
# ])

# # State 2: Spieler 0 verliert sicher
# pins_losing = jnp.array([
#     [-1, -1, -1, 11],   # Spieler 0: alle im Haus ← Sollte Value ≈ -1
#     [39, 44, 45, 46],   # Spieler 1: kurz vor Ziel
#     [1, 2, 3, 4],
#     [52, 55, 54, 53],
# ])

# # State 3: Zufälliger Mittelzustand
# pins_mid = jnp.array([
#     [10, 20, 30, -1],   # Spieler 0: mittelmäßig
#     [15, 25, -1, -1],
#     [5, 35, -1, -1],
#     [8, 18, -1, -1],
# ])

# print("\n=== TESTING SPECIFIC STATES V2 50 iterations===")
# for name, pins in [("WINNING", pins_winning), ("LOSING", pins_losing), ("MID", pins_mid)]:
#     board = set_pins_on_board(env_base.board, pins)
#     env = env_base.replace(board=board, pins=pins, current_player=0)
#     obs = encode_board(env)[None, ...]
#     invalid_actions = (~valid_action(env).flatten())[None, :]

#     latent = repr_net.apply(params['representation'], obs)
#     policy_logits, direct_value = pred_net.apply(params['prediction'], latent)

#     policy_out, mcts_value = run_muzero_mcts(
#         params, jax.random.PRNGKey(42), obs, invalid_actions,
#         num_simulations=100, max_depth=10, temperature=0.25
#     )

#     print(f"\n{name}:")
#     print(f"  Direct Value:  {float(direct_value.squeeze()):.4f}")
#     print(f"  MCTS Value:    {float(mcts_value.squeeze()):.4f}")
#     print(f"  Latent mean:   {float(latent.mean()):.4f}")
#     print(f"  Latent std:    {float(latent.std()):.4f}")
#     print(f"  Direct best action: {jnp.argmax(policy_logits[0])}")
#     print(f"  Top 3 Actions: {jnp.argsort(-policy_out.action_weights[0])[:3]}")
