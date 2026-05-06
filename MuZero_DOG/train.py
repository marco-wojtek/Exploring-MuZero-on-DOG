import gc
import os
import datetime
import numpy as np
import jax
from time import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
import os, sys
import pickle
import wandb
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MuZero_DOG.muzero_dog import repr_net, dynamics_net, pred_net, init_muzero_params, load_params_from_file
from DOG.dog import env_reset, encode_board, encode_board_with_belief
from MuZero_DOG.vec_replay_buffer import VectorizedReplayBuffer
from MuZero_DOG.game_agent import play_n_games_v3
# from MuZero_DOG.evaluate_agent import evaluate_agent_parallel

def get_temperature(iteration, total_iterations):
    """Phasenbasiert: nur 4 verschiedene Werte, garantiert gleiche Float-Instanz"""
    phase = int(iteration / total_iterations * len(TEMPERATURE_SCHEDULE))
    phase = min(phase, len(TEMPERATURE_SCHEDULE) - 1)
    return TEMPERATURE_SCHEDULE[phase]

def balanced_loss(ce, is_rare, mask, n_valid, w_rare=1.0, w_common=0.1):
    masked_rare = mask * is_rare
    n_rare    = jnp.maximum(jnp.sum(masked_rare), 1.0)
    n_common  = jnp.maximum(n_valid - n_rare, 1.0)
    loss_rare   = jnp.sum(masked_rare * ce) / n_rare
    loss_common = jnp.sum((mask - masked_rare) * ce) / n_common
    return w_rare * loss_rare + w_common * loss_common

# @jax.jit
def loss_fn(params, batch):
    """
    Loss Function für Stochastic MuZero.
    
    WICHTIGE ÄNDERUNGEN gegenüber deterministischem MuZero:
    1. Dynamics Network hat 2 Teile: action_dynamics und chance_dynamics
    2. Zusätzlicher Loss für chance_logits (Würfelverteilung vorhersagen)
    3. Unroll-Schritt ist: action → afterstate → chance → next_state
    4. KEIN Reward Loss (Brettspiele haben nur End-Rewards)
    
    Args:
        params: Dictionary mit 'representation', 'dynamics', 'prediction'
        batch: Dictionary mit:
            - observations: (B, T, H, W) oder (B, T, Features)
            - actions: (B, T)
            - target_values: (B, T)
            - policies: (B, T, 454) 
            - card_outcomes: (B, T)  # NEU: Tatsächliche Kartenereignisse
            - masks: (B, T)
            
    Returns:
        total_loss: Scalar
        (value_loss, policy_loss, chance_loss): Tuple of scalars
    """
    
    # 1. Root Encoding
    root_obs = batch['observations']  # (B, Features) - nur der erste Timestep
    latent_state = repr_net.apply(params['representation'], root_obs)
    
    num_unroll_steps = batch['actions'].shape[1]
    
    def unroll_step(carry, inputs):
        latent_state, total_loss = carry
        k, action, target_value, target_policy, card_outcome, mask, true_card_probs, target_discount, target_reward = inputs
        
        # ===== PREDICTION LOSS (am aktuellen State) =====
        pred_policy_logits, pred_value = pred_net.apply(params['prediction'], latent_state)
        pred_value = pred_value.squeeze(-1)
        
        # Policy Loss (Cross-Entropy über 4 Actions)
        l_policy = jnp.mean(mask * optax.softmax_cross_entropy(pred_policy_logits, target_policy))
        
        # Value Loss (MSE)
        l_value = jnp.mean(mask * (target_value - pred_value) ** 2)
        
        # prep targets for classification
        n_valid = jnp.sum(mask)
        target_reward_class = target_reward.astype(jnp.int32)
        target_discount_class = target_discount.astype(jnp.int32)
        # Erkennung: L2-Abstand von uniform > threshold → non-uniform
        is_deal = (true_card_probs[:, 0] < 0.99)    

        # ===== DYNAMICS LOSS (State Transition) =====
        # Schritt 1: Action Dynamics (Spieler wählt Aktion)
        def do_dynamics(state, action, card_outcome):
            afterstate, afterstate_value, pred_reward_logits, pred_chance_logits, pred_discount_logits = dynamics_net.apply(
                params['dynamics'], state, action, method=dynamics_net.action_dynamics
            )

            reward_ce   = optax.softmax_cross_entropy_with_integer_labels(pred_reward_logits,   target_reward_class)
            discount_ce = optax.softmax_cross_entropy_with_integer_labels(pred_discount_logits, target_discount_class)
            chance_ce   = optax.softmax_cross_entropy(pred_chance_logits, true_card_probs)

            l_reward   = balanced_loss(reward_ce,   target_reward_class != 1,   mask, n_valid) # != 1 → seltene Klassen (reward≠0)
            l_discount = balanced_loss(discount_ce, target_discount_class == 1, mask, n_valid) # == 1 → seltene Klasse (terminal)
            l_chance   = balanced_loss(chance_ce,   is_deal,             mask, n_valid)

            next_latent = dynamics_net.apply(
                params['dynamics'], afterstate, card_outcome, method=dynamics_net.chance_dynamics
            )
            return next_latent, l_chance, l_discount, l_reward
        
        def skip_dynamics(state, action, card_outcome):
            # Am Ende des Unrolls keine Dynamics mehr
            return state, 0.0, 0.0, 0.0
        
        # Nur Dynamics wenn nicht am Ende
        next_latent, l_chance, l_discount, l_reward = jax.lax.cond(
            k < num_unroll_steps,
            do_dynamics,
            skip_dynamics,
            latent_state, action, card_outcome
        )
        
        # Gradient Scaling (siehe MuZero Paper)
        next_latent = jax.lax.stop_gradient(next_latent * 0.5) + next_latent * 0.5
        
        # Gewichte die einzelnen Loss-Komponenten
        step_loss = (1.0 / config["unroll_steps"]) * (
            VALUE_SCALING * l_value +      # Value Loss dominiert (wie im Paper)
            POLICY_SCALING * l_policy +           # Policy Loss
            CHANCE_SCALING * l_chance +
            DISCOUNT_SCALING * l_discount +  # Discount Loss
            REWARD_SCALING * l_reward      # Reward Loss
        )       
        return (next_latent, total_loss + step_loss), (l_value, l_policy, l_chance, l_discount, l_reward)
    
    # Prepare scan inputs
    k_indices = jnp.arange(num_unroll_steps + 1)
    actions_padded = jnp.concatenate([batch['actions'], jnp.zeros((batch['actions'].shape[0], 1), dtype=jnp.int32)], axis=1)

    card_outcomes_padded = jnp.concatenate([
        batch['card_outcomes'],              # (B, K-2): card_outcomes[1..K-2]
        jnp.zeros((batch['card_outcomes'].shape[0], 1), dtype=jnp.int32)         # padding
    ], axis=1)  # shape: (B, K)

    card_probs_padded = jnp.concatenate([
        batch['card_probs'],                                             
        jnp.zeros((batch['card_probs'].shape[0], 1, 128)).at[:, :, 0].set(1.0)        
    ], axis=1)

    discount_targets_padded = jnp.concatenate([
        batch['discount_targets'],
        jnp.ones((batch['discount_targets'].shape[0], 1), dtype=jnp.int32) # Klasse 1 = discount=0 (neutral)
    ], axis=1)

    # ✅ NEU: Reward Targets padden
    reward_targets_padded = jnp.concatenate([
        batch['rewards'],
        jnp.ones((batch['rewards'].shape[0], 1), dtype=jnp.int32) # Klasse 1 = reward=0 (neutral)
    ], axis=1)

    scan_inputs = (
        k_indices,
        actions_padded.T,
        batch['target_values'].T,
        jnp.transpose(batch['policies'], (1, 0, 2)),
        card_outcomes_padded.T,
        batch['masks'].T,
        jnp.transpose(card_probs_padded, (1, 0, 2)),
        discount_targets_padded.T,
        reward_targets_padded.T
    )
    
    (final_state, total_loss), (v_losses, p_losses, c_losses, d_losses, r_losses) = jax.lax.scan(
        unroll_step,
        (latent_state, 0.0),
        scan_inputs
    )
    
    value_loss = jnp.sum(v_losses)
    policy_loss = jnp.sum(p_losses)
    chance_loss = jnp.sum(c_losses)
    discount_loss = jnp.sum(d_losses)
    reward_loss = jnp.sum(r_losses)
    
    return total_loss, (value_loss, policy_loss, chance_loss, discount_loss, reward_loss)

@jax.jit
def train_step(params, opt_state, batch):
    """Führt einen Trainingsschritt aus."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, (v_loss, p_loss, r_loss)), grads = grad_fn(params, batch)
    (loss, (v_loss, p_loss, c_loss, d_loss, r_loss)), grads = grad_fn(params, batch)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, {
        'total_loss': loss,
        'v_loss': v_loss, 
        'p_loss': p_loss, 
        'c_loss': c_loss,
        'd_loss': d_loss, 
        'r_loss': r_loss
    }

# --- Initialisierung (Beispiel) ---

def _print_gpu_stats(label: str):
    """Print GPU memory and live JAX array count for diagnosing fragmentation."""
    import subprocess
    # Temperature + clock speed: these queries don't require elevated permissions
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = [s.strip() for s in r.stdout.strip().split(',')]
        print(f"  [GPU/{label}] temp={parts[0]}\u00b0C  SM={parts[1]}MHz  mem_clk={parts[2]}MHz")
    except Exception as e:
        print(f"  [GPU/{label}] nvidia-smi unavailable: {e}")
    # JAX allocator stats. With BFC (default): shows pool bytes.
    # With platform allocator (XLA_PYTHON_CLIENT_ALLOCATOR=platform): returns None.
    try:
        device = jax.devices('gpu')[0]
        stats = device.memory_stats()
        if stats:
            used_gb  = stats.get('bytes_in_use', 0) / 1e9
            peak_gb  = stats.get('peak_bytes_in_use', 0) / 1e9
            limit_gb = stats.get('bytes_limit', 0) / 1e9
            print(f"  [GPU/{label}] BFC pool: {used_gb:.2f} GB in use  "
                  f"{peak_gb:.2f} GB peak  {limit_gb:.2f} GB limit")
        else:
            # platform allocator active — use nvidia-smi for process memory instead
            import subprocess
            try:
                r2 = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )
                pid = os.getpid()
                for line in r2.stdout.strip().splitlines():
                    if str(pid) in line:
                        print(f"  [GPU/{label}] platform-alloc live CUDA: {line.strip()}")
                        break
                else:
                    print(f"  [GPU/{label}] platform allocator active (no BFC pool stats)")
            except Exception:
                print(f"  [GPU/{label}] platform allocator active (no BFC pool stats)")
    except Exception as e:
        print(f"  [GPU/{label}] memory_stats unavailable: {e}")
    # Python-tracked JAX arrays (only covers objects with Python refs)
    try:
        lines = []
        for dev in jax.devices('gpu') + jax.devices('cpu'):
            arrs = jax.live_arrays(dev)
            if arrs:
                gb = sum(a.nbytes for a in arrs) / 1e9
                lines.append(f"{dev.device_kind}:{gb:.2f}GB({len(arrs)}arr)")
        print(f"  [GPU/{label}] live arrays: {' | '.join(lines) if lines else 'none'}")
    except Exception:
        pass

def test_training(config, params=None, opt_state=None):
    seed = config["seed"]
    iterations = config["iterations"]
    num_games = config["num_games_per_iteration"]
    buffer_capacity = config["Buffer_Capacity"]
    unroll_steps = config["unroll_steps"]
    td_steps = config["td_steps"]
    max_episode_length = config["max_episode_length"]
    num_simulation = config["MCTS_simulations"]
    max_depth = config["MCTS_max_depth"]
    schedule = config["Temperature_Schedule"]
    train_steps_per_iteration = config["train_steps_per_iteration"]
    switch_to_bootstrap_iteration = config["Bootstrap_Switch_Iteration"]


    print(f"JAX Devices: {jax.devices()}")
    print(f"JAX Backend: {jax.default_backend()}")

    env = env_reset(
        0,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=16,
        starting_player=0,
        seed=0,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=RULES['enable_teams'],
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        must_traverse_start=RULES['must_traverse_start'],
        disable_swapping=RULES['disable_swapping'],
        disable_hot_seven=RULES['disable_hot_seven'],
        disable_joker=RULES['disable_joker']
    )
    enc = encode_board(env)  # z.B. (8, 56)
    print(f"Observation shape: {enc.shape}")
    input_shape = enc.shape  # (8, 56)

    if params is None:
        params = init_muzero_params(jax.random.PRNGKey(seed), input_shape)  # Beispiel Input Shape
    
    if opt_state is None:
        opt_state = optimizer.init(params)

    replay = VectorizedReplayBuffer(
        capacity=buffer_capacity,
        batch_size=config["Buffer_batch_Size"], 
        unroll_steps=unroll_steps, 
        td_steps=td_steps,
        obs_shape=input_shape, 
        action_dim=454,
        max_episode_length=max_episode_length, 
        bootstrap_value_target=config["Bootstrap_Value_Target"]
    )
    
    dog_wandb_session.log({"games_in_replay_buffer": replay.size})
    # collect initial set of games
    print("Collecting initial games...")
    game_warmup = 5 # TODO: 
    for n in range(game_warmup):
        print(f"{n+1}/{game_warmup} Playing games to fill replay buffer...")
        buffers = play_n_games_v3(
            params, 
            jax.random.PRNGKey(seed*n), 
            input_shape, 
            num_envs=num_games, 
            num_simulation=num_simulation, 
            max_depth=max_depth, 
            max_steps=max_episode_length, 
            temp=get_temperature(0, iterations)
        )
        replay.save_games_from_buffers(buffers)
        dog_wandb_session.log({"games_in_replay_buffer": replay.size})

    times_per_iteration = []
    global_step = 0
    for it in range(50,iterations):
        start_time = time()
        print(f"Iteration {it+1}/{iterations}")
        if it == switch_to_bootstrap_iteration:
            print("=" * 60)
            print("SWITCHING TO BOOTSTRAP VALUE TARGETS")
            print("=" * 60)
            replay.bootstrap_value_target = True

        temp = get_temperature(it, iterations)
        start_time = time()

        game_start = time()
        buffers = play_n_games_v3(
            params, 
            jax.random.PRNGKey(seed+it**3), 
            input_shape, 
            num_envs=num_games, 
            num_simulation=num_simulation, 
            max_depth=max_depth, 
            max_steps=max_episode_length, 
            temp=temp
        )
        game_gen_time = time() - game_start
        episode_lengths = buffers['idx']
        ep_max = int(episode_lengths.max())
        ep_mean = float(episode_lengths.mean())
        pct_at_cap = float(100 * (episode_lengths >= max_episode_length).mean())
        print(f"  Game generation took {game_gen_time:.1f}s")
        print(f"  Episode lengths: min={int(episode_lengths.min())}, max={ep_max}, mean={ep_mean:.1f}, pct_at_cap={pct_at_cap:.1f}%")

        print("Saving collected games to replay buffer...")
        replay.save_games_from_buffers(buffers)
        dog_wandb_session.log({"games_in_replay_buffer": replay.size})

        print("Training on collected data...")
        train_start = time()
        for i in range(train_steps_per_iteration):
            batch = replay.sample_batch()
            params, opt_state, losses = train_step(params, opt_state, batch)
            current_lr = learning_rate_schedule(global_step)
            dog_wandb_session.log({**losses, 'learning_rate': float(current_lr)})
            global_step += 1
            if i % (train_steps_per_iteration // 4) == 0:
                log_losses = {k: float(v) for k, v in losses.items()}
                print(f"  Step {i}: total={log_losses['total_loss']:.2f} "
                      f"v={log_losses['v_loss']:.2f} p={log_losses['p_loss']:.2f} "
                      f"c={log_losses['c_loss']:.2f} d={log_losses['d_loss']:.2f} "
                      f"r={log_losses['r_loss']:.2f}")
        end_time = time()
        print(f"  Iteration {it+1} done in {end_time - start_time:.1f}s  "
              f"(game_gen={game_gen_time:.1f}s  train={end_time - train_start:.1f}s)")
        times_per_iteration.append(end_time - start_time)

        if (it == iterations - 1):
            print(f"Saving checkpoint at iteration {it+1}...")
            with open(f'MuZero_DOG/models/params/muzero_dog_params_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{it+1}_seed{config["seed"]}.pkl', 'wb') as f:
                pickle.dump(params, f)

            with open(f'MuZero_DOG/models/opt_state/muzero_dog_opt_state_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{it+1}_seed{config["seed"]}.pkl', 'wb') as f:
                pickle.dump(opt_state, f)

    # ============================================================
    # DIAGNOSTIC DUMP: Write replay buffer contents to file
    # ============================================================
    valid = replay.size
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_path = f'MuZero_DOG/models/replay_diag_{timestamp}.npz'

    rew    = replay.rewards[:valid]             # (valid, max_ep_len) int8, classes 0/1/2
    masks  = replay.masks[:valid]               # (valid, max_ep_len) bool
    clens  = replay.episode_lengths[:valid]     # (valid,) int32
    plrs   = replay.players[:valid]             # (valid, max_ep_len) int8
    cdist  = replay.card_distributions[:valid]  # (valid, max_ep_len, 128) float16
    pols   = replay.child_visits[:valid]        # (valid, max_ep_len, 454) float16

    print(f"\n{'='*60}")
    print(f"Replay buffer diagnostic ({valid} episodes, max_ep_len={max_episode_length}):")
    print(f"  Episode lengths : min={clens.min()}, max={clens.max()}, "
          f"mean={clens.mean():.1f}, pct_at_cap={100*(clens>=max_episode_length).mean():.1f}%")
    print(f"  Reward classes  : 0(loss)={np.sum(rew==0):,}  1(neutral)={np.sum(rew==1):,}  2(win)={np.sum(rew==2):,}")
    print(f"  Masks           : valid_steps={np.sum(masks):,}, valid_fraction={np.mean(masks):.4f}")
    print(f"  Player dist     : " + "  ".join(f"p{i}={np.sum(plrs==i):,}" for i in range(4)))
    # card_dist: count steps where a deal actually happened (non-degenerate distribution)
    cdist_f32 = cdist.astype(np.float32)
    deal_steps = np.sum(cdist_f32[:, :, 0] < 0.99)   # first bin < 0.99 → non-trivial dist
    print(f"  Card dist deals : {deal_steps:,} steps with non-trivial card distribution")
    top_cards = np.argsort(-cdist_f32[cdist_f32[:, :, 0] < 0.99].sum(axis=0))[:5] \
        if deal_steps > 0 else []
    if len(top_cards):
        print(f"  Top-5 card bins (by freq): {top_cards.tolist()}")
    # Policy / valid-action stats: a policy entry > 0 means that action was visited by MCTS
    # (invalid actions are masked → weight 0). Number of nonzero entries = valid actions seen.
    pols_f32 = pols.astype(np.float32)  # (valid, max_ep_len, 454)
    # Only look at masked (real MCTS) steps
    active_mask_3d = masks[:, :, None]  # (valid, max_ep_len, 1)
    valid_action_counts = np.sum(pols_f32 > 0, axis=-1)  # (valid, max_ep_len) — actions with any visit
    active_counts = valid_action_counts[masks]             # only real steps
    if len(active_counts) > 0:
        print(f"  Valid actions   : mean={active_counts.mean():.1f}  "
              f"min={active_counts.min()}  max={active_counts.max()}  "
              f"median={int(np.median(active_counts))}")
        # Distribution: how often 1, 2-5, 6-10, 11-20, 21+ valid actions
        buckets = [(1,1,'=1'), (2,5,'2-5'), (6,10,'6-10'), (11,20,'11-20'), (21,50,'21-50'), (51,454,'51+')]
        bucket_str = '  '.join(
            f"{lbl}:{np.sum((active_counts>=lo)&(active_counts<=hi)):,}"
            for lo, hi, lbl in buckets
        )
        print(f"  Action buckets  : {bucket_str}")
        # Most common single action (dominant policy concentration)
        mean_pol = pols_f32[masks].mean(axis=0)  # (454,)
        top3_actions = np.argsort(-mean_pol)[:3]
        print(f"  Top-3 actions by avg weight: {top3_actions.tolist()} "
              f"weights={np.round(mean_pol[top3_actions], 4).tolist()}")
    print(f"Saving replay diagnostic to {diag_path} ...")
    np.savez_compressed(
        diag_path,
        rewards=rew,
        masks=masks,
        card_distributions=cdist,
        episode_lengths=clens,
        players=plrs,
        policies=pols,
    )
    print(f"Saved ({os.path.getsize(diag_path)/1e6:.1f} MB)")
    print('='*60)

    return params, opt_state, times_per_iteration

# ============================================================================
# MAIN: Konfiguration und Training starten
# ============================================================================
# RULES = {
#     'enable_teams': True, # DOG-standard is True
#     'enable_initial_free_pin': True, # DOG-standard is False
#     'enable_circular_board': False, # DOG-standard is True
#     'enable_friendly_fire': True, # DOG-standard is True
#     'enable_start_blocking': True, # DOG-standard is True
#     'enable_jump_in_goal_area': False, # DOG-standard is False
#     'must_traverse_start': False, # DOG-standard is True
#     'disable_swapping': False, # DOG-standard is False
#     'disable_hot_seven': False, # DOG-standard is False
#     'disable_joker': False, # DOG-standard is False
# }
RULES = {
    'enable_teams': True, # DOG-standard is True
    'enable_initial_free_pin': False, # DOG-standard is False
    'enable_circular_board': False, # DOG-standard is True
    'enable_friendly_fire': True, # DOG-standard is True
    'enable_start_blocking': True, # DOG-standard is True
    'enable_jump_in_goal_area': False, # DOG-standard is False
    'must_traverse_start': True, # DOG-standard is True
    'disable_swapping': False, # DOG-standard is False
    'disable_hot_seven': False, # DOG-standard is False
    'disable_joker': False, # DOG-standard is False
}
TEMPERATURE_SCHEDULE = [2.0, 1.5, 1, 0.8, 0.6]#[1.0, 0.9, 0.8, 0.7]
VALUE_SCALING = 4.0  
POLICY_SCALING = 1.0
DISCOUNT_SCALING = 1.0
REWARD_SCALING = 1.0
CHANCE_SCALING = 1.0
config = {
    "seed": 40,
    "learning_rate": 0.001,
    "architecture": "Chance Classic no belief states no Bootstrapping",
    "num_games_per_iteration": 500,
    "iterations": 100,
    "optimizer": "adamw with piecewise constant learning rate schedule",
    "Buffer_Capacity": 5000,
    "Buffer_batch_Size": 128,
    "unroll_steps": 10,
    "td_steps": 50, 
    "max_episode_length": 1300,
    "MCTS_simulations": 50,
    "MCTS_max_depth": 25,
    "Bootstrap_Value_Target": False,   # Always bootstrap; MC-only (False) leaves timed-out episodes with target≈0
    "Bootstrap_Switch_Iteration": 101,   # Enable from iteration 0 (switch mechanism not needed)
    "Temperature_Schedule": TEMPERATURE_SCHEDULE,
    "train_steps_per_iteration": 1500,
    "rules": RULES,
    "Loss scaling": {
        "value": VALUE_SCALING, 
        "policy": POLICY_SCALING, 
        "discount": DISCOUNT_SCALING, 
        "reward": REWARD_SCALING,
        "chance": CHANCE_SCALING
    }
}

# # prep weights and biases
dog_wandb_session = wandb.init(entity="marco-wojtek-tu-dortmund",project="dog-muzero",config=config,)

# --- Setup Optimizer ---
learning_rate_schedule = optax.piecewise_constant_schedule(
    init_value=config["learning_rate"],  # 0.001
    boundaries_and_scales={
        15 * config["train_steps_per_iteration"]: 0.2,   # It 30: 0.001 → 0.0002
        40 * config["train_steps_per_iteration"]: 0.5,   # It 65: 0.0002 → 0.00004
        90 * config["train_steps_per_iteration"]: 0.5,   # It 90: 0.00004 → 0.00002
    }
)

optimizer = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.adamw(learning_rate_schedule, weight_decay=1e-4)
)
# --- Start Training ---
params = None
opt_state = None
filename = 'muzero_dog_params_lr0.001_g500_it50_seed40.pkl'
params = load_params_from_file(f"MuZero_DOG/models/params/{filename}")
with open(f"MuZero_DOG/models/opt_state/{filename.replace('params', 'opt_state')}", 'rb') as f:
    opt_state = pickle.load(f)
starttime = time()
params, opt_state, times_per_iteration = test_training(config=config, params=params, opt_state=opt_state)
endtime = time()
passed_time = endtime - starttime
print(f"{'='*60}\n")
print(f"Total training time: {int(passed_time / 3600)} hours and {int(passed_time % 3600 / 60)} minutes.")
print(f"Average time per iteration: {jnp.mean(jnp.array(times_per_iteration)) / 60:.2f} minutes.")
print(f"{'='*60}\n")
