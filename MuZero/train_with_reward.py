import os
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
from MuZero.muzero_deterministic_madn import repr_net, dynamics_net, pred_net, init_muzero_params, load_params_from_file
from MADN.deterministic_madn import env_reset, encode_board
from MuZero.vec_replay_buffer import VectorizedReplayBuffer
from MuZero.game_agent import play_n_games_v3

def get_temperature(iteration, total_iterations):
    """Phasenbasiert: nur 4 verschiedene Werte, garantiert gleiche Float-Instanz"""
    phase = int(iteration / total_iterations * len(TEMPERATURE_SCHEDULE))
    phase = min(phase, len(TEMPERATURE_SCHEDULE) - 1)
    return TEMPERATURE_SCHEDULE[phase]
# @jax.jit
def loss_fn(params, batch):
    """Vektorisierte Version mit scan statt Loop"""
    
    root_obs = batch['observations']
    latent_state = repr_net.apply(params['representation'], root_obs)
    
    num_unroll_steps = batch['actions'].shape[1]
    
    def unroll_step(carry, inputs):
        latent_state, total_loss = carry
        # k, action, target_value, target_policy, target_reward, mask = inputs
        k, action, target_value, target_policy, mask, target_discount, target_reward = inputs
        
        # Prediction
        pred_policy_logits, pred_value = pred_net.apply(params['prediction'], latent_state)
        pred_value = pred_value.squeeze(-1)
        
        # Losses
        l_value = jnp.mean(mask * (target_value - pred_value) ** 2)
        l_policy = jnp.mean(mask * optax.softmax_cross_entropy(pred_policy_logits, target_policy))
        
        step_loss = (1.0 / config["unroll_steps"]) * (VALUE_SCALING * l_value + POLICY_SCALING * l_policy) #* (1.0 / num_unroll_steps)
        
        # Dynamics (nur wenn nicht am Ende) Keine reward Vorhersage am Root
        def do_dynamics(state):
            new_state, pred_reward_logits, pred_discount_logits = dynamics_net.apply(
                params['dynamics'], state, action
            )
            
            # ✅ REWARD: Balanced per-class loss
            target_reward_class = target_reward.astype(jnp.int32)
            reward_ce = optax.softmax_cross_entropy_with_integer_labels(
                pred_reward_logits, target_reward_class
            )
            
            # Separate Mittelwerte pro Klasse → gleiche Gradient-Stärke
            is_neutral = (target_reward_class == 1)
            n_neutral = jnp.maximum(jnp.sum(mask * is_neutral), 1.0)
            n_non_neutral = jnp.maximum(jnp.sum(mask * (~is_neutral)), 1.0)
            
            loss_neutral = jnp.sum(mask * jnp.where(is_neutral, reward_ce, 0.0)) / n_neutral
            loss_non_neutral = jnp.sum(mask * jnp.where(~is_neutral, reward_ce, 0.0)) / n_non_neutral
            
            # 50/50 Gewichtung: Neutral lernt "default 0", Non-Neutral lernt Win/Lose
            l_reward = 0.1 * loss_neutral + 1.0 * loss_non_neutral
            
            # ✅ DISCOUNT: Balanced per-class loss (analog zu Reward)
            # Klasse 0=-1 (Gegner), Klasse 1=0 (Terminal), Klasse 2=+1 (eigener Zug)
            # Terminal (Klasse 1) ist extrem selten → separate Normierung
            target_discount_class = target_discount.astype(jnp.int32)
            discount_ce = optax.softmax_cross_entropy_with_integer_labels(
                pred_discount_logits, target_discount_class
            )
            
            is_terminal = (target_discount_class == 1)
            n_non_terminal = jnp.maximum(jnp.sum(mask * (~is_terminal)), 1.0)
            n_terminal = jnp.maximum(jnp.sum(mask * is_terminal), 1.0)
            
            loss_non_terminal = jnp.sum(mask * jnp.where(~is_terminal, discount_ce, 0.0)) / n_non_terminal
            loss_terminal = jnp.sum(mask * jnp.where(is_terminal, discount_ce, 0.0)) / n_terminal
            
            # Non-Terminal (6er-Regel) funktioniert schon gut → niedrige Gewichtung
            # Terminal muss stärker lernen
            l_discount = 0.1 * loss_non_terminal + 1.0 * loss_terminal
            
            return new_state, l_discount, l_reward
        
        def skip_dynamics(state):
            return state, 0.0, 0.0
        
        next_latent, l_discount, l_reward = jax.lax.cond(
            k < num_unroll_steps,
            do_dynamics,
            skip_dynamics,
            latent_state
        )

        discount_loss = (1.0 / config["unroll_steps"]) * DISCOUNT_SCALING * l_discount
        reward_loss = (1.0 / config["unroll_steps"]) * REWARD_SCALING * l_reward

        # Gradient scaling
        next_latent = jax.lax.stop_gradient(next_latent * 0.5) + next_latent * 0.5
        
        # return (next_latent, total_loss + step_loss + reward_loss), (l_value, l_policy, reward_loss)
        return (next_latent, total_loss + step_loss + discount_loss + reward_loss), (l_value, l_policy, l_discount, l_reward)
    
    # Prepare scan inputs
    k_indices = jnp.arange(num_unroll_steps + 1)
    actions_padded = jnp.concatenate([batch['actions'], jnp.zeros((batch['actions'].shape[0], 1), dtype=jnp.int32)], axis=1)
    #rewards_padded = jnp.concatenate([batch['rewards'], jnp.zeros((batch['rewards'].shape[0], 1))], axis=1)
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
        batch['masks'].T,
        discount_targets_padded.T,
        reward_targets_padded.T
    )

    (final_state, total_loss), (v_losses, p_losses, d_losses, r_losses) = jax.lax.scan(
        unroll_step,
        (latent_state, 0.0),
        scan_inputs
    )
    
    value_loss = jnp.sum(v_losses)
    policy_loss = jnp.sum(p_losses)
    discount_loss = jnp.sum(d_losses)
    reward_loss = jnp.sum(r_losses)
    
    return total_loss, (value_loss, policy_loss, discount_loss, reward_loss)

@jax.jit
def train_step(params, opt_state, batch):
    """Führt einen Trainingsschritt aus."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, (v_loss, p_loss, r_loss)), grads = grad_fn(params, batch)
    (loss, (v_loss, p_loss, d_loss, r_loss)), grads = grad_fn(params, batch)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, {
        'total_loss': loss,
        'v_loss': v_loss, 
        'p_loss': p_loss, 
        'd_loss': d_loss, 
        'r_loss': r_loss
    }

# --- Initialisierung (Beispiel) ---

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


    print(f"JAX Devices: {jax.devices()}")
    print(f"JAX Backend: {jax.default_backend()}")

    env = env_reset(
        0,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=10,
        starting_player=0,
        seed=0,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=RULES['enable_teams'],
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        enable_start_on_1=RULES['enable_start_on_1'],
        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
        must_traverse_start=RULES['must_traverse_start']
    )
    enc = encode_board(env)  # z.B. (8, 56)
    print(enc.shape)
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
        max_episode_length=max_episode_length, 
        bootstrap_value_target=config["Bootstrap_Value_Target"]
    )
    
    deterministic_madn_wandb_session.log({"games_in_replay_buffer": replay.size})
    # collect initial set of games
    print("Collecting initial games...")
    game_warmup = 3
    for n in range(game_warmup):
        print(f"{n+1}/{game_warmup} Playing games to fill replay buffer...")
        buffers = play_n_games_v3(params, jax.random.PRNGKey(seed*n), input_shape, num_envs=num_games, num_simulation=num_simulation, max_depth=max_depth, max_steps=max_episode_length, temp=get_temperature(0, iterations))
        replay.save_games_from_buffers(buffers)
        deterministic_madn_wandb_session.log({"games_in_replay_buffer": replay.size})

    times_per_iteration = []
    global_step = 0
    for it in range(iterations):
        start_time = time()
        print(f"Iteration {it+1}/{iterations}")
        # ✅ Automatically switch to bootstrap after Phase 1
        if (it) == 70:
            print("=" * 60)
            print("SWITCHING TO BOOTSTRAP VALUE TARGETS")
            print("=" * 60)
            replay.bootstrap_value_target = True

        temp = get_temperature(it, iterations)  # Phasenbasiert: nur 4 verschiedene Werte
        buffers = play_n_games_v3(params, jax.random.PRNGKey(seed+it**3), input_shape, num_envs=num_games, num_simulation=num_simulation, max_depth=max_depth, max_steps=max_episode_length, temp=temp)
        episode_lengths = buffers['idx']
        print(f"  Episode lengths: min={episode_lengths.min()}, max={episode_lengths.max()}, mean={episode_lengths.mean():.1f}")
        print("Saving collected games to replay buffer...")
        replay.save_games_from_buffers(buffers)
        deterministic_madn_wandb_session.log({"games_in_replay_buffer": replay.size})
        print("Training on collected data...")
        train_start = time()
        for i in range(train_steps_per_iteration):  
            batch = replay.sample_batch()
            params, opt_state, losses = train_step(params, opt_state, batch)
            current_lr = learning_rate_schedule(global_step)
            deterministic_madn_wandb_session.log({**losses, 'learning_rate': float(current_lr)})
            global_step += 1
            if i % (train_steps_per_iteration // 4) == 0:
                print(f"Step {i}, Losses: {{")
                print(f"  total_loss: {losses['total_loss']:.2f},")
                print(f"  v_loss: {losses['v_loss']:.2f} ({losses['v_loss']/unroll_steps:.3f} per step),")
                print(f"  p_loss: {losses['p_loss']:.2f} ({losses['p_loss']/unroll_steps:.3f} per step)")
                print(f"  d_loss: {losses['d_loss']:.2f} ({losses['d_loss']/unroll_steps:.3f} per step)")
                print(f"  r_loss: {losses['r_loss']:.2f} ({losses['r_loss']/unroll_steps:.3f} per step)")
                print(f"}}")
        end_time = time()
        print(f"""
              Iteration {it+1} completed in {end_time - start_time:.2f} seconds.
              Game playing + data collection time: {train_start - start_time:.2f} seconds.
              Training time: {end_time - train_start:.2f} seconds.
              """)
        times_per_iteration.append(end_time - start_time)

        # if ((it+1) % 50 == 0):
        #     print(f"Saving checkpoint at iteration {it+1}...")
        #     with open(f'models/params/gumbelmuzero_madn_params_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{it+1}_seed{config["seed"]}.pkl', 'wb') as f:
        #         pickle.dump(params, f)

        #     with open(f'models/opt_state/gumbelmuzero_madn_opt_state_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{it+1}_seed{config["seed"]}.pkl', 'wb') as f:
        #         pickle.dump(opt_state, f)
        if ((it+1) % 100 == 0):
            print(f"Saving checkpoint at iteration {it+1}...")
            with open(f'models/params/Experiment_{config["seed"]}_{it+1}.pkl', 'wb') as f:
                pickle.dump(params, f)

            with open(f'models/opt_state/Experiment_{config["seed"]}_{it+1}.pkl', 'wb') as f:
                pickle.dump(opt_state, f)

    return params, opt_state, times_per_iteration

RULES = {
    'enable_teams': True,
    'enable_initial_free_pin': True,
    'enable_circular_board': False,
    'enable_friendly_fire': False,
    'enable_start_blocking': False,
    'enable_jump_in_goal_area': True,
    'enable_start_on_1': True,
    'enable_bonus_turn_on_6': True,
    'must_traverse_start': False
}
TEMPERATURE_SCHEDULE = [2.0, 1.5, 1, 0.8, 0.6]#[1.0, 0.9, 0.8, 0.7]
VALUE_SCALING = 4.0  
POLICY_SCALING = 1.0
DISCOUNT_SCALING = 1.0
REWARD_SCALING = 1.0
config = {
    "seed": 29,
    "learning_rate": 0.005,
    "architecture": "Real Training with new RepNet2, DynNet4 and PredNet4. Reward loss with Cross-Entropy (3 Klassen: -1, 0, +1) + Per-Class Balanced Loss und Discount loss mit Cross-Entropy (3 Klassen: -1, 0, +1). Ziel: Stabileres Training durch diskrete Klassen für Rewards und Discounts, da kontinuierliche Werte in MADN oft sehr spiky und schwer vorherzusagen sind. Mit Klassengewichtung, um wichtige Ereignisse (Sieg/Niederlage) stärker zu gewichten und Priority Sampling im Replay Buffer (fixed).",
    "num_games_per_iteration": 1500,
    "iterations": 100,
    "optimizer": "adamw with piecewise_constant_schedule (similar as MuZero paper)",
    "Buffer_Capacity": 20000,
    "Buffer_batch_Size": 128,
    "unroll_steps": 10,
    "td_steps": 50, 
    "max_episode_length": 700,
    "MCTS_simulations": 100,
    "MCTS_max_depth": 50,
    "Bootstrap_Value_Target": False,
    "Temperature_Schedule": TEMPERATURE_SCHEDULE,
    "train_steps_per_iteration": 2500,
    "rules": RULES,
    "Loss scaling": {"value": VALUE_SCALING, "policy": POLICY_SCALING, "discount": DISCOUNT_SCALING, "reward": REWARD_SCALING}
}
# prep weights and biases
deterministic_madn_wandb_session = wandb.init(
    entity="marco-wojtek-tu-dortmund",
    project="deterministic-muzero-madn",
    config=config,
)

# --- Setup Optimizer ---
learning_rate_schedule = optax.piecewise_constant_schedule(
    init_value=config["learning_rate"],  # 0.005
    boundaries_and_scales={
        30 * config["train_steps_per_iteration"]: 0.2,    # It 50:  0.005 → 0.001
        60 * config["train_steps_per_iteration"]: 0.2,   # It 120: 0.001 → 0.0002
        80 * config["train_steps_per_iteration"]: 0.5,   # It 170: 0.0002 → 0.0001
    }
)

optimizer = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.adamw(learning_rate_schedule, weight_decay=1e-4)
)
# --- Start Training ---
params = None
opt_state = None
# params = load_params_from_file('muzero_madn_params_00001.pkl')
# with open('muzero_madn_opt_state_00001.pkl', 'rb') as f:
#     opt_state = pickle.load(f)
starttime = time()
params, opt_state, times_per_iteration = test_training(config=config, params=params, opt_state=opt_state)
endtime = time()
passed_time = endtime - starttime
print(f"{'='*60}\n")
print(f"Total training time: {int(passed_time / 3600)} hours and {int(passed_time % 3600 / 60)} minutes.")
print(f"Average time per iteration: {jnp.mean(jnp.array(times_per_iteration)) / 60:.2f} minutes.")
print(f"{'='*60}\n")