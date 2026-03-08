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
from MuZero_Classic_MADN.muzero_classic_madn import repr_net, dynamics_net, pred_net, decision_recurrent_fn, chance_recurrent_fn
from MADN.classic_madn import env_reset, dice_probabilities, encode_board
from MuZero_Classic_MADN.muzero_classic_madn import init_muzero_params
from MuZero_Classic_MADN.vec_replay_buffer_stochastic import VectorizedReplayBufferStochastic
from MuZero_Classic_MADN.game_agent_stochastic import play_n_games_v3

def get_temperature(iteration, total_iterations):
    """Phasenbasiert: nur 4 verschiedene Werte, garantiert gleiche Float-Instanz"""
    phase = int(iteration / total_iterations * len(TEMPERATURE_SCHEDULE))
    phase = min(phase, len(TEMPERATURE_SCHEDULE) - 1)
    return TEMPERATURE_SCHEDULE[phase]

# @jax.jit
def loss_fn_stochastic(params, batch):
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
            - policies: (B, T, 4)  # NUR 4 Actions für Pins!
            - dice_outcomes: (B, T)  # NEU: Tatsächliche Würfelergebnisse
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
        k, action, target_value, target_policy, dice_outcome, mask, true_dice_probs = inputs
        
        # ===== PREDICTION LOSS (am aktuellen State) =====
        pred_policy_logits, pred_value = pred_net.apply(params['prediction'], latent_state)
        pred_value = pred_value.squeeze(-1)
        
        # Policy Loss (Cross-Entropy über 4 Actions)
        l_policy = jnp.mean(mask * optax.softmax_cross_entropy(pred_policy_logits, target_policy))
        
        # Value Loss (MSE)
        l_value = jnp.mean(mask * (target_value - pred_value) ** 2)
        
        # ===== DYNAMICS LOSS (State Transition) =====
        # Schritt 1: Action Dynamics (Spieler wählt Aktion)
        def do_dynamics(state, action, dice_outcome):
            # a) Action Dynamics: latent_state + action → afterstate + reward + chance_logits
            afterstate, pred_reward, pred_chance_logits, discount_logits = dynamics_net.apply(params['dynamics'], state, action, method=dynamics_net.action_dynamics)
            
            # Chance Loss mit Maske
            l_chance = jnp.mean(mask * optax.softmax_cross_entropy(pred_chance_logits, true_dice_probs))
            
            # b) Chance Dynamics: afterstate + dice_outcome → next_latent_state
            next_latent = dynamics_net.apply(params['dynamics'], afterstate, dice_outcome, method=dynamics_net.chance_dynamics)
            
            return next_latent, l_chance
        
        def skip_dynamics(state, action, dice_outcome):
            # Am Ende des Unrolls keine Dynamics mehr
            return state, 0.0
        
        # Nur Dynamics wenn nicht am Ende
        next_latent, l_chance = jax.lax.cond(
            k < num_unroll_steps,
            do_dynamics,
            skip_dynamics,
            latent_state, action, dice_outcome
        )
        
        # Gradient Scaling (siehe MuZero Paper)
        next_latent = jax.lax.stop_gradient(next_latent * 0.5) + next_latent * 0.5
        
        # Gewichte die einzelnen Loss-Komponenten
        step_loss = (
            VALUE_SCALING * l_value +      # Value Loss dominiert (wie im Paper)
            POLICY_SCALING * l_policy +           # Policy Loss
            CHANCE_SCALING * l_chance         # Chance Loss (neue Komponente für Stochastic MuZero)
        )
        
        return (next_latent, total_loss + step_loss), (l_value, l_policy, l_chance)
    
    # Prepare scan inputs
    k_indices = jnp.arange(num_unroll_steps + 1)
    actions_padded = jnp.concatenate([batch['actions'], jnp.zeros((batch['actions'].shape[0], 1), dtype=jnp.int32)], axis=1)
    dice_padded = jnp.concatenate([batch['dice_outcomes'], jnp.zeros((batch['dice_outcomes'].shape[0], 1), dtype=jnp.int32)], axis=1)
    dice_prop_padded = jnp.concatenate([batch['dice_probs'], jnp.zeros((batch['dice_probs'].shape[0], 1, 6))], axis=1)  # NEU: Würfelverteilung für gepaddeten Schritt
    
    scan_inputs = (
        k_indices,
        actions_padded.T,
        batch['target_values'].T,
        jnp.transpose(batch['policies'], (1, 0, 2)),
        dice_padded.T,
        batch['masks'].T,
        jnp.transpose(dice_prop_padded, (1, 0, 2))  # Wahrscheinlichkeitsverteilungen für Würfelergebnisse
    )
    
    (final_state, total_loss), (v_losses, p_losses, c_losses) = jax.lax.scan(
        unroll_step,
        (latent_state, 0.0),
        scan_inputs
    )
    
    value_loss = jnp.sum(v_losses)
    policy_loss = jnp.sum(p_losses)
    chance_loss = jnp.sum(c_losses)
    
    return total_loss, (value_loss, policy_loss, chance_loss)

@jax.jit
def train_step(params, opt_state, batch):
    """Führt einen Trainingsschritt aus."""
    grad_fn = jax.value_and_grad(loss_fn_stochastic, has_aux=True)
    (loss, (v_loss, p_loss, c_loss)), grads = grad_fn(params, batch)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, {
        'total_loss': loss, 
        'v_loss': v_loss, 
        'p_loss': p_loss, 
        'c_loss': c_loss  # NEU: Chance Loss
    }

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
    train_steps_per_iteration = config["train_steps_per_iteration"]

    print(f"JAX Devices: {jax.devices()}")
    print(f"JAX Backend: {jax.default_backend()}")

    # Environment Setup
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
        must_traverse_start=RULES['must_traverse_start'],
        enable_dice_rethrow=RULES['enable_dice_rethrow']  # Wichtig für unterschiedliche Würfelverteilungen!
    )
    
    enc = encode_board(env)
    print(f"Observation shape: {enc.shape}")
    input_shape = enc.shape

    # Parameter Initialisierung
    if params is None:
        print("Initializing MuZero parameters...")
        params = init_muzero_params(jax.random.PRNGKey(seed), input_shape)
    
    if opt_state is None:
        opt_state = optimizer.init(params)

    # Replay Buffer Setup
    # WICHTIG: Für Stochastic MuZero brauchen wir dice_outcomes!
    replay = VectorizedReplayBufferStochastic(
        capacity=buffer_capacity, 
        batch_size=config["Buffer_batch_Size"], 
        unroll_steps=unroll_steps, 
        td_steps=td_steps,
        obs_shape=input_shape,
        action_dim=4,  # Nur 4 Actions (Pins) für stochastic!
        max_episode_length=max_episode_length, 
        bootstrap_value_target=config["Bootstrap_Value_Target"]
    )
    
    stochastic_madn_wandb_session.log({"games_in_replay_buffer": replay.size})
    
    # Collect initial games
    print("Collecting initial games...")
    game_warmup = 3
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
            temp=get_temperature(0, iterations)  # Warmup verwendet erste Temperature
        )
        replay.save_games_from_buffers(buffers)
        stochastic_madn_wandb_session.log({"games_in_replay_buffer": replay.size})
    # Training Loop
    times_per_iteration = []
    global_step = 0
    
    for it in range(iterations):
        start_time = time()
        print(f"\nIteration {it+1}/{iterations}")
        
        # Play games
        temp = get_temperature(it, iterations)
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
        episode_lengths = buffers['idx']
        print(f"  Episode lengths: min={episode_lengths.min()}, max={episode_lengths.max()}, mean={episode_lengths.mean():.1f}")
        
        print("Saving collected games to replay buffer...")
        replay.save_games_from_buffers(buffers)
        stochastic_madn_wandb_session.log({"games_in_replay_buffer": replay.size})
        
        # Training
        print("Training on collected data...")
        train_start = time()
        
        for i in range(train_steps_per_iteration):  
            batch = replay.sample_batch()
            params, opt_state, losses = train_step(params, opt_state, batch)
            
            current_lr = learning_rate_schedule(global_step)
            stochastic_madn_wandb_session.log({
               **losses,
               'learning_rate': float(current_lr)
            })
            global_step += 1
            
            if i % (train_steps_per_iteration // 3) == 0:
                print(f"Step {i}, Losses: {{")
                print(f"  total_loss: {losses['total_loss']:.2f},")
                print(f"  v_loss: {losses['v_loss']:.2f} ({losses['v_loss']/unroll_steps:.3f} per step),")
                print(f"  p_loss: {losses['p_loss']:.2f} ({losses['p_loss']/unroll_steps:.3f} per step),")
                print(f"  c_loss: {losses['c_loss']:.2f} ({losses['c_loss']/unroll_steps:.3f} per step)")  
                print(f"}}")
        
        end_time = time()
        print(f"""
            Iteration {it+1} completed in {end_time - start_time:.2f} seconds.
            Game playing time: {train_start - start_time:.2f} seconds.
            Training time: {end_time - train_start:.2f} seconds.
        """)
        times_per_iteration.append(end_time - start_time)

        # Save intermediate parameters every 50 iterations
        if (it + 1) % 30 == 0:
            print(f"Saving checkpoint at iteration {it+1}...")
            with open(f'models/params/stochastic_muzero_madn_params_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{it+1}_seed{config["seed"]}.pkl', 'wb') as f:
                pickle.dump(params, f)

            with open(f'models/opt_state/stochastic_muzero_madn_opt_state_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{it+1}_seed{config["seed"]}.pkl', 'wb') as f:
                pickle.dump(opt_state, f)
    return params, opt_state, times_per_iteration


# ============================================================================
# MAIN: Konfiguration und Training starten
# ============================================================================
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
    'enable_dice_rethrow': True  # NEU: Für unterschiedliche Würfelverteilungen!
}
TEMPERATURE_SCHEDULE = [1.5, 1.0, 0.7, 0.5]
VALUE_SCALING = 75.0  
POLICY_SCALING = 1.0
CHANCE_SCALING = 1.0 # NEU: Gewicht für Chance Loss
if __name__ == "__main__":
    config = {
        "seed": 68741,
        "learning_rate": 0.01,
        "architecture": "Stochastic MuZero Classic MADN with larger init lr, few more MCTS simulations, env before dice as target for chance loss",
        "num_games_per_iteration": 1500,
        "iterations": 100,
        "optimizer": "adamw with piecewise_constant_schedule",
        "Buffer_Capacity": 20000,
        "Buffer_batch_Size": 128,
        "unroll_steps": 5,
        "td_steps": 12,
        "max_episode_length": 800,
        "MCTS_simulations": 50, # less actions to evaluate (4 Pins) → less simulations needed
        "MCTS_max_depth": 25,
        "Bootstrap_Value_Target": True,
        "Temperature_Schedule": TEMPERATURE_SCHEDULE,
        "train_steps_per_iteration": 1500,
        "rules": RULES,
        "Loss Scaling": {
            "value_loss": VALUE_SCALING,
            "policy_loss": POLICY_SCALING,
            "chance_loss": CHANCE_SCALING
            }
    }
    
    # # Weights & Biases Setup
    stochastic_madn_wandb_session = wandb.init(
        entity="marco-wojtek-tu-dortmund",
        project="stochastic-muzero-madn",
        config=config,
    )

    # Optimizer Setup
    learning_rate_schedule = optax.piecewise_constant_schedule(
        init_value=config["learning_rate"],
        boundaries_and_scales={
            20 * config["train_steps_per_iteration"]: 0.1,   # Iteration: LR * 0.1
            50 * config["train_steps_per_iteration"]: 0.1,   # Iteration: LR * 0.01
            75 * config["train_steps_per_iteration"]: 0.1,   # Iteration: LR * 0.001
            90 * config["train_steps_per_iteration"]: 0.1,   # Iteration: LR * 0.0001
        }
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(learning_rate_schedule, weight_decay=1e-4)
    )
    
    # Start Training
    params = None
    opt_state = None
    
    # Optional: Load pretrained params
    # params = load_params_from_file('muzero_stochastic_madn_params_00001.pkl')
    # with open('muzero_stochastic_madn_opt_state_00001.pkl', 'rb') as f:
    #     opt_state = pickle.load(f)
    
    starttime = time()
    params, opt_state, times_per_iteration = test_training(
        config=config, 
        params=params, 
        opt_state=opt_state
    )
    endtime = time()
    
    passed_time = endtime - starttime
    print(f"\n{'='*60}")
    print(f"Total training time: {int(passed_time / 3600)} hours and {int(passed_time % 3600 / 60)} minutes.")
    print(f"Average time per iteration: {jnp.mean(jnp.array(times_per_iteration)) / 60:.2f} minutes.")
    print(f"{'='*60}\n")
    