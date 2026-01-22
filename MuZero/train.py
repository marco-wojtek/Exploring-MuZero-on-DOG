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
from MADN.deterministic_madn import env_reset, encode_board, old_encode_board
from MuZero.replay_buffer import ReplayBuffer, play_n_games_v3
from MuZero.vec_replay_buffer import VectorizedReplayBuffer
# @jax.jit
def loss_fn(params, batch):
    """Vektorisierte Version mit scan statt Loop"""
    
    root_obs = batch['observations']
    latent_state = repr_net.apply(params['representation'], root_obs)
    
    num_unroll_steps = batch['actions'].shape[1]
    
    def unroll_step(carry, inputs):
        latent_state, total_loss = carry
        # k, action, target_value, target_policy, target_reward, mask = inputs
        k, action, target_value, target_policy, mask = inputs
        
        # Prediction
        pred_policy_logits, pred_value = pred_net.apply(params['prediction'], latent_state)
        pred_value = pred_value.squeeze(-1)
        
        # Losses
        l_value = jnp.mean(mask * (target_value - pred_value) ** 2)
        l_policy = jnp.mean(mask * optax.softmax_cross_entropy(pred_policy_logits, target_policy))
        
        step_loss =  (l_value + l_policy) #* (1.0 / num_unroll_steps)
        
        # Dynamics (nur wenn nicht am Ende) Keine reward Vorhersage am Root
        def do_dynamics(state):
            new_state, pred_reward, _ = dynamics_net.apply(params['dynamics'], state, action)
            # pred_reward = pred_reward.squeeze(-1)
            # l_reward = jnp.mean(mask * (target_reward - pred_reward) ** 2)
            return new_state#, scale * l_reward
        
        def skip_dynamics(state):
            return state#, 0.0
        
        next_latent = jax.lax.cond(
            k < num_unroll_steps,
            do_dynamics,
            skip_dynamics,
            latent_state
        )
        
        # Gradient scaling
        next_latent = jax.lax.stop_gradient(next_latent * 0.5) + next_latent * 0.5
        
        # return (next_latent, total_loss + step_loss + reward_loss), (l_value, l_policy, reward_loss)
        return (next_latent, total_loss + step_loss), (l_value, l_policy)
    
    # Prepare scan inputs
    k_indices = jnp.arange(num_unroll_steps + 1)
    actions_padded = jnp.concatenate([batch['actions'], jnp.zeros((batch['actions'].shape[0], 1), dtype=jnp.int32)], axis=1)
    #rewards_padded = jnp.concatenate([batch['rewards'], jnp.zeros((batch['rewards'].shape[0], 1))], axis=1)
    
    scan_inputs = (
        k_indices,
        actions_padded.T,
        batch['target_values'].T,
        jnp.transpose(batch['policies'], (1, 0, 2)),
        #rewards_padded.T,
        batch['masks'].T
    )
    
    (final_state, total_loss), (v_losses, p_losses) = jax.lax.scan(
        unroll_step,
        (latent_state, 0.0),
        scan_inputs
    )
    
    value_loss = jnp.sum(v_losses)
    policy_loss = jnp.sum(p_losses)
    #reward_loss = jnp.sum(r_losses)
    
    return total_loss, (value_loss, policy_loss)#, reward_loss)

@jax.jit
def train_step(params, opt_state, batch):
    """Führt einen Trainingsschritt aus."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, (v_loss, p_loss, r_loss)), grads = grad_fn(params, batch)
    (loss, (v_loss, p_loss)), grads = grad_fn(params, batch)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, {'total_loss': loss, 'v_loss': v_loss, 'p_loss': p_loss}#, 'r_loss': r_loss}

# --- Initialisierung (Beispiel) ---

def test_training(num_games= 50, seed=42, iterations=100, params=None, opt_state=None):
    print(f"JAX Devices: {jax.devices()}")
    print(f"JAX Backend: {jax.default_backend()}")

    env = env_reset(
        0,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=10,
        starting_player=0,
        seed=0,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=False,
        enable_initial_free_pin=True,
        enable_circular_board=False
    )
    enc = old_encode_board(env)  # z.B. (8, 56)
    print(enc.shape)
    input_shape = enc.shape  # (8, 56)

    if params is None:
        params = init_muzero_params(jax.random.PRNGKey(seed), input_shape)  # Beispiel Input Shape
    
    if opt_state is None:
        opt_state = optimizer.init(params)

    replay = VectorizedReplayBuffer(capacity=50000, batch_size=128, unroll_steps=5)
    # collect initial set of games
    print("Collecting initial games...")
    for _ in range(3):
        print(f"Playing games to fill replay buffer...")
        buffers = play_n_games_v3(params, jax.random.PRNGKey(seed+1), input_shape, num_envs=num_games, temp=1.5)
    replay.save_games_from_buffers(buffers)
    times_per_iteration = []
    for it in range(iterations):
        start_time = time()
        print(f"Iteration {it+1}/{iterations}")
        temp = max(0.25, 1.5 - it * (1.25 / (iterations - 1)))  # Linearly decay temperature from 1.5 to 0.25
        buffers = play_n_games_v3(params, jax.random.PRNGKey(seed+it**3), input_shape, num_envs=num_games, temp=temp)
        print("Saving collected games to replay buffer...")
        replay.save_games_from_buffers(buffers)
        deterministic_madn_wandb_session.log({"games_in_replay_buffer": replay.size})
        print("Training on collected data...")
        train_start = time()
        train_steps = 1500
        for i in range(train_steps):  
            batch = replay.sample_batch()
            params, opt_state, losses = train_step(params, opt_state, batch)
            deterministic_madn_wandb_session.log(losses)
            if i % (train_steps // 3) == 0:
                print(f"Step {i}, Losses: {{")
                print(f"  total_loss: {losses['total_loss']:.2f},")
                print(f"  v_loss: {losses['v_loss']:.2f} ({losses['v_loss']/train_steps:.3f} per step),")
                print(f"  p_loss: {losses['p_loss']:.2f} ({losses['p_loss']/train_steps:.3f} per step)")
                print(f"}}")
        end_time = time()
        print(f"""
              Iteration {it+1} completed in {end_time - start_time:.2f} seconds.
              Game playing + data collection time: {train_start - start_time:.2f} seconds.
              Training time: {end_time - train_start:.2f} seconds.
              """)
        times_per_iteration.append(end_time - start_time)
    return params, opt_state, times_per_iteration

config = {
    "learning_rate": 5e-5,
    "architecture": "MuZero Deterministic MADN with Gumbel MCTS",
    "num_games_per_iteration": 1500,
    "iterations": 40,
    "optimizer": "adamw with warmup cosine decay"
}
# prep weights and biases
deterministic_madn_wandb_session = wandb.init(
    entity="marco-wojtek-tu-dortmund",
    project="deterministic-muzero-madn",
    config=config,
)

# --- Setup Optimizer ---
learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4,          # Warmup start
    peak_value=5e-3,          # Peak (reduziert wegen mehr Spielen)
    warmup_steps=4500,        # 3 Iterationen @ 1500 steps
    decay_steps=60000,        # 40 Iterationen @ 1500 steps
    end_value=config["learning_rate"]            # Niedriger End-Value
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
params, opt_state, times_per_iteration = test_training(num_games=config["num_games_per_iteration"], seed=7841, iterations=config["iterations"], params=params, opt_state=opt_state)
endtime = time()
passed_time = endtime - starttime
print(f"Total training time: {passed_time / 3600:.2f} hours and {passed_time % 3600 / 60:.2f} minutes.")
print(f"Average time per iteration: {jnp.mean(jnp.array(times_per_iteration)) / 60:.2f} minutes.")
# save trained parameters and optimizer state

with open(f'gumbelmuzero_madn_params_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{config["iterations"]}.pkl', 'wb') as f:
    pickle.dump(params, f)

with open(f'gumbelmuzero_madn_opt_state_lr{config["learning_rate"]}_g{config["num_games_per_iteration"]}_it{config["iterations"]}.pkl', 'wb') as f:
    pickle.dump(opt_state, f)
