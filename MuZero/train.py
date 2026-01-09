from time import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
import os, sys
import pickle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# Annahme: Deine Netzwerk-Klassen und init-Funktionen sind importiert
from MuZero.muzero_deterministic_madn import repr_net, dynamics_net, pred_net, init_muzero_params, run_muzero_mcts, load_params_from_file
from MADN.deterministic_madn import env_reset, encode_board, old_encode_board
from MuZero.replay_buffer import ReplayBuffer, Episode, play_deterministic_game_for_training, play_n_games, batch_encode, batch_env_step, batch_map_action, batch_reset, batch_valid_action

@jax.jit
def loss_fn(params, batch):
    """
    Berechnet den MuZero Loss für einen Batch.
    batch ist ein Dictionary/Struct mit:
    - observations: (B, Obs_Shape) -> Startzustand
    - actions: (B, Unroll_Steps) -> Aktionen im Unroll
    - target_values: (B, Unroll_Steps + 1) -> z
    - target_rewards: (B, Unroll_Steps) -> u
    - target_policies: (B, Unroll_Steps + 1, Num_Actions) -> pi
    - sample_weights: (B, Unroll_Steps + 1) -> Maskierung für Padding
    """
    
    # 1. Initial Inference (Representation Network)
    # Wir starten beim Schritt t=0
    root_obs = batch['observations']
    latent_state = repr_net.apply(params['representation'], root_obs)
    
    # Loss Akkumulatoren
    total_loss = 0.0
    value_loss = 0.0
    reward_loss = 0.0
    policy_loss = 0.0
    
    # Wir iterieren durch die Unroll-Schritte (K Schritte)
    # K = Anzahl der Schritte, die wir in die Zukunft schauen (z.B. 5)
    num_unroll_steps = batch['actions'].shape[1]
    
    for k in range(num_unroll_steps + 1):
        # A. Prediction (Policy & Value) für den aktuellen latenten Zustand
        pred_policy_logits, pred_value = pred_net.apply(params['prediction'], latent_state)
        pred_value = pred_value.squeeze(-1) # (B,)
        
        # B. Targets für diesen Schritt holen
        target_value = batch['target_values'][:, k]
        target_policy = batch['policies'][:, k]
        mask = batch['masks'][:, k]
        
        # C. Losses berechnen (Maskiert!)
        
        # Value Loss (MSE oder Cross-Entropy bei kategorischen Values)
        # Hier einfach MSE für den Anfang:
        l_value = jnp.mean(mask * (target_value - pred_value) ** 2)
        
        # Policy Loss (Cross Entropy)
        # target_policy sind Wahrscheinlichkeiten, pred_policy_logits sind Logits
        l_policy = jnp.mean(mask * optax.softmax_cross_entropy(pred_policy_logits, target_policy))
        
        # Reward Loss (nur wenn k > 0, da Reward zum Übergang gehört)
        if k > 0:
            # Der Reward wurde im VORHERIGEN Schritt (Dynamics) vorhergesagt
            # Wir vergleichen pred_reward (aus Schritt k-1) mit target_reward (aus Schritt k-1)
            # Hinweis: In der Schleife unten berechnen wir pred_reward für den NÄCHSTEN Schritt.
            # Daher müssen wir den Reward-Loss eigentlich dort berechnen oder speichern.
            # Einfacher: Wir berechnen Reward Loss direkt beim Dynamics Schritt.
            pass 

        # Skalierung der Losses (Policy oft weniger gewichtet am Anfang)
        scale = 1.0 if k == 0 else 0.5 # Gradient Scale für Recurrent Steps (MuZero Paper)
        
        total_loss += scale * (l_value + l_policy)
        value_loss += l_value
        policy_loss += l_policy
        
        # D. Dynamics Step (nur wenn wir nicht am Ende sind)
        if k < num_unroll_steps:
            action = batch['actions'][:, k] # Aktion, die tatsächlich gespielt wurde
            
            # Dynamics Network anwenden
            latent_state, pred_reward, _ = dynamics_net.apply(params['dynamics'], latent_state, action)
            pred_reward = pred_reward.squeeze(-1)
            
            # Reward Loss berechnen
            target_reward = batch['rewards'][:, k]
            l_reward = jnp.mean(mask * (target_reward - pred_reward) ** 2)
            
            total_loss += scale * l_reward
            reward_loss += l_reward
            
            # Gradient Scaling Hook für den latent state (optional, stabilisiert Training)
            latent_state = jax.lax.stop_gradient(latent_state * 0.5) + latent_state * 0.5

    # L2 Regularization (Weight Decay) macht meist der Optimizer (AdamW)
    
    return total_loss, (value_loss, policy_loss, reward_loss)

@jax.jit
def train_step(params, opt_state, batch):
    """Führt einen Trainingsschritt aus."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (v_loss, p_loss, r_loss)), grads = grad_fn(params, batch)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, {'total_loss': loss, 'v_loss': v_loss, 'p_loss': p_loss, 'r_loss': r_loss}

# --- Setup Optimizer ---
learning_rate = 1e-5
optimizer = optax.adamw(learning_rate)

# --- Initialisierung (Beispiel) ---


def test_training(num_games= 50, seed=42, iterations=100, params=None, opt_state=None):
    env = env_reset(
        seed,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=10,
        starting_player=0,
        seed=seed,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=False,
        enable_initial_free_pin=True,
        enable_circular_board=False
    )
    enc = old_encode_board(env)  # z.B. (8, 56)
    print(enc.shape)
    input_shape = enc.shape  # (8, 56)

    if params is None:
        params = init_muzero_params(jax.random.PRNGKey(0), input_shape)  # Beispiel Input Shape
    
    if opt_state is None:
        opt_state = optimizer.init(params)

    replay = ReplayBuffer(capacity=1000, batch_size=4, unroll_steps=5)
    for it in range(iterations):
        start_time = time()
        print(f"Iteration {it+1}/{iterations}: Playing games to collect training data...")
        eps = play_n_games(params, jax.random.PRNGKey(it**3), num_envs=num_games)
        print("Saving collected games to replay buffer...")
        replay.save_games(eps)
        # for game_idx in range(num_games):
        #     if (game_idx+1) % 10 == 0:
        #         print(f"Playing game {game_idx+1}/{num_games} for training data...")
        #     #print(f"Playing game {game_idx+1}/{num_games} for training data...")
        #     env = env_reset(0, num_players=4, distance=10, enable_initial_free_pin=True, enable_circular_board=False)
        #     episode = play_deterministic_game_for_training(env, params, jax.random.PRNGKey(game_idx))
        #     replay.save_game(episode)

        print("Training on collected data...")
        train_start = time()
        train_steps = 1000
        for i in range(train_steps):  
            batch = replay.sample_batch()
            params, opt_state, losses = train_step(params, opt_state, batch)
            if i % (train_steps // 10) == 0:
                print(f"Step {i}, Losses: {losses}")
        end_time = time()
        print(f"""
              Iteration {it+1} completed in {end_time - start_time:.2f} seconds.
              Game playing + data collection time: {train_start - start_time:.2f} seconds.
              Training time: {end_time - train_start:.2f} seconds.
              """)
    return params, opt_state

params = None
opt_state = None
# params = load_params_from_file('muzero_madn_params_00001.pkl')
# with open('muzero_madn_opt_state_00001.pkl', 'rb') as f:
#     opt_state = pickle.load(f)
params, opt_state = test_training(30, seed=42, iterations=6, params=params, opt_state=opt_state)
# save trained parameters and optimizer state

with open('muzero_madn_params_lr5_g30_it6.pkl', 'wb') as f:
    pickle.dump(params, f)

with open('muzero_madn_opt_state_lr5_g30_it6.pkl', 'wb') as f:
    pickle.dump(opt_state, f)