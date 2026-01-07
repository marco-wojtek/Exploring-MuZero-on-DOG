import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
from time import time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MADN.deterministic_madn import *
from MuZero.muzero_deterministic_madn import *
@struct.dataclass
class Episode:
    # Beobachtungen (Encoded Board States)
    # Shape: (T, H, W, C) oder (T, Features)
    observations: chex.Array 
    
    # Aktionen, die in diesen Zuständen gewählt wurden
    # Shape: (T,)
    actions: chex.Array
    
    # Rewards, die NACH der Aktion erhalten wurden
    # Shape: (T,)
    rewards: chex.Array
    
    # Der "Value", der vom MCTS für diesen Zustand berechnet wurde (Target für Value-Netz)
    # Shape: (T,)
    root_values: chex.Array
    
    # Die Policy-Verteilung vom MCTS (Target für Policy-Netz)
    # Shape: (T, Num_Actions)
    child_visits: chex.Array
    
    # Maske, um Padding am Ende zu ignorieren (falls Spiele unterschiedlich lang sind)
    # Shape: (T,)
    mask: chex.Array
    
    # Optional: Würfelwürfe (für Stochastic MuZero Analyse, nicht zwingend für Training)
    chance_outcomes: chex.Array

import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, unroll_steps: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.buffer = []
        self.position = 0

    def save_game(self, episode: Episode):
        """Speichert eine fertige Episode."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity
    
    def save_games(self, episodes):
        """Speichert mehrere Episoden."""
        for episode in episodes:
            self.save_game(episode)

    def sample_batch(self):
        """
        Zieht einen Batch für das Training.
        Wichtig: Wir brauchen für jeden Eintrag im Batch eine Startposition (t)
        und dann die Daten für t, t+1, ..., t+unroll_steps.
        """
        batch = []
        for _ in range(self.batch_size):
            episode = random.choice(self.buffer)
            
            # Wähle einen zufälligen Startpunkt im Spiel
            # Wir müssen sicherstellen, dass wir noch 'unroll_steps' weit gehen können
            # oder wir padden später.
            game_len = len(episode.actions)
            t_start = random.randint(0, game_len - 1)
            
            batch.append(self._extract_sequence(episode, t_start))
            
        # Stacken zu JAX Arrays
        return jax.tree_util.tree_map(lambda *x: np.stack(x), *batch)

    def _extract_sequence(self, episode, t_start):
        """Extrahiert eine Sequenz der Länge unroll_steps + 1."""
        K = self.unroll_steps + 1
        end = t_start + K
        seq = {}

        indices = jnp.arange(t_start, end)
        
        # Beobachtungen extrahieren
        obs = episode.observations[t_start]

        actions = []
        rewards = []
        policies = []
        values = [] # Bootstrap values
        masks = []

        game_len = len(episode.actions)
        for k in range(K):
            idx = t_start + k
            if idx < game_len:
                if k < K - 1:
                    actions.append(episode.actions[idx])
                    rewards.append(episode.rewards[idx])
                
                policies.append(episode.child_visits[idx])
                values.append(episode.root_values[idx])
                masks.append(1.0)
            else:
                if k < K - 1:
                    actions.append(0)  # Dummy Aktion
                    rewards.append(0.0)  # Dummy Reward
                
                policies.append(jnp.zeros_like(episode.child_visits[0]))  # Dummy Policy
                values.append(0.0)  # Dummy Value
                masks.append(0.0)  # Padding Maske
                
        # Target values berechen (Bootstrapping)
        # z_t = u_{t+1} + gamma * u_{t+2} + ... + gamma^(n-1) * v_{t+n}

        target_values = []
        gamma = 0.99
        steps = 5 # Anzahl der Schritte für das Bootstrapping

        for k in range(K):
            bootstrap_idx = t_start + k + steps

            value = 0
            current_gamma = 1.0

            for n in range(steps):
                reward_idx = t_start + k + n
                if reward_idx < game_len:
                    reward = episode.rewards[reward_idx]
                    value += current_gamma * reward
                    current_gamma *= gamma
                else:
                    break

        if bootstrap_idx < game_len:
            value += current_gamma * episode.root_values[bootstrap_idx]
        target_values.append(value)

        seq['observations'] = obs
        seq['actions'] = jnp.array(actions)
        seq['rewards'] = jnp.array(rewards)
        seq['policies'] = jnp.array(policies)
        seq['values'] = jnp.array(values)
        seq['masks'] = jnp.array(masks)
        seq['target_values'] = jnp.array(target_values)

        return seq

def play_classic_game_for_training(env, params, rng_key):
    observations = []
    actions = []
    rewards = []
    root_values = []
    child_visits = []
    chance_outcomes = []
    
    # Sicherstellen, dass wir einen initialen Würfelwurf haben
    if env.die == 0:
        rng_key, subkey = jax.random.split(rng_key)
        env = throw_die(env, subkey)

    while not env.done:
        # 1. Prüfen ob Züge möglich sind
        valid_mask = valid_action(env)
        
        if not jnp.any(valid_mask):
            # SKIP-LOGIK:
            # Wenn keine Aktion möglich ist, führen wir no_step aus und würfeln neu.
            # Wir speichern diesen Zustand NICHT für das Training.
            env, _, _ = no_step(env)
            
            # Wichtig: Der nächste Spieler braucht einen Würfelwurf
            rng_key, subkey = jax.random.split(rng_key)
            env = throw_die(env, subkey)
            continue

        # 2. Observation speichern
        obs = encode_board(env)
        observations.append(obs)
        
        # 3. MCTS ausführen
        rng_key, subkey = jax.random.split(rng_key)
        policy_output = run_mcts(env, params, subkey)
        
        # 4. MCTS Daten speichern
        root_values.append(policy_output.value) 
        child_visits.append(policy_output.action_weights)
        
        # 5. Aktion wählen & ausführen
        action = policy_output.action_selected
        actions.append(action)
        
        # Decision Step (Pin bewegen)
        # env_step gibt (env, reward, done) zurück
        env_next, reward, done = env_step(env, map_action(env, action))
        rewards.append(reward)
        
        # 6. Chance Step (Würfeln für den nächsten Zustand)
        # Das ist entscheidend für Stochastic MuZero:
        # Der Übergang ist: State + Action -> Afterstate -> Chance -> Next State
        if not done:
            rng_key, subkey = jax.random.split(rng_key)
            env_next = throw_die(env_next, subkey)
            # Wir speichern, was gewürfelt wurde (0-5), damit das Netz es lernen kann
            chance_outcomes.append(env_next.die - 1)
        else:
            chance_outcomes.append(0) # Dummy wert am Spielende

        env = env_next

    return Episode(
        observations=jnp.stack(observations),
        actions=jnp.array(actions),
        rewards=jnp.array(rewards),
        root_values=jnp.array(root_values),
        child_visits=jnp.stack(child_visits),
        chance_outcomes=jnp.array(chance_outcomes),
        mask=jnp.ones(len(actions))
    )

def play_deterministic_game_for_training(env, params, rng_key):
    observations = []
    actions = []
    rewards = []
    root_values = []
    child_visits = []
    chance_outcomes = [] # Wird hier nicht benötigt, aber für Einheitlichkeit gefüllt
    
    while not env.done:
        # 1. Prüfen ob Züge möglich sind
        valid_mask = valid_action(env)
        
        if not jnp.any(valid_mask):
            # SKIP-LOGIK:
            # Einfach zum nächsten Spieler wechseln, nichts speichern.
            env, _, _ = no_step(env)
            continue

        # 2. Observation speichern
        obs = encode_board(env)[None, ...]  # Batch-Dimension hinzufügen   
        observations.append(obs)
        
        # 3. MCTS ausführen
        rng_key, subkey = jax.random.split(rng_key)
        valid_mask = valid_action(env).flatten()  # (24,)
        invalid_mask = ~valid_mask[None, :] 
        policy_output, root_value = run_muzero_mcts(params, subkey, obs, invalid_actions=invalid_mask)
        
        # 4. MCTS Daten speichern
        # jax.debug.print("Policy: {}", policy_output)
        root_values.append(root_value[0])  # Aus dem Batch extrahieren
        child_visits.append(policy_output.action_weights[0])  # Aus dem Batch extrahieren
        
        # 5. Aktion wählen & ausführen
        action = policy_output.action[0] # Aus dem Batch extrahieren
        actions.append(action)
        
        # Environment Schritt (beinhaltet Spielerwechsel und Action-Set Update)
        env_next, reward, done = env_step(env, map_action(action))
        rewards.append(reward)
        
        # Deterministic: Kein Würfelwurf nötig
        chance_outcomes.append(0) 
        
        env = env_next

    # print("Final Board:", env.board)
    return Episode(
        observations=jnp.stack(observations),
        actions=jnp.array(actions),
        rewards=jnp.array(rewards),
        root_values=jnp.array(root_values),
        child_visits=jnp.stack(child_visits),
        chance_outcomes=jnp.array(chance_outcomes),
        mask=jnp.ones(len(actions))
    )


# test play function with untrained NNs
# env = env_reset(0, num_players=4, distance=10, enable_initial_free_pin=True, enable_circular_board=False)
# enc = encode_board(env)  # z.B. (8, 56)
# print(enc.shape)
# input_shape = enc.shape  # (8, 56)
# parameters = init_muzero_params(jax.random.PRNGKey(0), input_shape)

# eps = play_deterministic_game_for_training(env, parameters, jax.random.PRNGKey(1))

# print("Episode observations shape:", eps.observations.shape)
# print("Episode actions shape:", eps.actions.shape)
# print("Episode actions:", eps.actions)
# print("Episode rewards shape:", eps.rewards.shape)
# print("Episode rewards:", eps.rewards)
# print("Episode root values shape:", eps.root_values.shape)
# print("Episode root values:", eps.root_values)
# print("Episode child visits shape:", eps.child_visits.shape)
# print("Episode chance outcomes shape:", eps.chance_outcomes.shape)

def env_reset_batched(seed):
    return env_reset(
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

# 2. Vektorisierte Funktionen vorbereiten
batch_reset = jax.vmap(env_reset_batched)
batch_valid_action = jax.vmap(valid_action)
batch_encode = jax.vmap(old_encode_board)
batch_env_step = jax.vmap(env_step, in_axes=(0, 0))
batch_map_action = jax.vmap(map_action)

@jax.jit
def actor_step(envs, params, rng_key):
    """
    Führt einen Schritt für N parallele Spiele aus.
    """
    # A. Observations
    obs = batch_encode(envs)
    
    # B. MCTS
    # Maskierung für beendete Spiele:
    # Wenn env.done ist, sind alle Actions invalid. Das führt zu NaNs im MCTS.
    # Wir erzwingen eine valide Action (z.B. 0) für done-Envs, damit MCTS durchläuft.
    # Das Ergebnis wird später ignoriert.
    
    val_actions = batch_valid_action(envs).reshape(envs.board.shape[0], -1)
    dones = envs.done

    def perform_action(env, obs, val_act, done, params, rng_key):
        def mcts_step():
            obs_batched = obs[None, ...] 
            invalid_actions_batched = (~val_act)[None, ...]
            
            policy_output, root_values = run_muzero_mcts(params, rng_key, obs_batched, invalid_actions=invalid_actions_batched)
            
            # Ergebnisse aus dem Batch (Größe 1) extrahieren
            act = policy_output.action[0]
            action_weights = policy_output.action_weights[0]
            root_value = root_values[0]
            
            mapped_act = map_action(act)
            next_env, reward, next_done = env_step(env, mapped_act)
            return next_env, obs, act, reward, root_value, action_weights, next_done
        def no_action_step():
            next_env, reward, next_done = no_step(env)
            dummy_action = jnp.int32(-1)
            dummy_policy = jnp.zeros_like(val_act, dtype=jnp.float32)
            dummy_root_value = 0.0
            return next_env, obs, dummy_action, reward, dummy_root_value, dummy_policy, next_done

        return jax.lax.cond(
            jnp.any(val_act) & (~done),
            mcts_step,
            no_action_step
        )
    
    # # Falls done, setze Action 0 als valid (Dummy), damit MCTS nicht crasht
    # force_valid = jnp.zeros_like(val_actions).at[:, 0].set(1)
    # safe_val_actions = jnp.where(dones[:, None], force_valid, val_actions)
    
    # invalid_mask = ~safe_val_actions.astype(jnp.bool_)
    
    # policy_output, root_values = run_muzero_mcts(params, rng_key, obs, invalid_actions=invalid_mask)
    
    # # C. Action Selection
    # actions = policy_output.action
    
    # # D. Environment Step
    # mapped_actions = batch_map_action(actions)
    
    # # Step ausführen
    # next_envs, rewards, next_dones = batch_env_step(envs, mapped_actions)

    next_envs, obs, actions, rewards, root_values, policy_output_action_weights, next_dones = jax.vmap(perform_action, in_axes=(0, 0, 0, 0, None, 0))(
        envs, obs, val_actions, dones, params, jax.random.split(rng_key, envs.board.shape[0])
    )
    
    # Falls ein Environment schon 'done' war, setzen wir Reward auf 0
    rewards = jnp.where(dones, 0.0, rewards)
    final_dones = jnp.logical_or(dones, next_dones)
    
    return next_envs, (obs, actions, rewards, root_values, policy_output_action_weights, final_dones)

def play_n_games(params, rng_key, num_envs=20):
    """
    Spielt num_envs Spiele parallel und gibt eine Liste von Episoden zurück.
    """
    # 1. Initialisierung der Environments
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs,), 0, 1000000)
    envs = batch_reset(seeds)
    
    # Buffer für jedes Environment (Liste von Listen)
    buffers = [ {'obs': [], 'act': [], 'rew': [], 'val': [], 'pol': []} for _ in range(num_envs) ]
    
    active_mask = np.ones(num_envs, dtype=bool)
    completed_episodes = [None] * num_envs
    
    step_counter = 0
    MAX_STEPS = 2000 # Sicherheitsabbruch, falls Spiele hängen
    
    # Loop solange noch mindestens ein Spiel läuft
    while np.any(active_mask) and step_counter < MAX_STEPS:
        step_counter += 1
        rng_key, subkey = jax.random.split(rng_key)
        
        # JIT-Step ausführen (läuft auf GPU/TPU für alle Envs gleichzeitig)
        envs, data = actor_step(envs, params, subkey)
        
        # Daten auf CPU holen für Listen-Operationen
        obs, acts, rews, vals, pols, dones = jax.device_get(data)
        
        for i in range(num_envs):
            if active_mask[i]:
                # Daten speichern
                buffers[i]['obs'].append(obs[i])
                buffers[i]['act'].append(acts[i])
                buffers[i]['rew'].append(rews[i])
                buffers[i]['val'].append(vals[i])
                buffers[i]['pol'].append(pols[i])
                
                # Check ob fertig
                if dones[i]:
                    active_mask[i] = False
                    
                    # Episode erstellen
                    ep = Episode(
                        observations=np.stack(buffers[i]['obs']),
                        actions=np.array(buffers[i]['act']),
                        rewards=np.array(buffers[i]['rew']),
                        root_values=np.array(buffers[i]['val']),
                        child_visits=np.stack(buffers[i]['pol']),
                        mask=np.ones(len(buffers[i]['act'])),
                        chance_outcomes=np.zeros(len(buffers[i]['act']))
                    )
                    completed_episodes[i] = ep
    
    # Filtern von None (falls MAX_STEPS erreicht wurde)
    return [ep for ep in completed_episodes if ep is not None]

# env = env_reset(0, num_players=4, distance=10, enable_initial_free_pin=True, enable_circular_board=False)
# enc = old_encode_board(env)  # z.B. (8, 56)
# input_shape = enc.shape  # (8, 56)
# print(input_shape)
# parameters = init_muzero_params(jax.random.PRNGKey(0), input_shape)
# batch_size = 1
# print(f"Spiele {batch_size} Spiele parallel...")
# start_time = time()
# eps = play_n_games(parameters, jax.random.PRNGKey(192), num_envs=batch_size)
# print(f"Played {len(eps)} games in parallel.")
# end_time = time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")
# for i, ep in enumerate(eps):
#     print(f"Episode {i}, Length: {len(ep.actions)}")
#     print("Actions:", ep.actions)
#     print("Rewards:", ep.rewards)

# for a in ep.actions:
#     if a == -1:
#         print(env.action_set)
#         print(env.current_player)
#         print(env.board)
#         env, r, d = no_step(env)
#         print(f"Action: {a}, No Step executed. Reward: {r}, Done: {d}")
        
#         continue
#     ma = map_action(a)
#     env, r, d = env_step(env, ma)
#     print(f"Action: {a}, Mapped: {ma}, Reward: {r}, Done: {d}")
#     print(env.board)
# rng_key, subkey = jax.random.split(jax.random.PRNGKey(1))
# seeds = jax.random.randint(subkey, (batch_size,), 0, 1000000)
# envs = batch_reset(seeds)

# # flattened valid actions
# val_actions = batch_valid_action(envs).reshape(batch_size, -1)

# # get indices of valid actions for each env
# val_act_idx = [jnp.where(val_actions[i])[0] for i in range(10)]
# # print("Valid Actions Indices:", val_act_idx)

# # pick random action for each env from valid actions
# random_actions = jnp.array([random.choice(val_act_idx[i]) for i in range(batch_size)])

# # map selected actions
# mapped = batch_map_action(random_actions)

# # print("Mapped Actions:", mapped)

# # perform step and read rewards
# next_envs, rewards, dones = batch_env_step(envs, mapped)
# print("Rewards after step:", rewards)

# print("Board of all env after step:\n", next_envs.board)