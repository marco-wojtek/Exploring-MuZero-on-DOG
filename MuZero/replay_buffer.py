import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
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
        return jax.tree_map(lambda *x: np.stack(x), *batch)

    def _extract_sequence(self, episode, t_start):
        """Extrahiert eine Sequenz der Länge unroll_steps + 1."""
        # Wir brauchen unroll_steps + 1 Datenpunkte für das Target Value
        # (Bootstrap Value am Ende)
        
        # Hier Logik implementieren, um Slices aus den Arrays zu schneiden
        # und mit Nullen aufzufüllen (Padding), wenn das Spiel zu Ende ist.
        # ...
        pass

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
        policy_output = run_muzero_mcts(params, subkey, obs, invalid_actions=invalid_mask)
        
        # 4. MCTS Daten speichern
        #jax.debug.print("Policy: {}", policy_output)
        #root_values.append(policy_output.value[0])  # Aus dem Batch extrahieren
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

    print("Final Board:", env.board)
    return Episode(
        observations=jnp.stack(observations),
        actions=jnp.array(actions),
        rewards=jnp.array(rewards),
        root_values=jnp.array([0.0]*len(actions)),  # Dummy Werte, da nicht genutzt
        child_visits=jnp.stack(child_visits),
        chance_outcomes=jnp.array(chance_outcomes),
        mask=jnp.ones(len(actions))
    )


# test play function with untrained NNs
env = env_reset(0, num_players=4, distance=10, enable_initial_free_pin=True, enable_circular_board=False)
enc = encode_board(env)  # z.B. (8, 56)
print(enc.shape)
input_shape = enc.shape  # (8, 56)
parameters = init_muzero_params(jax.random.PRNGKey(0), input_shape)

eps = play_deterministic_game_for_training(env, parameters, jax.random.PRNGKey(1))

print("Episode observations shape:", eps.observations.shape)
print("Episode actions shape:", eps.actions.shape)
print("Episode actions:", eps.actions)
print("Episode rewards shape:", eps.rewards.shape)
print("Episode root values shape:", eps.root_values.shape)
print("Episode child visits shape:", eps.child_visits.shape)
print("Episode chance outcomes shape:", eps.chance_outcomes.shape)