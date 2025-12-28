import chex
import jax
import jax.numpy as jnp
from flax import struct

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

def play_game_for_training(env, params, rng_key):
    observations = []
    actions = []
    rewards = []
    root_values = []
    child_visits = []
    chance_outcomes = []
    
    step = 0
    while not env.done:
        # 1. Observation speichern (bevor wir agieren)
        obs = encode_board(env)
        observations.append(obs)
        
        # 2. MCTS ausführen
        rng_key, subkey = jax.random.split(rng_key)
        policy_output = run_mcts(env, params, subkey)
        
        # 3. Daten aus MCTS speichern
        # Der geschätzte Wert des aktuellen Zustands
        root_values.append(policy_output.value) 
        # Die Wahrscheinlichkeiten der Aktionen (Besuchszahlen)
        child_visits.append(policy_output.action_weights)
        
        # 4. Aktion wählen (basierend auf MCTS Policy)
        action = policy_output.action_selected
        actions.append(action)
        
        # 5. Environment Schritt
        # Hier ist der Trick bei Stochastic MuZero:
        # Wir müssen wissen, was gewürfelt wurde, um es dem Netz beizubringen.
        
        # a) Würfeln (Chance Node)
        rng_key, subkey = jax.random.split(rng_key)
        env_after_die = throw_die(env, subkey)
        chance_outcomes.append(env_after_die.die - 1) # 0-5 Index
        
        # b) Ziehen (Decision Node)
        env_next, reward, done = env_step(env_after_die, action)
        rewards.append(reward)
        
        env = env_next
        step += 1

    # Am Ende alles in ein Episode-Objekt packen
    return Episode(
        observations=jnp.stack(observations),
        actions=jnp.array(actions),
        rewards=jnp.array(rewards),
        root_values=jnp.array(root_values),
        child_visits=jnp.stack(child_visits),
        chance_outcomes=jnp.array(chance_outcomes),
        mask=jnp.ones(len(actions))
    )