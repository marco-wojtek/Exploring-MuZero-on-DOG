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

    players: chex.Array  # Spieler bei jedem Zeitschritt
    teams: chex.Array    # Team bei jedem Zeitschritt (0 oder 1)

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

    def save_games_from_buffers(self, all_buffers):
        """Speichert mehrere Episoden aus Buffers."""
        num_envs = all_buffers['idx'].shape[0]
        for i in range(num_envs):
            length = all_buffers['idx'][i]
        
            ep = Episode(
                observations=all_buffers['obs'][i, :length],
                actions=all_buffers['act'][i, :length],
                rewards=all_buffers['rew'][i, :length],
                root_values=all_buffers['val'][i, :length],
                child_visits=all_buffers['pol'][i, :length],
                mask=all_buffers['mask'][i, :length],
                chance_outcomes=jnp.zeros(length),
                players=all_buffers['player'][i, :length],
                teams=all_buffers['team'][i, :length]
            ) 
            self.save_game(ep)

    def sample_batch(self):
        """Optimierte Version mit weniger Overhead."""
        episodes = random.choices(self.buffer, k=self.batch_size)

        game_lens = [len(ep.actions) for ep in episodes]
        t_starts = [random.randint(0, max(0, gl - 1)) for gl in game_lens]

        batch = [self._extract_sequence(ep, t) for ep, t in zip(episodes, t_starts)]
        
        # Stacken zu JAX Arrays
        return jax.tree_util.tree_map(lambda *x: np.stack(x), *batch)

    def _extract_sequence(self, episode, t_start):
        """
        Flexible Implementierung: 
        - teams = -1: Single-Player (mit negativen Rewards für Verlierer)
        - teams = 0 oder 1: Team-Play (Zero-Sum)
        """
        K = self.unroll_steps + 1
        game_len = len(episode.actions)
        
        obs = episode.observations[t_start]
        root_player = episode.players[t_start]
        root_team = episode.teams[t_start]
        
        # Bestimme finalen Reward aus Root-Perspektive
        z = 0.0
        if game_len > 0 and episode.rewards[-1] > 0:
            final_player = episode.players[-1]
            final_team = episode.teams[-1]
            
            if root_team == -1:  # Single-Player
                z = 1.0 if final_player == root_player else -1.0
            else:  # Team-Play
                z = 1.0 if final_team == root_team else -1.0
        
        # Extrahiere Sequenz-Daten
        actions = []
        rewards = []
        policies = []
        values = []
        masks = []
        target_values = []
        gamma = 0.997
        
        for k in range(K):
            idx = t_start + k
            if idx < game_len:
                if k < K - 1:
                    actions.append(episode.actions[idx])
                    rewards.append(episode.rewards[idx])
                
                policies.append(episode.child_visits[idx])
                values.append(episode.root_values[idx])
                masks.append(episode.mask[idx])
                
                steps_until_end = game_len - 1 - idx
            
                if steps_until_end >= K:
                    # Bootstrap mit MCTS Value nach K Steps
                    target_value = (gamma ** K) * episode.root_values[idx + K]
                else:
                    # Bootstrap mit finalem Outcome z
                    target_value = (gamma ** steps_until_end) * z
                    
                target_values.append(target_value)
            else:
                # Padding
                if k < K - 1:
                    actions.append(0)
                    rewards.append(0.0)
                policies.append(jnp.zeros(24))
                values.append(0.0)
                masks.append(0.0)
                target_values.append(0.0)
        
        return {
            'observations': obs,
            'actions': jnp.array(actions),
            'rewards': jnp.array(rewards),  # Wird für Reward Loss verwendet (wenn aktiv)
            'policies': jnp.array(policies),
            'values': jnp.array(values),
            'masks': jnp.array(masks),
            'target_values': jnp.array(target_values)
        }