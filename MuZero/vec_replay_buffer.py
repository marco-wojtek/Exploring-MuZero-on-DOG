import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
from time import time
import numpy as np
import random
class VectorizedReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, unroll_steps: int,
                 obs_shape=(14, 56), action_dim=24, max_episode_length=500):
        self.capacity = capacity
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.max_episode_length = max_episode_length
        
        # ✅ Alle Daten als zusammenhängende NumPy Arrays
        self.observations = np.zeros((capacity, max_episode_length, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, max_episode_length), dtype=np.int32)
        self.rewards = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.root_values = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.child_visits = np.zeros((capacity, max_episode_length, action_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.players = np.zeros((capacity, max_episode_length), dtype=np.int32)
        self.teams = np.zeros((capacity, max_episode_length), dtype=np.int32)
        self.episode_lengths = np.zeros(capacity, dtype=np.int32)
        
        self.position = 0
        self.size = 0
    
    def save_games_from_buffers(self, all_buffers):
        """Speichert Batch von Spielen direkt."""
        num_games = all_buffers['idx'].shape[0]
        episode_lengths = np.array(all_buffers['idx'])
        
        for i in range(num_games):
            pos = self.position
            length = int(episode_lengths[i])
            
            if length == 0:
                continue
            
            # Kopiere Daten (NumPy ist hier schnell)
            self.observations[pos, :length] = np.array(all_buffers['obs'][i, :length])
            self.actions[pos, :length] = np.array(all_buffers['act'][i, :length])
            self.rewards[pos, :length] = np.array(all_buffers['rew'][i, :length])
            self.root_values[pos, :length] = np.array(all_buffers['val'][i, :length])
            self.child_visits[pos, :length] = np.array(all_buffers['pol'][i, :length])
            self.masks[pos, :length] = np.array(all_buffers['mask'][i, :length])
            self.players[pos, :length] = np.array(all_buffers['player'][i, :length])
            self.teams[pos, :length] = np.array(all_buffers['team'][i, :length])
            self.episode_lengths[pos] = length
            
            self.position = (pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self):
        """
        Vollständig vektorisierte Sampling-Funktion.
        KEIN Python-Loop über batch_size!
        """
        K = self.unroll_steps + 1
        gamma = 0.99
        
        # ========================================
        # SCHRITT 1: Sample Episode-Indizes
        # ========================================
        # Vorher: episodes = random.choices(self.buffer, k=self.batch_size)
        # Jetzt: Direkte Integer-Indizes
        ep_indices = np.random.randint(0, self.size, size=self.batch_size)
        # Shape: (batch_size,)
        # Beispiel: [42, 187, 9, 531, ...]
        
        # ========================================
        # SCHRITT 2: Sample Start-Positionen
        # ========================================
        # Vorher: t_starts = [random.randint(0, max(0, gl - 1)) for gl in game_lens]
        # Jetzt: Vektorisiert über alle Episodes
        
        ep_lengths = self.episode_lengths[ep_indices]  # (batch_size,)
        max_starts = np.maximum(ep_lengths - 1, 0)     # (batch_size,)
        
        # Random start positions (uniform über [0, max_start])
        t_starts = (np.random.rand(self.batch_size) * max_starts).astype(np.int32)
        # Shape: (batch_size,)
        # Beispiel: [23, 87, 5, 142, ...]
        
        # ========================================
        # SCHRITT 3: Extrahiere Root Observations
        # ========================================
        # Vorher: obs = episode.observations[t_start]
        # Jetzt: Batch-Indexing (sehr schnell!)
        
        root_obs = self.observations[ep_indices, t_starts]
        # Shape: (batch_size, 14, 56)
        
        root_players = self.players[ep_indices, t_starts]  # (batch_size,)
        root_teams = self.teams[ep_indices, t_starts]      # (batch_size,)
        
        # ========================================
        # SCHRITT 4: Berechne finale Rewards (z)
        # ========================================
        # Für jeden Batch-Eintrag den finalen Zustand holen
        final_indices = ep_lengths - 1  # Letzter Timestep jeder Episode
        
        final_rewards = self.rewards[ep_indices, final_indices]  # (batch_size,)
        final_players = self.players[ep_indices, final_indices]  # (batch_size,)
        final_teams = self.teams[ep_indices, final_indices]      # (batch_size,)
        
        # Vektorisierte Reward-Berechnung
        game_won = final_rewards > 0
        is_single_player = root_teams == -1
        player_won = final_players == root_players
        team_won = final_teams == root_teams
        
        # z = 1.0 wenn gewonnen, -1.0 wenn verloren, 0.0 sonst
        z = np.where(
            game_won,
            np.where(
                is_single_player,
                np.where(player_won, 1.0, -1.0),
                np.where(team_won, 1.0, -1.0)
            ),
            0.0
        )
        # Shape: (batch_size,)
        
        # ========================================
        # SCHRITT 5: Extrahiere Sequenzen (K Steps)
        # ========================================
        # Vorher: Python-Loop über K
        # Jetzt: Broadcasting!
        
        # Erstelle Offset-Array für K Steps
        k_offsets = np.arange(K)  # [0, 1, 2, 3, 4, 5] wenn K=6
        # Shape: (K,)
        
        # Broadcast: t_starts[:, None] + k_offsets[None, :]
        seq_indices = t_starts[:, None] + k_offsets[None, :]
        # Shape: (batch_size, K)
        # Beispiel für batch_size=2:
        # [[23, 24, 25, 26, 27, 28],
        #  [87, 88, 89, 90, 91, 92]]
        
        # Clip zu Episode-Längen (für Padding am Ende)
        valid_mask = seq_indices < ep_lengths[:, None]
        seq_indices_clipped = np.minimum(seq_indices, ep_lengths[:, None] - 1)
        # Shape: (batch_size, K)
        
        # ========================================
        # SCHRITT 6: Extrahiere alle Daten mit Advanced Indexing
        # ========================================
        # NumPy Advanced Indexing: super schnell!
        
        # Für ep_indices müssen wir auf gleiche Shape bringen
        ep_indices_broadcast = ep_indices[:, None]  # (batch_size, 1)
        ep_indices_expanded = np.broadcast_to(ep_indices_broadcast, (self.batch_size, K))
        # Shape: (batch_size, K)
        
        # Extrahiere Actions (nur für k=0..K-2)
        action_indices = seq_indices_clipped[:, :-1]  # (batch_size, K-1)
        ep_for_actions = ep_indices_expanded[:, :-1]
        
        actions = self.actions[ep_for_actions, action_indices]
        # Shape: (batch_size, K-1)
        
        rewards_seq = self.rewards[ep_for_actions, action_indices]
        # Shape: (batch_size, K-1)
        
        # Extrahiere Policies, Values, Masks (für alle K Steps)
        policies = self.child_visits[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K, 24)
        
        values = self.root_values[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K)
        
        masks = self.masks[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K)
        
        # ========================================
        # SCHRITT 7: Berechne Target Values (Bootstrap)
        # ========================================
        # Vorher: Python-Loop mit if-else
        # Jetzt: Vektorisiert!
        
        # Steps bis zum Ende der Episode
        steps_until_end = ep_lengths[:, None] - 1 - seq_indices
        # Shape: (batch_size, K)
        
        # Bootstrap-Condition: steps_until_end >= K
        bootstrap_from_value = steps_until_end >= K
        
        # Bootstrap-Indizes (idx + K, aber clipped)
        bootstrap_indices = np.minimum(seq_indices + K, ep_lengths[:, None] - 1)
        bootstrap_values = self.root_values[ep_indices_expanded, bootstrap_indices]
        # Shape: (batch_size, K)
        
        # Berechne Target Values mit np.where (wie ternary operator)
        target_values = np.where(
            bootstrap_from_value,
            # Falls True: Bootstrap mit Value
            (gamma ** K) * bootstrap_values,
            # Falls False: Bootstrap mit z
            (gamma ** np.maximum(steps_until_end, 0)) * z[:, None]
        )
        # Shape: (batch_size, K)
        
        # ========================================
        # SCHRITT 8: Padding für ungültige Positionen
        # ========================================
        # Setze Werte auf 0, wo seq_indices >= episode_length
        
        actions = np.where(valid_mask[:, :-1], actions, 0)
        rewards_seq = np.where(valid_mask[:, :-1], rewards_seq, 0.0)
        policies = np.where(valid_mask[:, :, None], policies, 0.0)
        values = np.where(valid_mask, values, 0.0)
        masks = np.where(valid_mask, masks, 0.0)
        target_values = np.where(valid_mask, target_values, 0.0)
        
        # ========================================
        # SCHRITT 9: Return Batch
        # ========================================
        return {
            'observations': jnp.array(root_obs),       # (batch_size, 14, 56)
            'actions': jnp.array(actions),             # (batch_size, K-1)
            'rewards': jnp.array(rewards_seq),         # (batch_size, K-1)
            'policies': jnp.array(policies),           # (batch_size, K, 24)
            'values': jnp.array(values),               # (batch_size, K)
            'masks': jnp.array(masks),                 # (batch_size, K)
            'target_values': jnp.array(target_values)  # (batch_size, K)
        }