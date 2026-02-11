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
                 obs_shape=(14, 56), action_dim=24, max_episode_length=500, bootstrap_value_target=True):
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
        self.bootstrap_value_target = bootstrap_value_target
    
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
        gamma = 1 # Paper sagt 1 für Brettspiele
        
        # ========================================
        # SCHRITT 1: Sample Episode-Indizes
        # ========================================
        ep_indices = np.random.randint(0, self.size, size=self.batch_size)
        # Shape: (batch_size,)
        
        # ========================================
        # SCHRITT 2: Sample Start-Positionen
        # ========================================
        ep_lengths = self.episode_lengths[ep_indices]  # (batch_size,)
        max_starts = np.maximum(ep_lengths - K, 0)     # (batch_size,) minimaler Startpunkt, damit K Schritte möglich sind
        
        # t_starts = (np.random.rand(self.batch_size) * max_starts).astype(np.int32)
        t_starts = np.random.randint(0, max_starts + 1)
        # Shape: (batch_size,)
        
        # ========================================
        # SCHRITT 3: Extrahiere Root Observations
        # ========================================
        root_obs = self.observations[ep_indices, t_starts]
        # Shape: (batch_size, 14, 56)
        
        # ========================================
        # SCHRITT 4: Finale Episode-Daten (für Bootstrap)
        # ========================================
        final_indices = ep_lengths - 1  # Letzter Timestep jeder Episode
        
        final_rewards = self.rewards[ep_indices, final_indices]  # (batch_size,)
        final_players = self.players[ep_indices, final_indices]  # (batch_size,)
        final_teams = self.teams[ep_indices, final_indices]      # (batch_size,)
        
        # ========================================
        # SCHRITT 5: Extrahiere Sequenzen (K Steps)
        # ========================================
        k_offsets = np.arange(K)  # [0, 1, 2, 3, 4, 5] wenn K=6
        seq_indices = t_starts[:, None] + k_offsets[None, :]
        # Shape: (batch_size, K)
        
        # Clip zu Episode-Längen (für Padding am Ende)
        valid_mask = seq_indices < ep_lengths[:, None]
        seq_indices_clipped = np.minimum(seq_indices, ep_lengths[:, None] - 1)
        # Shape: (batch_size, K)
        
        # ========================================
        # SCHRITT 6: Extrahiere alle Daten mit Advanced Indexing
        # ========================================
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
        # SCHRITT 7: Berechne Target Values (Bootstrap) - KORRIGIERT!
        # ========================================
        
        # 7.1: Extrahiere Players/Teams für JEDEN Timestep der Sequenz
        seq_players = self.players[ep_indices_expanded, seq_indices_clipped]  # (batch_size, K)
        seq_teams = self.teams[ep_indices_expanded, seq_indices_clipped]      # (batch_size, K)
        
        # 7.2: Expand finale Werte für Broadcasting
        final_rewards_expanded = final_rewards[:, None]  # (batch_size, 1)
        final_players_expanded = final_players[:, None]  # (batch_size, 1)
        final_teams_expanded = final_teams[:, None]      # (batch_size, 1)
        
        # 7.3: Berechne z FÜR JEDEN TIMESTEP (nicht nur Root!)
        game_won_seq = final_rewards_expanded > 0                    # (batch_size, K)
        is_single_player_seq = seq_teams == -1                       # (batch_size, K)
        player_won_seq = (final_players_expanded == seq_players)     # (batch_size, K)
        team_won_seq = (final_teams_expanded == seq_teams)           # (batch_size, K)
        
        # z = 1.0 wenn gewonnen, -1.0 wenn verloren, 0.0 sonst
        # Jetzt individuell für JEDEN Timestep!
        z_seq = np.where(
            game_won_seq,
            np.where(
                is_single_player_seq,
                np.where(player_won_seq, 1.0, -1.0),
                np.where(team_won_seq, 1.0, -1.0)
            ),
            0.0
        )
        # Shape: (batch_size, K) ← WICHTIG: Nicht mehr (batch_size,)!
        
        # 7.4: Steps bis zum Ende der Episode
        steps_until_end = ep_lengths[:, None] - 1 - seq_indices  # (batch_size, K)
        
        # 7.5: Bootstrap-Condition: steps_until_end >= K
        bootstrap_from_value = (steps_until_end >= K) | (z_seq == 0)  # Bootstrap mit Value wenn noch K Schritte übrig ODER unentschieden (z=0) - KORRIGIERT! (sonst bootstrapped er nur bei unentschieden, was zu wenig ist)
        
        # 7.6: Bootstrap-Indizes (idx + K, aber clipped)
        bootstrap_indices = np.minimum(seq_indices + K, ep_lengths[:, None] - 1)
        bootstrap_values_raw = self.root_values[ep_indices_expanded, bootstrap_indices]
        # Shape: (batch_size, K)
        
        # ✅ NEU: 7.6b - Perspektiven-Flip für Bootstrap Values
        # Extrahiere Spieler bei Bootstrap-Position
        bootstrap_players = self.players[ep_indices_expanded, bootstrap_indices]  # (batch_size, K)
        bootstrap_teams = self.teams[ep_indices_expanded, bootstrap_indices] 

        
        # Check: Gleicher Spieler ODER gleiches Team (wenn Teams aktiv)
        is_team_mode = seq_teams != -1  # (batch_size, K)
        same_player_bootstrap = (seq_players == bootstrap_players)  # (batch_size, K)
        same_team_bootstrap = (seq_teams == bootstrap_teams)   

        same_perspective = np.where(
            is_team_mode,
            same_team_bootstrap,    # Team-Modus: Check Team
            same_player_bootstrap   # Free-For-All: Check Player
        )

        bootstrap_values = np.where(
            same_perspective,
            bootstrap_values_raw,   # Gleiche Perspektive: Value behalten
            -bootstrap_values_raw   # Andere Perspektive: Value negieren
        )
        # 7.7: Berechne Target Values
        target_values = np.where(
            bootstrap_from_value & self.bootstrap_value_target,
            # Falls True: Bootstrap mit Value nach K Steps
            bootstrap_values,  # = bootstrap_values (da gamma=1)
            # Falls False: Bootstrap mit z AUS PERSPEKTIVE DIESES TIMESTEPS
            z_seq  # ← KEIN [:, None] mehr nötig, schon (batch_size, K)!
        )
        # Shape: (batch_size, K)
        
        # ========================================
        # SCHRITT 8: Padding für ungültige Positionen
        # ========================================
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