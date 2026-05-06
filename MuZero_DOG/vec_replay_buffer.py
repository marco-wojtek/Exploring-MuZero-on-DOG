import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
from time import time
import numpy as np
import random

class VectorizedReplayBuffer:
    """
    Replay Buffer für MuZero DOG.
    Speichert vollständige Episoden (bis max_episode_length).
    
    Besonderheiten gegenüber SessionReplayBuffer:
    1. Speichert card_outcomes und card_distributions (stochastische Kartenverteilung)
    2. Kein Session-Splitting — vollständige Spiele werden gespeichert
    3. Target values werden vektorisiert beim Speichern (save_games_from_buffers) berechnet,
       nicht erst beim Sampling
    """
    def __init__(self, capacity: int, batch_size: int, unroll_steps: int, td_steps: int,
                 obs_shape: tuple, action_dim=454, max_episode_length=500, bootstrap_value_target=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim 
        self.max_episode_length = max_episode_length
        
        # float16: halves RAM; repr_net upcasts to float32 at its first line
        self.observations = np.zeros((capacity, max_episode_length, *obs_shape), dtype=np.float16)
        # int16: range -1..997 fits in [-32768, 32767]; initial fill -1
        self.actions = np.full((capacity, max_episode_length), -1, dtype=np.int16)
        # int8: class labels 0/1/2 fit in [-128, 127]
        self.rewards = np.zeros((capacity, max_episode_length), dtype=np.int8)
        self.root_values = np.zeros((capacity, max_episode_length), dtype=np.float16)
        # float16: halves RAM; upcast to float32 in sample_batch return for softmax CE
        self.child_visits = np.zeros((capacity, max_episode_length, action_dim), dtype=np.float16)
        self.masks = np.zeros((capacity, max_episode_length), dtype=np.bool_)
        self.card_outcomes = np.zeros((capacity, max_episode_length), dtype=np.uint8)  # 0..127
        self.card_distributions = np.zeros((capacity, max_episode_length, 128), dtype=np.float16)
        self.players = np.zeros((capacity, max_episode_length), dtype=np.int8)  # 0..3
        self.teams = np.zeros((capacity, max_episode_length), dtype=np.int8)    # -1/0/1
        self.episode_lengths = np.zeros(capacity, dtype=np.int32)
        self.discounts = np.zeros((capacity, max_episode_length), dtype=np.int8)  # class labels 0/1/2
        self.target_values = np.zeros((capacity, max_episode_length), dtype=np.float16)
        
        self.position = 0
        self.size = 0
        self.bootstrap_value_target = bootstrap_value_target
    
    def save_games_from_buffers(self, all_buffers):
        """Speichert Batch von Spielen direkt. Berechnet target values vektorisiert beim Speichern."""
        all_buffers = jax.device_get(all_buffers)
        N = all_buffers['idx'].shape[0]
        episode_lengths = all_buffers['idx']
        max_len = int(episode_lengths.max())
        td = self.td_steps
        GAMMA = 1.0

        t = np.arange(max_len)[None, :]                                    # (1, max_len)
        valid = t < episode_lengths[:, None]                               # (N, max_len)

        final_idx  = episode_lengths - 1
        row        = np.arange(N)
        final_rew  = all_buffers['rew'][row, final_idx]                    # (N,)
        final_plr  = all_buffers['player'][row, final_idx]                 # (N,)
        final_team = all_buffers['team'][row, final_idx]                   # (N,)

        players = all_buffers['player'][:, :max_len]                       # (N, max_len)
        teams   = all_buffers['team'][:, :max_len]                         # (N, max_len)
        game_won = (final_rew[:, None] == 2)
        is_team  = teams != -1
        z = np.where(game_won,
                np.where(is_team,
                    np.where(teams == final_team[:, None], 1.0, -1.0),
                    np.where(players == final_plr[:, None], 1.0, -1.0)),
                0.0)                                                       # (N, max_len)

        steps_to_end = np.maximum(0, episode_lengths[:, None] - 1 - t)    # (N, max_len)

        boot_idx      = np.minimum(t + td, episode_lengths[:, None] - 1)
        vals          = all_buffers['val'][:, :max_len].astype(np.float32)
        boot_vals_raw = vals[row[:, None], boot_idx]
        boot_plr  = all_buffers['player'][row[:, None], boot_idx]
        boot_team = all_buffers['team'][row[:, None], boot_idx]
        same_persp = np.where(is_team, teams == boot_team, players == boot_plr)
        boot_vals  = np.where(same_persp, boot_vals_raw, -boot_vals_raw)

        use_bootstrap = ((steps_to_end >= td) & self.bootstrap_value_target) | (z == 0)
        target_values = np.where(use_bootstrap,
            boot_vals * (GAMMA ** np.minimum(td, steps_to_end)),
            z * (GAMMA ** steps_to_end))
        target_values = np.clip(target_values, -1.0, 1.0)
        target_values = np.where(valid, target_values, 0.0)               # (N, max_len)

        for i in range(N):
            length = int(episode_lengths[i])
            if length == 0:
                continue

            pos = self.position
            self.observations[pos, :length]       = all_buffers['obs'][i, :length]
            self.actions[pos, :length]            = all_buffers['act'][i, :length]
            self.rewards[pos, :length]            = all_buffers['rew'][i, :length]
            self.root_values[pos, :length]        = vals[i, :length]
            self.child_visits[pos, :length]       = all_buffers['pol'][i, :length]
            self.masks[pos, :length]              = all_buffers['mask'][i, :length]
            self.card_outcomes[pos, :length]      = all_buffers['card_outcome'][i, :length]
            self.card_distributions[pos, :length] = all_buffers['card_dist'][i, :length]
            self.players[pos, :length]            = all_buffers['player'][i, :length]
            self.teams[pos, :length]              = all_buffers['team'][i, :length]
            self.discounts[pos, :length]          = all_buffers['discount'][i, :length]
            self.target_values[pos, :length]      = target_values[i, :length]
            self.episode_lengths[pos]             = length

            self.position = (pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self):
        """
        Vollständig vektorisierte Sampling-Funktion für MuZero DOG.
        KEIN Python-Loop über batch_size!
        Gibt card_outcomes und card_distributions zurück für das Training des chance_dynamics.
        """
        K = self.unroll_steps + 1
        TERMINAL_RATIO = 0.25  # 25% des Batches enthält Terminal-Steps

        n_terminal = int(self.batch_size * TERMINAL_RATIO)
        n_normal = self.batch_size - n_terminal
        
        # ========================================
        # SCHRITT 1: Sample Episode-Indizes
        # ========================================
        # --- Normal Sampling: kann an JEDER Position starten ---
        # Auch nahe am Ende! Dann gibt es partielle Windows (mask=0 für padding)
        # aber Terminal-Steps können natürlich im Dynamics-Bereich landen
        ep_indices_normal = np.random.randint(0, self.size, size=n_normal)
        ep_lengths_normal = self.episode_lengths[ep_indices_normal]
        max_starts_normal = ep_lengths_normal - 1  # kann überall starten
        t_starts_normal = np.random.randint(0, max_starts_normal + 1, size=n_normal)
        
        # --- Terminal Sampling: Terminal-Step an ZUFÄLLIGER Position k im Fenster ---
        # Nicht immer k=9! Bei k=0 kommt latent_state direkt aus RepNet → beste Qualität
        ep_indices_terminal = np.random.randint(0, self.size, size=n_terminal)
        ep_lengths_terminal = self.episode_lengths[ep_indices_terminal]
        # terminal_k = zufällige Position (0..K-2) wo der letzte Step der Episode landen soll
        # K-1 = 10 Positionen für Actions (k=0..9), davon nutzen wir k=0..K-2
        max_terminal_k = np.minimum(self.unroll_steps - 1, ep_lengths_terminal - 1)  # kann nicht vor Episode-Start
        terminal_k = np.floor(np.random.uniform(0, 1, size=n_terminal) * (max_terminal_k + 1)).astype(np.int32)
        # t_start so setzen dass ep_length-1 (letzter Step) bei Position terminal_k liegt
        t_starts_terminal = np.maximum(ep_lengths_terminal - 1 - terminal_k, 0)
        
        # --- Zusammenführen ---
        ep_indices = np.concatenate([ep_indices_normal, ep_indices_terminal])
        t_starts = np.concatenate([t_starts_normal, t_starts_terminal])
        ep_lengths = self.episode_lengths[ep_indices]
        # Shape: (batch_size,)
        
        # ========================================
        # SCHRITT 2: Extrahiere Root Observations
        # ========================================
        root_obs = self.observations[ep_indices, t_starts]
        # Shape: (batch_size, *obs_shape)
        
        # ========================================
        # SCHRITT 3: Extrahiere Sequenzen (K Steps)
        # ========================================
        k_offsets = np.arange(K)  # [0, 1, 2, 3, 4, 5] wenn K=6
        seq_indices = t_starts[:, None] + k_offsets[None, :]
        # Shape: (batch_size, K)
        
        # Clip zu Episode-Längen (für Padding am Ende)
        valid_mask = seq_indices < ep_lengths[:, None]
        seq_indices_clipped = np.minimum(seq_indices, ep_lengths[:, None] - 1)
        # Shape: (batch_size, K)
        
        # ========================================
        # SCHRITT 4: Extrahiere alle Daten mit Advanced Indexing
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
        
        # Extrahiere Card Outcomes (nur für k=0..K-2)
        card_outcomes = self.card_outcomes[ep_for_actions, action_indices]
        # Shape: (batch_size, K-1)
        card_probs_seq = self.card_distributions[ep_for_actions, action_indices]
        # Shape: (batch_size, K-1, 128)
        
        # Extrahiere Policies, Values, Masks (für alle K Steps)
        policies = self.child_visits[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K, action_dim)
        
        values = self.root_values[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K)
        
        masks = self.masks[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K)
        
        discount_targets = self.discounts[ep_for_actions, action_indices]
        # ========================================
        # SCHRITT 5: Vorberechnete Target Values lesen
        # ========================================
        target_values = self.target_values[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K)

        # ========================================
        # SCHRITT 6: Padding für ungültige Positionen
        # ========================================
        actions = np.where(valid_mask[:, :-1], actions, 0)
        rewards_seq = np.where(valid_mask[:, :-1], rewards_seq, 1)  # Klasse 1 = reward=0 (neutral)
        card_outcomes = np.where(valid_mask[:, :-1], card_outcomes, 0)
        # Uniform padding (1/128): verhindert is_non_uniform=True für gepaddte Positionen
        uniform_dist = np.zeros(128, dtype=np.float32)
        uniform_dist[0] = 1.0  # Alle Karten weg → nur leere Hand möglich
        card_probs_seq = np.where(valid_mask[:, :-1, None], card_probs_seq, uniform_dist)
        policies = np.where(valid_mask[:, :, None], policies, 0.0)
        values = np.where(valid_mask, values, 0.0)
        masks = np.where(valid_mask, masks, 0.0)
        target_values = np.where(valid_mask, target_values, 0.0)
        discount_targets = np.where(valid_mask[:, :-1], discount_targets, 1)  # Klasse 1 = discount=0 (neutral)
        
        # ========================================
        # SCHRITT 7: Return Batch
        # ========================================
        return {
            'observations': jnp.array(root_obs),                            
            'actions': jnp.array(actions),                                  
            'rewards': jnp.array(rewards_seq),                                
            'card_outcomes': jnp.array(card_outcomes),                       
            'card_probs': jnp.array(card_probs_seq, dtype=jnp.float32),      
            'policies': jnp.array(policies, dtype=jnp.float32),              
            'values': jnp.array(values),                             
            'masks': jnp.array(masks, dtype=jnp.float32),                     
            'target_values': jnp.array(target_values),                        
            'discount_targets': jnp.array(discount_targets)                   
        }
