import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
from time import time
import numpy as np
import random

class SessionReplayBuffer:
    """
    Session based buffer. Only condiders sessions of card swapping + subsequent play, not the initial deal or subsequent distributions.
    """
    def __init__(self, capacity: int, batch_size: int, unroll_steps: int, td_steps: int,
                 obs_shape: tuple, action_dim=454, bootstrap_value_target=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim 
        max_episode_length = 28 # Maximum 4 swaps + 4*6 cards played
        self.max_episode_length = max_episode_length
        
        self.observations = np.zeros((capacity, max_episode_length, *obs_shape), dtype=np.float16)
        self.actions = np.full((capacity, max_episode_length), -1, dtype=np.int16)
        self.rewards = np.zeros((capacity, max_episode_length), dtype=np.int8)
        self.root_values = np.zeros((capacity, max_episode_length), dtype=np.float16)
        self.target_values = np.zeros((capacity, max_episode_length), dtype=np.float16)
        self.policies = np.zeros((capacity, max_episode_length, action_dim), dtype=np.float16)
        self.masks = np.zeros((capacity, max_episode_length), dtype=np.bool_)
        self.episode_lengths = np.zeros(capacity, dtype=np.int32)
        self.discount_targets = np.zeros((capacity, max_episode_length), dtype=np.int8)  # class labels 0/1/2
        self.is_terminal = np.zeros(capacity, dtype=np.bool_)

        self.position = 0
        self.size = 0
        self.bootstrap_value_target = bootstrap_value_target
    
    def save_games_from_buffers(self, all_buffers):
        # Single bulk device-to-host transfer (statt vieler np.array()-Aufrufe im Loop)
        all_buffers = jax.device_get(all_buffers)
        
        N = all_buffers['idx'].shape[0] # number of games in the batch
        lengths = all_buffers['idx']          # (N,)
        max_len = int(lengths.max())
        td = self.td_steps
        GAMMA = 1.0

        t = np.arange(max_len)[None, :]       # (1, max_len)
        valid = t < lengths[:, None]          # (N, max_len) — echte Schritte

        # Finale Werte: kein Slicing, nur Indexzugriff
        final_idx  = lengths - 1                                      # (N,)
        row        = np.arange(N)
        final_rew  = all_buffers['rew'][row, final_idx]               # (N,)
        final_plr  = all_buffers['player'][row, final_idx]            # (N,)
        final_team = all_buffers['team'][row, final_idx]              # (N,)

        # z für jeden Schritt
        players = all_buffers['player'][:, :max_len]                  # (N, max_len)
        teams   = all_buffers['team'][:, :max_len]                    # (N, max_len)
        game_won = (final_rew[:, None] == 2)                          # (N, 1) → broadcast
        is_team  = teams != -1
        z = np.where(game_won,
                np.where(is_team,
                    np.where(teams == final_team[:, None], 1.0, -1.0),
                    np.where(players == final_plr[:, None], 1.0, -1.0)),
                0.0)                                                   # (N, max_len)

        # steps_to_end — kein np.arange(length)[::-1] mehr nötig
        steps_to_end = np.maximum(0, lengths[:, None] - 1 - t)       # (N, max_len)

        # Bootstrap-Indices und -Values
        boot_idx      = np.minimum(t + td, lengths[:, None] - 1)     # (N, max_len)
        vals          = all_buffers['val'][:, :max_len].astype(np.float32)
        boot_vals_raw = vals[row[:, None], boot_idx]                  # (N, max_len)
        boot_plr  = all_buffers['player'][row[:, None], boot_idx]
        boot_team = all_buffers['team'][row[:, None], boot_idx]
        same_persp = np.where(is_team, teams == boot_team, players == boot_plr)
        boot_vals  = np.where(same_persp, boot_vals_raw, -boot_vals_raw)

        # Target Values
        use_bootstrap = ((steps_to_end >= td) & self.bootstrap_value_target ) | (z == 0)
        target_values = np.where(use_bootstrap,
            boot_vals * (GAMMA ** np.minimum(td, steps_to_end)),
            z * (GAMMA ** steps_to_end))
        target_values = np.clip(target_values, -1.0, 1.0)            # (N, max_len)
        target_values = np.where(valid, target_values, 0.0)           # ungültige Positionen nullen

        deal_mask = all_buffers['deal_happened'][:, :max_len] & valid  # (N, max_len)

        for i in range(N):
            L = int(lengths[i])
            deal_positions = np.where(deal_mask[i, :L])[0]
            starts = np.r_[0, deal_positions + 1]
            ends   = np.r_[deal_positions + 1, L]

            for s, e in zip(starts, ends):
                sess_len = e - s
                if sess_len <= 0:
                    continue

                slot = self.position

                # Felder auf 0/Defaultwert vorinitialisieren ist nicht nötig —
                # nur die gültigen Positionen beschreiben, Rest bleibt Altdaten
                # aber episode_lengths schützt sample_batch vor Altdaten-Lesen
                self.observations[slot, :sess_len]     = all_buffers['obs'][i, s:e]
                self.actions[slot, :sess_len]          = all_buffers['act'][i, s:e]
                self.rewards[slot, :sess_len]          = all_buffers['rew'][i, s:e]
                self.root_values[slot, :sess_len]      = vals[i, s:e]
                self.target_values[slot, :sess_len]    = target_values[i, s:e]
                self.policies[slot, :sess_len]         = all_buffers['pol'][i, s:e]
                self.masks[slot, :sess_len]            = all_buffers['mask'][i, s:e]
                self.discount_targets[slot, :sess_len] = all_buffers['discount'][i, s:e]
                self.discount_targets[slot, sess_len - 1] = 1  # Letzter Schritt: neutral (Klasse 1), da kein Folgeschritt mehr
                self.episode_lengths[slot]             = sess_len
                self.is_terminal[slot] = (e == L)

                self.position = (self.position + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)

    
    def sample_batch(self):
        K = self.unroll_steps + 1

        terminal_slots = np.where(self.is_terminal[:self.size])[0]
        n_term = self.batch_size // 4   # 25% terminale Sessions
        n_norm = self.batch_size - n_term
        ep_t   = terminal_slots[np.random.randint(0, len(terminal_slots), size=n_term)]
        ep_n   = np.random.randint(0, self.size, size=n_norm)
        ep_indices = np.concatenate([ep_n, ep_t])
        
        ep_lengths = self.episode_lengths[ep_indices]                          # (B,)
        t_starts   = np.floor(
            np.random.uniform(0, 1, size=self.batch_size) * ep_lengths
        ).astype(np.int32)          

        # Index-Matrizen aufbauen
        k_off              = np.arange(K)                                         # (K,)
        seq_idx            = t_starts[:, None] + k_off[None, :]                   # (B, K)
        valid_mask         = seq_idx < ep_lengths[:, None]                        # (B, K)
        seq_idx_c          = np.minimum(seq_idx, ep_lengths[:, None] - 1)         # (B, K) geclippt

        ep_exp  = np.broadcast_to(ep_indices[:, None], (self.batch_size, K))      # (B, K)
        ep_act  = ep_exp[:, :-1]                                                   # (B, K-1)
        act_idx = seq_idx_c[:, :-1]                                                # (B, K-1)

        # Daten extrahieren
        root_obs      = self.observations[ep_indices, t_starts]                   # (B, *obs_shape)
        actions       = self.actions[ep_act, act_idx]                             # (B, K-1)
        rewards       = self.rewards[ep_act, act_idx]                             # (B, K-1)
        policies      = self.policies[ep_exp, seq_idx_c]                          # (B, K, action_dim)
        masks         = self.masks[ep_exp, seq_idx_c]                             # (B, K)
        target_values = self.target_values[ep_exp, seq_idx_c]                     # (B, K)
        discount_tgts = self.discount_targets[ep_act, act_idx]                    # (B, K-1)

        # Ungültige Positionen (außerhalb der Session) auffüllen
        actions       = np.where(valid_mask[:, :-1], actions,       0)
        rewards       = np.where(valid_mask[:, :-1], rewards,       1)   # Klasse 1 = neutral
        policies      = np.where(valid_mask[:, :, None], policies,  0.0)
        masks         = np.where(valid_mask, masks,                 False)
        target_values = np.where(valid_mask, target_values,         0.0)
        discount_tgts = np.where(valid_mask[:, :-1], discount_tgts, 1)   # Klasse 1 = neutral

        return {
            'observations':     jnp.array(root_obs),
            'actions':          jnp.array(actions),
            'rewards':          jnp.array(rewards),
            'policies':         jnp.array(policies,      dtype=jnp.float32),
            'masks':            jnp.array(masks,         dtype=jnp.float32),
            'target_values':    jnp.array(target_values, dtype=jnp.float32),
            'discount_targets': jnp.array(discount_tgts),
        }