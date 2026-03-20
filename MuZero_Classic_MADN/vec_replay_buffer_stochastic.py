import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
from time import time
import numpy as np
import random

class VectorizedReplayBufferStochastic:
    """
    Replay Buffer für Stochastic MuZero mit Unterstützung für dice_outcomes.
    
    WICHTIGE ÄNDERUNGEN gegenüber deterministischem Buffer:
    1. Speichert dice_outcomes (Würfelergebnisse)
    2. Action dimension ist 4 (Pins) statt 24 (Pins × Dice)
    3. Gibt dice_outcomes im Batch zurück
    """
    def __init__(self, capacity: int, batch_size: int, unroll_steps: int, td_steps: int,
                 obs_shape=(11, 56), action_dim=4, max_episode_length=500, bootstrap_value_target=True):
        self.capacity = capacity
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim  # 4 für stochastic (nur Pins)
        self.max_episode_length = max_episode_length
        
        self.observations = np.zeros((capacity, max_episode_length, *obs_shape), dtype=np.float32)
        self.actions = np.full((capacity, max_episode_length), -1, dtype=np.int32)
        self.rewards = np.zeros((capacity, max_episode_length), dtype=np.int32)  # Klassen-Index statt Float
        self.root_values = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.child_visits = np.zeros((capacity, max_episode_length, action_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.dice_outcomes = np.full((capacity, max_episode_length), -1, dtype=np.int32)
        self.dice_distributions = np.zeros((capacity, max_episode_length, 6), dtype=np.float32)  # NEU: Würfelverteilungen (6 mögliche Ergebnisse)
        self.players = np.zeros((capacity, max_episode_length), dtype=np.int32)
        self.teams = np.zeros((capacity, max_episode_length), dtype=np.int32)
        self.episode_lengths = np.zeros(capacity, dtype=np.int32)
        self.discounts = np.zeros((capacity, max_episode_length), dtype=np.int32)  # Klassen-Index statt Float
        
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
            self.dice_outcomes[pos, :length] = np.array(all_buffers['dice'][i, :length])  # NEU!
            self.dice_distributions[pos, :length] = np.array(all_buffers['dice_dist'][i, :length])  # NEU!
            self.players[pos, :length] = np.array(all_buffers['player'][i, :length])
            self.teams[pos, :length] = np.array(all_buffers['team'][i, :length])
            self.discounts[pos, :length] = np.array(all_buffers['discount'][i, :length])
            self.episode_lengths[pos] = length
            
            self.position = (pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self):
        """
        Vollständig vektorisierte Sampling-Funktion für Stochastic MuZero.
        KEIN Python-Loop über batch_size!
        
        NEU: Gibt auch dice_outcomes zurück für das Training des chance_dynamics.
        """
        K = self.unroll_steps + 1
        TD = self.td_steps
        GAMMA = 1.0
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
        t_starts_normal = np.random.randint(0, max_starts_normal + 1)
        
        # --- Terminal Sampling: Terminal-Step an ZUFÄLLIGER Position k im Fenster ---
        # Nicht immer k=9! Bei k=0 kommt latent_state direkt aus RepNet → beste Qualität
        ep_indices_terminal = np.random.randint(0, self.size, size=n_terminal)
        ep_lengths_terminal = self.episode_lengths[ep_indices_terminal]
        # terminal_k = zufällige Position (0..K-2) wo der letzte Step der Episode landen soll
        # K-1 = 10 Positionen für Actions (k=0..9), davon nutzen wir k=0..K-2
        max_terminal_k = np.minimum(self.unroll_steps - 1, ep_lengths_terminal - 1)  # kann nicht vor Episode-Start
        terminal_k = np.array([np.random.randint(0, int(m) + 1) for m in max_terminal_k])
        # t_start so setzen dass ep_length-1 (letzter Step) bei Position terminal_k liegt
        t_starts_terminal = np.maximum(ep_lengths_terminal - 1 - terminal_k, 0)
        
        # --- Zusammenführen ---
        ep_indices = np.concatenate([ep_indices_normal, ep_indices_terminal])
        t_starts = np.concatenate([t_starts_normal, t_starts_terminal])
        ep_lengths = self.episode_lengths[ep_indices]
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
        
        # NEU: Extrahiere Dice Outcomes (nur für k=0..K-2)
        dice_seq = self.dice_outcomes[ep_for_actions, action_indices]
        # Shape: (batch_size, K-1)
        # WICHTIG: dice_seq[i, k] ist der Würfelwert der VOR der Action actions[i, k] gewürfelt wurde
        # Das Netzwerk muss lernen: action_dynamics(state, action) → chance_logits
        # Dann: chance_dynamics(afterstate, dice_seq[i, k]) → next_state
        
        # Extrahiere Policies, Values, Masks (für alle K Steps)
        policies = self.child_visits[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K, 4)  # 4 Actions für Pins!
        
        values = self.root_values[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K)
        
        masks = self.masks[ep_indices_expanded, seq_indices_clipped]
        # Shape: (batch_size, K)
        
        discount_targets = self.discounts[ep_for_actions, action_indices]
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
        z_seq = np.where(
            game_won_seq,
            np.where(
                is_single_player_seq,
                np.where(player_won_seq, 1.0, -1.0),
                np.where(team_won_seq, 1.0, -1.0)
            ),
            0.0
        )
        # Shape: (batch_size, K)
        
        # 7.4: Steps bis zum Ende der Episode
        steps_until_end = ep_lengths[:, None] - 1 - seq_indices  # (batch_size, K)
        
        # 7.5: Bootstrap-Condition: steps_until_end >= TD
        bootstrap_from_value = (steps_until_end >= TD)
        
        # 7.6: Bootstrap-Indizes (idx + TD, aber clipped)
        bootstrap_indices = np.minimum(seq_indices + TD, ep_lengths[:, None] - 1)
        bootstrap_values_raw = self.root_values[ep_indices_expanded, bootstrap_indices]
        # Shape: (batch_size, K)
        
        # ✅ NEU: 7.6b - Perspektiven-Flip für Bootstrap Values
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

        # Temporaler Discount anwenden
        steps_to_end = np.maximum(steps_until_end, 0)
        temporal_discount = GAMMA ** steps_to_end
        z_seq = z_seq * temporal_discount

        target_values = np.where(
            (z_seq == 0) | (bootstrap_from_value & self.bootstrap_value_target),
            # Falls True: Bootstrap mit Value nach K Steps
            bootstrap_values * (GAMMA ** np.minimum(TD, steps_until_end)),
            # Falls False: Bootstrap mit z AUS PERSPEKTIVE DIESES TIMESTEPS
            z_seq  # ← KEIN [:, None] mehr nötig, schon (batch_size, K)!
        )
        target_values = np.clip(target_values, -1.0, 1.0)
        # Shape: (batch_size, K)
        
        # ========================================
        # SCHRITT 8: Padding für ungültige Positionen
        # ========================================
        actions = np.where(valid_mask[:, :-1], actions, 0)
        rewards_seq = np.where(valid_mask[:, :-1], rewards_seq, 1)  # Klasse 1 = reward=0 (neutral)
        dice_seq = np.where(valid_mask[:, :-1], dice_seq, 0)  # NEU: Dice padding
        # Uniform padding (1/6): verhindert is_non_uniform=True für gepaddte Positionen
        uniform_dist = np.full(6, 1.0 / 6.0, dtype=np.float32)
        dice_probs_seq = np.where(valid_mask[:, :-1, None], self.dice_distributions[ep_for_actions, action_indices], uniform_dist)  # NEU: Dice distribution padding
        policies = np.where(valid_mask[:, :, None], policies, 0.0)
        values = np.where(valid_mask, values, 0.0)
        masks = np.where(valid_mask, masks, 0.0)
        target_values = np.where(valid_mask, target_values, 0.0)
        discount_targets = np.where(valid_mask[:, :-1], discount_targets, 1)  # Klasse 1 = discount=0 (neutral)
        
        # ========================================
        # SCHRITT 9: Konvertiere Dice Values (1-6) zu Indizes (0-5)
        # ========================================
        # WICHTIG: Im Environment sind Würfel 1-6, im Netzwerk verwenden wir 0-5
        dice_seq_indices = np.maximum(dice_seq - 1, 0)  # 1→0, 2→1, ..., 6→5
        # Bei padding (dice_seq=0 nach Maske) wird das zu -1, dann zu 0 (harmlos wegen Mask)
        
        # ========================================
        # SCHRITT 10: Return Batch
        # ========================================
        return {
            'observations': jnp.array(root_obs),           # (batch_size, 14, 56)
            'actions': jnp.array(actions),                 # (batch_size, K-1)
            'rewards': jnp.array(rewards_seq),             # (batch_size, K-1)
            'dice_outcomes': jnp.array(dice_seq_indices),  # (batch_size, K-1) NEU! (0-5)
            'dice_probs': jnp.array(dice_probs_seq),       # (batch_size, K-1, 6) NEU! (0-5)
            'policies': jnp.array(policies),               # (batch_size, K, 4)
            'values': jnp.array(values),                   # (batch_size, K)
            'masks': jnp.array(masks),                     # (batch_size, K)
            'target_values': jnp.array(target_values),    # (batch_size, K)
            'discount_targets': jnp.array(discount_targets)  # (batch_size, K-1)
        }
