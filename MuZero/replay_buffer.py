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

@functools.partial(jax.jit, static_argnames=['num_envs', 'input_shape', 'max_steps'])
def play_batch_of_games_jitted(envs, num_envs, input_shape, params, rng_key, max_steps=500, temp=1.0):
    """MCTS parallel + Early Exit + XLA optimiert
    Verwende play_batch_of_games_jitted, wenn du viele Spiele parallel simulieren möchtest, insbesondere für Training oder Datengewinnung.
    """
    
    def body_fn(carry):
        envs_state, buffers, dones, step_count, rng_keys = carry
        
        # Neue Keys für diesen Step generieren
        rng_key, *step_keys = jax.random.split(rng_keys, num_envs + 1)
        step_keys = jnp.array(step_keys)

        # ✅ PARALLEL: vmap über alle aktiven Envs
        def step_single_env(env, buffer, done, key):
            def do_active_step(env, buffer):
                obs = old_encode_board(env)[None, ...]
                valid_mask = valid_action(env).flatten()
                invalid_mask = (~valid_mask)[None, :]
                has_valid = jnp.any(valid_mask)
                
                # Unterscheidung: MCTS oder no_step
                def do_mcts(env):
                    policy_output, root_value = run_muzero_mcts(
                        params, key, obs, invalid_actions=invalid_mask, temperature=temp
                    )
                    action = policy_output.action[0]
                    next_env, reward, next_done = env_step(env, map_action(action))
                    return next_env, obs[0], action, reward, root_value[0], policy_output.action_weights[0], next_done, 1
                
                def do_skip(env):
                    # Keine validen Actions → no_step
                    next_env, reward, next_done = no_step(env)
                    dummy_obs = jnp.zeros_like(obs[0])
                    return next_env, dummy_obs, jnp.int32(-1), reward, 0.0, jnp.zeros(24), next_done, 0
                
                # Wähle zwischen MCTS und no_step
                next_env, step_obs, action, reward, value, policy, next_done, mask = jax.lax.cond(
                    has_valid,
                    do_mcts,
                    do_skip,
                    env
                )
                
                # Buffer Update
                idx = buffer['idx']
                current_player = env.current_player
                team = jax.lax.cond(env.rules['enable_teams'], lambda: jnp.int8(current_player%2), lambda: jnp.int8(-1))
                new_buffer = {
                    'obs': buffer['obs'].at[idx].set(step_obs),
                    'act': buffer['act'].at[idx].set(action),
                    'rew': buffer['rew'].at[idx].set(reward),
                    'val': buffer['val'].at[idx].set(value),
                    'pol': buffer['pol'].at[idx].set(policy),
                    'mask': buffer['mask'].at[idx].set(mask),
                    'player': buffer['player'].at[idx].set(current_player),
                    'team': buffer['team'].at[idx].set(team),
                    'idx': idx + 1
                }
                return next_env, new_buffer, next_done
            
            def do_skip_step(env, buffer):
                # Game ist fertig, nichts tun
                return env, buffer, done
            
            return jax.lax.cond(~done, do_active_step, do_skip_step, env, buffer)
        
        # ✅ HIER: vmap über alle Envs gleichzeitig!
        new_envs, new_buffers, new_dones = jax.vmap(step_single_env)(
            envs_state, buffers, dones, step_keys  # keys muss pro Step neu sein!
        )
        
        return (new_envs, new_buffers, new_dones, step_count + 1, rng_key)
    
    # Initialisierung
    init_buffers = {
        'obs': jnp.zeros((num_envs, max_steps, *input_shape)),
        'act': jnp.zeros((num_envs, max_steps), dtype=jnp.int32),
        'rew': jnp.zeros((num_envs, max_steps)),
        'val': jnp.zeros((num_envs, max_steps)),
        'pol': jnp.zeros((num_envs, max_steps, 24)),
        'mask': jnp.zeros((num_envs, max_steps)),
        'player': jnp.zeros((num_envs, max_steps), dtype=jnp.int32),
        'team': jnp.full((num_envs, max_steps), -1, dtype=jnp.int32),
        'idx': jnp.zeros(num_envs, dtype=jnp.int32)    
    }
    init_dones = jnp.zeros(num_envs, dtype=jnp.bool_)
    
    def cond_fn(carry):
        _, _, dones, step_count, _ = carry
        # Stoppe wenn ALLE done ODER max_steps erreicht
        return jnp.any(~dones) & (step_count < max_steps)
    
    final_envs, final_buffers, final_dones, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (envs, init_buffers, init_dones, 0, rng_key)
    )
    
    return final_buffers

def play_n_games_v3(params, rng_key, input_shape, num_envs=50, max_steps=500, temp=1.0):
    """Bester Ansatz: Alles in JAX, aber mit bedingter Ausführung"""
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs,), 0, 1000000)
    envs = batch_reset(seeds)
    
    all_buffers = play_batch_of_games_jitted(envs, num_envs, input_shape, params, subkey, max_steps, temp)
    return all_buffers
    # Episode Extraction wie in v2
    episodes = []
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
        episodes.append(ep)
    return episodes

def play_n_games_v3_batched(params, rng_key, input_shape, num_envs=2048, batch_size=256):
    """Spiele in Batches von 256"""
    all_episodes = []
    for i in range(0, num_envs, batch_size):
        rng_key, subkey = jax.random.split(rng_key)
        batch_eps = play_n_games_v3(params, subkey, input_shape, num_envs=batch_size)
        all_episodes.extend(batch_eps)
    return all_episodes