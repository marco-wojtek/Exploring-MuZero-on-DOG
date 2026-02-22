import chex
import jax
import jax.numpy as jnp
from flax import struct
import functools
import sys, os
from time import time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MADN.classic_madn import *
from MuZero.muzero_classic_madn import *

RULES = {
    'enable_teams': False,
    'enable_initial_free_pin': True,
    'enable_circular_board': False,
    'enable_friendly_fire': False,
    'enable_start_blocking': False,
    'enable_jump_in_goal_area': True,
    'enable_start_on_1': True,
    'enable_bonus_turn_on_6': True,
    'must_traverse_start': False,
    'enable_dice_rethrow': True  # NEU: Für unterschiedliche Würfelverteilungen!
}
def env_reset_batched(seed):
    return env_reset(
        seed,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=10,
        starting_player=0,
        seed=seed,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=RULES['enable_teams'],
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        enable_start_on_1=RULES['enable_start_on_1'],
        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
        must_traverse_start=RULES['must_traverse_start'],
        enable_dice_rethrow=RULES['enable_dice_rethrow']  # Wichtig für unterschiedliche Würfelverteilungen!
    )

# 2. Vektorisierte Funktionen vorbereiten
batch_reset = jax.vmap(env_reset_batched)
batch_valid_action = jax.vmap(valid_action)
batch_encode = jax.vmap(encode_board)  # Verwende board_to_matrix für classic MADN
batch_env_step = jax.vmap(env_step, in_axes=(0, 0))
batch_throw_die = jax.vmap(throw_die)

@functools.partial(jax.jit, static_argnames=['num_envs', 'input_shape', 'max_steps', 'num_simulations', 'max_depth', 'temp'])
def play_batch_of_games_jitted(envs, num_envs, input_shape, params, rng_key, num_simulations, max_depth, max_steps, temp):
    """
    Spielt einen Batch von Spielen parallel mit Stochastic MuZero.
    
    WICHTIG: Der Ablauf ist:
    1. Würfel werfen (automatisch in Environment)
    2. State speichern (nach dem Würfeln)
    3. MCTS ausführen und Aktion wählen
    4. Aktion ausführen
    5. Nächster Spieler (wiederholen)
    
    Args:
        envs: Batch von Environments
        num_envs: Anzahl der parallelen Environments
        input_shape: Shape der Observation (z.B. (14, 56))
        params: MuZero Netzwerk-Parameter
        rng_key: JAX PRNG Key
        num_simulations: Anzahl der MCTS Simulationen
        max_depth: Maximale MCTS Suchtiefe
        max_steps: Maximale Schritte pro Spiel
        
    Returns:
        final_buffers: Dictionary mit allen gesammelten Daten
    """
    
    def body_fn(carry):
        envs_state, buffers, dones, step_count, rng_key = carry
        
        # Neue Keys für diesen Step generieren
        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)

        # ✅ PARALLEL: vmap über alle aktiven Envs
        def step_single_env(env, buffer, done, key):
            def do_active_step(env, buffer):
                # 1. WÜRFELN (automatisch in der Environment)
                key1, key2 = jax.random.split(key)
                env_after_dice = throw_die(env)
                dice_value = env_after_dice.die  # Speichern für Replay
                
                # 2. State nach dem Würfeln speichern (DAS ist der Decision Node!)
                obs = encode_board(env_after_dice)[None, ...]
                valid_mask = valid_action(env_after_dice).flatten()
                invalid_mask = (~valid_mask)[None, :]
                has_valid = jnp.any(valid_mask)
                
                # 3. Unterscheidung: MCTS oder no_step
                def do_mcts(env):
                    # Stochastic MuZero MCTS
                    policy_output, root_value = run_stochastic_muzero_mcts(
                        params, key2, obs, invalid_actions=invalid_mask, num_simulations=num_simulations, max_depth=max_depth, temperature=temp
                    )
                    # Action ist ein Index (0-3 für 4 Pins)
                    action = policy_output.action[0]
                    next_env, reward, next_done = env_step(env, action)
                    return next_env, obs[0], action, reward, root_value[0], policy_output.action_weights[0], next_done, 1, dice_value
                
                def do_skip(env):
                    # Keine validen Actions → no_step
                    next_env, reward, next_done = no_step(env)
                    dummy_obs = jnp.zeros_like(obs[0])
                    return next_env, dummy_obs, jnp.int32(-1), reward, 0.0, jnp.zeros(4), next_done, 0, dice_value
                
                # Wähle zwischen MCTS und no_step
                next_env, step_obs, action, reward, value, policy, next_done, mask, dice = jax.lax.cond(
                    has_valid,
                    do_mcts,
                    do_skip,
                    env_after_dice
                )
                
                # Buffer Update
                idx = buffer['idx']
                current_player = env_after_dice.current_player
                team = jax.lax.cond(env_after_dice.rules['enable_teams'], lambda: jnp.int8(current_player%2), lambda: jnp.int8(-1))
                dice_dist = dice_probabilities(env_after_dice)  # Würfelverteilung speichern
                new_buffer = {
                    'obs': buffer['obs'].at[idx].set(step_obs),
                    'act': buffer['act'].at[idx].set(action),
                    'rew': buffer['rew'].at[idx].set(reward),
                    'val': buffer['val'].at[idx].set(value),
                    'pol': buffer['pol'].at[idx].set(policy),
                    'mask': buffer['mask'].at[idx].set(mask),
                    'dice': buffer['dice'].at[idx].set(dice),  # Würfelergebnis speichern
                    'dice_dist': buffer['dice_dist'].at[idx].set(dice_dist),  # Würfelverteilung speichern
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
            envs_state, buffers, dones, step_keys
        )
        
        return (new_envs, new_buffers, new_dones, step_count + 1, rng_key)
    
    # Initialisierung
    init_buffers = {
        'obs': jnp.zeros((num_envs, max_steps, *input_shape)),
        'act': jnp.zeros((num_envs, max_steps), dtype=jnp.int32),
        'rew': jnp.zeros((num_envs, max_steps)),
        'val': jnp.zeros((num_envs, max_steps)),
        'pol': jnp.zeros((num_envs, max_steps, 4)),  # NUR 4 Actions für Pins!
        'mask': jnp.zeros((num_envs, max_steps)),
        'dice': jnp.zeros((num_envs, max_steps), dtype=jnp.int32),  
        'dice_dist': jnp.zeros((num_envs, max_steps, 6)),  # NEU: Würfelverteilung (6 mögliche Ergebnisse)
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

def play_n_games_v3(params, rng_key, input_shape, num_envs, num_simulation, max_depth, max_steps, temp):
    """
    Spielt n Spiele mit Stochastic MuZero.
    
    Args:
        params: MuZero Netzwerk-Parameter
        rng_key: JAX PRNG Key
        input_shape: Shape der Observation (z.B. (14, 56))
        num_envs: Anzahl der parallelen Spiele
        num_simulation: Anzahl der MCTS Simulationen
        max_depth: Maximale MCTS Suchtiefe
        max_steps: Maximale Schritte pro Spiel
        temp: Temperatur für die Aktionsauswahl
    Returns:
        all_buffers: Dictionary mit allen gesammelten Daten (für Replay Buffer)
    """
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs,), 0, 1000000)
    envs = batch_reset(seeds)
    
    all_buffers = play_batch_of_games_jitted(
        envs, num_envs, input_shape, params, subkey, 
        num_simulation, max_depth, max_steps, temp
    )
    return all_buffers

def play_n_games_v3_batched(params, rng_key, input_shape, num_envs=2048, batch_size=256, num_simulation=50, max_depth=25, max_steps=500):
    """
    Spiele in Batches von 256 (um Memory zu sparen).
    
    Args:
        params: MuZero Netzwerk-Parameter
        rng_key: JAX PRNG Key
        input_shape: Shape der Observation
        num_envs: Gesamtzahl der Spiele
        batch_size: Batch-Größe pro Durchlauf
        num_simulation: Anzahl der MCTS Simulationen
        max_depth: Maximale MCTS Suchtiefe
        max_steps: Maximale Schritte pro Spiel
        
    Returns:
        all_buffers: Kombinierte Buffers aus allen Batches
    """
    all_buffers_list = []
    for i in range(0, num_envs, batch_size):
        rng_key, subkey = jax.random.split(rng_key)
        batch_buffers = play_n_games_v3(
            params, subkey, input_shape, 
            num_envs=batch_size, 
            num_simulation=num_simulation, 
            max_depth=max_depth, 
            max_steps=max_steps
        )
        all_buffers_list.append(batch_buffers)
    
    # Kombiniere alle Batches
    # Stack entlang der Env-Dimension (axis=0)
    combined_buffers = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x, axis=0), 
        *all_buffers_list
    )
    
    return combined_buffers