import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
from time import time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *
from MuZero_DOG.muzero_dog import *

RULES = {
    'enable_teams': True,
    'enable_initial_free_pin': False,
    'enable_circular_board': True,
    'enable_friendly_fire': True,
    'enable_start_blocking': True,
    'enable_jump_in_goal_area': False,
    'must_traverse_start': True,
    'disable_swapping': False,
    'disable_hot_seven': False,
    'disable_joker': False,
}

def env_reset_batched(seed):
    return env_reset(
        0,  # <- Das wird an '_' übergeben
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
        must_traverse_start=RULES['must_traverse_start'],
        disable_swapping=RULES['disable_swapping'],
        disable_hot_seven=RULES['disable_hot_seven'],
        disable_joker=RULES['disable_joker']
    )

# 2. Vektorisierte Funktionen vorbereiten
batch_reset = jax.vmap(env_reset_batched)
batch_valid_action = jax.vmap(valid_actions)
batch_encode = jax.vmap(encode_board)
batch_env_step = jax.vmap(env_step, in_axes=(0, 0))
batch_map_action = jax.vmap(map_action_to_move)

@functools.partial(jax.jit, static_argnames=['num_envs', 'input_shape', 'num_simulations', 'max_depth', 'max_steps', 'temp'])
def play_batch_of_games_jitted(envs, num_envs, input_shape, params, rng_key, num_simulations, max_depth, max_steps, temp):
    """MCTS parallel + Early Exit + XLA optimiert
    Verwende play_batch_of_games_jitted, wenn du viele Spiele parallel simulieren möchtest, insbesondere für Training oder Datengewinnung.
    """
    pass

def play_n_games_v3(params, rng_key, input_shape, num_envs, num_simulation, max_depth, max_steps, temp):
    """Bester Ansatz: Alles in JAX, aber mit bedingter Ausführung"""
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs,), 0, 1000000)
    envs = batch_reset(seeds)
    
    all_buffers = play_batch_of_games_jitted(envs, num_envs, input_shape, params, subkey, num_simulation, max_depth, max_steps, temp)
    return all_buffers