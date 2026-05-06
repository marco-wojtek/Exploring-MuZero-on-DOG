"""
Cross-group evaluation engine.

Supports mixed observation encodings within the same game:
- Players on Team A can use encode_board_with_belief (belief=True)
- Players on Team B can use encode_board (belief=False)

This is needed when comparing agents trained WITH belief states
against agents trained WITHOUT belief states.
"""
import functools
import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import chex
import jax
import jax.numpy as jnp
from time import time
from functools import partial
import pickle
import numpy as np
import math
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *
from MuZero_DOG.eval_config.classic import run_muzero_mcts as classic_muzero, init_muzero_params as init_classic_params
from MuZero_DOG.eval_config.sample import run_muzero_mcts as sample_muzero, init_muzero_params as init_sample_params
from MuZero_DOG.eval_config.session import run_muzero_mcts as session_muzero, init_muzero_params as init_session_params


def manual_get_winner(board: Board, num_players, goal, rules) -> chex.Array:
    collect_winners = jax.vmap(is_player_done, in_axes=(None, None, None, 0))
    players_done = collect_winners(num_players, board, goal, jnp.arange(4, dtype=jnp.int8))

    def four_players_case():
        team_0 = players_done[0] & players_done[2]
        team_1 = players_done[1] & players_done[3]
        both = team_0 & team_1
        none = ~(team_0 | team_1)
        return jax.lax.cond(
            both | none,
            lambda: jnp.full(players_done.shape, False, dtype=jnp.bool_),
            lambda: jax.lax.cond(
                team_0,
                lambda: jnp.array([True, False, True, False], dtype=jnp.bool_),
                lambda: jnp.array([False, True, False, True], dtype=jnp.bool_),
            ),
        )

    return jax.lax.cond(rules['enable_teams'], four_players_case, lambda: players_done)


def env_reset_batched(seed, starting_player):
    return env_reset(
        seed,
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=16,
        starting_player=starting_player,
        seed=seed,
        enable_teams=RULES['enable_teams'],
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        must_traverse_start=RULES['must_traverse_start'],
        disable_swapping=RULES['disable_swapping'],
        disable_hot_seven=RULES['disable_hot_seven'],
        disable_joker=RULES['disable_joker'],
    )


batch_reset = jax.vmap(env_reset_batched, in_axes=(0, 0))


def calculate_progress(env, player_idx):
    board_size = env.board_size
    distance = board_size // env.num_players
    pins = env.pins[player_idx]
    goals = env.goal[player_idx]
    rules = env.rules
    travers_start_enabled = rules['must_traverse_start']

    rotated_pins = jnp.where(
        pins < 0,
        pins - 5,
        jnp.where(
            pins < board_size,
            (pins - distance * player_idx) % board_size - jnp.int32(travers_start_enabled),
            board_size + (pins - goals[0]),
        ),
    )
    rotated_goals = jnp.arange(board_size, board_size + 4)
    sorted_pins = jnp.sort(rotated_pins)
    distance_matrix = jnp.abs(sorted_pins[:, None] - rotated_goals[None, :])

    def match_iteration(i, carry):
        total_dist, mask = carry
        masked_distances = jnp.where(mask, distance_matrix, jnp.inf)
        flat_idx = jnp.argmin(masked_distances)
        row = flat_idx // 4
        col = flat_idx % 4
        min_dist = distance_matrix[row, col]
        new_total = total_dist + min_dist
        new_mask = mask.at[row, :].set(False)
        new_mask = new_mask.at[:, col].set(False)
        return new_total, new_mask

    initial_mask = jnp.ones((4, 4), dtype=jnp.bool_)
    initial_total = jnp.float32(0.0)
    final_total, _ = jax.lax.fori_loop(0, 4, match_iteration, (initial_total, initial_mask))
    return final_total


@jax.jit
def calculate_player_progress(envs):
    def player_progress_single(env):
        def single_player_progress(player_idx):
            return calculate_progress(env, player_idx)
        return jax.vmap(single_player_progress)(jnp.arange(env.num_players))
    x = jax.vmap(player_progress_single)(envs)
    return jnp.mean(x, axis=0), x


def _build_player_step_fn(agent_type: int, use_belief: bool):
    """Build a per-player step closure with MCTS variant AND encoding resolved at trace time.

    agent_type (static int):
        0 = random
        1 = rule_based
        2 = classic_muzero
        3 = sample_muzero
        4 = session_muzero

    use_belief (static bool):
        True  → encode_board_with_belief  (82, 80) obs
        False → encode_board              (43, 80) obs
    """
    def _step(params, key, env):
        if use_belief:
            obs = encode_board_with_belief(env)[None, ...]
        else:
            obs = encode_board(env)[None, ...]

        valid_mask = valid_actions(env).flatten()
        invalid_mask = (~valid_mask)[None, :]

        if agent_type == 2:
            def do_mcts():
                policy_output, _ = classic_muzero(
                    params, key, obs,
                    invalid_actions=invalid_mask,
                    num_simulations=NUM_SIMULATIONS,
                    max_depth=MAX_DEPTH,
                    temperature=TEMPERATURE,
                )
                action = policy_output.action[0]
                next_env, _, next_done = env_step(env, action)
                return next_env, next_done
        elif agent_type == 3:
            def do_mcts():
                policy_output, _ = sample_muzero(
                    params, key, obs,
                    invalid_actions=invalid_mask,
                    num_simulations=NUM_SIMULATIONS,
                    max_depth=MAX_DEPTH,
                    temperature=TEMPERATURE,
                )
                action = policy_output.action[0]
                next_env, _, next_done = env_step(env, action)
                return next_env, next_done
        else:  # type 4 or unknown → session
            def do_mcts():
                policy_output, _ = session_muzero(
                    params, key, obs,
                    invalid_actions=invalid_mask,
                    num_simulations=NUM_SIMULATIONS,
                    max_depth=MAX_DEPTH,
                    temperature=TEMPERATURE,
                )
                action = policy_output.action[0]
                next_env, _, next_done = env_step(env, action)
                return next_env, next_done

        def do_random():
            logits = jnp.where(valid_mask, 0.0, -1e9)
            action = jax.random.categorical(key, logits)
            next_env, _, next_done = env_step(env, action)
            return next_env, next_done

        def do_no_step():
            next_env, _, next_done = no_step(env)
            return next_env, next_done

        if agent_type == 0:
            def _action(env):
                return jax.lax.cond(jnp.any(valid_mask), do_random, do_no_step)
        else:
            def _action(env):
                return jax.lax.cond(jnp.any(valid_mask), do_mcts, do_no_step)

        return _action(env)

    return _step


@functools.partial(jax.jit, static_argnames=['num_envs', 'agent_types', 'belief_flags'])
def play_eval_loop_jitted(envs, params_tuple, rng_key, num_envs, agent_types, belief_flags):
    """JIT-compiled game loop with per-player encoding and MCTS variant.

    agent_types:  static tuple of 4 ints  (2/3/4 for MCTS variants)
    belief_flags: static tuple of 4 bools (True=with_belief, False=without)
    """
    step_fns = [_build_player_step_fn(agent_types[i], belief_flags[i]) for i in range(4)]

    def body_fn(carry):
        envs, winners, dones, step_count, rng_key = carry
        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)

        def step_single_env(env, done, key, winner):
            def do_step(env, winner):
                current_player = env.current_player
                next_env, next_done = jax.lax.switch(
                    current_player,
                    [lambda fn=step_fns[i], par=params_tuple[i]: fn(par, key, env)
                     for i in range(4)],
                )

                def update_winner(env, winner):
                    win = manual_get_winner(env.board, env.num_players, env.goal, env.rules)
                    return winner + win.astype(jnp.int32)

                new_winner = jax.lax.cond(
                    next_done,
                    lambda: update_winner(next_env, winner),
                    lambda: winner,
                )
                return next_env, next_done, new_winner

            def skip_step(env, winner):
                return env, done, winner

            return jax.lax.cond(
                ~done,
                lambda: do_step(env, winner),
                lambda: skip_step(env, winner),
            )

        new_envs, new_dones, new_winners = jax.vmap(step_single_env)(
            envs, dones, step_keys, winners
        )
        return (new_envs, new_winners, new_dones, step_count + 1, rng_key)

    init_dones = envs.done
    init_winners = jnp.zeros((num_envs, 4), dtype=jnp.int32)

    def cond_fn(carry):
        _, _, dones, step_count, _ = carry
        return jnp.any(~dones) & (step_count < 2000)

    final_envs, final_winners, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (envs, init_winners, init_dones, 0, rng_key),
    )
    return final_envs, final_winners


def evaluate_cross_group(params_a, params_b, type_a, type_b, belief_a, belief_b, batch_size=25):
    """Evaluate Team A (players 0&2) vs Team B (players 1&3) with different encodings.

    Args:
        params_a: loaded params dict for team A agent
        params_b: loaded params dict for team B agent
        type_a:   int (2=classic, 3=sample, 4=session) for team A
        type_b:   int (2=classic, 3=sample, 4=session) for team B
        belief_a: bool — True if team A uses belief encoding
        belief_b: bool — True if team B uses belief encoding
        batch_size: parallel games per call
    """
    # Assign types and belief flags per player position
    agents = [params_a, params_b, params_a, params_b]
    for p, t in zip(agents, [type_a, type_b, type_a, type_b]):
        p['type'] = t

    agent_types = (type_a, type_b, type_a, type_b)
    belief_flags = (belief_a, belief_b, belief_a, belief_b)

    rng_key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
    rng_key, subkey = jax.random.split(rng_key)
    num_total = batch_size * 4
    seeds = jax.random.randint(subkey, (num_total,), 0, 1_000_000)
    envs = batch_reset(seeds, jnp.repeat(jnp.arange(4), batch_size))

    params_tuple = tuple(agents)

    final_envs, winners_flat = play_eval_loop_jitted(
        envs, params_tuple, subkey, num_total, agent_types, belief_flags
    )

    progress_mean, progress = calculate_player_progress(final_envs)

    # Aggregate by starting player
    winners = jnp.zeros((4, 4), dtype=jnp.int32)
    average_progress = jnp.zeros((4, 4), dtype=jnp.float32)
    winners_split = jnp.array_split(winners_flat, 4, axis=0)
    progress_split = jnp.array_split(progress, 4, axis=0)
    for i in range(4):
        winners = winners.at[i].set(jnp.sum(winners_split[i], axis=0))
        average_progress = average_progress.at[i].set(jnp.mean(progress_split[i], axis=0))

    total_wins = jnp.sum(winners, axis=0)
    team_a_wins = int(total_wins[0] + total_wins[2]) // 2  # each team member gets a win
    team_b_wins = int(total_wins[1] + total_wins[3]) // 2

    print(f"  Team A wins: {team_a_wins}  |  Team B wins: {team_b_wins}  (of {batch_size * 4} games)")
    print(f"  Total Wins per Player: {total_wins}")
    print(f"  Avg Pin Distance per Player: {jnp.sum(average_progress, axis=0) / 4}")

    return {
        'winners': winners,
        'total_wins': total_wins,
        'team_a_wins': team_a_wins,
        'team_b_wins': team_b_wins,
        'average_progress': average_progress,
    }


# Rules for evaluation games
RULES = {
    'enable_teams': True,
    'enable_initial_free_pin': False,
    'enable_circular_board': False,
    'enable_friendly_fire': True,
    'enable_start_blocking': True,
    'enable_jump_in_goal_area': False,
    'must_traverse_start': True,
    'disable_swapping': False,
    'disable_hot_seven': False,
    'disable_joker': False,
}
NUM_SIMULATIONS = 50
MAX_DEPTH = 25
TEMPERATURE = 0.0
