import functools
# from deterministic_madn import *
from classic_madn import *
import jax
import jax.numpy as jnp
from time import time

@functools.partial(jax.jit, static_argnums=(2,))
def run_gumbel(rng_key:chex.PRNGKey, env:classic_MADN, num_simulations:int):
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    policy_output = mctx.gumbel_muzero_policy(
        params= None,
        rng_key=key1,
        root = jax.vmap(root_fn, (None, 0))(env,jax.random.split(key2, batch_size)),
        recurrent_fn=jax.vmap(recurrent_fn, (None,None,0,0)),
        num_simulations=num_simulations,
        max_depth=350,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1)
    )
    return policy_output

def simulate_game(env, key):
    rng_key = jax.random.PRNGKey(key)
    steps = 0
    while not env.done:
        print("Current player: ", env.current_player)
        print("Current board:\n", env.board)
        print("Current pins:\n", env.pins)
        rng_key, subkey = jax.random.split(rng_key)
        env = throw_die(env, subkey)
        print("Die throw:\n", env.die)
        valid_actions = valid_action(env)
        print("Valid actions:\n", valid_actions)
        if not jnp.any(valid_actions):
            env, reward, done = no_step(env)
            print("No valid actions available. Skipping turn.")
        else:
            valid_action_indices = jnp.argwhere(valid_actions)
            N = valid_action_indices.shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            idx = jax.random.randint(subkey, (), 0, N)
            chosen = valid_action_indices[idx][0]
            print("Chosen action: ", chosen)
            env, reward, done = env_step(env, chosen)
        print("Reward: ", reward)
        print("Done: ", done)
        print("-"*30)
        steps += 1
    print("Game over after ", steps, " steps.")
    print("Final board:\n", env.board)
    print("Final pins:\n", env.pins)

@functools.partial(jax.jit, static_argnames=('game', 'policy_apply_fn'))
def play_game(env, game, num_players, distance, policy_apply_fn, params, rng_key):
    steps = 0
    states, actions, players = [], [], []
    while not env.done:
        valid_actions = valid_action(env)
        if not jnp.any(valid_actions):
            action = None
            env, reward, done = no_step(env)
        else:
            logits = policy_apply_fn(params, encode_board(env.board))

            # Ungültige Züge maskieren
            flat_valid_mask = valid_actions.flatten()
            logits = jnp.where(flat_valid_mask, logits, -jnp.inf)

            rng_key, subkey = jax.random.split(rng_key)
            action_idx = jax.random.categorical(subkey, logits).astype(jnp.int8)
            action = map_action(action_idx)

            env, reward, done = game.env_step(env, action)
        
        states.append(env.board)
        actions.append(action)
        players.append(env.current_player)

    return {
        'states': jnp.array(states),
        'actions': jnp.array(actions),
        'players': jnp.array(players)
    }


def simulate_mcts_game(env, key, iterations = 500):
    rng_key = jax.random.PRNGKey(key)
    steps = 0
    while not env.done:
        print("Current player: ", env.current_player)
        print("Current board:\n", env.board)
        print("Current pins:\n", env.pins)
        print("Current actionset:\n", env.action_set)
        valid_actions = valid_action(env)
        print("Valid actions:\n", valid_actions)
        if not jnp.any(valid_actions):
            env, reward, done = no_step(env)
            print("No valid actions available. Skipping turn.")
        else:
            # if env.current_player == 1:
            #     print("Skipping MCTS for player 1, choosing random action.")
            #     valid_action_indices = jnp.argwhere(valid_actions)
            #     N = valid_action_indices.shape[0]
            #     rng_key, subkey = jax.random.split(rng_key)
            #     idx = jax.random.randint(subkey, (), 0, N)
            #     chosen = valid_action_indices[idx]
            #     chosen = chosen.at[1].set(chosen[1]+1)  # Ensure action index is int8
            #     print("Chosen action: ", chosen)
            #     env, reward, done = env_step(env, chosen)
            #     steps += 1
            #     continue
            valid_action_indices = jnp.argwhere(valid_actions)
            N = valid_action_indices.shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            policy_output = run_gumbel(subkey, env, iterations)
            w = policy_output.action_weights
            w = w.mean(axis=0)
            chosen = jnp.argmax(w)
            chosen = map_action(chosen)
            print("Chosen action: ", chosen)
            env, reward, done = env_step(env, chosen)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Winner: ", get_winner(env.board, env.goal))
        print("-"*30)
        steps += 1
    print("Game over after ", steps, " steps.")
    print("Final board:\n", env.board)
    print("Final pins:\n", env.pins)


'''Simulate a random game'''
# simulate_mcts_game(999, 1500)
env = env_reset(0, num_players=jnp.int8(2), distance=jnp.int8(10), enable_initial_free_pin=True)
simulate_game(env, 99)

'''Test MCTS on a specific board state'''
# env = env_reset(0, num_players=jnp.int8(2), distance=jnp.int8(10))
# print(env.rules)
# print("Current board:\n", env.board)
# env.pins = jnp.array([
#     [7, 5, -1, -1],
#     [27, 15, -1, 8]
#     ], dtype=jnp.int8)
# env.board = set_pins_on_board(env.board, env.pins)
# env.current_player = 1
# print("Current board:\n", env.board)
# print("Current player:\n", env.current_player)
# print("Current pins:\n", env.pins)
# print("Current actionset:\n", env.action_set)


# # Simuliere beide Strategien manuell
# policy_output = run_gumbel(jax.random.PRNGKey(20), env, 10000)
# end = time()
# w = policy_output.action_weights
# w = w.mean(axis=0)
# print(w)
# print(jnp.argmax(w))
# print(map_action(jnp.argmax(w)))

# start = time()
# env.current_player = 0
# policy_output = run_gumbel(jax.random.PRNGKey(20), env, 10000)
# end = time()
# w = policy_output.action_weights
# w = w.mean(axis=0)
# print(w)
# print(jnp.argmax(w))
# print(map_action(jnp.argmax(w)))

# print("Time taken for MCTS: ", end - start)

'''Test a specific board state and following transitions'''
# env = env_reset(0, num_players=jnp.int8(2), distance=jnp.int8(10), enable_initial_free_pin=True)
# env.action_set = jnp.array([
#     [4, 4, 4, 4, 4, 4],
#     [4, 4, 4, 4, 4, 4]
#     ], dtype=jnp.int8)
# key = jax.random.PRNGKey(53)
# print(rollout(env, key))
# env.done = True
# env, reward, done = env_step(env, jnp.array([1,6]))
# print("After action:")
# print("Winner: ", get_winner(env.board, env.goal))
# print(env.pins)
# print(reward)
# print(done)
# print(env.board)

# print("After action:")
# print(get_winner(env))
# print(env.pins)
# print(reward)
# print(done)
# print(env.board)
# print(env.board)
# print(encode_board(env))
# # new_pins = env.pins[jnp.arange(env.current_player, env.current_player + env.num_players) % env.num_players]
# print([jnp.arange(env.current_player, env.current_player + env.num_players) % env.num_players])
# home_positions = jnp.ones((env.num_players, env.board.shape[0]), dtype=jnp.int8) * jnp.count_nonzero(env.pins == -1, axis=1)[:, None]  # (4, board_size)
# print(home_positions)
# print(home_positions[jnp.arange(env.current_player, env.current_player + env.num_players) % env.num_players])