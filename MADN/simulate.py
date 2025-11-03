import functools
from deterministic_madn import *
import jax
import jax.numpy as jnp
from time import time

@functools.partial(jax.jit, static_argnums=(2,))
def run_gumbel(rng_key:chex.PRNGKey, env:deterministic_MADN, num_simulations:int):
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    policy_output = mctx.gumbel_muzero_policy(
        params= None,
        rng_key=key1,
        root = jax.vmap(root_fn, (None, 0))(env,jax.random.split(key2, batch_size)),
        recurrent_fn=jax.vmap(recurrent_fn, (None,None,0,0)),
        num_simulations=num_simulations,
        max_depth=300,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1)
    )
    return policy_output

def simulate_game(key):
    env = env_reset(0, num_players=jnp.int8(2), distance=jnp.int8(10))
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
            valid_action_indices = jnp.argwhere(valid_actions)
            N = valid_action_indices.shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            idx = jax.random.randint(subkey, (), 0, N)
            chosen = valid_action_indices[idx]
            chosen = chosen.at[1].set(chosen[1]+1)  # Ensure action index is int8
            print("Chosen action: ", chosen)
            env, reward, done = env_step(env, chosen)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Winner: ", get_winner(env))
        print("-"*30)
        steps += 1
    print("Game over after ", steps, " steps.")
    print("Final board:\n", env.board)
    print("Final pins:\n", env.pins)

@functools.partial(jax.jit, static_argnames=('game', 'policy_apply_fn'))
def play_game(game, num_players, distance, policy_apply_fn, params, rng_key):
    env = game.env_reset(0, num_players=num_players, distance=distance)
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

# simulate_game(12)
env = env_reset(0, num_players=jnp.int8(2), distance=jnp.int8(10))
print("Current board:\n", env.board)
env, _, _ = env_step(env, jnp.array([1, 1], dtype=jnp.int8))
env, _, _ = env_step(env, jnp.array([1, 1], dtype=jnp.int8))
env, _, _ = env_step(env, jnp.array([1, 2], dtype=jnp.int8))
env, _, _ = env_step(env, jnp.array([1, 3], dtype=jnp.int8))
env, _, _ = env_step(env, jnp.array([0, 1], dtype=jnp.int8))
env, _, _ = env_step(env, jnp.array([0, 1], dtype=jnp.int8))
print("Current board:\n", env.board)
print("Current player:\n", env.current_player)
print("Current pins:\n", env.pins)
print("Current actionset:\n", env.action_set)
start = time()
policy_output = run_gumbel(jax.random.PRNGKey(20), env, 5000)
end = time()
print("Time taken for MCTS: ", end - start)
w = policy_output.action_weights
print(w.mean(axis=0))
# env.board = env.board.at[24].set(1)
# env.board = env.board.at[25].set(1)
# env.board = env.board.at[26].set(1)
# env.board = env.board.at[27].set(1)
# env.board = env.board.at[10].set(0)
# env.board = env.board.at[15].set(0)
# env.board = env.board.at[3].set(0)
# env.pins = env.pins.at[1, 0].set(24)
# env.pins = env.pins.at[1, 3].set(25)
# env.pins = env.pins.at[1, 1].set(26)
# env.pins = env.pins.at[1, 2].set(27)
# env.pins = env.pins.at[0, 0].set(3)
# env.pins = env.pins.at[0, 2].set(10)
# env.pins = env.pins.at[0, 3].set(15)
# env.current_player = 1
# print(env.board)
# print(get_winner(env))
# print(env.pins)
# env, reward, done = env_step(env, jnp.array([2 ,1], dtype=jnp.int8))

# print("After action:")
# print(get_winner(env))
# print(env.pins)
# print(reward)
# print(done)
# print(env.board)

# env.current_player = 1
# print(env)
# print(jnp.isin(env.pins[env.current_player], env.goal[env.current_player]))
# done = jnp.all(jnp.isin(env.pins[env.current_player], env.goal[env.current_player])) # check if the current player has won, order of the pins does not matter
# print("Done: ", done)
