from deterministic_madn import deterministic_MADN, env_reset, env_step, refill_action_set, valid_action, no_step
import jax
import jax.numpy as jnp


def simulate_game():
    env = env_reset(0, num_players=jnp.int8(2), distance=jnp.int8(10))
    rng_key = jax.random.PRNGKey(0)
    steps = 0
    while not env.done:
        print("Current player: ", env.current_player)
        print("Current board:\n", env.board)
        print("Current pins:\n", env.pins)
        print("Current actionset:\n", env.action_set)
        valid_actions = valid_action(env)
        print("Valid actions:\n", valid_actions)
        if not jnp.any(valid_actions):
            refill_action_set(env)
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
        print("-"*30)
        steps += 1
    print("Game over after ", steps, " steps.")
    print("Final board:\n", env.board)
    print("Final pins:\n", env.pins)

simulate_game()
# env = env_reset(0, num_players=jnp.int8(2), distance=jnp.int8(10))
# env.board = env.board.at[24].set(1)
# env.board = env.board.at[26].set(1)
# env.board = env.board.at[27].set(1)
# env.board = env.board.at[25].set(1)
# env.board = env.board.at[10].set(0)
# env.board = env.board.at[15].set(0)
# env.board = env.board.at[3].set(0)
# env.pins = env.pins.at[1, 0].set(24)
# env.pins = env.pins.at[1, 1].set(26)
# env.pins = env.pins.at[1, 2].set(27)
# env.pins = env.pins.at[1, 3].set(25)
# env.pins = env.pins.at[0, 0].set(3)
# env.pins = env.pins.at[0, 2].set(10)
# env.pins = env.pins.at[0, 3].set(15)

# env.current_player = 1
# print(env)
# print(jnp.isin(env.pins[env.current_player], env.goal[env.current_player]))
# done = jnp.all(jnp.isin(env.pins[env.current_player], env.goal[env.current_player])) # check if the current player has won, order of the pins does not matter
# print("Done: ", done)
