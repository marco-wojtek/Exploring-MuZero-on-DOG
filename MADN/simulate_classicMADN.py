import functools
from classic_madn import *
import jax
import mctx


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


def run_mcts_search(env: classic_MADN, rng_key: chex.PRNGKey, num_simulations: int = 100):
    """
    Führt MCTS Suche mit stochastic MuZero aus
    """
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    # MCTS Policy mit chance function
    mcts_policy = mctx.stochastic_muzero_policy(
        params=None,  # Keine gelernten Parameter in diesem Beispiel
        rng_key=key1,
        root=jax.vmap(root_fn, (None, 0))(env,jax.random.split(key2, batch_size)),
        decision_recurrent_fn=recurrent_fn,
        chance_recurrent_fn=recurrent_chance_fn,
        num_simulations=num_simulations,
        max_depth=300,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1)
        )
    
    return mcts_policy

def madn_mcts_example():
    rng_key = jax.random.PRNGKey(12)
    env = env_reset(0, num_players=2, distance=10)
    pins = jnp.array([[3,2,6,18],
                      [10, 4, 19, 0]])
    env.pins = pins
    env.board = set_pins_on_board(env.board, pins)    
    print(env.board)    
    # Würfel werfen
    rng_key, subkey = jax.random.split(rng_key)
    env = throw_die(env, subkey)
    print("Die throw:\n", env.die)
    
    # MCTS Suche
    rng_key, subkey = jax.random.split(rng_key)
    policy_output = run_mcts_search(env, subkey, num_simulations=200)
    
    # Beste Aktion wählen
    action = jnp.argmax(policy_output.action_weights)
    
    return action, policy_output.action_weights

print(madn_mcts_example())
# simulate_game(env_reset(0, num_players=2, distance=10, enable_initial_free_pin=True), key=42)

rng_key = jax.random.PRNGKey(12)
env = env_reset(0, num_players=2, distance=10)
pins = jnp.array([[3,2,6,18],
                    [10, 4, 19, 0]])
env.board = set_pins_on_board(env.board, pins)    
print(env.board)    
# Würfel werfen
rng_key, subkey = jax.random.split(rng_key)
env = throw_die(env, subkey)
print("Die throw:\n", env.die)
valid_actions = valid_action(env)
print("Valid actions:\n", valid_actions)
print(value_function(env,rng_key))
# env, reward, done = env_step(env, jnp.array(2, dtype=jnp.int8))
