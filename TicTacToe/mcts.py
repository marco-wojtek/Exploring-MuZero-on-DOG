import jax
import jax.numpy as jnp
from TicTacToeV2 import TicTacToeV2, root_fn, recurrent_fn, env_reset, env_step
# from TicTacToe import TicTacToe, root_fn, recurrent_fn, env_reset, env_step
import functools
import chex
import mctx

@functools.partial(jax.jit, static_argnums=(2,))
def run_mcts(rng_key:chex.PRNGKey, env:TicTacToeV2, num_simulations:int):
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    policy_output = mctx.muzero_policy(
        params= None,
        rng_key=key1,
        root = jax.vmap(root_fn, (None, 0))(env,jax.random.split(key2, batch_size)),
        recurrent_fn=jax.vmap(recurrent_fn, (None,None,0,0)),
        num_simulations=num_simulations,
        max_depth=9,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        dirichlet_fraction=0.0,
    )
    return policy_output

# env = env_reset(0)
# env, reward, done = env_step(env, 0)
# print(env.board)
# env, reward, done = env_step(env, 1)
# print(env.board)
# env, reward, done = env_step(env, 2)
# print(env.board)
# env, reward, done = env_step(env, 3)
# print(env.board)
# env, reward, done = env_step(env, 4)
# print(env.board)
# env, reward, done = env_step(env, 5)
# print(env.board)
# env, reward, done = env_step(env, 7)
# print(env.board)
# policy_output = run_mcts(jax.random.PRNGKey(0), env, 50)
# w = policy_output.action_weights
# print(w.mean(axis=0))