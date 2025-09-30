import chex
import jax
import jax.numpy as jnp
import mctx

Board = chex.Array
Action = chex.Array
Player = chex.Array
Reward = chex.Array
Done = chex.Array

@chex.dataclass
class TicTacToe:
    board: Board  # shape (3, 3), values in {0, 1, -1} for empty, player 1, player -1
    current_player: Player  # scalar, 1 or -1
    reward: Reward  # scalar, reward for the current player
    done: Done  # scalar, whether the game is over

def get_winner(board:Board) -> Player:
    '''
    returns the winning player or 0 if tie or not Done
    '''
    lines = jnp.concatenate([
        board,  # rows
        board.T,  # columns
        jnp.array([[board[i,i] for i in range(3)]]),  # main diagonal
        jnp.array([[board[i,2-i] for i in range(3)]])  # anti diagonal
    ])
    line_sums = jnp.sum(lines, axis=1)
    winner = jnp.where(jnp.any(line_sums == 3), 1, 0)
    winner = jnp.where(jnp.any(line_sums == -3), -1, winner)
    return winner

def env_reset(_):
    return TicTacToe(
        board = jnp.zeros((3,3), dtype=jnp.int8),
        current_player=jnp.int8(1),
        done = jnp.bool_(False),
        reward=jnp.int8(0),
    )

def env_step(env:TicTacToe, action:Action) -> tuple[TicTacToe, Reward, Done]:
    row, column = action // 3, action % 3
    invalid_move = env.board[row, column] != 0

    # change board if game is not done and the action is valid
    board = env.board.at[row, column].set(jnp.where(env.done | invalid_move, env.board[row, column], env.current_player))

    reward = jnp.where(env.done, 0, jnp.where(invalid_move, -1, get_winner(board)*env.current_player)).astype(jnp.int8)
    
    done = env.done | reward != 0 | invalid_move | jnp.all(board != 0)

    env = TicTacToe(
        board=board,
        current_player=jnp.where(done, env.current_player, - env.current_player),
        done= done,
        reward=reward
    )
    return env, reward, done

def valid_action_mask(env:TicTacToe) -> chex.Array:
    return jnp.where(env.done, jnp.full(env.board.shape, False), env.board == 0)

def winning_action_mask(env:TicTacToe, player:Player) -> chex.Array:
    env = TicTacToe(
        board=env.board,
        current_player=player,
        done=env.done,
        reward=env.reward
    )

    env, reward, done = jax.vmap(env_step, (None, 0))(env, jnp.arange(9, dtype=jnp.int8))
    return reward == 1

def policy_function(env:TicTacToe) -> chex.Array:
    'assign value to moves; 0 for invalid, 100 for legal moves, 200 for opp winning moves and 300 for own winning moves'
    return sum((
        valid_action_mask(env).reshape(-1).astype(jnp.float32) * 100,
        winning_action_mask(env, -env.current_player) * 200,
        winning_action_mask(env, env.current_player) * 300
    ))

def rollout(env:TicTacToe, rng_key: chex.PRNGKey) -> Reward:

    def cond(a):
        env, key = a
        return ~env.done
    def step(a):
        env, key = a
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, policy_function(env))
        env, reward, done = env_step(env, action)
        return env, key
    leaf, key = jax.lax.while_loop(cond, step, (env, rng_key))
    return leaf.reward * leaf.current_player * env.current_player

def value_function(env:TicTacToe, rng_key:chex.PRNGKey) -> chex.Array:
    return rollout(env, rng_key).astype(jnp.float32)

def root_fn(env:TicTacToe, rng_key:chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = policy_function(env),
        value=value_function(env, rng_key),
        embedding=env,
    )

def recurrent_fn(params, rng_key, action, embedding):
    env = embedding

    env, reward, done = env_step(env, action)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward= reward,
        discount= jnp.where(done, 0, -1).astype(jnp.float32),
        prior_logits = policy_function(env),
        value = jnp.where(done, 0, value_function(env, rng_key)).astype(jnp.float32),
    )

    return recurrent_fn_output, env