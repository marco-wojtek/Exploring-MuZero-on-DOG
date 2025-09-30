import chex
import jax
import jax.numpy as jnp
import mctx

Board = chex.Array
Start = chex.Array
Target = chex.Array
Goal = chex.Array
Action = chex.Array
Player = chex.Array
Reward = chex.Array
Done = chex.Array
Action_set = chex.Array

@chex.dataclass
class deterministic_MADN:
    board: Board  # shape (11, 11), values in {0, 1, -1} for empty, player 1, player 2, player 3, player 4
    current_player: Player  # scalar, 1 or -1
    reward: Reward  # scalar, reward for the current player
    done: Done  # scalar, whether the game is over
    action_set: Action_set  # available actions, 1-6 each 3x until empty, then refilled
    start: Start  # shape (4,), starting positions of the players
    target: Target  # shape (4,), positions before the goals of the players
    goal: Goal  # shape (4,4), goal positions of the players

def get_winner(goal: Goal) -> Player:
    '''
    returns the index of the winning player or 0 if tie or not Done
    '''
    player_goals = jnp.all(goal > 0, axis=1)
    return jnp.where(jnp.any(player_goals), jnp.argmax(player_goals)+1, 0)

def env_reset(_):
    return deterministic_MADN(
        board = jnp.zeros((11,11), dtype=jnp.int8),
        current_player=jnp.int8(1),
        done = jnp.bool_(False),
        reward=jnp.int8(0),
        action_set = jnp.array([1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6], dtype=jnp.int8),
        start = jnp.array([0, 10, 110, 120], dtype=jnp.int8),
        target = jnp.array([39, 49, 89, 99], dtype=jnp.int8),
        goal = jnp.zeros((4,4), dtype=jnp.int8)
    )

def env_step(state: deterministic_MADN, action: Action) -> deterministic_MADN:
    pass

def valid_action(env:deterministic_MADN) -> chex.Array:
    pass

def winning_action(env:deterministic_MADN) -> chex.Array:
    pass

def policy_function(env:deterministic_MADN) -> chex.Array:
    pass    

def rollout(env:deterministic_MADN, rng_key:chex.PRNGKey, policy_function, max_steps:int=100) -> tuple[deterministic_MADN, chex.PRNGKey]:
    pass

def value_function(env:deterministic_MADN) -> chex.Array:
    pass

def root_fn(params, rng_key, env:deterministic_MADN):
    pass

def recurrent_fn(params, rng_key, action: Action, embedding:deterministic_MADN):
    pass    
