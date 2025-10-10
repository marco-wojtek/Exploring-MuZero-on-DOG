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
Pins = chex.Array
Num_players = chex.Array

@chex.dataclass
class deterministic_MADN:
    board: Board  # shape (11, 11), values in {0, 1, -1} for empty, player 1, player 2, player 3, player 4
    num_players: Num_players
    current_player: Player  # scalar, 1 or -1
    pins : Pins  # shape (4,4), positions of the players' pins
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

def env_reset(_, num_players:int=4) -> deterministic_MADN:
    return deterministic_MADN(
        board = jnp.zeros(num_players*16, dtype=jnp.int8),
        num_players = num_players,
        pins = jnp.zeros((num_players,4), dtype=jnp.int8)-1,
        current_player=jnp.int8(1),
        done = jnp.bool_(False),
        reward=jnp.int8(0),
        action_set=jnp.tile(jnp.arange(1, 6 + 1, dtype=jnp.int8), (num_players, 3)),
        start = jnp.array(jnp.arange(num_players)*16, dtype=jnp.int8),
        target = jnp.array(jnp.arange(num_players)*16 - 1, dtype=jnp.int8),
        goal = jnp.zeros((num_players,4), dtype=jnp.int8)
    )

def env_step(state: deterministic_MADN, action: Action) -> deterministic_MADN:
    pass

def valid_action(env:deterministic_MADN) -> chex.Array:
    #return valid_action for each pin of the current player
    current_player_index = env.current_player - 1
    valid_values = env.action_set[current_player_index]
    actions = jnp.unique(valid_values)

    return jnp.where(
        (env.pins[current_player_index] == -1)[:, None],
        jnp.isin(actions, jnp.array([1, 6])),  # only 1 or 6 can move from home
        jnp.ones_like(actions, dtype=jnp.bool)  # all values can move if not at home
    )
    
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

