import chex
import jax
import jax.numpy as jnp

Board = chex.Array
Start = chex.Array
Target = chex.Array
Goal = chex.Array
Action = chex.Array
Player = chex.Array
Reward = chex.Array
Done = chex.Array
Pins = chex.Array
Num_players = chex.Array
Size = chex.Array
Die = chex.Array

@chex.dataclass
class MADN:
    board: Board  # shape (64,), values in {0, 1, 2, 3, 4} for empty, player 1, player 2, player 3, player 4
    num_players: Num_players
    current_player: Player  # scalar, 1, 2, 3, or 4
    pins : Pins  # shape (num_players,4), positions of the players' pins
    reward: Reward  # scalar, reward for the current player
    done: Done  # scalar, whether the game is over
    start: Start  # shape (num_players,), starting indices of the players
    target: Target  # shape (num_players,), positions before the goals of the players
    goal: Goal  # shape (num_players,4), goal positions of the players
    board_size: Size  # scalar, size of the board (num_players * distance)
    total_board_size: Size  # scalar, size of the board + goal areas (num_players * distance + num_players * 4)
    die : Die

def get_winner(goal: Goal) -> Player:
    '''
    returns the index of the winning player or 0 if tie or not Done
    '''
    player_goals = jnp.all(goal > 0, axis=1)
    return jnp.where(jnp.any(player_goals), jnp.argmax(player_goals)+1, 0)

def env_reset(_, num_players=jnp.int8(4), distance=jnp.int8(10)) -> MADN:
    board_size = num_players * distance
    total_board_size = board_size + num_players * 4 # add goal areas
    num_pins = 4
    return MADN(
        board = - jnp.ones(total_board_size, dtype=jnp.int8), # board is filled with -1 (empty) or 0-3 (player index)
        num_players = num_players, # number of players
        pins = - jnp.ones((num_players,num_pins), dtype=jnp.int8), # index of each players' pins, -1 means in start area
        current_player=jnp.int8(0), # index of current player, 0-3
        done = jnp.bool_(False), # whether the game is over
        reward=jnp.int8(0), # reward for the current player
        start = jnp.array(jnp.arange(num_players)*distance, dtype=jnp.int8), # starting positions of each player
        target = jnp.array((jnp.arange(num_players)*distance - 1)%board_size, dtype=jnp.int8),
        goal = jnp.reshape(jnp.arange(board_size, board_size + num_players*4, dtype=jnp.int8), (num_players, 4)),
        board_size=jnp.int8(board_size),
        total_board_size=jnp.int8(total_board_size),
        die=jnp.int8(0)
    )

@jax.jit
def env_step(env: MADN, action: Action) -> MADN:
    pin = action.astype(jnp.int8)
    move = env.die
    # check if the action is valid
    invalid_action = ~valid_action(env)[pin]

    moved_positions = env.pins[env.current_player, pin] + move
    new_positions = moved_positions % env.board_size
    x = moved_positions - env.target[env.current_player]

    new_position = jnp.where(
        env.pins[env.current_player, pin] == -1,
        env.start[env.current_player], # move from start area to starting position
        jnp.where(
            (4 >= x) & (x > 0) & (env.board[env.goal[env.current_player, x-1]] != env.current_player),
            env.goal[env.current_player, x-1], # move to goal position
            new_positions
        )
    )

    # update pins
    pin_at_pos = env.board[new_position] # check if a player was at the new position
    pins = env.pins.at[env.current_player, pin].set(jnp.where(invalid_action, env.pins[env.current_player, pin], new_position))
    pins = jax.lax.cond(
        (pin_at_pos != -1) & (pin_at_pos != env.current_player) & ~invalid_action, # if a player was at the new position and it's not the current player and the action is valid
        lambda p: p.at[pin_at_pos, jnp.where(p[pin_at_pos] == new_position, jnp.arange(4, dtype=jnp.int8), p[pin_at_pos])].set(-1), # send the pin of that player back to start area
        lambda p: p,
        pins
    )
    board = env.board.at[new_position].set(jnp.where(invalid_action, env.board[new_position], env.current_player))

    board = jax.lax.cond(
        env.pins[env.current_player, pin] != -1, # if the pin was moved
        lambda b: b.at[env.pins[env.current_player, pin]].set(-1), # set the old position to empty
        lambda b: b, board
    )

    reward = jnp.where(env.done, 0, jnp.where(invalid_action, -1, get_winner(env.goal))) # reward is 0 if game is done, -1 if action is invalid, else the index of the winning player (1-4) or 0 if no winner yet
    # check if the game is done
    done = jnp.all(jnp.isin(env.pins[env.current_player], env.goal[env.current_player])) # check if the current player has won, order of the pins does not matter
    current_player = jnp.where(done | (move == 6) | invalid_action, env.current_player, (env.current_player + 1) % env.num_players) # if the game is not done or the player played a 6, switch to the next player

    env = MADN(
        board=board,
        num_players=env.num_players,
        pins=pins,
        current_player=current_player,
        done= done,
        reward=reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        die=env.die
    )
    return env, reward, done

def valid_action(env:MADN) -> chex.Array:
    '''
    Returns a mask of shape (4, 6) indicating which actions are valid for each pin of the current player
    '''
    #return valid_action for each pin of the current player
    current_player = env.current_player
    current_pins = env.pins[current_player]
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]
    die = env.die

    # calculate possible actions
    moved_positions = current_pins + die # shape (4, 1)
    new_positions = moved_positions % env.board_size
    x = moved_positions - target

    # filter out invalid moves blocked by own pins
    result = (board[new_positions] != current_player) # check move to any board position

    result = jnp.where(
        (4 >= x) & (x > 0),
        result | (board[goal[x-1]] != current_player), # if goal is possible, check if goal position is free
        result
    )

    # filter actions for pins in goal area
    result = jnp.where(
        jnp.isin(current_pins, goal),
        current_pins + die <= goal[-1],
        result
    )
    # filter actions for pins in start area
    result = jnp.where(
        (current_pins == -1),
        jnp.isin(die, jnp.array([1, 6])),
        result
    )
    return result # filter possible actions with available actions

def dice_roll(env:MADN, rng_key:chex.PRNGKey) -> chex.Array:
    env.die = jax.random.randint(rng_key, shape=(), minval=1, maxval=7).astype(jnp.int8)
    return env.die

def winning_action(env:MADN) -> chex.Array:
    pass

def policy_function(env:MADN) -> chex.Array:
    pass    

def rollout(env:MADN, rng_key:chex.PRNGKey, policy_function, max_steps:int=100) -> tuple[MADN, chex.PRNGKey]:
    pass

def value_function(env:MADN) -> chex.Array:
    pass

def root_fn(params, rng_key, env:MADN):
    pass

def recurrent_fn(params, rng_key, action: Action, embedding:MADN):
    pass    


env = env_reset(0)
d = dice_roll(env, jax.random.PRNGKey(2))
print(d)
env, _, _ = env_step(env, jnp.int8(0))
print(env)