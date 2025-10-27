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
Action_set = chex.Array
Pins = chex.Array
Num_players = chex.Array
Size = chex.Array

@chex.dataclass
class deterministic_MADN:
    board: Board  # shape (64,), values in {0, 1, 2, 3, 4} for empty, player 1, player 2, player 3, player 4
    num_players: Num_players
    current_player: Player  # scalar, 1, 2, 3, or 4
    pins : Pins  # shape (num_players,4), positions of the players' pins
    reward: Reward  # scalar, reward for the current player
    done: Done  # scalar, whether the game is over
    action_set: Action_set  # available actions, 1-6 each 3x until empty, then refilled
    start: Start  # shape (num_players,), starting indices of the players
    target: Target  # shape (num_players,), positions before the goals of the players
    goal: Goal  # shape (num_players,4), goal positions of the players
    board_size: Size  # scalar, size of the board (num_players * distance)
    total_board_size: Size  # scalar, size of the board + goal areas (num_players * distance + num_players * 4)

def get_winner(goal: Goal) -> Player:
    '''
    returns the index of the winning player or 0 if tie or not Done
    '''
    player_goals = jnp.all(goal > 0, axis=1)
    return jnp.where(jnp.any(player_goals), jnp.argmax(player_goals)+1, 0)

def env_reset(_, num_players=jnp.int8(4), distance=jnp.int8(10)) -> deterministic_MADN:
    board_size = num_players * distance
    total_board_size = board_size + num_players * 4 # add goal areas
    num_pins = 4
    return deterministic_MADN(
        board = - jnp.ones(total_board_size, dtype=jnp.int8), # board is filled with -1 (empty) or 0-3 (player index)
        num_players = num_players, # number of players
        pins = - jnp.ones((num_players,num_pins), dtype=jnp.int8), # index of each players' pins, -1 means in start area
        current_player=jnp.int8(0), # index of current player, 0-3
        done = jnp.bool_(False), # whether the game is over
        reward=jnp.int8(0), # reward for the current player
        action_set= num_pins * jnp.ones((num_players, 6), dtype=jnp.int8), # each player starts with 4 actions 1-6
        start = jnp.array(jnp.arange(num_players)*distance, dtype=jnp.int8), # starting positions of each player
        target = jnp.array((jnp.arange(num_players)*distance - 1)%board_size, dtype=jnp.int8),
        goal = jnp.reshape(jnp.arange(board_size, board_size + num_players*4, dtype=jnp.int8), (num_players, 4)),
        board_size=jnp.int8(board_size),
        total_board_size=jnp.int8(total_board_size),
    )

@jax.jit
def env_step(env: deterministic_MADN, action: Action) -> deterministic_MADN:
    pin = action[0].astype(jnp.int8)
    move = action[1].astype(jnp.int8) # action is in {1, 2, 3, 4, 5, 6}
    # check if the action is valid
    invalid_action = ~valid_action(env)[pin, move-1]

    moved_positions = env.pins[env.current_player, pin] + move
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - env.target[env.current_player]

    new_position = jnp.where(
        env.pins[env.current_player, pin] == -1,
        env.start[env.current_player], # move from start area to starting position
        jnp.where(jnp.isin(env.pins[env.current_player, pin], env.goal[env.current_player]),
                    moved_positions,  
                    jnp.where(
                        (4 >= x) & (x > 0) & (env.board[env.goal[env.current_player, x-1]] != env.current_player),
                        env.goal[env.current_player, x-1], # move to goal position
                        fitted_positions
                    )
        )
    )

    # update pins
    pin_at_pos = env.board[new_position] # check if a player was at the new position, also consider start area
    pins = env.pins.at[env.current_player, pin].set(jnp.where(invalid_action, env.pins[env.current_player, pin], new_position))
    pins = jax.lax.cond(
        (pin_at_pos != -1) & (pin_at_pos != env.current_player) & ~invalid_action, # if a player was at the new position and it's not the current player and the action is valid
        lambda p: p.at[pin_at_pos].set(jnp.where(p[pin_at_pos] == new_position, -1, p[pin_at_pos])), # send the pin of that player back to start area
        lambda p: p,
        pins
    )
    board = env.board.at[new_position].set(jnp.where(invalid_action, env.board[new_position], env.current_player))

    board = jax.lax.cond(
        env.pins[env.current_player, pin] != -1, # if the pin was moved
        lambda b: b.at[env.pins[env.current_player, pin]].set(-1), # set the old position to empty
        lambda b: b, board
    )

    # update action set, change one instance of the played action to 0 (no action)
    curr_state = env.action_set.at[env.current_player, move-1].get()
    action_set = env.action_set.at[env.current_player, move-1].set(jnp.where(invalid_action | (curr_state == 0), curr_state, curr_state-1))
    action_set = jax.lax.cond(
        jnp.all(action_set[env.current_player] == 0), # if all actions are 0, refill the action set
        lambda a: a.at[env.current_player].set(env.pins.shape[1] * jnp.ones(6, dtype=jnp.int8)),
        lambda a: a,
        action_set
    )
    reward = jnp.where(env.done, 0, jnp.where(invalid_action, -1, get_winner(env.goal))) # reward is 0 if game is done, -1 if action is invalid, else the index of the winning player (1-4) or 0 if no winner yet
    # check if the game is done
    done = jnp.all(jnp.isin(pins[env.current_player], env.goal[env.current_player])) # check if the current player has won, order of the pins does not matter
    current_player = jnp.where(done | (move == 6) | invalid_action, env.current_player, (env.current_player + 1) % env.num_players) # if the game is not done or the player played a 6, switch to the next player

    env = deterministic_MADN(
        board=board,
        num_players=env.num_players,#/
        pins=pins,#
        current_player=current_player,
        done= done,#
        reward=reward,#
        action_set=action_set,#/
        start=env.start,#
        target=env.target,#
        goal=env.goal,#
        board_size=env.board_size,#
        total_board_size=env.total_board_size,#
    )
    return env, reward, done

def refill_action_set(env:deterministic_MADN) -> chex.Array:
    '''
    Refills the action set for the current player if all actions are used up.
    '''
    env.action_set = env.action_set.at[env.current_player].set(env.pins.shape[1] * jnp.ones(6, dtype=jnp.int8))

def no_step(env:deterministic_MADN) -> deterministic_MADN:
    """
    No-op step function for the environment.
    """
    env = deterministic_MADN(
        board=env.board,
        num_players=env.num_players,
        pins=env.pins,
        current_player=(env.current_player + 1) % env.num_players,
        done=env.done,
        reward=env.reward,
        action_set=env.action_set,
        start=env.start,
        target=env.target,
        goal=env.goal,
        board_size=env.board_size,
        total_board_size=env.total_board_size,
    )
    return env, 0, env.done

@jax.jit
def valid_action(env:deterministic_MADN) -> chex.Array:
    '''
    Returns a mask of shape (4, 6) indicating which actions are valid for each pin of the current player
    '''
    #return valid_action for each pin of the current player
    current_player = env.current_player
    current_pins = env.pins[current_player]
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]
    action_set = env.action_set[current_player]
    valid_actions = jnp.where(action_set>0, True, False) # available actions for each pin

    # calculate possible actions
    moved_positions = current_pins[:, None] + jnp.arange(1, 7)
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - target

    # filter out invalid moves blocked by own pins
    result = (board[fitted_positions] != current_player) # check move to any board position

    result = jnp.where(
        (4 >= x) & (x > 0),
        result | (board[goal[x-1]] != current_player), # if goal is possible, check if goal position is free
        result
    )
    # filter actions for pins in goal area
    result = jnp.where(
        jnp.isin(current_pins, goal)[:, None],
        (current_pins[:, None] + jnp.arange(1, 7) <= goal[-1]) & (board[moved_positions] != current_player),
        result
    )

    # filter actions for pins in start area
    result = jnp.where(
        (current_pins == -1)[:, None],
        jnp.isin(jnp.arange(1, 7), jnp.array([1, 6])) & (env.board[env.start[current_player]] != env.current_player),
        result
    )
    return result & valid_actions # filter possible actions with available actions

def encode_board(env:deterministic_MADN) -> chex.Array:
    '''
    returns a 4xN array encoding the board state for each player
    '''
    start = env.start
    num_players = env.num_players
    board = env.board

    current_pin_positions = (board == jnp.arange(num_players)[:, None]).astype(jnp.int8)
    current_player = jnp.ones_like(board, dtype=jnp.int8) * env.current_player
    available_actions = jax.vmap(lambda p: jnp.ones_like(board, dtype=jnp.int8) * env.action_set[p])(jnp.arange(jnp.arange(len(env.action_set))))
    board_encoding = jnp.concatenate([current_pin_positions, current_player[None, :], available_actions], axis=0)
    return board_encoding

    # encoding = jax.vmap(lambda s: jnp.roll(board, -s))(start)
    # board_encoding = (encoding == jnp.arange(num_players)[:, None]).astype(jnp.int8)
    # return board_encoding


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

# action = jnp.array([0, 2], dtype=jnp.int8)
# env = env_reset(0, num_players=4, distance=10)

# # env_step ist mit @jax.jit dekoriert; das Original ist unter __wrapped__
# res_py = env_step.__wrapped__(env, action)    # ungejittete Ausführung
# res_jit = env_step(env, action)               # jittete Ausführung (erster Aufruf kompiliert)

# # warten bis fertig und vom Gerät holen
# res_jit = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, res_jit)
# res_py = jax.device_get(res_py)
# res_jit = jax.device_get(res_jit)

# # Prüfen auf Gleichheit
# chex.assert_trees_all_close(res_py, res_jit, atol=0, rtol=0)
# print("JIT und Non-JIT Ergebnisse sind gleich.")

# # Debug: Inhalte anzeigen
# res_py = jax.tree_util.tree_map(lambda x: jnp.array(x), res_py)
# res_jit = jax.tree_util.tree_map(lambda x: jnp.array(x), res_jit)
# print("res_py:", res_py)
# print("res_jit:", res_jit)
# # optional: JAXPR der nicht-jittbaren Version ansehen
# print("jaxpr env_step (non-jit):")
# print(jax.make_jaxpr(env_step.__wrapped__)(env, action))

