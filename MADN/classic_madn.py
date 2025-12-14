import chex
import jax
import jax.numpy as jnp
import mctx

NORMAL_DICE_DISTRIBUTION = jnp.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
# Options for soft-locked players and rethrowing dice
OUT_ON_SIX_DICE_DISTRIBUTION = jnp.array([25/216, 25/216, 25/216, 25/216, 25/216, 91/216])
OUT_ON_ONE_DICE_DISTRIBUTION = jnp.array([91/216, 25/216, 25/216, 25/216, 25/216, 25/216])
OUT_ON_ONE_AND_SIX_DICE_DISTRIBUTION = jnp.array([76/216, 16/216, 16/216, 16/216, 16/216, 76/216])

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
class classic_MADN:
    board: Board  # shape (64,), values in {0, 1, 2, 3, 4} for empty, player 1, player 2, player 3, player 4
    num_players: Num_players
    current_player: Player  # scalar, 1, 2, 3, or 4
    pins : Pins  # shape (num_players,4), positions of the players' pins
    reward: Reward  # scalar, reward for the current player
    done: Done  # scalar, whether the game is over
    start: Start  # shape (num_players,), starting indices of the players
    target: Target  # shape (num_players,), positions before the goals of the players
    goal: Goal  # shape (num_players,4), goal positions of the players
    die: Die
    board_size: Size  # scalar, size of the board (num_players * distance)
    total_board_size: Size  # scalar, size of the board + goal areas (num_players * distance + num_players * 4)
    rules : dict  # game rules

def env_reset(
        _,
        num_players=jnp.int8(4),
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=jnp.int8(10),
        starting_player = jnp.int8(0), # -1, 0, 1, 2, 3
        key = jax.random.PRNGKey(0),
        enable_teams = False,
        enable_initial_free_pin = False,
        enable_circular_board = True,
        enable_start_blocking = False,
        enable_jump_in_goal_area = True,
        enable_friendly_fire = False, # friendly fire only in board space, not goal area
        enable_start_on_1 = True,
        enable_bonus_turn_on_6 = True,
        enable_dice_rethrow = False,
            ) -> classic_MADN:
    
    subkey, _ = jax.random.split(key)
    starting_player = jnp.where((starting_player < 0) | (starting_player >= num_players), jax.random.randint(subkey, (), 0, num_players), starting_player)
    
    board_size = 4 * distance
    total_board_size = board_size + 4 * 4 # add goal areas
    num_pins = 4

    enable_teams = enable_teams & (num_players == 4)
    # board indicator positions
    # layout must work for number of players
    layout = jax.lax.cond(
        (jnp.sum(layout)!=num_players) | (jnp.all(layout) & (num_players < 4)),
        lambda: jnp.array([False, False, False, False], dtype=jnp.bool_).at[:num_players].set(True),
        lambda: layout
    )
    
    start = jnp.array(jnp.arange(4)*distance, dtype=jnp.int8)[layout]
    target = (start - 1)%board_size
    goal = jnp.reshape(jnp.arange(board_size, board_size + 4*4, dtype=jnp.int8), (4, 4))[layout, :]


    pins = - jnp.ones((num_players,num_pins), dtype=jnp.int8)
    pins = jax.lax.cond(
        enable_initial_free_pin,
        lambda: pins.at[:,0].set(start),
        lambda: pins
    )
    board = - jnp.ones(total_board_size, dtype=jnp.int8)
    board = jax.lax.cond(
        enable_initial_free_pin,
        lambda: set_pins_on_board(board, pins),
        lambda: board
    )
    return classic_MADN(
        board = board, # board is filled with -1 (empty) or 0-3 (player index)
        num_players = jnp.array(num_players, dtype=jnp.int8), # number of players
        pins = pins,
        current_player=jnp.array(starting_player, dtype=jnp.int8), # index of current player, 0-3
        done = jnp.bool_(False), # whether the game is over
        reward=jnp.array(0, dtype=jnp.int8), # reward for the current player
        start = start,
        target = target,
        goal = goal,
        die = jnp.array(0, dtype=jnp.int8),
        board_size=jnp.array(board_size, dtype=jnp.int8),
        total_board_size=jnp.array(total_board_size, dtype=jnp.int8),
        rules = {
        'enable_teams': enable_teams,
        'enable_initial_free_pin':enable_initial_free_pin,
        'enable_circular_board':enable_circular_board,
        'enable_start_blocking':enable_start_blocking,
        'enable_jump_in_goal_area':enable_jump_in_goal_area,
        'enable_friendly_fire':enable_friendly_fire,
        'enable_start_on_1':enable_start_on_1,
        'enable_bonus_turn_on_6':enable_bonus_turn_on_6,
        'enable_dice_rethrow':enable_dice_rethrow,
        }
    )

def is_player_done(num_players, board:Board, goal:Goal, player: Player) -> chex.Array:
    '''
    returns winner as jnp.array to support multiple winners in team mode    
    '''
    return jax.lax.cond(
        player >= num_players,
        lambda: False,
        lambda: jnp.all(board[goal[player]] >= 0)
    )

def get_winner(env: classic_MADN, board: Board) -> chex.Array:
    collect_winners = jax.vmap(is_player_done, in_axes=(None, None, None, 0))
    players_done = collect_winners(env.num_players, board, env.goal, jnp.arange(4, dtype=jnp.int8))  # (4,)

    def four_players_case():
        team_0 = players_done[0] & players_done[2]  # Team 0&2 fertig
        team_1 = players_done[1] & players_done[3]  # Team 1&3 fertig
        both = team_0 & team_1  # Beide Teams fertig (unentschieden)
        none = ~(team_0 | team_1)  # Kein Team fertig
        
        return jax.lax.cond(
            both | none,  # Bei Unentschieden oder keinem Gewinner
            lambda: jnp.full(players_done.shape, False, dtype=jnp.bool_),  # [-1, -1]
            lambda: jax.lax.cond(
                team_0,  # Falls Team 0&2 gewonnen hat
                lambda:jnp.array([False, True, False, True], dtype=jnp.bool_),  # [0, 2]
                lambda: jnp.array([True, False, True, False], dtype=jnp.bool_)   # [1, 3]
            )
        )


    return jax.lax.cond(env.rules['enable_teams'], four_players_case, lambda: players_done)

def is_soft_locked(env: classic_MADN) -> chex.Array:
    '''
    Checks if the current player is in a soft locked position
    A soft locked position is defined as having all pins either in the start area or in the goal area and not being able to move any pin out of the start area or within the goal area.
    '''
    current_player = env.current_player
    pins = env.pins[current_player]
    board = env.board
    goal_pos = env.goal[current_player]

    pins_not_at_home = len(pins) - jnp.count_nonzero(pins == -1) # number of pins not at home
    locked_condition = jnp.where(
        pins_not_at_home >0,
        jnp.all(board[goal_pos[-pins_not_at_home:]] == current_player),  # check if pins in goal area are locked
        True
    )
    return locked_condition

def dice_probabilities(env:classic_MADN) -> chex.Array:
    soft_locked = is_soft_locked(env)
    #print("Soft locked: ", soft_locked)
    a= jax.lax.cond(
        soft_locked & env.rules['enable_dice_rethrow'],
        lambda: jax.lax.cond(
            env.rules['enable_start_on_1'],
            lambda: OUT_ON_ONE_AND_SIX_DICE_DISTRIBUTION,
            lambda: OUT_ON_SIX_DICE_DISTRIBUTION
        ),
        lambda: NORMAL_DICE_DISTRIBUTION
    )
    #print("Dice probabilities: ", a)
    return a

def throw_die(env: classic_MADN, rng_key: chex.PRNGKey) -> classic_MADN:
    return classic_MADN(
        board=env.board,
        num_players=env.num_players,
        pins=env.pins,
        current_player=env.current_player,
        done=env.done,
        reward=env.reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        die=jax.random.choice(rng_key, jnp.array([1,2,3,4,5,6], dtype=jnp.int8), p = dice_probabilities(env)),
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )

def set_die(env: classic_MADN, die_value: chex.Array) -> classic_MADN:
    return classic_MADN(
        board=env.board,
        num_players=env.num_players,
        pins=env.pins,
        current_player=env.current_player,
        done=env.done,
        reward=env.reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        die=die_value.astype(jnp.int8),
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )

def check_goal_path_for_pin(start, x_val, goal, board, current_player):
    """Prüft für einen einzelnen Pin ob der Pfad im Goal frei ist
    Args:
        start: Position im Ziel, falls Pin im Ziel ist, sonst -1 wenn pin nicht im Ziel Startet
        x_val: Anzahl der Felder im Ziel die begangen werden sollen
        goal: Array der Goal Positionen für den aktuellen Spieler
        board: Aktuelles Spielfeld
        current_player: Aktueller Spieler
        """
    goal_area = jnp.arange(len(goal))
    return jnp.all(
            jnp.where(
                (start < goal_area) & (goal_area < x_val),
                board[goal] != current_player,
                True  # Positionen außerhalb von x_val ignorieren
            )
        )

@jax.jit
def env_step(env: classic_MADN, pin: Action) -> classic_MADN:
    pin = pin.astype(jnp.int8)
    move = env.die.astype(jnp.int8)
    # currently player ID
    player_id = env.current_player
    # ID of the players' pins to be moved (important for teams)
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    # check if the action is valid
    invalid_action = ~valid_action(env)[pin]

    current_positions = env.pins[current_player, pin]
    moved_positions = current_positions + move
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - env.target[current_player]

    a = jax.lax.cond(
        jnp.isin(current_positions, env.goal[current_player]),
        lambda: check_goal_path_for_pin(current_positions - env.goal[current_player,0], moved_positions - env.goal[current_player,0] +1, env.goal[current_player], env.board, current_player),
        lambda: check_goal_path_for_pin(- jnp.ones(4, dtype=jnp.int8), x, env.goal[current_player], env.board, current_player)
    )
    A = (env.board[env.goal[current_player, x-1]] != current_player) & (env.rules['enable_jump_in_goal_area'] | a)
    new_position = jnp.where(
        current_positions == -1,
        env.start[current_player], # move from start area to starting position
        jnp.where(jnp.isin(current_positions, env.goal[current_player]),
                    moved_positions,
                    jnp.where(
                        # ~jnp.isin(current_player, env.board[env.goal[current_player, x]]), # check if current player has pins in goal area
                        (4 >= x) & (x > 0) & A & (current_positions <= env.target[current_player]),
                        env.goal[current_player, x-1], # move to goal position
                        fitted_positions
                    )
        )
    )
    
    # update pins
    # pin at new position
    pin_at_pos = env.board[new_position]
    # if a player is at the new position and it's not the current player, send that pin back to start area
    # pins = env.pins.at[current_player, pin].set(jnp.where(invalid_action, env.pins[current_player, pin], new_position))
    pins = jax.lax.cond(
        (pin_at_pos != -1) & ((pin_at_pos != current_player) | env.rules['enable_friendly_fire']) & ~invalid_action, # if a player was at the new position and it's not the current player and the action is valid
        lambda p: p.at[pin_at_pos].set(jnp.where(p[pin_at_pos] == new_position, -1, p[pin_at_pos])), # send the pin of that player back to start area
        lambda p: p,
        env.pins
    )
    #set the moved pin to the new position
    pins = pins.at[current_player, pin].set(jnp.where(invalid_action, env.pins[current_player, pin], new_position))

    board = jax.lax.cond(
        ~invalid_action,
        lambda b: set_pins_on_board(-jnp.ones_like(b, dtype=jnp.int8), pins),
        lambda b: b,
        env.board
    )

    winner = get_winner(env, board)
    reward = jnp.array(jnp.where(env.done, 0, jnp.where(invalid_action, -1, winner[current_player])), dtype=jnp.int8) # reward is 0 if game is done, -1 if action is invalid, else the index of the winning player (1-4) or 0 if no winner yet
    # check if the game is done
    done = env.done | jnp.any(winner)
    # player changes on invalid action
    current_player = jnp.where(done | (env.rules['enable_bonus_turn_on_6'] & (move == 6)), player_id, (player_id + 1) % env.num_players) # if the game is not done or the player played a 6, switch to the next player

    env = classic_MADN(
        board=board,
        num_players=env.num_players,#/
        pins=pins,#
        current_player=current_player,
        done= done,#
        reward=reward,#
        start=env.start,#
        target=env.target,#
        goal=env.goal,#
        die=env.die,#
        board_size=env.board_size,#
        total_board_size=env.total_board_size,#
        rules=env.rules,#/
    )
    return env, reward, done

@jax.jit
def set_pins_on_board(board, pins):
    num_players, num_pins = pins.shape

    def body(idx, board):
        player = idx // num_pins
        pin = idx % num_pins
        pos = pins[player, pin]
        board = jax.lax.cond(
            pos != -1,
            lambda b: b.at[pos].set(player),
            lambda b: b,
            board
        )
        return board

    board = jax.lax.fori_loop(0, num_players * num_pins, body, jnp.ones_like(board, dtype=jnp.int8) * -1)
    return board

def no_step(env:classic_MADN) -> classic_MADN:
    """
    No-op step function for the environment.
    """                  
    env = classic_MADN(
        board=env.board,
        num_players=env.num_players,
        pins=env.pins,
        current_player=(env.current_player + 1) % env.num_players,
        done=env.done,
        reward=env.reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        die=env.die,
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )
    return env, jnp.array(0, dtype=jnp.int8), env.done

@jax.jit
def valid_action(env:classic_MADN) -> chex.Array:
    '''
    Returns a mask of shape (4, ) indicating which actions are valid for each pin of the current player
    '''
    #return valid_action for each pin of the current player
    current_player = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, current_player), (current_player + 2)%4, current_player)
    current_pins = env.pins[current_player]
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]
    start = env.start
    die = env.die
    num_players_static = start.shape[0]          # statisch für JIT
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)
    pins_on_start = (board[start] == player_ids)#check which players have pins on start positions and block with them

    # calculate possible actions
    current_positions = current_pins
    moved_positions = current_pins + die
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - target


    # filter out invalid moves blocked by own pins
    result = (board[fitted_positions] != current_player) | env.rules['enable_friendly_fire'] # check move to any board position

    # filter actions where a pin on a start spot would block others
    distance = env.board_size // 4
    nearest_start_before = ((current_pins//distance)+1)%num_players_static # nearest start before is the next start field in front of a pin
    nearest_start_after = fitted_positions//distance
    cond = start[nearest_start_before] == start[nearest_start_after] # if cond: pin traverses a start position
    result = jnp.where(
        env.rules['enable_start_blocking'] & cond,
        (~pins_on_start[nearest_start_after] | (current_pins == start[current_player])) & result, # true if start not blocked and new pos is free
        result
    )

    result = jax.lax.cond(
        env.rules['enable_circular_board'],
        lambda: result,
        lambda: jnp.where(
            (current_positions <= target) & (moved_positions > (target + len(current_pins))), # if moving beyond target is not allowed
            False,
            result
        )
    )

    # check if goal position is free or on circular board, the other board position is possible 
    check_all_pins = jax.vmap(check_goal_path_for_pin, in_axes=(0, 0, None, None, None))
    A = (env.rules["enable_circular_board"] & result) # if rule enabled, consider circle rotation, else only goal area
    B = (board[goal[x-1]] != current_player)
    # Entweder man darf im Ziel überspringen oder auf dem Weg (im Zielbereich) ist kein eigener Pin 
    # wenn C true ist ist B auch true da B eine Teil-Bedingung davon ist
    C = (env.rules['enable_jump_in_goal_area'] | check_all_pins(- jnp.ones(4, dtype=jnp.int8), x, goal, board, current_player))
    result = jnp.where(
        (4 >= x) & (x > 0) & (current_positions <= target),
        A | (B & C),
        result
    )
    # filter actions for pins in goal area
    # Entweder man darf im Ziel überspringen oder auf dem Weg (im Zielbereich) ist kein eigener Pin 
    
    D = (env.rules['enable_jump_in_goal_area'] | check_all_pins(current_pins - goal[0], moved_positions - goal[0] +1, goal, board, current_player))
    result = jnp.where(
        jnp.isin(current_pins, goal),
        (moved_positions <= goal[-1]) & (board[moved_positions] != current_player) & D,
        result
    )

    # filter actions for pins in start area
    start_moves = jax.lax.cond(
        env.rules['enable_start_on_1'],
        lambda: jnp.array([1, 6]),
        lambda: jnp.array([-1, 6])
    )
    result = jnp.where(
        (current_pins == -1),
        jnp.isin(die, start_moves) & (~pins_on_start[current_player]),
        result
    )

    return result # filter possible actions with available actions



def encode_board(env: classic_MADN) -> chex.Array:
    num_players = env.num_players
    board = env.board
    distance = env.board_size // 4
    current_player = env.current_player
    current_pins = env.pins[current_player]
    
    #rolled idx
    rolled_idx = jnp.arange(env.current_player, env.current_player + env.num_players) % env.num_players
    # Pin Channel (current_player only), position jedes einzelnen pins (4xboard_size) leer wenn -1
    pin_channel = jax.nn.one_hot(
        jnp.clip(current_pins, 0, board.shape[0]-1),  # Werte außerhalb vermeiden
        board.shape[0],
        dtype=jnp.int8
    )
    pin_channel = jnp.where(current_pins[:, None] == -1, 0, pin_channel)
    # Spielerpositionen (One-hot)
    # with roll over for current player
    new_board = jnp.roll(env.board[0:env.board_size], shift=-distance*current_player, axis=0)
    new_pins = jnp.roll(env.board[env.board_size:env.total_board_size], shift=-4*current_player, axis=0)
    board = jnp.concatenate([new_board, new_pins], axis=0)
    player_channels = (board == rolled_idx[:, None]).astype(jnp.int8)[1:]  # (4, board_size)

    # Spielerposition im Haus
    home_positions = jnp.ones((num_players, board.shape[0]), dtype=jnp.int8) * jnp.count_nonzero(env.pins == -1, axis=1)[:, None]  # (4, board_size)
    home_positions = home_positions[rolled_idx]  # (4, board_size)
    # Aktueller Spieler (optional)
    # current_player_channel = jnp.ones((1, board.shape[0]), dtype=jnp.int8) * current_player  # (1, board_size)

    # Würfel Kanal
    die_channel = jnp.ones((1, board.shape[0]), dtype=jnp.int8) * env.die  # (1, board_size)

    # Alles zusammenfügen
    board_encoding = jnp.concatenate([pin_channel, player_channels, home_positions, die_channel], axis=0)  # (features, board_size)
    return board_encoding

def encode_board_linear(env: classic_MADN) -> chex.Array:
    num_players = env.num_players
    board = env.board

    # Spielerpositionen (One-hot)
    player_channels = (board == jnp.arange(num_players)[:, None]).astype(jnp.int8)
    player_channels = player_channels.reshape(-1)  # (features * board_size,)

    # Spielerposition im Haus
    home_positions = jnp.count_nonzero(env.pins == -1, axis=1)
    home_positions = home_positions.reshape(-1)  # (num_players,)

    # Aktueller Spieler (optional)
    current_player_channel = jnp.zeros(num_players, dtype=jnp.int8).at[env.current_player].set(1) 

    # Würfel Kanal
    die_channel = jnp.zeros(6, dtype=jnp.int8)
    die_channel = die_channel.at[env.die - 1].set(1)

    # Alles zusammenfügen
    board_encoding = jnp.concatenate([player_channels, home_positions, current_player_channel, die_channel], axis=0)  # (features, board_size)
    return board_encoding

def map_action(env:classic_MADN, board_position: chex.Array) -> Action:
    '''
    Returns the pin index (0-3) corresponding to the board position
    '''
    pins = env.pins[env.current_player]
    pin_index = jnp.argwhere(pins == board_position)
    return pin_index[0][0]

def winning_action(env:classic_MADN) -> chex.Array:
    env_copy = classic_MADN(
        board=env.board,
        num_players=env.num_players,
        current_player=env.current_player,
        pins=env.pins,
        done=env.done,
        reward=env.reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        die=env.die,
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )

    env_copy, reward, done = jax.vmap(env_step, (None, 0))(env_copy, jnp.array([jnp.array(pin, dtype=jnp.int8) for pin in range(4)], dtype=jnp.int8))

    return reward == 1

def policy_function(env:classic_MADN) -> chex.Array:
    return sum(
        (valid_action(env).flatten().astype(jnp.float32) * 100,
        winning_action(env).astype(jnp.float32) * 200)
    )    

@jax.jit
def rollout(env:classic_MADN, rng_key:chex.PRNGKey) -> tuple[classic_MADN, chex.PRNGKey]:

    def cond(a):
        env, key, steps = a
        return (~env.done) & (steps < 300)
    def step(a):
        env, key, steps = a
        key, subkey = jax.random.split(key)
        env = throw_die(env, subkey)
        def step_env(e):
            action = jax.random.categorical(subkey, policy_function(e)).astype(jnp.int8)
            return env_step(e, action)
        env, reward, done = jax.lax.cond(
            jnp.all(valid_action(env) == False),
            lambda e: no_step(e),
            lambda e: step_env(e),
            env)

        return env, key, steps + 1

    leaf, key, steps = jax.lax.while_loop(cond, step, (env, rng_key, 0))
    winner = get_winner(leaf, leaf.board)
    root_player = env.current_player
    # +1 für Sieg, -1 für Niederlage, 0 sonst
    return jnp.where(winner == -1, 0.0, jnp.where(winner[root_player], 1.0, -1.0))

def value_function(env:classic_MADN, rng_key:chex.PRNGKey) -> chex.Array:
    return rollout(env, rng_key).astype(jnp.float32)

def recurrent_chance_fn(params, rng_key, chance_outcome, afterstate: classic_MADN):
    """
    Chance node: Würfelwurf simulieren, für jeden Würfelwert einen neuen Zustand erzeugen.
    """
    dice_value = chance_outcome + 1  # Konvertiere 0-5 zu 1-6
    
    # Setze den Würfelwert im Afterstate
    env = set_die(afterstate, dice_value)
    
    # Berechne die resultierenden Werte für diesen spezifischen Würfelwert
    action_logits = valid_action(env).astype(jnp.float32)
    value = value_function(env, rng_key)
    reward = env.reward.astype(jnp.float32)
    discount = jnp.where(env.done, 0.0, 1.0)
    
    return mctx.ChanceRecurrentFnOutput(
        action_logits=action_logits,  # [A] - Aktionswahrscheinlichkeiten
        value=value,                  # scalar - Zustandswert
        reward=reward,                # scalar - Reward
        discount=discount             # scalar - Discount
    ), env

# Angepasste recurrent_fn ohne Würfelwurf
def recurrent_fn(params, rng_key, action: Action, embedding: classic_MADN):
    """
    Recurrent function - führt nur die Spieleraktion aus, kein Würfelwurf
    """
    env = embedding
    
    # Prüfe ob gültige Aktionen verfügbar sind
    valid_actions = valid_action(env)
    
    # bestimme Afterstate (state nach einer aktion aber vor dem nächsten würfelwurf)
    afterstate, reward, done = jax.lax.cond(
        jnp.all(~valid_actions),
        lambda _: no_step(env),
        lambda _: env_step(env, action),
        operand=None
    )
    
    chance_logits = jnp.ones(6) * jnp.log(1.0 / 6.0)  # Uniform

    afterstate_value = value_function(afterstate, rng_key)
    return mctx.DecisionRecurrentFnOutput(
        chance_logits=chance_logits,
        afterstate_value=afterstate_value
    ), afterstate

# Root function anpassen
def root_fn(env: classic_MADN, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    """
    Root function für MCTS - Startzustand ohne Würfelwurf
    """

    # if env.die <= 0:
    #     # Keine Aktionen möglich ohne Würfelwert
    #     prior_logits = jnp.full(4, -jnp.inf, dtype=jnp.float32)
    # else:
    #     prior_logits = policy_function(env)
    
    return mctx.RootFnOutput(
        prior_logits=policy_function(env),
        value=value_function(env, rng_key),
        embedding=env,
    )

def all_pin_distributions(total=7, num_pins=4):
    # Erzeuge alle möglichen Werte für die ersten drei Pins
    a = jnp.arange(total + 1)
    b = jnp.arange(total + 1)
    c = jnp.arange(total + 1)
    # Erzeuge alle Kombinationen (a, b, c)
    grid = jnp.array(jnp.meshgrid(a, b, c, indexing='ij')).reshape(3, -1).T
    # Berechne den vierten Wert
    d = total - grid[:, 0] - grid[:, 1] - grid[:, 2]
    # Filtere gültige Kombinationen (d >= 0)
    mask = d >= 0
    result = jnp.concatenate([grid[mask], d[mask][:, None]], axis=1)
    return result
