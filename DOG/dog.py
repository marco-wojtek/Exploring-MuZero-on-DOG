import chex
import jax
import jax.numpy as jnp
import mctx

def all_pin_distributions(total=7):
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

DISTS_7_4 = all_pin_distributions(7)  # (120,4), lex nach a0,a1,a2

def index_to_dist(idx: int) -> jnp.ndarray:
    return DISTS_7_4[idx]  # (4,)

def dist_to_index(dist: jnp.ndarray):
    # JAX-kompatibel: lineare Suche über konstante Tabelle
    mask = jnp.all(DISTS_7_4 == dist[None, :], axis=1)
    return jnp.int32(jnp.argmax(mask)) 

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
Deck = chex.Array
Card = chex.Array
Hand = chex.Array

@chex.dataclass
class DOG:
    board: Board
    num_players: Num_players
    current_player: Player 
    pins : Pins 
    reward: Reward 
    done: Done  
    start: Start  
    target: Target  
    goal: Goal  
    deck: Deck  
    hands: Hand
    board_size: Size 
    total_board_size: Size  
    rules : dict  

def env_reset(
        _,
        num_players=jnp.int8(4),
        distance=jnp.int8(10),
        enable_initial_free_pin = False,
        enable_circular_board = True,
        enable_friendly_fire = False,
        enable_dice_rethrow = False,
        disable_swapping = False,
        disable_hot_seven = False,
        disable_joker = False,
            ) -> DOG:
    
    board_size = num_players * distance
    total_board_size = board_size + num_players * 4 # add goal areas
    num_pins = 4

    start = jnp.array(jnp.arange(num_players)*distance, dtype=jnp.int8)
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

    # prepare deck
    num_cards = 14 - disable_joker.astype(jnp.int8) - disable_hot_seven.astype(jnp.int8) - disable_swapping.astype(jnp.int8)
    deck = jnp.ones(num_cards, dtype=jnp.int8)*8
    deck = deck.at[0].set(6 + (2*disable_joker.astype(jnp.int8)))  # Idx 0 ist nur joker wenn dieser enabled ist

    return DOG(
        board = board, # board is filled with -1 (empty) or 0-3 (player index)
        num_players = jnp.array(num_players, dtype=jnp.int8), # number of players
        pins = pins,
        current_player=jnp.array(0, dtype=jnp.int8), # index of current player, 0-3
        done = jnp.bool_(False), # whether the game is over
        reward=jnp.array(0, dtype=jnp.int8), # reward for the current player
        start = start,
        target = jnp.array((jnp.arange(num_players)*distance - 1)%board_size, dtype=jnp.int8),
        goal = jnp.reshape(jnp.arange(board_size, board_size + num_players*4, dtype=jnp.int8), (num_players, 4)),
        deck = deck,
        hands = jnp.zeros((num_players, num_cards), dtype=jnp.int8),
        board_size=jnp.array(board_size, dtype=jnp.int8),
        total_board_size=jnp.array(total_board_size, dtype=jnp.int8),
        rules = {
        'enable_initial_free_pin':enable_initial_free_pin,
        'enable_circular_board':enable_circular_board,
        'enable_friendly_fire':enable_friendly_fire,
        'enable_dice_rethrow':enable_dice_rethrow,
        'disable_swapping': disable_swapping,
        'disable_hot_seven': disable_hot_seven,
        'disable_joker': disable_joker,
        }
    )

def distribute_cards(env: DOG, quantity: int):
    pass

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

# returns a boolean array indicating valid swap positions
def valid_swap(env):
    current_player = env.current_player
    Num_players = env.num_players
    current_pins = env.pins[current_player]
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]
    start = env.start
    num_players_static = start.shape[0]          # statisch für JIT
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)

    swap_mat = jnp.tile(board[:-Num_players*4], (4,1))
    
    condA = jnp.where(~jnp.isin(swap_mat, jnp.array([-1, current_player])), True, False)
    condA = condA.at[:,start].set(board[start] == player_ids)

    condB = (~jnp.isin(current_pins, jnp.array([-1, start[current_player]])))[:, None] 
    return  condA & condB

def val_action_7(env:DOG, seven_dist) -> chex.Array:
    '''
    Returns a mask of shape (4, ) indicating which actions are valid for each pin of the current player
    '''
    #return valid_action for each pin of the current player
    current_player = env.current_player
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]

    # calculate possible actions
    current_positions = env.pins[current_player]
    moved_positions = current_positions + seven_dist
    fitted_position = moved_positions % env.board_size
    x = moved_positions - target


    # Überlaufen der Zielposition verhindern falls kein Rundbrett
    result = jax.lax.cond(
        env.rules['enable_circular_board'],
        lambda: jnp.ones_like(current_positions, dtype=bool),
        lambda: ~((current_positions <= target) & (moved_positions > (target + 4)))
    )
    result = jnp.where(
        (4 >= x) & (x > 0) & (current_positions <= env.target[current_player]),
        (env.rules["enable_circular_board"] | result),#(env.rules["enable_circular_board"] & result) | (board[goal[x-1]] != current_player), # if goal is possible, check if goal position is free
        result
    )
    # filter actions for pins in goal area
    result = jnp.where(
        jnp.isin(current_positions, goal),
        (moved_positions <= goal[-1]),# & (board[moved_positions%env.total_board_size] != current_player),
        result
    )

    # alle Aktionen müssenrechenrisch möglich sein und es dürfen keine zwei Pins auf die gleiche Position ziehen
    board_mover = jnp.where(current_positions == -1, moved_positions==-1, True)# prüfe dass kein pin im startbereich bewegt werden würde 

    return jnp.all(result & board_mover) 

@jax.jit
def valid_actions(env: DOG) -> chex.Array:
    """
    Returns a boolean array indicating valid actions for the current player.
    An action is valid if the corresponding card is present in the player's hand.
    """
    current_player = env.current_player
    current_pins = env.pins[current_player]
    hand = env.hands[current_player]

    # valid_actions based on cards in hand
    valid_action = jnp.where(hand > 0, True, False)

    # filter actions based on effect (handle special cards seperatly if necessary)
    
    pass

def map_action_to_card(action: Action, env: DOG) -> Card:
    """
    Maps an action index to a card value based on the current player's hand.
    """
    hand = env.hands[env.current_player]
    card_indices = jnp.where(hand > 0, jnp.arange(len(hand)), -1)
    valid_card_indices = card_indices[card_indices != -1]
    card = valid_card_indices[action]
    return card

def no_step(env:DOG) ->  DOG:
    """
    No-op step function for the environment.
    On no step the hand of the current player is emptied and the turn passes to the next player.
    """               
    hand_cards = jnp.sum(env.hands, axis=1) 
    def body(i, pnext):
        cand = (env.current_player + i + 1) % env.num_players
        take = (pnext == -1) & (hand_cards[cand] > 0)
        return jnp.where(take, cand, pnext)
    next_player = jax.lax.fori_loop(0, env.num_players, body, -jnp.array(1, dtype=jnp.int8))  
    print("Next player found:", next_player)
    env = DOG(
        board=env.board,
        num_players=env.num_players,
        pins=env.pins,
        current_player=next_player,
        done=env.done,
        reward=env.reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        deck=env.deck,
        hands=env.hands.at[env.current_player].set(jnp.zeros(len(env.deck), dtype=jnp.int8)),
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )
    return env, jnp.array(0, dtype=jnp.int8), env.done

@jax.jit
def env_step(env: DOG, action: Action) -> tuple[DOG, Reward, Done]:
    """
    Placeholder step function for the environment.
    Currently implements a no-op step that just passes the turn to the next player.
    """               
    # WICHTIG: Wenn dog gespielt wird, muss man über das startfeld ins Ziel gehen, ohne darauf loszugehen. 
    # target ist das feld vor dem startfeld.
    # das heißt x muss noch einen wert abgezogen bekommen, damit r.g. eine 5 auf target, ins Ziel führt
    # ergo: Man kann nicht mi einer 1 vom startfeld ins Ziel gehen!!
    env, reward, done = no_step(env)
    
    return env, reward, done