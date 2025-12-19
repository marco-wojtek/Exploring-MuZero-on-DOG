import chex
import jax
import jax.numpy as jnp
import mctx
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.utility_funcs import *

DISTS_7_4 = all_pin_distributions(7)  # (120,4), lex nach a0,a1,a2

def index_to_dist(idx: int) -> jnp.ndarray:
    '''
    Wandelt einen Index in die entsprechende Pin-Verteilung um.
    Args:
        idx: Index der Verteilung
    Returns:
        Ein Array der Form (4,) mit der Pin-Verteilung.
    '''
    return DISTS_7_4[idx]  # (4,)

def dist_to_index(dist: jnp.ndarray):
    '''
    Wandelt eine Pin-Verteilung in den entsprechenden Index um.
    Args:
        dist: Ein Array der Form (4,) mit der Pin-Verteilung.
    Returns:
        Der Index der Verteilung.
    '''
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
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=jnp.int32(10),
        starting_player = jnp.int8(0), # -1, 0, 1, 2, 3
        key = jax.random.PRNGKey(0),
        enable_teams = False,
        enable_initial_free_pin = False,
        enable_circular_board = True,
        enable_start_blocking = False,
        enable_jump_in_goal_area = True,
        enable_friendly_fire = False,
        disable_swapping = False,
        disable_hot_seven = False,
        disable_joker = False,
            ) -> DOG:
    
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
    
    start = jnp.array(jnp.arange(4)*distance, dtype=jnp.int32)[layout]
    target = (start - 1)%board_size
    goal = jnp.reshape(jnp.arange(board_size, board_size + 4*4, dtype=jnp.int32), (4, 4))[layout, :]


    pins = - jnp.ones((num_players,num_pins), dtype=jnp.int32)
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
    num_cards = 14 - jnp.int8(disable_joker) - jnp.int8(disable_hot_seven) - jnp.int8(disable_swapping)
    deck = jnp.ones(num_cards, dtype=jnp.int8)*8
    deck = deck.at[0].set(6 + (2*jnp.int8(disable_joker)))  # Idx 0 ist nur joker wenn dieser enabled ist

    return DOG(
        board = board, # board is filled with -1 (empty) or 0-3 (player index)
        num_players = jnp.array(num_players, dtype=jnp.int8), # number of players
        pins = pins,
        current_player=jnp.array(starting_player, dtype=jnp.int8), # index of current player, 0-3
        done = jnp.bool_(False), # whether the game is over
        reward=jnp.array(0, dtype=jnp.int8), # reward for the current player
        start = start,
        target = target,
        goal = goal,
        deck = deck,
        hands = jnp.zeros((num_players, num_cards), dtype=jnp.int8),##
        board_size=jnp.array(board_size, dtype=jnp.int16),
        total_board_size=jnp.array(total_board_size, dtype=jnp.int16),
        rules = {
        'enable_teams':enable_teams,
        'enable_initial_free_pin':enable_initial_free_pin,
        'enable_circular_board':enable_circular_board,
        'enable_start_blocking':enable_start_blocking,
        'enable_jump_in_goal_area':enable_jump_in_goal_area,
        'enable_friendly_fire':enable_friendly_fire,
        'disable_swapping': disable_swapping,
        'disable_hot_seven': disable_hot_seven,
        'disable_joker': disable_joker,
        }
    )

def reset_deck(env: DOG) -> Deck:
    num_cards = 14 - jnp.int8(env.rules['disable_joker']) - jnp.int8(env.rules['disable_hot_seven']) - jnp.int8(env.rules['disable_swapping'])
    deck = jnp.ones(num_cards, dtype=jnp.int8)*8
    deck = deck.at[0].set(6 + (2*jnp.int8(env.rules['disable_joker'])))
    return deck

def distribute_cards(env: DOG, quantity: int, key: jax.random.PRNGKey) -> DOG:
    '''
    Distributes `quantity` cards to each player's hand from the deck.
    Args:
        env: DOG environment
        quantity: Number of cards to distribute to each player
        key: JAX random key
    Returns:
        Updated DOG environment with cards distributed.
    '''
    num_players = env.num_players
    num_card_types = len(env.deck)
    
    # Erstelle einen "flachen" Pool aller Karten im Deck
    # deck[i] gibt an, wie oft Karte i vorhanden ist
    # Wir erstellen einen Index-Array wo jede Karte entsprechend ihrer Häufigkeit vorkommt
    card_indices = jnp.arange(num_card_types)
    
    # Wiederhole jeden Kartenindex entsprechend seiner Häufigkeit
    # Maximale Deckgröße = sum(deck) 
    max_deck_size = jnp.sum(env.deck)

    if max_deck_size < quantity * num_players:
        new_deck = reset_deck(env)
    else:
        new_deck = env.deck
    
    # Erstelle expandierten Pool: [0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, ...]
    expanded_pool = jnp.repeat(card_indices, new_deck, total_repeat_length=max_deck_size)
    
    # Mische den Pool
    key, subkey = jax.random.split(key)
    shuffled_indices = jax.random.permutation(subkey, max_deck_size)
    shuffled_pool = expanded_pool[shuffled_indices]
    
    # Verteile Karten an jeden Spieler
    total_cards_to_distribute = quantity * num_players
    cards_to_distribute = shuffled_pool[:total_cards_to_distribute]
    
    # Reshape zu (num_players, quantity)
    cards_per_player = cards_to_distribute.reshape(num_players, quantity)
    
    # Zähle Karten pro Spieler und Kartentyp
    def count_cards_for_player(player_cards):
        # One-hot encoding und summieren
        one_hot = jax.nn.one_hot(player_cards, num_card_types, dtype=jnp.int8)
        return jnp.sum(one_hot, axis=0)
    
    new_hand_additions = jax.vmap(count_cards_for_player)(cards_per_player)
    
    # Aktualisiere Hände und Deck
    new_hands = env.hands + new_hand_additions
    
    # Berechne wie viele Karten insgesamt pro Typ verteilt wurden
    total_distributed = jnp.sum(new_hand_additions, axis=0)
    new_deck = new_deck - total_distributed
    
    return DOG(
        board=env.board,
        num_players=env.num_players,
        pins=env.pins,
        current_player=env.current_player,
        done=env.done,
        reward=env.reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        deck=new_deck,
        hands=new_hands,
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )

def is_player_done(num_players, board:Board, goal:Goal, player: Player) -> chex.Array:
    '''
    Prüft ob ein Spieler fertig ist (alle Pins im Goal).
    Args:
        num_players: Anzahl der Spieler im Spiel
        board: Aktuelles Spielfeld
        goal: Array der Goal Positionen für alle Spieler
        player: Spielerindex der geprüft werden soll
    Returns:
        Boolean, ob der Spieler fertig ist
    '''
    return jax.lax.cond(
        player >= num_players,
        lambda: False,
        lambda: jnp.all(board[goal[player]] >= 0)
    )

def get_winner(env: DOG, board: Board) -> chex.Array:
    '''
    Bestimmt die Gewinner des Spiels basierend auf dem aktuellen Board-Zustand.
    Args:
        env: DOG environment
        board: Aktuelles Spielfeld
    Returns:
        Ein boolean-Array der Form (4,), das für jeden Spieler angibt, ob er gewonnen hat.
    '''
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

@jax.jit
def set_pins_on_board(board, pins):
    '''
    Setzt die Pins auf dem Spielfeld.
    Args:
        board: Aktuelles Spielfeld
        pins: Array der Pin-Positionen für alle Spieler
    Returns:
        Aktualisiertes Spielfeld mit gesetzten Pins
    '''
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

# @jax.jit
def val_swap(env):
    '''
    Gibt eine Maske zurück, die gültige Swap-Positionen für den aktuellen Spieler angibt.
    Args:
        env: DOG environment
    Returns:
        Ein boolean-Array der Form (total_board_size, ), das für jede Position angibt, ob sie für einen Swap gültig ist.
    '''
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    current_pins = env.pins[current_player]
    board = env.board
    goal = env.goal
    start = env.start
    num_players_static = start.shape[0]
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)

    swap_mat = jnp.tile(board, (4,1))
    cond_a = jnp.where(~jnp.isin(swap_mat, jnp.array([-1, current_player])), True, False)
    cond_b = cond_a.at[:,start].set(~((board[start] == player_ids) & env.rules['enable_start_blocking']))  # start positions cannot be swapped if blocked except the rule is disabled
    cond_c = cond_b.at[:, goal].set(False)  # goal positions cannot be swapped
    condA = cond_a & cond_b & cond_c

    disallowed_pos = jax.lax.cond(
        env.rules['enable_start_blocking'],
        lambda: jnp.concatenate([jnp.array([-1]), jnp.array([start[current_player]]), goal[current_player]]),
        lambda: jnp.concatenate([jnp.array([-1]), jnp.array([-1]), goal[current_player]])
    )
    condB = (~jnp.isin(current_pins, disallowed_pos))[:, None]
    return condA & condB

# @jax.jit
def val_action_7(env:DOG, seven_dist) -> chex.Array:
    '''
    Gibt eine Maske zurück, die gültige Aktionen für die 7 Aktion des aktuellen Spielers angibt.

    Args:
        env: DOG environment
        seven_dist: Die Distanz, die mit der 7 Aktion bewegt werden soll (1-7)
    Returns:
        Ein boolean-Array der Form (4,), das für jeden Pin angibt, ob die 7 Aktion gültig ist.

    Die 7 Aktion ist ein Sonderfall. Normalerweise können Figuren im Ziel nicht geschlagen werden. Wenn aber die Regel, dass im 
    Ziel übersrungen werden darf, aktiviert ist, kann eine Figur im Ziel übersprungen werden und somit auch durch die 7 geschlagen werden.
    Will ein Spieler nicht das eine Figur auf Feld 40 von einer auf Feld 38 mit einer 4 geschlagen wird, so muss die auf 40 1-2 bewegt werden und die andere entsprechend weniger.
    Die 7 Aktion wird nur beschränkt wenn start-Block oder Jump im Ziel Verbot gilt.
    '''
    #return valid_action for each pin of the current player
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]
    start = env.start
    num_players_static = start.shape[0]          # statisch für JIT
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)
    pins_on_start = (board[start] == player_ids)#check which players have pins on start positions and block with them
    

    # calculate possible actions
    current_positions = env.pins[current_player]
    moved_positions = current_positions + seven_dist
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - target -1

    pins_on_start = pins_on_start.at[current_player].set(jnp.any(jnp.where(current_positions == start[current_player], moved_positions == start[current_player], False))) # if any pin does not move, check if it is on start
    # Überlaufen der Zielposition verhindern falls kein Rundbrett
    result = jax.lax.cond(
        env.rules['enable_circular_board'],
        lambda: jnp.ones_like(current_positions, dtype=bool),
        lambda: ~((current_positions <= target) & ((moved_positions > (target + 4)) | (x == 0))) # if moved_pos > target + 4 or x = 0 means overrun and new round start
    )
    distance = env.board_size // 4
    nearest_start_before = ((current_positions  //distance)+1)%num_players_static # nearest start before is the next start field in front of a pin
    nearest_start_after = fitted_positions//distance
    traverses_start_position = start[nearest_start_before] == start[nearest_start_after] # if cond: pin traverses a start position
    result = jnp.where(
        env.rules['enable_start_blocking'] & traverses_start_position,
        ~pins_on_start[nearest_start_after] & result, # true if start not blocked and new pos is free
        result
    )
    # Wenn start blocked ist, dürfen keine Pins ins Ziel ziehen, d.h. die Ziel traversalmenge x wird auf 0 gesetzt
    x = jnp.where(
        env.rules['enable_start_blocking'] & traverses_start_position & pins_on_start[nearest_start_after],
        0, 
        x
    )
    # print(result)
    check_all_pins = jax.vmap(check_goal_path_for_pin, in_axes=(0, 0, None, None, None))
    A = (env.rules["enable_circular_board"] & result) # if rule enabled, consider circle rotation, else only goal area
    # Entweder man darf im Ziel überspringen oder auf dem Weg (im Zielbereich) ist kein eigener Pin 
    # wenn C true ist ist B auch true da B eine Teil-Bedingung davon ist
    # Für alle pins die sich im Ziel bewegen sollten die neuen positionen geprüft werden, da die alten nicht blockieren könnten
    tmp_pins = env.pins.at[current_player].set(jnp.where(jnp.isin(current_positions, goal), moved_positions, current_positions))
    tmp_board = set_pins_on_board(board, tmp_pins)
    # print(tmp_board)
    # print(result)
    # B = (tmp_board[goal[x-1]] != current_player)
    C = (env.rules['enable_jump_in_goal_area'] | check_all_pins(- jnp.ones(4, dtype=jnp.int8), x, goal, tmp_board, current_player))
    # print("A:", A)
    # print("B:", B)
    # print("C:", C)
    # print(x)
    result = jnp.where(
        (4 >= x) & (x > 0) & (current_positions <= target),
        A | C,
        result
    )
    # print(result)
    # filter actions for pins in goal area
    D = (env.rules['enable_jump_in_goal_area'] | check_relative_order_preserved(current_positions, moved_positions, env.board_size))
    result = jnp.where(
        jnp.isin(current_positions, goal),
        (moved_positions <= goal[-1])  & D,
        result
    )
    # print("D:", D)
    # print(result)
    # alle Aktionen müssenrechenrisch möglich sein und es dürfen keine zwei Pins auf die gleiche Position ziehen
    board_mover = jnp.where(current_positions == -1, moved_positions==-1, True)# prüfe dass kein pin im startbereich bewegt werden würde 
    return jnp.all(result & board_mover) 

def val_action_normal_move(env:DOG, move: int):
    '''
    Gibt eine Maske zurück, die gültige Aktionen für eine normale Bewegung des aktuellen Spielers angibt.
    Args:
        env: DOG environment
        move: Die Distanz, die bewegt werden soll
    Returns:
        Ein boolean-Array der Form (4,), das für jeden Pin angibt, ob die Aktion gültig ist.
    '''
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    current_pins = env.pins[current_player]
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]
    start = env.start
    num_players_static = start.shape[0]          # statisch für JIT
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)
    pins_on_start = (board[start] == player_ids)

    # calculate possible actions
    current_positions = current_pins
    moved_positions = current_pins + move
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - target -1 #(start feld muss auch überlaufen werden)

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
    # every pin that would travers start position when its blocked cannot enter the goal
    x = jnp.where(
        env.rules['enable_start_blocking'] & cond,
        0, # true if start not blocked and new pos is free
        x
    )

    result = jax.lax.cond(
        env.rules['enable_circular_board'],
        lambda: result,
        lambda: jnp.where(
            (current_positions <= target) & ((x > 4) | (x == 0)), # if moving beyond target is not allowed
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
    result = jnp.where(
        (current_pins == -1),
        jnp.isin(move, jnp.array([11, 13])) & (~pins_on_start[current_player]),
        result
    )

    return result & (move > 0)

def val_neg_move(env:DOG, move:int):
    '''
    Gibt eine Maske zurück, die gültige Aktionen für eine negative Bewegung des aktuellen Spielers angibt.
    Args:
        env: DOG environment
        move: Die negative Distanz, die bewegt werden soll
    Returns:
        Ein boolean-Array der Form (4,), das für jeden Pin angibt, ob die Aktion gültig ist.
    '''
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    current_pins = env.pins[current_player]
    board = env.board
    target = env.target[current_player]
    goal = env.goal[current_player]
    start = env.start
    num_players_static = start.shape[0]          # statisch für JIT
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)
    pins_on_start = (board[start] == player_ids)

    # calculate possible actions
    current_positions = current_pins
    moved_positions = current_pins + move
    fitted_positions = moved_positions % env.board_size

    result = (board[fitted_positions] != current_player) | env.rules['enable_friendly_fire'] # check move to any board position

    # filter actions where a pin on a start spot would block others
    distance = env.board_size // 4
    nearest_start_before = (current_pins//distance) # nearest start before is the next start field behind of a pin
    nearest_start_after = ((fitted_positions//distance)+1)%num_players_static
    cond = start[nearest_start_before] == start[nearest_start_after] # if cond: pin traverses a start position
    result = jnp.where(
        env.rules['enable_start_blocking'] & cond,
        (~pins_on_start[nearest_start_after] | (current_pins == start[current_player])) & result, # true if start not blocked and new pos is free
        result
    )
    result = result & (env.rules['enable_circular_board'] | (moved_positions >= (start[current_player]))) # if circular board not enabled, prevent moving beyond start position backwards

    # filter actions for pins in start area
    result = jnp.where(
        jnp.isin(current_pins, jnp.concatenate([jnp.array([-1]), goal])),
        False,
        result
    )

    return result

# @jax.jit
def valid_actions(env: DOG) -> chex.Array:
    """
    Gibt eine Maske zurück, die alle gültigen Aktionen für den aktuellen Spieler angibt. Berücksichtigt alle valid_action functions.
    Args:
        env: DOG environment
    Returns:
        Ein boolean-Array der Form (num_total_actions,), das für jede Aktion angibt, ob sie gültig ist.
    """
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    current_pins = env.pins[current_player]
    hand = env.hands[current_player]

    # valid_actions based on cards in hand
    valid_action = jnp.where(hand > 0, True, False)
    
    num_total_actions = 4 * (12 + 1 + env.total_board_size) + 120 # actions without joker copy :==>  num_pins * (num_normal_moves + -4 move + swap moves) + move 7 distributions
    all_actions = jnp.full((num_total_actions,), False)

    # filter actions based on effect (handle special cards seperatly if necessary)
    num_swaps = 4*env.total_board_size
    valid_swaps = jax.lax.cond(
        valid_action[1],
        lambda: val_swap(env).flatten(),
        lambda: jnp.full((num_swaps,), False)
    )
    all_actions = all_actions.at[:num_swaps].set(valid_swaps)

    traversed_moves = num_swaps + len(DISTS_7_4)
    valid_hot_7 = jax.lax.cond(
        valid_action[7],
        lambda: jax.vmap(val_action_7, in_axes=(None, 0))(env, DISTS_7_4).flatten(),
        lambda: jnp.full((len(DISTS_7_4),), False)
    )
    all_actions = all_actions.at[num_swaps:traversed_moves].set(valid_hot_7)


    normal_card_indices = jnp.array([2,3,4,5,6,8,9,10,11,12,13])  # 2-6, 8-13

    # Maske für normale Karten
    normal_mask = hand[normal_card_indices] > 0

    # Maske für die 1: True, wenn 11 vorhanden ist
    one_mask = hand[11] > 0  # Index 11 entspricht Karte 11

    # Kombinierte Maske: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    # 1: one_mask, Rest: normal_mask
    final_mask = jnp.concatenate([one_mask[None], normal_mask])

    moves = jnp.where(final_mask, jnp.array([1,2,3,4,5,6,8,9,10,11,12,13]), 0)
    normal_actions = jax.vmap(val_action_normal_move, in_axes=(None, 0))(env, final_mask).flatten()
    all_actions = all_actions.at[traversed_moves:-4].set(normal_actions)

    valid_neg_4 = jax.lax.cond(hand[4] > 0, lambda: val_neg_move(env, -4), lambda: jnp.full((4,), False))
    all_actions = all_actions.at[-4:].set(valid_neg_4)

    # joker can replicate any action
    valid_joker = jax.lax.cond(
        hand[0] > 0,
        lambda: all_actions,
        lambda: jnp.zeros_like(all_actions, dtype=bool)
    )
    return jnp.concatenate([valid_joker, all_actions])

def no_step(env:DOG) ->  DOG:
    """
    Führt keinen Schritt aus und setzt die Hand des aktuellen Spielers auf leer.
    Args:
        env: DOG environment
    Returns:
        Aktualisiertes DOG environment mit leerer Hand für den aktuellen Spieler.
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

# @jax.jit
def step_swap(env: DOG, pin_idx: Action, swap_pos: Action) -> DOG:
    '''
    Führt einen Swap-Schritt im DOG-Spiel aus.
    Args:
        env: DOG environment
        pin_idx: Index des Pins des aktuellen Spielers, der getauscht werden soll
        swap_pos: Position auf dem Spielfeld, mit der getauscht werden soll
    Returns:
        Aktualisiertes Spielfeld und Pin-Positionen nach dem Swap
    '''
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    invalid_action = ~val_swap(env)[pin_idx, swap_pos]
    # print("Swap valid:", ~invalid_action)
    
    swapped_player = env.board[swap_pos]
    pin_pos = env.pins[current_player, pin_idx]
    board = env.board.at[swap_pos].set(current_player)
    board = board.at[pin_pos].set(swapped_player)
    pins = env.pins.at[current_player, pin_idx].set(swap_pos)
    new_pin_pos = jnp.where(pins[swapped_player] == swap_pos, pin_pos, pins[swapped_player])
    pins = pins.at[swapped_player].set(new_pin_pos)

    board, pins = jax.lax.cond(
        invalid_action,
        lambda: (env.board, env.pins),
        lambda: (board, pins)
    )

    winner = get_winner(env, board)
    done = env.done | jnp.any(winner)
    reward = jnp.array(jnp.where(env.done, 0, jnp.where(invalid_action, -1, winner[current_player])), dtype=jnp.int8)
    return board, pins, reward, done

# @jax.jit
def step_normal_move(env: DOG, pin: Action, move: Action) -> DOG:
    '''
    Führt einen normalen Bewegungsschritt im DOG-Spiel aus.
    Args:
        env: DOG environment
        pin: Index des Pins des aktuellen Spielers, der bewegt werden soll
        move: Die Distanz, die bewegt werden soll
    Returns:
        Aktualisiertes Spielfeld und Pin-Positionen nach der Bewegung
    '''
    pin = pin.astype(jnp.int32)
    move = move.astype(jnp.int32)
    # currently player ID
    player_id = env.current_player
    # ID of the players' pins to be moved (important for teams)
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    # check if the action is valid
    invalid_action = ~val_action_normal_move(env, move)[pin]

    current_positions = env.pins[current_player, pin]
    moved_positions = current_positions + move
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - env.target[current_player] - 1 #(start feld muss auch überaufen werden)

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
    #print("Normal move valid:", ~invalid_action)
    winner = get_winner(env, board)
    done = env.done | jnp.any(winner)
    reward = jnp.array(jnp.where(env.done, 0, jnp.where(invalid_action, -1, winner[current_player])), dtype=jnp.int8)
    return board, pins, reward, done

# @jax.jit
def step_neg_move(env: DOG, pin: Action, move: Action) -> DOG:
    '''
    Führt einen negativen Bewegungsschritt im DOG-Spiel aus.
    Args:
        env: DOG environment
        pin: Index des Pins des aktuellen Spielers, der bewegt werden soll
        move: Die negative Distanz, die bewegt werden soll
    Returns:
        Aktualisiertes Spielfeld und Pin-Positionen nach der Bewegung
    '''
    pin = pin.astype(jnp.int32)
    move = move.astype(jnp.int32)
    # currently player ID
    player_id = env.current_player
    # ID of the players' pins to be moved (important for teams)
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    # check if the action is valid
    invalid_action = ~val_neg_move(env, move)[pin]

    current_positions = env.pins[current_player, pin]
    moved_positions = current_positions + move
    fitted_positions = moved_positions % env.board_size

    new_position = fitted_positions
    
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
    # print("Backward move valid:", ~invalid_action)
    winner = get_winner(env, board)
    done = env.done | jnp.any(winner)
    reward = jnp.array(jnp.where(env.done, 0, jnp.where(invalid_action, -1, winner[current_player])), dtype=jnp.int8)
    return board, pins, reward, done

# @jax.jit
def step_hot_7(env:DOG, seven_dist):
    '''
    Führt einen Hot 7 Bewegungsschritt im DOG-Spiel aus.
    Args:
        env: DOG environment
        seven_dist: Die Distanz, die mit der 7 Aktion bewegt werden soll (1-7)
    Returns:
        Aktualisiertes Spielfeld und Pin-Positionen nach der Bewegung
    '''
    player_id = env.current_player
    # ID of the players' pins to be moved (important for teams)
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    # check if the action is valid
    invalid_action = ~jnp.all(val_action_7(env, seven_dist))
    current_pins = env.pins
    current_positions = current_pins[current_player]
    moved_positions = current_positions + seven_dist
    fitted_positions = moved_positions % env.board_size
    x = moved_positions - env.target[current_player] - 1 #(start feld muss auch überaufen werden)

    ###########################
    tmp_pins = env.pins.at[current_player].set(jnp.where(jnp.isin(current_positions, env.goal[current_player]), moved_positions, current_positions))
    tmp_board = set_pins_on_board(env.board, tmp_pins)
    a = jax.vmap(
        lambda pos, xi: jnp.where(
            jnp.isin(pos, env.goal[current_player]),
            True,
            check_goal_path_for_pin(-1, xi, env.goal[current_player], tmp_board, current_player)
        )
    )(current_positions, x)
    A = (env.rules['enable_jump_in_goal_area'] | a) #& (env.board[env.goal[current_player, x-1]] != current_player)
    new_positions = jnp.where(
        current_positions == -1, #pins in start cannot be moved with hot 7
        -1, # pins cannot leave the starting position with hot 7
        jnp.where(jnp.isin(current_positions, env.goal[current_player]),
                    moved_positions,
                    jnp.where(
                        (4 >= x) & (x > 0) & A & (current_positions <= env.target[current_player]),
                        env.goal[current_player, x-1], # move to goal position
                        fitted_positions
                    )
        )
    )
        
    # update pins
    # Liste von abgelaufenen Feldern. Jede Figur die in diesen Feldern ist wird zurück geschickt
    # bei den figuren des aktuellen Spielers muss die alte und neue position abgedeckt werden
    # Zielbereiche müssen extra behandelt werden
    pins = current_pins.at[current_player].set(jnp.where(invalid_action, current_positions, new_positions))
    hit_paths = calc_paths(current_positions, new_positions, env.start[current_player], env.goal[current_player], env.target[current_player], env.board_size, traversal_over_start=True)
    hit_pins = jnp.isin(env.pins, hit_paths)
    curr_pins_hit = calc_active_players_pins_hit(current_positions, new_positions, env.start[current_player], env.goal[current_player], env.target[current_player], env.board_size, traversal_over_start=True)
    hit_pins = hit_pins.at[current_player].set(curr_pins_hit)
    # if a player is at the new position and it's not the current player, send that pin back to start area
    pins = jnp.where(
        hit_pins & ~invalid_action,
        pins.at[jnp.where(hit_pins)].set(-1),
        pins
    )
    
    board = jax.lax.cond(
        ~invalid_action,
        lambda b: set_pins_on_board(-jnp.ones_like(b, dtype=jnp.int8), pins),
        lambda b: b,
        env.board
    )
    # print("Hot 7 move valid:", ~invalid_action)
    winner = get_winner(env, board)
    done = env.done | jnp.any(winner)
    reward = jnp.array(jnp.where(env.done, 0, jnp.where(invalid_action, -1, winner[current_player])), dtype=jnp.int8)
    return board, pins, reward, done

# @jax.jit
def env_step(env: DOG, action: Action) -> tuple[DOG, Reward, Done]:
    """
    Führt einen Schritt im DOG-Spiel basierend auf der gegebenen Aktion aus.
    Args:
        env: DOG environment
        action: Die Aktion, die ausgeführt werden soll
    Returns:
        Aktualisiertes DOG environment, Belohnung und Done-Status
    """   
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)            
    
    mapped_action = map_action_to_move(env, action)
    card_used = map_action_to_card(mapped_action)

    is_joker = mapped_action[0] == 1
    is_swap = mapped_action[1] == 1
    move_dists = mapped_action[2:]

    def swap_step():
        pin_idx = jnp.argmax(move_dists >= 0)
        swap_pos = move_dists[pin_idx]
        return step_swap(env, jnp.array(pin_idx), jnp.array(swap_pos))
    
    def hot_7_step():
        return step_normal_move(env, jnp.array(0), jnp.array(0))
        return step_hot_7(env, move_dists)
    
    def move_step():
        pin_idx = jnp.argmax(move_dists != 0)
        move = move_dists[pin_idx]
        return jax.lax.cond(
            move < 0,
            lambda: step_neg_move(env, jnp.array(pin_idx), jnp.array(move)),
            lambda: step_normal_move(env, jnp.array(pin_idx), jnp.array(move))
        )
    
    board, pins, reward, done =jax.lax.cond(
                                    is_swap,
                                    lambda: swap_step(),
                                    lambda: jax.lax.cond(
                                        jnp.sum(move_dists) == 7,
                                        lambda: hot_7_step(),
                                        lambda: move_step()
                                    )
                                )

    hands = env.hands.at[current_player, card_used].add(jnp.where(reward == -1, 0, -1))  # only remove card if action was valid
    
    hand_cards = jnp.sum(hands, axis=1) 
    def body(i, pnext):
        cand = (env.current_player + i + 1) % env.num_players
        take = (pnext == -1) & (hand_cards[cand] > 0)
        return jnp.where(take, cand, pnext)
    next_player = jax.lax.fori_loop(0, env.num_players, body, -jnp.array(1, dtype=jnp.int8))  

    current_player = jnp.where(reward == -1, current_player, next_player)
    env = DOG(
        board=board,
        num_players=env.num_players,
        pins=pins,
        current_player=current_player,
        done=done,
        reward=reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        deck=env.deck,
        hands=hands,
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )
    return env, reward, done

# @jax.jit
def map_action_to_move(env: DOG, action: Action) -> jnp.array:
    """
    Maps a given action index to the corresponding move distance in the DOG game.
    Args:
        action: The action index to map.
        env: DOG environment
    Returns:
        An array indicating the card and corresponding move.
    """
    action_space = 2* ( 4 * (12 + 1 + env.total_board_size) + 120)  # total action space
    is_joker = (action - (action_space // 2)) < 0

    # Aktion ohne Joker-Anteil
    act = action % (action_space // 2)

    pins_x_board = (4 * env.total_board_size)

    is_swap = act < pins_x_board
    is_hot_7 = (act >= pins_x_board) & (act < (pins_x_board + 120))
    is_normal_move = (act >= (pins_x_board + 120)) & (act < (action_space // 2 -4))

    def swap_action_details(act):
        """
        Get pin and swap position from swap action.
        Args:
            act: The action index for swap.
        Returns:
            An array indicating the pin index and swap position.
        """
        pin_idx = act // env.total_board_size
        swap_pos = act % env.total_board_size
        dist = - jnp.ones(4, dtype=jnp.int32)
        return dist.at[pin_idx].set(swap_pos)
    
    def normal_move_action_details(act):
        """
        Get card and move distance from normal move action.
        Args:
            act: The action index for normal move.
        Returns:
            An array indicating the card and move distance.
        """
        normal_act = act - (pins_x_board + 120) # wert im bereich 0 - (4*12 -1)
        pin_idx = normal_act // 12
        move = (normal_act % 12)
        move = move + 1 + (move >= 7).astype(jnp.int32)  # skip 7
        dist = jnp.zeros(4, dtype=jnp.int32)
        return dist.at[pin_idx].set(move)
    dist = jax.lax.cond(
        is_swap,
        lambda: swap_action_details(act),  # Swap actions have no move distance
        lambda: jax.lax.cond(
            is_hot_7,
            lambda: jnp.array(DISTS_7_4[(act - pins_x_board)]),  # Hot 7 actions
            lambda: jax.lax.cond(
                is_normal_move,
                lambda: normal_move_action_details(act),  # Normal move actions
                lambda: jnp.zeros(4, dtype=jnp.int32).at[act - ((action_space // 2) -4)].set(-4)  # Negative move actions
            )
        )
    )
    return jnp.concatenate([is_joker[None].astype(jnp.int8), is_swap[None].astype(jnp.int8), dist])

def map_action_to_card(action: Action) -> Card:
    """
    Maps a given action index to the corresponding card in the current player's hand.
    Args:
        action: The action to map, shape (6,): [is_joker, is_swap, dist_pin0, dist_pin1, dist_pin2, dist_pin3]
    Returns:
        The card corresponding to the given action index.
    """
    moved_spaces = jnp.sum(action[2:])
    return jax.lax.cond(
        action[0] == 1,
        lambda: 0,  # Joker card
        lambda: jax.lax.cond(
            action[1] == 1,
            lambda: 1,  # Swap card
            lambda: jax.lax.cond(
                moved_spaces == -4,
                lambda: 4,  # Negative move card
                lambda: jnp.where(moved_spaces==1, 11, moved_spaces)  # Normal move card
            )
        )
    )
