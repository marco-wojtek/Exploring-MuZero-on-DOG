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
        hands = jnp.zeros((num_players, num_cards), dtype=jnp.int8),
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

def distribute_cards(env: DOG, quantity: int):
    pass

def is_player_done(num_players, board:Board, goal:Goal, player: Player) -> chex.Array:
    '''
    returns winner as jnp.array to support multiple winners in team mode    
    '''
    return jax.lax.cond(
        player >= num_players,
        lambda: False,
        lambda: jnp.all(board[goal[player]] >= 0)
    )

def get_winner(env: DOG, board: Board) -> chex.Array:
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

def check_relative_order_preserved(old_pos: jnp.ndarray, new_pos: jnp.ndarray, board_size: int) -> jnp.ndarray:
    """
    Prüft, ob die relative Reihenfolge der Pins im Zielbereich erhalten bleibt.

    Args:
        old_pos: Die alten Positionen der Pins.
        new_pos: Die neuen Positionen der Pins.
        board_size: Die Größe des Hauptspielbretts (z.B. 40). Alles darüber ist Zielbereich.

    Returns:
        Ein boolean-Array, das für jeden Pin anzeigt, ob die Bedingung erfüllt ist.
    """
    # Bedingung 1: Alle Pins, die nicht im Zielbereich starten, sind immer gültig.
    # Dies schließt auch Pins im Start (-1) ein.
    valid_outside_goal = (old_pos < board_size)

    # Bedingung 2: Für Pins im Zielbereich muss die relative Reihenfolge erhalten bleiben.
    
    # Erstelle eine Maske für Pins, die sich im Zielbereich befinden.
    in_goal_mask = (old_pos >= board_size)

    # Erweitere die Dimensionen, um paarweise Vergleiche zu ermöglichen.
    # Shape: (num_pins, 1) und (1, num_pins)
    old_pos_col = old_pos[:, None]
    old_pos_row = old_pos[None, :]
    new_pos_col = new_pos[:, None]
    new_pos_row = new_pos[None, :]

    # Berechne die Vorzeichen der Differenzen für alle Paare.
    # sign(a - b) gibt an, ob a > b (+1), a < b (-1) oder a == b (0).
    sign_diff_old = jnp.sign(old_pos_col - old_pos_row)
    sign_diff_new = jnp.sign(new_pos_col - new_pos_row)

    # Die Reihenfolge ist nur dann erhalten, wenn die Vorzeichen aller Vergleiche gleich bleiben.
    order_preserved_matrix = (sign_diff_old == sign_diff_new)

    # Erstelle eine Maske für die paarweisen Vergleiche, die nur Pins im Zielbereich berücksichtigt.
    # Ein Paar (i, j) ist relevant, wenn sowohl Pin i als auch Pin j im Ziel sind.
    goal_pairs_mask = in_goal_mask[:, None] & in_goal_mask[None, :]

    # Ein Pin im Zielbereich ist gültig, wenn für ihn die Reihenfolge zu allen
    # anderen Pins im Zielbereich erhalten bleibt.
    # Wir verwenden jnp.where, um nur die relevanten Paare zu prüfen.
    # jnp.all prüft dann pro Zeile (pro Pin), ob alle seine Vergleiche stimmen.
    valid_in_goal = jnp.all(jnp.where(goal_pairs_mask, order_preserved_matrix, True), axis=1)

    # Das Endergebnis ist True, wenn der Pin entweder außerhalb des Ziels war
    # oder wenn er im Ziel war und seine Reihenfolge beibehalten wurde.
    return valid_outside_goal | valid_in_goal

# returns a boolean array indicating valid swap positions
# @jax.jit
def val_swap(env):
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    Num_players = env.num_players
    current_pins = env.pins[current_player]
    board = env.board
    board_size = env.board_size
    target = env.target[current_player]
    goal = env.goal[current_player]
    start = env.start
    num_players_static = start.shape[0]
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)

    swap_mat = jnp.tile(board[:board_size], (4,1))
    condA = jnp.where(~jnp.isin(swap_mat, jnp.array([-1, current_player])), True, False)
    condA = condA & condA.at[:,start].set(board[start] != player_ids)
    condB = (~jnp.isin(current_pins, jnp.array([-1, start[current_player]])))[:, None]
    return condA & condB

@jax.jit
def val_action_7(env:DOG, seven_dist) -> chex.Array:
    '''
    Returns a mask of shape (4, ) indicating which actions are valid for each pin of the current player
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
    x = moved_positions - target

    pins_on_start = pins_on_start.at[current_player].set(jnp.all(jnp.where(seven_dist == 0, moved_positions == start[current_player], True))) # if any pin does not move, check if it is on start
    # Überlaufen der Zielposition verhindern falls kein Rundbrett
    result = jax.lax.cond(
        env.rules['enable_circular_board'],
        lambda: jnp.ones_like(current_positions, dtype=bool),
        lambda: ~((current_positions <= target) & (moved_positions > (target + 4)))
    )

    distance = env.board_size // num_players_static
    nearest_start_before = ((current_positions  //distance)+1)%num_players_static # nearest start before is the next start field in front of a pin
    nearest_start_after = fitted_positions//distance
    cond = start[nearest_start_before] == start[nearest_start_after] # if cond: pin traverses a start position
    result = jnp.where(
        env.rules['enable_start_blocking'] & cond,
        ~pins_on_start[nearest_start_after] & result, # true if start not blocked and new pos is free
        result
    )

    check_all_pins = jax.vmap(check_goal_path_for_pin, in_axes=(0, 0, None, None, None))
    A = (env.rules["enable_circular_board"] & result) # if rule enabled, consider circle rotation, else only goal area
    # Entweder man darf im Ziel überspringen oder auf dem Weg (im Zielbereich) ist kein eigener Pin 
    # wenn C true ist ist B auch true da B eine Teil-Bedingung davon ist
    # Für alle pins die sich im Ziel bewegen sollten die neuen positionen geprüft werden, da die alten nicht blockieren könnten
    tmp_pins = env.pins.at[current_player].set(jnp.where(jnp.isin(current_positions, goal), moved_positions, current_positions))
    tmp_board = set_pins_on_board(board, tmp_pins)
    B = (tmp_board[goal[x-1]] != current_player)
    C = (env.rules['enable_jump_in_goal_area'] | check_all_pins(- jnp.ones(4, dtype=jnp.int8), x, goal, tmp_board, current_player))
    result = jnp.where(
        (4 >= x) & (x > 0) & (current_positions <= target),
        A | (B & C),
        result
    )
    # filter actions for pins in goal area
    D = (env.rules['enable_jump_in_goal_area'] | check_relative_order_preserved(current_positions, moved_positions, env.board_size))
    result = jnp.where(
        jnp.isin(current_positions, goal),
        (moved_positions <= goal[-1]) & D,
        result
    )
    # alle Aktionen müssenrechenrisch möglich sein und es dürfen keine zwei Pins auf die gleiche Position ziehen
    board_mover = jnp.where(current_positions == -1, moved_positions==-1, True)# prüfe dass kein pin im startbereich bewegt werden würde 
    return jnp.all(result & board_mover) 

@jax.jit
def val_action_normal_move(env:DOG, move: int):
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
    x = moved_positions - target

    result = (board[fitted_positions] != current_player) | env.rules['enable_friendly_fire'] # check move to any board position

    # filter actions where a pin on a start spot would block others
    distance = env.board_size // num_players_static
    nearest_start_before = ((current_pins//distance)+1)%num_players_static # nearest start before is the next start field in front of a pin
    nearest_start_after = fitted_positions//distance
    cond = start[nearest_start_before] == start[nearest_start_after] # if cond: pin traverses a start position
    result = jnp.where(
        env.rules['enable_start_blocking'] & cond,
        ~pins_on_start[nearest_start_after] & result, # true if start not blocked and new pos is free
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
    result = jnp.where(
        (current_pins == -1),
        jnp.isin(move, jnp.array([11, 13])) & (~pins_on_start[current_player]),
        result
    )

    return result

# @jax.jit
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
    
    num_total_actions = 4 * (12 + 1 + env.board_size) + 120 # actions without joker copy :==>  num_pins * (num_normal_moves + -4 move + swap moves) + move 7 distributions
    all_actions = jnp.full((num_total_actions,), False)

    # filter actions based on effect (handle special cards seperatly if necessary)
    if valid_action[1]: # Card 1: Swap
        num_swaps = 4*env.board_size
        all_actions = all_actions.at[:num_swaps].set(val_swap(env).flatten())
    if valid_action[7]: # Card 7: Move 7 with distribution
        traversed_moves = num_swaps + len(DISTS_7_4)
        all_actions = all_actions.at[num_swaps:traversed_moves].set(jax.vmap(val_action_7, in_axes=(None, 0))(env, DISTS_7_4))    
    normal_actions = jax.vmap(val_action_normal_move, in_axes=(None, 0))(env, jnp.array([1,2,3,4,-4,5,6,8,9,10,11,12,13])).flatten()
    all_actions = all_actions.at[traversed_moves:].set(normal_actions)
    return all_actions

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