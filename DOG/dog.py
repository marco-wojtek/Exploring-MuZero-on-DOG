import chex
import jax
import jax.numpy as jnp
import mctx

def get_all_paths_compact(start, end, N):
    """
    Berechnet alle Pfadpositionen zwischen start und end Positionen für mehrere Pins gleichzeitig.
    
    Args:
        start: Array der Startpositionen
        end: Array der Endpositionen  
        N: Board-Größe für Modulo-Rechnung (optional)
    
    Returns:
        Array aller Pfadpositionen zwischen start und end (exklusive start, inklusive end)
    """
    valid_mask = end != start

    use_modulo = (start < N) & (end < N)

    # Modulo-Logik für Rundbrett
    distance = (end - start) % N
    distance = jnp.where(distance == 0, N, distance)  # Vollrunde = N Schritte
    distance = jnp.where(valid_mask, distance, 0)  # Keine Bewegung wenn start == end
    
    max_len = jnp.max(distance)
    
    # Erstelle alle möglichen Pfade für alle start/end Paare
    i, j = jnp.meshgrid(jnp.arange(len(start)), jnp.arange(max_len), indexing='ij')
    path_values_normal = start[i] + j + 1
    path_values_modulo = (start[i] + j + 1) % N
    path_values = jnp.where(use_modulo[i], path_values_modulo, path_values_normal)
    
    valid_positions = valid_mask[i] & (j < distance[i])
    
    return path_values, valid_positions

def calc_paths(start, end, goal, target, N):
    '''
    Berechnet alle Pfadpositionen zwischen start und end Positionen. 
    Bei Übergang ins Goal wird der Pfad bis zum Target berechnet und anschließend alle Goal-Positionen hinzugefügt.
    
    Args:
        start: (4,) Array der Startpositionen
        end: (4,) Array der Endpositionen  
        goal: (4,) Array der Goal-Positionen
        target: int Target-Position (Eingang zum Goal-Bereich)
        N: int Board-Größe
    '''
    A = jnp.isin(start, goal)  # start in goal
    B = jnp.isin(end, goal)    # end in goal  

    # Berechne alle Pfade für same area (both in goal or both not in goal)
    same_area_condition = A == B
    same_area_paths, same_area_mask = get_all_paths_compact(start, end, N)
    same_area_valid = same_area_condition[:, None] & same_area_mask
    
    # Berechne Pfade für different area (traverse to goal)
    diff_area_condition = A != B
    
    # Pfade bis zum Target für Pins die ins Goal wechseln
    target_array = jnp.full_like(end, target)
    diff_area_paths_to_target, diff_area_to_target_mask = get_all_paths_compact(start, target_array, N)
    diff_area_to_target_valid = diff_area_condition[:, None] & diff_area_to_target_mask
    
    # Kombiniere alle gültigen Pfadpositionen
    all_same_area = same_area_paths[same_area_valid]
    all_diff_area_to_target = diff_area_paths_to_target[diff_area_to_target_valid]
    
    # Goal-Positionen für Übergänge ins Goal
    transition_to_goal = diff_area_condition
    if jnp.any(transition_to_goal):
        goal_start = goal[0]
        goal_end = jnp.max(jnp.where(transition_to_goal, end, goal[0]))
        goal_range = jnp.arange(goal_start, goal_end + 1, dtype=jnp.int8)
        all_path_positions = jnp.concatenate([all_same_area, all_diff_area_to_target, goal_range])
    else:
        all_path_positions = jnp.concatenate([all_same_area, all_diff_area_to_target])
    
    return jnp.unique(all_path_positions)

def prototype(start, end, goal, target, N):
    '''
    Prototype-Funktion zur Validierung der Pfadberechnung.
    Args:
        start: (4,) Array der Startpositionen
        end: (4,) Array der Endpositionen  
        goal: (4,) Array der Goal-Positionen
        target: int Target-Position (Eingang zum Goal-Bereich)
        N: int Board-Größe
    '''
    x = jnp.array([start, end])
    A = jnp.isin(start, goal)  # start in goal
    B = jnp.isin(end, goal)    # end in goal
    
    paths = []
    for i in range(4):
        if A[i] == B[i]:
            p, _ = get_all_paths_compact(jnp.array([start[i]]), jnp.array([end[i]]), N)
            paths.append(p[0])
        else:
            p, _ = get_all_paths_compact(jnp.array([start[i]]), jnp.array([target]), N)
            goal_range = jnp.arange(goal[0], end[i] + 1, dtype=jnp.int8)
            paths.append(jnp.concatenate([p[0], goal_range]))

    other_paths_0 = jnp.concatenate([jnp.concatenate(paths[1:3]), paths[3]])
    other_paths_1 = jnp.concatenate([paths[0], jnp.concatenate(paths[2:])])
    other_paths_2 = jnp.concatenate([jnp.concatenate(paths[:2]), paths[3]])
    other_paths_3 = jnp.concatenate(paths[:3])
    
    a = jnp.all(jnp.isin(jnp.array([start[0], end[0]]), other_paths_0))
    b = jnp.all(jnp.isin(jnp.array([start[1], end[1]]), other_paths_1))
    c = jnp.all(jnp.isin(jnp.array([start[2], end[2]]), other_paths_2))
    d = jnp.all(jnp.isin(jnp.array([start[3], end[3]]), other_paths_3))
    return jnp.array([a,b,c,d])

def all_pin_distributions(total=7):
    '''
    Erzeugt alle möglichen Verteilungen von `total` Pins auf 4 Pins (a0, a1, a2, a3).
    Returns:
        Ein Array der Form (num_distributions, 4) mit allen möglichen Verteilungen.
    '''
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
    '''
    Distributes `quantity` cards to each player's hand from the deck.
    Args:
        env: DOG environment
        quantity: Number of cards to distribute to each player

    '''
    pass

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

def check_goal_path_for_pin(start, x_val, goal, board, current_player):
    """
    Prüft, ob der Pfad eines Pins im Zielbereich frei von eigenen Pins ist.
    Args:
        start: Startposition des Pins im Zielbereich.
        x_val: Zielposition des Pins im Zielbereich.
        goal: Array der Goal-Positionen für den aktuellen Spieler.
        board: Aktuelles Spielfeld.
        current_player: Index des aktuellen Spielers.
    Returns:
        Boolean, ob der Pfad frei von eigenen Pins ist.
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
    '''
    Gibt eine Maske zurück, die gültige Swap-Positionen für den aktuellen Spieler angibt.
    Args:
        env: DOG environment
    Returns:
        Ein boolean-Array der Form (board_size, ), das für jede Position angibt, ob sie für einen Swap gültig ist.
    '''
    player_id = env.current_player
    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
    Num_players = env.num_players
    current_pins = env.pins[current_player]
    board = env.board
    board_size = env.board_size
    target = env.target[current_player]
    goal = env.goal
    start = env.start
    num_players_static = start.shape[0]
    player_ids = jnp.arange(num_players_static, dtype=board.dtype)

    swap_mat = jnp.tile(board[:board_size], (4,1))
    condA = jnp.where(~jnp.isin(swap_mat, jnp.array([-1, current_player])), True, False)
    condA = condA & condA.at[:,start].set(~((board[start] == player_ids) & env.rules['enable_start_blocking']))  # start positions cannot be swapped if blocked except the rule is disabled
    disallowed_pos = jax.lax.cond(
        env.rules['enable_start_blocking'],
        lambda: jnp.concatenate([jnp.array([-1]), jnp.array([start[current_player]]), goal[current_player]]),
        lambda: jnp.concatenate([jnp.array([-1]), jnp.array([-1]), goal[current_player]])
    )
    condB = (~jnp.isin(current_pins, disallowed_pos))[:, None]
    return condA & condB

@jax.jit
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
    cond = start[nearest_start_before] == start[nearest_start_after] # if cond: pin traverses a start position
    result = jnp.where(
        env.rules['enable_start_blocking'] & cond,
        ~pins_on_start[nearest_start_after] & result, # true if start not blocked and new pos is free
        result
    )
    # x = jnp.where(
    #     env.rules['enable_start_blocking'] & cond,
    #     0, 
    #     x
    # )
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
    B = (tmp_board[goal[x-1]] != current_player)
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

@jax.jit
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
@jax.jit
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

@jax.jit
def valid_actions(env: DOG) -> chex.Array:
    """
    Gibt eine Maske zurück, die alle gültigen Aktionen für den aktuellen Spieler angibt. Berücksichtigt alle valid_action functions.
    Args:
        env: DOG environment
    Returns:
        Ein boolean-Array der Form (num_total_actions,), das für jede Aktion angibt, ob sie gültig ist.
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
    normal_actions = jax.vmap(val_action_normal_move, in_axes=(None, 0))(env, jnp.array([1,2,3,4,5,6,8,9,10,11,12,13])).flatten()
    all_actions = all_actions.at[traversed_moves:-4].set(normal_actions)
    all_actions = all_actions.at[-4:].set(val_neg_move(env, -4))
    return all_actions

def map_action_to_card(action: Action, env: DOG) -> Card:
    """
    Maps a given action index to the corresponding card in the current player's hand.
    Args:
        action: The action index to map.
        env: DOG environment
    Returns:
        The card corresponding to the given action index.
    """
    hand = env.hands[env.current_player]
    card_indices = jnp.where(hand > 0, jnp.arange(len(hand)), -1)
    valid_card_indices = card_indices[card_indices != -1]
    card = valid_card_indices[action]
    return card

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

def step_swap(env, pin_idx, swap_pos):
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
    print("Swap valid:", ~invalid_action)
    
    swapped_player = env.board[swap_pos]
    pin_pos = env.pins[current_player, pin_idx]
    board = env.board.at[swap_pos].set(current_player)
    board = board.at[pin_pos].set(swapped_player)
    pins = env.pins.at[current_player, pin_idx].set(swap_pos)
    pins = pins.at[swapped_player, jnp.where(pins[swapped_player] == swap_pos)].set(pin_pos)

    return jax.lax.cond(
        invalid_action,
        lambda: (env.board, env.pins),
        lambda: (board, pins)
    )

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
    pin = pin.astype(jnp.int8)
    move = move.astype(jnp.int8)
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
    return (board, pins)

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
    pin = pin.astype(jnp.int8)
    move = move.astype(jnp.int8)
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
    print("Backward move valid:", ~invalid_action)
    return (board, pins)

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
        -1, # move from start area to starting position
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
    pins = current_pins.at[current_player].set(jnp.where(invalid_action, current_pins[current_player], new_positions))
    hit_paths = calc_paths(current_positions, new_positions, env.goal[current_player], env.target[current_player], env.board_size)
    hit_pins = jnp.isin(env.pins, hit_paths)
    curr_pins_hit = prototype(current_positions, new_positions, env.goal[current_player], env.target[current_player], env.board_size)
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
    print("Hot 7 move valid:", ~invalid_action)
    return (board, pins)

@jax.jit
def env_step(env: DOG, action: Action) -> tuple[DOG, Reward, Done]:
    """
    Führt einen Schritt im DOG-Spiel basierend auf der gegebenen Aktion aus.
    Args:
        env: DOG environment
        action: Die Aktion, die ausgeführt werden soll
    Returns:
        Aktualisiertes DOG environment, Belohnung und Done-Status
    """               
    # WICHTIG: Wenn dog gespielt wird, muss man über das startfeld ins Ziel gehen, ohne darauf loszugehen. 
    # target ist das feld vor dem startfeld.
    # das heißt x muss noch einen wert abgezogen bekommen, damit r.g. eine 5 auf target, ins Ziel führt
    # ergo: Man kann nicht mi einer 1 vom startfeld ins Ziel gehen!!
    env, reward, done = no_step(env)
    
    return env, reward, done