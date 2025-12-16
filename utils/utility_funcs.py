import jax
from jax import numpy as jnp

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

def calc_paths(start, end, start_idx, goal, target, N, traversal_over_start=False):
    '''
    Berechnet alle Pfadpositionen zwischen start und end Positionen. 
    Bei Übergang ins Goal wird der Pfad bis zum Target (ggf. mit dem eigenen Startfeld) berechnet und anschließend alle Goal-Positionen hinzugefügt.
    
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
        if traversal_over_start:
            goal_range = jnp.concatenate([jnp.array([start_idx]), goal_range])
        all_path_positions = jnp.concatenate([all_same_area, all_diff_area_to_target, goal_range])
    else:
        all_path_positions = jnp.concatenate([all_same_area, all_diff_area_to_target])
    
    return jnp.unique(all_path_positions)

def calc_active_players_pins_hit(starting_position, final_position, start_index, goal_area, target, board_size, traversal_over_start=False):
    '''
    Berechne, welche von den eigenen Pins auf den Pfaden der anderen eigenen Pins liegen und somit geschlagen werden.
    Args:
        starting_position: (4,) Array der Startpositionen
        final_position: (4,) Array der Endpositionen  
        goal_area: (4,) Array der Goal-Positionen
        target: int Target-Position (Eingang zum Goal-Bereich)
        board_size: int Board-Größe
    '''
    x = jnp.array([starting_position, final_position])
    A = jnp.isin(starting_position, goal_area)  # start in goal
    B = jnp.isin(final_position, goal_area)    # end in goal
    
    paths = []
    for i in range(4):
        if A[i] == B[i]: # player bleibt in der gleichen area
            p, _ = get_all_paths_compact(jnp.array([starting_position[i]]), jnp.array([final_position[i]]), board_size)
            paths.append(p[0])
        else: # player wechselt in goal area
            p, _ = get_all_paths_compact(jnp.array([starting_position[i]]), jnp.array([target]), board_size)
            goal_range = jnp.arange(goal_area[0], final_position[i] + 1, dtype=jnp.int8)
            if traversal_over_start:
                paths.append(jnp.concatenate([p[0], goal_range, jnp.array([start_index])]))
            else:
                paths.append(jnp.concatenate([p[0], goal_range]))
            

    other_paths_0 = jnp.concatenate([jnp.concatenate(paths[1:3]), paths[3]])
    other_paths_1 = jnp.concatenate([paths[0], jnp.concatenate(paths[2:])])
    other_paths_2 = jnp.concatenate([jnp.concatenate(paths[:2]), paths[3]])
    other_paths_3 = jnp.concatenate(paths[:3])
    
    a = jnp.all(jnp.isin(jnp.array([starting_position[0], final_position[0]]), other_paths_0))
    b = jnp.all(jnp.isin(jnp.array([starting_position[1], final_position[1]]), other_paths_1))
    c = jnp.all(jnp.isin(jnp.array([starting_position[2], final_position[2]]), other_paths_2))
    d = jnp.all(jnp.isin(jnp.array([starting_position[3], final_position[3]]), other_paths_3))
    return jnp.array([a,b,c,d])

def check_goal_path_for_pin2(start, x_val, goal, board, current_player):
    '''
    Überprüft, ob der Pfad eines Pins im Zielbereich frei von eigenen Pins ist.
        Args:
            start: Startposition des Pins
            x_val: Zielposition des Pins
            goal: Zielpositionen des Spielers
            board: Das aktuelle Spielfeld
            current_player: Der aktuelle Spieler
        Returns:
            Ein boolescher Wert, der angibt, ob der Pfad frei von eigenen Pins ist.
    '''
    goal_area = jnp.arange(len(goal))[None, :] # shape (1, 4)

    return jnp.all(
        jnp.where(
            (start[:, None] < goal_area) & (goal_area < x_val[:, None]),
            board[goal[goal_area]]!=current_player,
            True
        ),
        axis=1
    )

def check_goal_path_for_pin(start, x_val, goal, board, current_player):
    '''
    Überprüft, ob der Pfad eines Pins im Zielbereich frei von eigenen Pins ist.
        Args:
            start: Startposition des Pins
            x_val: Zielposition des Pins
            goal: Zielpositionen des Spielers
            board: Das aktuelle Spielfeld
            current_player: Der aktuelle Spieler
        Returns:
            Ein boolescher Wert, der angibt, ob der Pfad frei von eigenen Pins ist.
    '''
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