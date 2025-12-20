import jax
import jax.numpy as jnp
import sys, os, time

def board_to_matrix(env):
    '''
    VERALTET -> ERWARTET DYNAMISCHE BRETT-GRÖßE
    Wandelt das Brett der MADN-Umgebung in eine Matrix-Darstellung um.
        Args:
            env: Die aktuelle Spielumgebung
        Returns:
            Eine Matrix-Darstellung des Brettes.
    '''
    board = env.board
    num_players = int(env.num_players)
    n = int(env.board_size // num_players)
    start = env.start
    start_area = board[start]
    goal = env.goal
    goal_area = board[goal]
    if num_players == 2:
        board_matrix = jnp.ones((3, n))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n])
        board_matrix = board_matrix.at[1, :].set(jnp.ones(n)*9)
        board_matrix = board_matrix.at[2, :].set(jnp.flip(board[n:2*n]))

        board_matrix = board_matrix.at[0, 0].set(jnp.where(start_area[0]==-1, 10, start_area[0]))
        board_matrix = board_matrix.at[-1, -1].set(jnp.where(start_area[1]==-1, 11, start_area[1]))

        board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
        board_matrix = board_matrix.at[-1, 1:5].set(jnp.where(goal_area[0]==-1, 10, goal_area[0]))
        board_matrix = board_matrix.at[0, -5: -1].set(jnp.flip(jnp.where(goal_area[1]==-1, 11, goal_area[1])))
    elif num_players == 3:
        board_matrix = jnp.ones((n+1, n+1))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n+1])
        board_matrix = board_matrix.at[jnp.arange(n+1), n - jnp.arange(n+1)].set(board[n:2*n+1])
        board_matrix = board_matrix.at[1:, 0].set(jnp.flip(board[2*n:3*n]))

        board_matrix = board_matrix.at[0, 0].set(jnp.where(start_area[0]==-1, 10, start_area[0]))
        board_matrix = board_matrix.at[0, -1].set(jnp.where(start_area[1]==-1, 11, start_area[1]))
        board_matrix = board_matrix.at[-1, 0].set(jnp.where(start_area[2]==-1, 12, start_area[2]))


        board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
        board_matrix = board_matrix.at[2:6, 0].set(jnp.where(goal_area[0]==-1, 10, goal_area[0]))
        board_matrix = board_matrix.at[0, -6: -2].set(jnp.flip(jnp.where(goal_area[1]==-1, 11, goal_area[1])))
        board_matrix = board_matrix.at[-3, 3:7].set(jnp.where(goal_area[2]==-1, 12, goal_area[2]))
    elif num_players == 4:
        board_matrix = jnp.ones((n+1, n+1))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n+1])
        board_matrix = board_matrix.at[:, -1].set(board[n:2*n+1])
        board_matrix = board_matrix.at[-1, :].set(jnp.flip(board[2*n:3*n+1]))
        board_matrix = board_matrix.at[1:, 0].set(jnp.flip(board[3*n:4*n]))

        board_matrix = board_matrix.at[0, 0].set(jnp.where(start_area[0]==-1, 10, start_area[0]))
        board_matrix = board_matrix.at[0, -1].set(jnp.where(start_area[1]==-1, 11, start_area[1]))
        board_matrix = board_matrix.at[-1, -1].set(jnp.where(start_area[2]==-1, 12, start_area[2]))
        board_matrix = board_matrix.at[-1, 0].set(jnp.where(start_area[3]==-1, 13, start_area[3]))
        

        board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
        board_matrix = board_matrix.at[2:6, 0].set(jnp.where(goal_area[0]==-1, 10, goal_area[0]))
        board_matrix = board_matrix.at[0, -6: -2].set(jnp.flip(jnp.where(goal_area[1]==-1, 11, goal_area[1])))
        board_matrix = board_matrix.at[-6:-2, -1].set(jnp.flip(jnp.where(goal_area[2]==-1, 12, goal_area[2])))
        board_matrix = board_matrix.at[-1, 2:6].set(jnp.where(goal_area[3]==-1, 13, goal_area[3]))
    return board_matrix

def replace_rows_simple(x, y):
    '''
    Ersetzt Zeilen in y basierend auf dem Booleschen Array x.
        Args:
            x: Ein Boolesches Array der Form (4,), das angibt, welche Zeilen ersetzt werden sollen
            y: Eine Matrix der Form (n, m), aus der die n Zeilen entnommen werden
        Returns:
            Eine Matrix der Form (4, m), in der die n Zeilen entsprechend x ersetzt wurden.
    '''
    base_matrix = -jnp.ones_like(y, dtype=jnp.int32)
    
    # Zähle kumulative True-Werte für y-Indexierung
    cumsum_x = jnp.cumsum(x) - 1
    
    # Für jede Zeile: ersetzen falls x[i] True
    def get_row(i):
        return jnp.where(
            x[i],
            y[cumsum_x[i]],  # Zeile aus y
            base_matrix[i]   # Original -1 Zeile
        )
    
    return jax.vmap(get_row)(jnp.arange(4))

def board_to_mat(env, layout):
    '''
    Wandelt das Brett der MADN-Umgebung in eine Matrix-Darstellung um, basierend auf dem Layout.
        Args:
            env: Die aktuelle Spielumgebung
            layout: Ein Boolesches Array der Form (4,), das angibt, welche Spieler im Spiel sind   
        Returns:
            Eine Matrix-Darstellung des Brettes.
    '''
    # setze pins auf indizierbare werte: Spieler 1 hat pins 10,11,12,13; Spieler 2 hat pins 20,21,22,23; etc.
    board = jnp.array(jnp.where(env.board!=-1, (env.board+1)*10, -1), dtype=jnp.int32)
    num_players = int(env.num_players)
    pin_ids = jnp.tile(jnp.arange(1, 5), (num_players,1))
    pins = env.pins
    board = board.at[pins].add(jnp.where(pins!=-1, pin_ids, 0))

    n = int(env.board_size // 4)

    layout = jax.lax.cond(
        (jnp.sum(layout)!=num_players) | (jnp.all(layout) & (num_players < 4)),
        lambda: jnp.array([False, False, False, False], dtype=jnp.bool_).at[:num_players].set(True),
        lambda: layout
    )

    start = jnp.arange(4)*n
    start_area = board[start]
    goal = env.goal
    goal_area = replace_rows_simple(layout, board[goal])
    pin_shape = env.pins.shape[0]
    idx = jnp.arange(pin_shape)[:, None] + 1 # Shape (n,1) für Broadcasting
    home = jnp.where(env.pins == -1, pin_ids + (10*idx), -1)
    home_area = replace_rows_simple(layout, home)

    colour_ids = replace_rows_simple(layout, jnp.arange(10, 50, step=10, dtype=jnp.int8)) 
    colour_ids = jnp.where(
        colour_ids == -1,
        7,
        colour_ids
    )
    board_matrix = jnp.ones((n+1, n+1))*8
    board_matrix = board_matrix.at[0, :].set(board[0:n+1])
    board_matrix = board_matrix.at[:, -1].set(board[n:2*n+1])
    board_matrix = board_matrix.at[-1, :].set(jnp.flip(board[2*n:3*n+1]))
    board_matrix = board_matrix.at[1:, 0].set(jnp.flip(board[3*n:4*n]))


    board_matrix = board_matrix.at[0, 0].set(jnp.where(start_area[0]==-1, colour_ids[0], start_area[0]))
    board_matrix = board_matrix.at[0, -1].set(jnp.where(start_area[1]==-1, colour_ids[1], start_area[1]))
    board_matrix = board_matrix.at[-1, -1].set(jnp.where(start_area[2]==-1, colour_ids[2], start_area[2]))
    board_matrix = board_matrix.at[-1, 0].set(jnp.where(start_area[3]==-1, colour_ids[3], start_area[3]))
    

    board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
    board_matrix = board_matrix.at[2:6, 0].set(jnp.where(goal_area[0]==-1, colour_ids[0], goal_area[0]))
    board_matrix = board_matrix.at[0, -6: -2].set(jnp.flip(jnp.where(goal_area[1]==-1, colour_ids[1], goal_area[1])))
    board_matrix = board_matrix.at[-6:-2, -1].set(jnp.flip(jnp.where(goal_area[2]==-1, colour_ids[2], goal_area[2])))
    board_matrix = board_matrix.at[-1, 2:6].set(jnp.where(goal_area[3]==-1, colour_ids[3], goal_area[3]))

    board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
    board_matrix = board_matrix.at[:2, :2].set(jnp.where(home_area[0].reshape((2,2))==-1, jnp.full((2,2), colour_ids[0]), home_area[0].reshape((2,2))))
    board_matrix = board_matrix.at[:2, -2:].set(jnp.where(home_area[1].reshape((2,2))==-1, jnp.full((2,2), colour_ids[1]), home_area[1].reshape((2,2))))
    board_matrix = board_matrix.at[-2:, -2:].set(jnp.where(home_area[2].reshape((2,2))==-1, jnp.full((2,2), colour_ids[2]), home_area[2].reshape((2,2))))
    board_matrix = board_matrix.at[-2:, :2].set(jnp.where(home_area[3].reshape((2,2))==-1, jnp.full((2,2), colour_ids[3]), home_area[3].reshape((2,2))))

    return board_matrix

def matrix_to_string(matrix):
    '''
    Wandelt eine Matrix-Darstellung des MADN-Bretts in einen String um.
        Args:
            matrix: Eine Matrix-Darstellung des Brettes
        Returns:
            Ein String, der die Matrix darstellt.
    '''
    str_repr = ""
    pin_rep = ["♠", "♥", "♦", "♣"]
    pin_colors = ["\033[94m", "\033[91m", "\033[93m", "\033[92m", "\033[90m"]
    reset = "\033[0m"
    for row in matrix:
        for cell in row:
            if cell == -1:
                str_repr += " \u25A1 "
            if cell == 7:
                str_repr += f" {pin_colors[4]}\u25A1{reset} "
            elif cell == 8:
                str_repr += " . "
            elif cell == 9:
                str_repr += "   "
            elif cell >= 10:
                idx = int(cell//10) -1
                is_field = (cell % 10) == 0
                if is_field:
                    color = pin_colors[idx % len(pin_colors)]
                    str_repr += f" {color}\u25A1{reset} "
                else:
                    color = pin_colors[idx % len(pin_colors)]
                    str_repr += f" {color}{pin_rep[idx]}{reset} "
        str_repr += "\n"
    return str_repr

def animate_terminal(matrices, delay=0.25):
    '''
    Animiert eine Sequenz von Matrix-Darstellungen des MADN-Bretts im Terminal.
        Args:
            matrices: Eine Liste von Matrix-Darstellungen des Brettes
            delay: Die Verzögerung zwischen den Frames in Sekunden
        Returns:
            None
    '''
    for M in matrices:
        s = matrix_to_string(M) # Clear + Home
        sys.stdout.write(s)
        sys.stdout.flush()
        time.sleep(delay)
        os.system('cls')

def matrices_to_gif(matrices, path="madn_run.gif", scale=32):
    '''
    Erstellt ein GIF aus einer Sequenz von Matrix-Darstellungen des MADN-Bretts.
        Args:
            matrices: Eine Liste von Matrix-Darstellungen des Brettes
            path: Der Pfad, unter dem das GIF gespeichert wird
            scale: Die Skalierung der einzelnen Zellen im GIF
        
        Returns:
            Der Pfad zum gespeicherten GIF.
    '''
    from PIL import Image
    color_map = {
        -1: (240,240,240),  # leeres Feld
         8: (200,200,200),  # Leerplatz / Hintergrund
         9: (255,255,255),  # Abstand
         0: (50,120,255),
         1: (255,60,60),
         2: (255,200,40),
         3: (60,200,60),
         10: (0,80,150),
         11: (140,0,0),
         12: (140,100,0),
         13: (0,100,0)
    }
    frames = []
    for M in matrices:
        h, w = M.shape
        img = Image.new("RGB", (w*scale, h*scale), (0,0,0))
        px = img.load()
        for y in range(h):
            for x in range(w):
                mat_val = int(M[y,x])
                v = mat_val//10 if mat_val >= 10 else mat_val
                c = color_map.get(v, (0,0,0))
                for dy in range(scale):
                    for dx in range(scale):
                        px[x*scale+dx, y*scale+dy] = c
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=1)
    return path