import jax
import jax.numpy as jnp
import sys, os, time

def board_to_matrix(env):
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
    board = env.board
    num_players = int(env.num_players)
    n = int(env.board_size // 4)

    layout = jax.lax.cond(
        (jnp.sum(layout)!=num_players) | (jnp.all(layout) & (num_players < 4)),
        lambda: jnp.array([False, False, False, False], dtype=jnp.bool_).at[:num_players].set(True),
        lambda: layout
    )

    start = env.start
    start_area = replace_rows_simple(layout, board[start])
    goal = env.goal
    goal_area = replace_rows_simple(layout, board[goal])

    colour_ids = replace_rows_simple(layout, jnp.arange(10, 14, dtype=jnp.int8))
    colour_ids = jnp.where(
        colour_ids == -1,
        14,
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

    return board_matrix

def matrix_to_string(matrix):
    str_repr = ""
    pin_rep = ["♠", "♥", "♦", "♣"]
    pin_colors = ["\033[94m", "\033[91m", "\033[93m", "\033[92m", "\033[90m"]
    reset = "\033[0m"
    for row in matrix:
        for cell in row:
            if cell == -1:
                str_repr += " \u25A1 "
            elif cell == 8:
                str_repr += " . "
            elif cell == 9:
                str_repr += "   "
            elif cell >= 10:
                idx = int(cell-10)
                color = pin_colors[idx % len(pin_colors)]
                str_repr += f" {color}\u25A1{reset} "
            else:
                idx = int(cell)
                color = pin_colors[idx % len(pin_colors)]
                str_repr += f" {color}{pin_rep[idx]}{reset} "
        str_repr += "\n"
    return str_repr

def animate_terminal(matrices, delay=0.25):
    for M in matrices:
        s = matrix_to_string(M) # Clear + Home
        sys.stdout.write(s)
        sys.stdout.flush()
        time.sleep(delay)
        os.system('cls')

def matrices_to_gif(matrices, path="madn_run.gif", scale=32):
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
                v = int(M[y,x])
                c = color_map.get(v, (0,0,0))
                for dy in range(scale):
                    for dx in range(scale):
                        px[x*scale+dx, y*scale+dy] = c
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=1)
    return path