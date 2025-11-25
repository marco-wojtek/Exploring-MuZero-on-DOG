import jax
import jax.numpy as jnp
import sys, os, time

def board_to_matrix(env):
    board = env.board
    num_players = int(env.num_players)
    n = int(env.board_size // num_players)
    goal = env.goal
    if num_players == 2:
        board_matrix = jnp.ones((3, n))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n])
        board_matrix = board_matrix.at[1, :].set(jnp.ones(n)*9)
        board_matrix = board_matrix.at[2, :].set(jnp.flip(board[n:2*n]))

        board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
        board_matrix = board_matrix.at[-1, 2:6].set(board[goal[0]])
        board_matrix = board_matrix.at[0, -6: -2].set(jnp.flip(board[goal[1]]))
    elif num_players == 3:
        board_matrix = jnp.ones((n+1, n+1))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n+1])
        board_matrix = board_matrix.at[jnp.arange(n+1), n - jnp.arange(n+1)].set(board[n:2*n+1])
        board_matrix = board_matrix.at[1:, 0].set(jnp.flip(board[2*n:3*n]))

        board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
        board_matrix = board_matrix.at[2:6, 0].set(board[goal[0]])
        board_matrix = board_matrix.at[0, -6: -2].set(jnp.flip(board[goal[1]]))
        board_matrix = board_matrix.at[-3, 3:7].set(board[goal[2]])
    elif num_players == 4:
        board_matrix = jnp.ones((n+1, n+1))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n+1])
        board_matrix = board_matrix.at[:, -1].set(board[n:2*n+1])
        board_matrix = board_matrix.at[-1, :].set(jnp.flip(board[2*n:3*n+1]))
        board_matrix = board_matrix.at[1:, 0].set(jnp.flip(board[3*n:4*n]))

        board_matrix = jnp.pad(board_matrix, ((1,1),(1,1)), constant_values=9)
        board_matrix = board_matrix.at[2:6, 0].set(board[goal[0]])
        board_matrix = board_matrix.at[0, -6: -2].set(jnp.flip(board[goal[1]]))
        board_matrix = board_matrix.at[-6:-2, -1].set(jnp.flip(board[goal[2]]))
        board_matrix = board_matrix.at[-1, 2:6].set(board[goal[3]])

    return board_matrix

def matrix_to_string(matrix):
    str_repr = ""
    pin_rep = ["♠", "♥", "♦", "♣"]
    pin_colors = ["\033[94m", "\033[91m", "\033[93m", "\033[92m"]
    reset = "\033[0m"
    for row in matrix:
        for cell in row:
            if cell == -1:
                str_repr += " \u25A1 "
            elif cell == 8:
                str_repr += " . "
            elif cell == 9:
                str_repr += "   "
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
         3: (60,200,60)
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