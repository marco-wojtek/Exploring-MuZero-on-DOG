import jax
import jax.numpy as jnp

def board_to_matrix(env):
    board = env.board
    num_players = int(env.num_players)
    n = int(env.board_size // num_players)
    board_str = ""
    if num_players == 2:
        board_matrix = jnp.ones((3, n))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n])
        board_matrix = board_matrix.at[1, :].set(jnp.ones(n)*9)
        board_matrix = board_matrix.at[2, :].set(jnp.flip(board[n:2*n]))
    elif num_players == 3:
        board_matrix = jnp.ones((n+1, n+1))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n+1])
        board_matrix = board_matrix.at[jnp.arange(n+1), n - jnp.arange(n+1)].set(board[n:2*n+1])
        board_matrix = board_matrix.at[1:, 0].set(jnp.flip(board[2*n:3*n]))
    elif num_players == 4:
        board_matrix = jnp.ones((n+1, n+1))*8
        board_matrix = board_matrix.at[0, :].set(board[0:n+1])
        board_matrix = board_matrix.at[:, -1].set(board[n:2*n+1])
        board_matrix = board_matrix.at[-1, :].set(jnp.flip(board[2*n:3*n+1]))
        board_matrix = board_matrix.at[1:, 0].set(jnp.flip(board[3*n:4*n]))
    return board_matrix

def matrix_to_string(matrix):
    str_repr = ""
    pin_rep = ["♠", "♥", "♦", "♣"]
    for row in matrix:
        for cell in row:
            if cell == -1:
                str_repr += " \u25A1 "
            elif cell == 8:
                str_repr += " . "
            elif cell == 9:
                str_repr += "  "
            else:
                str_repr += f" {pin_rep[int(cell)]} "
        str_repr += "\n"
    return str_repr