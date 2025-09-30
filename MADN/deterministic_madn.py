import chex
import jax
import jax.numpy as jnp
import mctx

Board = chex.Array
Action = chex.Array
Player = chex.Array
Reward = chex.Array
Done = chex.Array

@chex.dataclass
class deterministic_MADN:
    board: Board  # shape (11, 11), values in {0, 1, -1} for empty, player 1, player -1
    current_player: Player  # scalar, 1 or -1
    reward: Reward  # scalar, reward for the current player
    done: Done  # scalar, whether the game is over