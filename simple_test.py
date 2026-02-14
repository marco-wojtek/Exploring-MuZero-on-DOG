from jax import numpy as jnp
import jax
from MADN.deterministic_madn import env_reset, encode_board

env = env_reset(0, num_players=4, layout=jnp.array([True, True, True, True]), distance=10, starting_player=0, seed=0, enable_teams=True, enable_initial_free_pin=True, enable_circular_board=False)
env = env.replace(action_set=env.action_set.at[1, :].set(0), current_player=1)  # Setze Spieler 1 auf eine Position ohne gültige Aktionen
obs = encode_board(env)
print("Encoded Board Shape:", obs.shape)  # Sollte (features, board_size) sein
print(obs[9:20])