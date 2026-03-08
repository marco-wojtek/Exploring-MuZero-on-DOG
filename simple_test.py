from jax import numpy as jnp
import jax
from DOG.dog import env_reset, valid_actions, env_step, get_play_action_size, map_action_to_move, set_pins_on_board, valid_step_actions

env = env_reset(0, num_players=4,
                    distance=jnp.int32(10),
                    enable_circular_board=True,
                    enable_jump_in_goal_area=True,
                    enable_start_blocking=True,
                    enable_friendly_fire=False,
                    enable_teams=True)
env = env.replace(pins=jnp.array([[-1, 10, 7, 3], [12, 20, 21, 8], [-1, -1, -1, -1], [-1, -1, -1, -1]]), board=set_pins_on_board(env.board, jnp.array([[-1, 10, 7, 3], [12, 20, 21, 8], [-1, -1, -1, -1], [-1, -1, -1, -1]])), current_player=0)
print(env.hands)
print(env.phase)
print(valid_step_actions(env))
# env, _, _ = env_step(env, jnp.array(get_play_action_size(env)+4))
# print(env.swap_choices)
# env, _, _ = env_step(env, jnp.array(get_play_action_size(env)))
# print(env.swap_choices)
# env, _, _ = env_step(env, jnp.array(get_play_action_size(env)+5))
# print(env.swap_choices)
# env, _, _ = env_step(env, jnp.array(get_play_action_size(env)+1))
# print(env.swap_choices)
# print(env.hands)
# print(env.phase)
# print(map_action_to_move(env, jnp.array(757)))
# env, r, d = env_step(env, jnp.array(757))
# print(env.pins)
# print(env.current_player)
# print(env.hands)