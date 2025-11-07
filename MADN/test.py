"""
Tests for deterministic MADN implementation.
Covers various move scenarios to ensure correctness of game logic.
"""
import pytest
import jax
import jax.numpy as jnp
from deterministic_madn import *

'''
Every move must also test:
- updating the board state correctly
    - new board positions
    - removing hit opponent pins
    - old board position of moved pins
- updating the pin positions correctly
    - new pin positions
    - removing hit opponent pins
    - old pin positions of moved pins
- updating the action set correctly
- handling turn changes correctly
- detecting win conditions correctly
'''

@pytest.mark.parametrize("pin", range(4))
@pytest.mark.parametrize("move", range(1, 7))
@pytest.mark.parametrize("player", range(4))
def test_start_moves(pin, move, player):
    '''
    Test moving out of the start area:
    - moving out of start area to starting position
    - hitting opponent pin on starting position
    - blocked by own pin on starting position
    '''
    env = env_reset(0, num_players=jnp.int8(4), distance=jnp.int8(10))
    env.current_player = jnp.int8(player)
    action = jnp.array([pin, move], dtype=jnp.int8)
    is_valid = valid_action(env)[pin, move-1]
    env_new, reward, done = env_step(env, action)
    
    # Test: Nach gültigem Zug ist Pin nicht mehr im Startbereich (wenn er vorher draußen war)
    if is_valid and env.pins[env.current_player, pin] != -1:
        assert env_new.pins[env.current_player, pin] != -1
        assert env_new.board[env_new.pins[env.current_player, pin]] == env.current_player

    # Test: Nach ungültigem Zug bleibt alles gleich
    if not is_valid:
        assert jnp.all(env_new.board == env.board)
        assert jnp.all(env_new.pins == env.pins)

    # Test: Keine doppelten Pins auf dem Board
    unique, counts = jnp.unique(env_new.board[env_new.board != -1], return_counts=True)
    assert jnp.all(counts == 1)

def normal_move():
    '''
    Test normal move cases:
    - moving on the board
    - hitting opponent pins
    - blocked by own pins
    - moving into goal area
    '''
    pass

def goal_move():
    '''
    Test goal move cases:
    - moving within goal area
    - hitting opponent pins in goal area
    - blocked by own pins in goal area
    - overshooting goal area
    '''
    pass