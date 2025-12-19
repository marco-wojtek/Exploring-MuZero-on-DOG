import jax
import jax.numpy as jnp
from dog import *
import pytest

@pytest.mark.parametrize(
    "pins, player, pin, move, rules, expected_valid",
    [
      # Testfall 0: Figur aus start holen mit Zug 11, erfolgreich
      (jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(11),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 0, -1], [6, 14, 44, -1]])),
      # Testfall 1: Figur aus start holen mit Zug 13, erfolgreich
      (jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(13),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 0, -1, -1], [6, 14, 44, -1]])),
      # Testfall 2: Figur aus start holen mit Zug 11 Spieler 1, erfolgreich
      (jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(11),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, -1, -1], [6, 14, 44, 10]])),
      # Testfall 3: Figur aus start holen mit Zug 13 Spieler 1, erfolgreich
      (jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(13),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, -1, -1], [6, 14, 44, 10]])),
      # Testfall 4: Figur aus start holen mit Zug 13, start besetzt, nicht erfolgreich
      (jnp.array([[-1, -1, 0, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(3),
        jnp.array(13),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 0, -1], [6, 14, 44, -1]])),  
      # Testfall 5: Figur aus start holen mit Zug 13 Spieler 1, start besetzt, nicht erfolgreich
      (jnp.array([[-1, -1, 0, -1], [6, 10, 44, -1]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(13),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 0, -1], [6, 10, 44, -1]])),
      # Testfall 6: Figur aus start holen mit Zug 13, start besetzt von Gegner, erfolgreich
      (jnp.array([[-1, -1, -1, -1], [6, 14, 44, 0]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(13),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 0, -1], [6, 14, 44, -1]])),
      # Testfall 7: Figur aus start holen mit Zug 13, start besetzt von Gegner, erfolgreich
      (jnp.array([[-1, -1, 10, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(13),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, -1, -1], [6, 14, 44, 10]])), 
      # Testfall 8: Figur auf dem Feld normal bewegen, kein Gegner auf dem Weg
      (jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(1),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[13, 5, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 9: Figur auf dem Feld normal bewegen, eigener, Pin auf dem Weg
      (jnp.array([[12, 10, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(3),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[12, 13, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 10: Figur auf dem Feld normal bewegen, Gegner auf dem Weg
      (jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(5),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[17, 5, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 11: Figur auf dem Feld normal bewegen, eigener Pin auf Zielposition, nicht erfolgreich
      (jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(7),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]])),  
        # Testfall 12: Figur auf dem Feld normal bewegen, Gegner auf Zielposition, wird geschlagen
      (jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(2),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[14, 5, 40, -1], [6, -1, 44, -1]])),  
        # Testfall 13: Figur im Ziel Bewegen, gültig
      (jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(2),
        jnp.array(2),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[12, 5, 40, -1], [6, 14, 46, -1]])),  
      # Testfall 14: Figur im Ziel Bewegen, zu weit, nicht gültig
      (jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(2),
        jnp.array(4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[12, 5, 40, -1], [6, 14, 44, -1]])),  
        # Testfall 15: Figur von Board ins Ziel Bewegen, erfolgreich
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(5),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[41, 35, 40, -1], [6, 14, 44, -1]])),
        # Testfall 16: Figur von Board ins Ziel Bewegen, nicht erfolgreich, pin im weg
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 17: Figur von Board ins Ziel Bewegen, Spieler 1, erfolgreich
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(6),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [45, 14, 44, -1]])),   
        # Testfall 18: Figur von Board ins Ziel Bewegen, Spieler 1, nicht erfolgreich, pin im weg
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(5),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]])), 
        #### AB HIER WERDEN REGELN GETRESTET ####
        # Testfall 19: Ziel "unterlaufen"
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(3),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[0, 35, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 20: Ziel "unterlaufen", nicht circular
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(3),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]])),  
        # Testfall 21: Ziel "überlaufen"
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(8),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[5, 35, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 22: Ins Ziel, aber blockiert durch eigenen Pin
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[1, 35, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 23: Ziel "unterlaufen", Spieler 1
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [10, 14, 44, -1]])),  
        # Testfall 24: Ziel "unterlaufen", nicht circular, Spieler 1
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]])), 
        # Testfall 25: Ziel "überlaufen", Spieler 1
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(9),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [15, 14, 44, -1]])), 
        # Testfall 26: Ins Ziel, aber blockiert durch eigenen Pin, Spieler 1
      (jnp.array([[37, 35, 40, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(5),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, -1], [11, 14, 44, -1]])), 
        # Testfall 27: Ins Ziel
      (jnp.array([[37, 35, 42, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(5),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[41, 35, 42, -1], [6, 14, 44, -1]])),
        # Testfall 28: Ins Ziel, blockiert durch eigenen Pin, nicht circular
      (jnp.array([[37, 35, 42, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(7),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 42, -1], [6, 14, 44, -1]])),  
        # Testfall 29: Ins Ziel, blockiert durch eigenen Pin, circular
      (jnp.array([[37, 35, 42, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(7),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[4, 35, 42, -1], [6, 14, 44, -1]])), 
        # Testfall 30: Im Ziel, blockiert durch eigenen Pin
      (jnp.array([[40, 35, 42, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(3),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[40, 35, 42, -1], [6, 14, 44, -1]])),  
      # Testfall 31: Ins Ziel, Spieler 1
      (jnp.array([[37, 35, 42, -1], [6, 14, 46, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(5),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 42, -1], [44, 14, 46, -1]])),
        # Testfall 32: Ins Ziel, blockiert durch eigenen Pin, nicht circular, Spieler 1
      (jnp.array([[37, 35, 42, -1], [6, 14, 46, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(8),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 42, -1], [6, 14, 46, -1]])),  
        # Testfall 33: Ins Ziel, blockiert durch eigenen Pin, circular, Spieler 1
      (jnp.array([[37, 35, 42, -1], [6, 15, 46, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(8),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 42, -1], [14, 15, 46, -1]])), 
        # Testfall 34: Im Ziel, blockiert durch eigenen Pin, Spieler 1
      (jnp.array([[40, 35, 42, -1], [44, 14, 46, -1]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(3),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[40, 35, 42, -1], [44, 14, 46, -1]])), 
        # Testfall 35: Ziel "unterlaufen", start blocked
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(3),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, 0], [6, 14, 44, -1]])),  
        # Testfall 36: Ziel "überlaufen", start blocked
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(8),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, 0], [6, 14, 44, -1]])), 
        # Testfall 37: Ins Ziel, aber blockiert durch eigenen Pin, start blocked
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, 0], [6, 14, 44, -1]])), 
      # Testfall 38: Ins Ziel, aber start blocked
      (jnp.array([[37, 35, 43, 0], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(5),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 43, 0], [6, 14, 44, 10]])), 
      # Testfall 39: Ziel "unterlaufen", Spieler 1, start blocked
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]])),  
        # Testfall 40: Ziel "überlaufen", Spieler 1, start blocked
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(8),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]])), 
        # Testfall 41: Ins Ziel, aber blockiert durch eigenen Pin, Spieler 1, start blocked
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]]),
        jnp.array(1),
        jnp.array(0),
        jnp.array(4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]])),  
        # Testfall 42: Friendly Fire, normal
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(2),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': True},
        jnp.array([[-1, 37, 40, 0], [6, 14, 44, 10]])),   
         # Testfall 43: Friendly Fire, im Ziel
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': True},
        jnp.array([[1, 35, 40, 0], [6, 14, 44, 10]])),  
         # Testfall 44: Friendly Fire, start
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(3),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': True},
        jnp.array([[0, 35, 40, -1], [6, 14, 44, 10]])),   
        # Testfall 45: Friendly Fire, start, start blocked
      (jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(3),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': True},
        jnp.array([[37, 35, 40, 0], [6, 14, 44, 10]])),
      # Testfall 46: Move Pin on start board when start blocks
      (jnp.array([[37, 35, 3, 0], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(3),
        jnp.array(3),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': True},
        jnp.array([[37, 35, -1, 3], [6, 14, 44, 10]])),
        # Testfall 47: Move Pin on start board when start blocks, Spieler 1
      (jnp.array([[37, 35, 3, 0], [6, 14, 44, 10]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': True},
        jnp.array([[37, 35, 3, 0], [6, -1, 44, 14]])),
    ]
)
def test_normal_move(pins, player, pin, move, rules, expected_valid):
    env = env_reset(0, num_players=len(pins),
                    distance=jnp.int32(10),
                    enable_circular_board=rules['enable_circular_board'],
                    enable_jump_in_goal_area=rules['enable_jump_in_goal_area'],
                    enable_start_blocking=rules['enable_start_blocking'],
                    enable_friendly_fire=rules['enable_friendly_fire'])
    env.pins = pins
    env.board = set_pins_on_board(env.board, env.pins)
    env.current_player = player
    board, pins = step_normal_move(env, pin, move)
    print(pins)
    assert jnp.array_equal(pins, expected_valid)


@pytest.mark.parametrize(
    "pins, player, pin, move, rules, expected_valid",
    [
      # Testfall 0: Figur aus start holen mit Zug -4, nicht möglich
      (jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]])),
      # Testfall 1: Figur aus start holen mit Zug -4, nicht möglich, Spieler 1
      (jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, -1, -1], [6, 14, 44, -1]])),
      # Testfall 2: Figur normal bewegen mit Zug -4
      (jnp.array([[-1, -1, 13, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 9, -1], [6, 14, 44, -1]])),
      # Testfall 3: Mit -4 Gegner Schlagen
      (jnp.array([[-1, -1, 18, -1], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 14, -1], [6, -1, 44, -1]])),
        # Testfall 4: Mit -4 über Gegner laufen, nicht schlagen
      (jnp.array([[-1, -1, 18, -1], [6, 20, 44, -1]]),
        jnp.array(1),
        jnp.array(1),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 18, -1], [6, 16, 44, -1]])),
        # Testfall 5: Mit -4 im Ziel, nicht möglich
      (jnp.array([[-1, -1, 40, -1], [6, 20, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 40, -1], [6, 20, 44, -1]])),
        # Testfall 6: Mit -4 im Ziel, nicht möglich, Spieler 1
      (jnp.array([[-1, -1, 40, -1], [6, 20, 44, -1]]),
        jnp.array(1),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 40, -1], [6, 20, 44, -1]])),
        # Testfall 7: Mit -4 am eigenen Start vorbei, nicht zirkulär
      (jnp.array([[-1, -1, 2, -1], [6, 20, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 2, -1], [6, 20, 44, -1]])),
        # Testfall 8: Mit -4 am eigenen Start vorbei, Spieler 1, nicht zirkulär
      (jnp.array([[-1, -1, 40, -1], [6, 20, 12, -1]]),
        jnp.array(1),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 40, -1], [6, 20, 12, -1]])),
        # Testfall 9: Mit -4 am eigenen Start vorbei, zirkulär
      (jnp.array([[-1, -1, 2, -1], [6, 20, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 38, -1], [6, 20, 44, -1]])),
        # Testfall 10: Mit -4 am eigenen Start vorbei, Spieler 1, zirkulär
      (jnp.array([[-1, -1, 40, -1], [6, 20, 12, -1]]),
        jnp.array(1),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 40, -1], [6, 20, 8, -1]])),
        # Testfall 11: Mit -4 am eigenen Start vorbei, zirkulär, start blocked
      (jnp.array([[-1, 0, 2, -1], [6, 20, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[-1, 0, 2, -1], [6, 20, 44, -1]])),
        # Testfall 12: Mit -4 am eigenen Start vorbei, Spieler 1, zirkulär, start blocked
      (jnp.array([[-1, -1, 40, -1], [6, 20, 12, 10]]),
        jnp.array(1),
        jnp.array(2),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 40, -1], [6, 20, 12, 10]])),
        # Testfall 13: Mit -4 friendly fire
      (jnp.array([[1, 5, 40, -1], [6, 20, 12, 10]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': True},
        jnp.array([[-1, 1, 40, -1], [6, 20, 12, 10]])),
        # Testfall 14: Mit -4 friendly fire, von eigenem Startfeld, während start blocked
      (jnp.array([[36, 0, 40, -1], [6, 20, 12, 10]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': True},
        jnp.array([[-1, 36, 40, -1], [6, 20, 12, 10]])),
         # Testfall 15: Mit -4 friendly fire, Spieler 1
      (jnp.array([[1, 5, 40, -1], [6, 20, 12, 16]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': True},
        jnp.array([[1, 5, 40, -1], [6, 20, -1, 12]])),
        # Testfall 16: Mit -4 friendly fire, von eigenem Startfeld, Spieler 1
      (jnp.array([[1, 5, 40, -1], [6, 20, 12, 10]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(-4),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': True},
        jnp.array([[1, 5, 40, -1], [-1, 20, 12, 6]])),
    ]
)
def test_neg_move(pins, player, pin, move, rules, expected_valid):
    env = env_reset(0, num_players=len(pins),
                    distance=jnp.int32(10),
                    enable_circular_board=rules['enable_circular_board'],
                    enable_jump_in_goal_area=rules['enable_jump_in_goal_area'],
                    enable_start_blocking=rules['enable_start_blocking'],
                    enable_friendly_fire=rules['enable_friendly_fire'])
    env.pins = pins
    env.board = set_pins_on_board(env.board, env.pins)
    env.current_player = player
    board, pins = step_neg_move(env, pin, move)
    print(pins)
    assert jnp.array_equal(pins, expected_valid)

@pytest.mark.parametrize(
    "pins, player, pin, pos, rules, expected_valid",
    [
      # Testfall 0: Tausch mit Figur im Start
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(6),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]])),
        # Testfall 1: Tausch mit leerem Feld
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(12),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]])),
        # Testfall 2: Tausch mit eigene Ziel-Figur
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(6),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]])),
      # Testfall 3: Tausch mit geschützte Ziel-Figur
      (jnp.array([[-1, 1, 41, 38], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(10),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[-1, 1, 41, 38], [6, 14, 44, 10]])),
        # Testfall 4: Tausch mit geschützte Ziel-Figur, aber schutz disabled
      (jnp.array([[-1, 1, 41, 38], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(10),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, 1]])),
      # Testfall 5: Tausch mit Figur im Start, Spieler 1
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(10),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]])),
        # Testfall 6: Tausch mit leerem Feld, Spieler 1
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(1),
        jnp.array(1),
        jnp.array(11),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]])),
        # Testfall 7: Tausch mit eigene Ziel-Figur, Spieler 1
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(2),
        jnp.array(6),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]])),
      # Testfall 8: Tausch mit geschützte Ziel-Figur, Spieler 1
      (jnp.array([[-1, 1, 41, 38], [6, 14, 44, 10]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(1),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[-1, 1, 41, 38], [6, 14, 44, 10]])),
        # Testfall 9: Tausch mit geschützte Ziel-Figur, aber schutz disabled, Spieler 1
      (jnp.array([[-1, 1, 41, 38], [6, 14, 44, 10]]),
        jnp.array(1),
        jnp.array(3),
        jnp.array(1),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, 1]])),
        # Testfall 10: korrekter Tausch
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(6),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 6, 41, 38], [10, 14, 44, -1]])),
        # Testfall 11: korrekter Tausch
      (jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(38),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 10, 41, 38], [6, 14, 44, -1]])),
        # Testfall 12: Tausch mit eigene geschützte Figur
      (jnp.array([[-1, 0, 41, 38], [6, 14, 44, 10]]),
        jnp.array(0),
        jnp.array(1),
        jnp.array(14),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[-1, 0, 41, 38], [6, 14, 44, 10]])),
        # Testfall 13: Tausch mit eigene geschützte Figur, Spieler 1
      (jnp.array([[-1, 0, 41, 38], [6, 14, 44, 10]]),
        jnp.array(1),
        jnp.array(1),
        jnp.array(0),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[-1, 0, 41, 38], [6, 14, 44, 10]])),
    ]
)
def test_swap_move(pins, player, pin, pos, rules, expected_valid):
    env = env_reset(0, num_players=len(pins),
                    distance=jnp.int32(10),
                    enable_circular_board=rules['enable_circular_board'],
                    enable_jump_in_goal_area=rules['enable_jump_in_goal_area'],
                    enable_start_blocking=rules['enable_start_blocking'],
                    enable_friendly_fire=rules['enable_friendly_fire'])
    env.pins = pins
    env.board = set_pins_on_board(env.board, env.pins)
    env.current_player = player
    board, pins = step_swap(env, pin, pos)
    print(pins)
    assert jnp.array_equal(pins, expected_valid)

@pytest.mark.parametrize(
    "pins, player, dist, rules, expected_valid",
    [
      # Testfall 0: Ein pin aus start holen, nicht möglich
      (jnp.array([[-1, 41, 38, 6], [7, 14, 44, -1]]),
        jnp.array(0),
        jnp.array([1, 1, 3, 2]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 41, 38, 6], [7, 14, 44, -1]])),
        # Testfall 1: Ein pin aus start holen, nicht möglich, Spieler 1
      (jnp.array([[-1, 41, 38, 6], [7, 14, 44, -1]]),
        jnp.array(1),
        jnp.array([1, 1, 3, 2]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 41, 38, 6], [7, 14, 44, -1]])),
        # Testfall 2: Normal gültig, ohne schlagen
      (jnp.array([[-1, 41, 38, 6], [10, 14, 44, -1]]),
        jnp.array(0),
        jnp.array([0, 1, 1, 3]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 42, 39, 9], [10, 14, 44, -1]])),
        # Testfall 3: Normal gültig, mit schlagen (direkt)
      (jnp.array([[-1, 41, 38, 6], [9, 14, 44, -1]]),
        jnp.array(0),
        jnp.array([0, 1, 1, 3]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 42, 39, 9], [-1, 14, 44, -1]])),
        # Testfall 4: Normal gültig, mit schlagen (indirekt)
      (jnp.array([[-1, 41, 38, 6], [9, 14, 44, -1]]),
        jnp.array(0),
        jnp.array([0, 1, 1, 4]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 42, 39, 10], [-1, 14, 44, -1]])),
        # Testfall 5: Normal gültig, einer ins Ziel
      (jnp.array([[-1, 41, 38, 6], [9, 14, 44, -1]]),
        jnp.array(0),
        jnp.array([0, 1, 4, 4]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, 42, 41, 10], [-1, 14, 44, -1]])),
        # Testfall 6: Normal gültig, einer ins Ziel, schlägt eigene Figur
      (jnp.array([[-1, 40, 38, 6], [9, 14, 44, -1]]),
        jnp.array(0),
        jnp.array([0, 0, 4, 4]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[-1, -1, 41, 10], [-1, 14, 44, -1]])),
        # Testfall 7: Normal gültig, schlägt eigene Figur
      (jnp.array([[5, 40, 38, 6], [9, 14, 44, -1]]),
        jnp.array(0),
        jnp.array([5, 0, 1, 2]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[10, 40, 39, -1], [-1, 14, 44, -1]])),
        # Testfall 8: Normal gültig, schlägt alles
      (jnp.array([[5, 8, 7, 6], [9, 10, 11, 12]]),
        jnp.array(0),
        jnp.array([7, 0, 1, 2]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[12, -1, -1, -1], [-1, -1, -1, -1]])),
        # Testfall 9: Einer Start blocked
      (jnp.array([[5, 8, 7, 6], [9, 10, 11, 12]]),
        jnp.array(0),
        jnp.array([7, 0, 1, 2]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[5, 8, 7, 6], [9, 10, 11, 12]])),
        # Testfall 10: Einer Ziel blocked, aber circular
      (jnp.array([[37, 36, 41, 1], [9, 10, 11, 12]]),
        jnp.array(0),
        jnp.array([0, 7, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': False, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[-1, 3, 41, -1], [9, 10, 11, 12]])),
        # Testfall 11: Einer Ziel blocked, nicht circular
      (jnp.array([[37, 36, 41, 1], [9, 10, 11, 12]]),
        jnp.array(0),
        jnp.array([0, 7, 0, 0]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': False, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [9, 10, 11, 12]])),
        # Testfall 12: Ziel unterlaufen, nicht circular
      (jnp.array([[37, 36, 41, 1], [9, 10, 11, 12]]),
        jnp.array(0),
        jnp.array([0, 4, 0, 0]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [9, 10, 11, 12]])),
        # Testfall 13: Ziel überlaufen, nicht circular
      (jnp.array([[37, 36, 41, 1], [9, 10, 11, 12]]),
        jnp.array(0),
        jnp.array([8, 0, 0, 0]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [9, 10, 11, 12]])),
        # Testfall 14: Ziel unterlaufen, circular, Spieler 1
      (jnp.array([[37, 36, 41, 1], [9, 6, 11, 12]]),
        jnp.array(1),
        jnp.array([0, 4, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [-1, 10, 11, 12]])),
        # Testfall 15: Ziel überlaufen, circular, Spieler 1
      (jnp.array([[37, 36, 41, 1], [9, 13, 11, 12]]),
        jnp.array(1),
        jnp.array([6,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [15, -1, -1, -1]])),
        # Testfall 16: Läuft ins Ziel, schlägt pin auf dem eigenen Startfeld
      (jnp.array([[37, 36, 41, 0], [9, 13, 11, 12]]),
        jnp.array(0),
        jnp.array([4,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[40, 36, 41, -1], [9, 13, 11, 12]])),
        # Testfall 17: Läuft ins Ziel, schlägt Gegner pin auf dem eigenen Startfeld
      (jnp.array([[37, 36, 41, 1], [0, 13, 11, 12]]),
        jnp.array(0),
        jnp.array([4,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[40, 36, 41, 1], [-1, 13, 11, 12]])),
        # Testfall 18: Läuft ins Ziel, schlägt Gegner pin auf dem eigenen Startfeld, Spieler 1
      (jnp.array([[37, 36, 41, 1], [8, 13, 10, 12]]),
        jnp.array(1),
        jnp.array([3,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [44, 13, -1, 12]])),
        # Testfall 19: Läuft ins Ziel, schlägt Gegner pin auf dem eigenen Startfeld, Spieler 1
      (jnp.array([[37, 36, 41, 10], [8, 13, 1, 12]]),
        jnp.array(1),
        jnp.array([3,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, -1], [44, 13, 1, 12]])),
         # Testfall 20: Ziel überlaufen, circular, Spieler 1
      (jnp.array([[37, 36, 15, 10], [9, 13, 11, 12]]),
        jnp.array(1),
        jnp.array([6,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 36, -1, -1], [15, -1, -1, -1]])),
        # Testfall 21: Läuft ins Ziel, schlägt pin auf dem eigenen Startfeld nicht, wegen blocking
      (jnp.array([[37, 36, 41, 0], [9, 13, 11, 12]]),
        jnp.array(0),
        jnp.array([4,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 0], [9, 13, 11, 12]])),
        # Testfall 22: Läuft ins Ziel, schlägt Gegner pin auf dem eigenen Startfeld, block regel an aber hier nicht relevant
      (jnp.array([[37, 36, 41, 1], [0, 13, 11, 12]]),
        jnp.array(0),
        jnp.array([4,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[40, 36, 41, 1], [-1, 13, 11, 12]])),
        # Testfall 23: Läuft ins Ziel, aber start blocked, Spieler 1
      (jnp.array([[37, 36, 41, 1], [9, 13, 10, 12]]),
        jnp.array(1),
        jnp.array([4,0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': True, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [9, 13, 10, 12]])),
        # Testfall 24: Einer Ziel blocked, aber circular, Spieler 1
      (jnp.array([[37, 36, 41, 1], [9, 2, 45, 12]]),
        jnp.array(1),
        jnp.array([4, 0, 0, 0]),
        {'enable_circular_board': True, 'enable_jump_in_goal_area': False, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [13, 2, 45, -1]])),
        # Testfall 25: Einer Ziel blocked, nicht circular, Spieler 1
      (jnp.array([[37, 36, 41, 1], [9, 2, 45, 12]]),
        jnp.array(1),
        jnp.array([4, 0, 0, 0]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': False, 'enable_start_blocking': True, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [9, 2, 45, 12]])),
        # Testfall 26: Ziel unterlaufen, nicht circular, Spieler 1
      (jnp.array([[37, 36, 41, 1], [9, 2, 45, 12]]),
        jnp.array(1),
        jnp.array([1, 0, 0, 0]),
        {'enable_circular_board': False, 'enable_jump_in_goal_area': False, 'enable_start_blocking': False, 'enable_friendly_fire': False},
        jnp.array([[37, 36, 41, 1], [9, 2, 45, 12]])),
    ]
)
def test_7_move(pins, player, dist, rules, expected_valid):
    env = env_reset(0, num_players=len(pins),
                    distance=jnp.int32(10),
                    enable_circular_board=rules['enable_circular_board'],
                    enable_jump_in_goal_area=rules['enable_jump_in_goal_area'],
                    enable_start_blocking=rules['enable_start_blocking'],
                    enable_friendly_fire=rules['enable_friendly_fire'])
    env.pins = pins
    env.board = set_pins_on_board(env.board, env.pins)
    env.current_player = player
    board, pins = step_hot_7(env, dist)
    print(pins)
    assert jnp.array_equal(pins, expected_valid)