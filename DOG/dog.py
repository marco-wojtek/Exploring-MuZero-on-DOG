import chex
import jax
import jax.numpy as jnp
import mctx

Board = chex.Array
Start = chex.Array
Target = chex.Array
Goal = chex.Array
Action = chex.Array
Player = chex.Array
Reward = chex.Array
Done = chex.Array
Pins = chex.Array
Num_players = chex.Array
Size = chex.Array
Deck = chex.Array
Card = chex.Array
Hand = chex.Array

@chex.dataclass
class DOG:
    board: Board
    num_players: Num_players
    current_player: Player 
    pins : Pins 
    reward: Reward 
    done: Done  
    start: Start  
    target: Target  
    goal: Goal  
    deck: Deck  
    hands: Hand
    board_size: Size 
    total_board_size: Size  
    rules : dict  

def env_reset(
        _,
        num_players=jnp.int8(4),
        distance=jnp.int8(10),
        enable_initial_free_pin = False,
        enable_circular_board = True,
        enable_friendly_fire = False,
        enable_dice_rethrow = False,
        disable_swapping = False,
        disable_hot_seven = False,
        disable_joker = False,
            ) -> DOG:
    
    board_size = num_players * distance
    total_board_size = board_size + num_players * 4 # add goal areas
    num_pins = 4

    start = jnp.array(jnp.arange(num_players)*distance, dtype=jnp.int8)
    pins = - jnp.ones((num_players,num_pins), dtype=jnp.int8)
    pins = jax.lax.cond(
        enable_initial_free_pin,
        lambda: pins.at[:,0].set(start),
        lambda: pins
    )
    board = - jnp.ones(total_board_size, dtype=jnp.int8)
    board = jax.lax.cond(
        enable_initial_free_pin,
        lambda: set_pins_on_board(board, pins),
        lambda: board
    )

    # prepare deck
    num_cards = 14 - disable_joker.astype(jnp.int8) - disable_hot_seven.astype(jnp.int8) - disable_swapping.astype(jnp.int8)
    deck = jnp.ones(num_cards, dtype=jnp.int8)*8
    deck = deck.at[0].set(6 + (2*disable_joker.astype(jnp.int8)))  # Idx 0 ist nur joker wenn dieser enabled ist

    return DOG(
        board = board, # board is filled with -1 (empty) or 0-3 (player index)
        num_players = jnp.array(num_players, dtype=jnp.int8), # number of players
        pins = pins,
        current_player=jnp.array(0, dtype=jnp.int8), # index of current player, 0-3
        done = jnp.bool_(False), # whether the game is over
        reward=jnp.array(0, dtype=jnp.int8), # reward for the current player
        start = start,
        target = jnp.array((jnp.arange(num_players)*distance - 1)%board_size, dtype=jnp.int8),
        goal = jnp.reshape(jnp.arange(board_size, board_size + num_players*4, dtype=jnp.int8), (num_players, 4)),
        deck = deck,
        hands = jnp.zeros((num_players, num_cards), dtype=jnp.int8),
        board_size=jnp.array(board_size, dtype=jnp.int8),
        total_board_size=jnp.array(total_board_size, dtype=jnp.int8),
        rules = {
        'enable_initial_free_pin':enable_initial_free_pin,
        'enable_circular_board':enable_circular_board,
        'enable_friendly_fire':enable_friendly_fire,
        'enable_dice_rethrow':enable_dice_rethrow,
        'disable_swapping': disable_swapping,
        'disable_hot_seven': disable_hot_seven,
        'disable_joker': disable_joker,
        }
    )

@jax.jit
def set_pins_on_board(board, pins):
    num_players, num_pins = pins.shape

    def body(idx, board):
        player = idx // num_pins
        pin = idx % num_pins
        pos = pins[player, pin]
        board = jax.lax.cond(
            pos != -1,
            lambda b: b.at[pos].set(player),
            lambda b: b,
            board
        )
        return board

    board = jax.lax.fori_loop(0, num_players * num_pins, body, board)
    return board

def no_step(env:DOG) ->  DOG:
    """
    No-op step function for the environment.
    On no step the hand of the current player is emptied and the turn passes to the next player.
    """               
    hand_cards = jnp.sum(env.hands, axis=1) 
    def body(i, pnext):
        cand = (env.current_player + i + 1) % env.num_players
        take = (pnext == -1) & (hand_cards[cand] > 0)
        return jnp.where(take, cand, pnext)
    next_player = jax.lax.fori_loop(0, env.num_players, body, -jnp.array(1, dtype=jnp.int8))  
    print("Next player found:", next_player)
    env = DOG(
        board=env.board,
        num_players=env.num_players,
        pins=env.pins,
        current_player=next_player,
        done=env.done,
        reward=env.reward,
        start=env.start,
        target=env.target,
        goal=env.goal,
        deck=env.deck,
        hands=env.hands.at[env.current_player].set(jnp.zeros(len(env.deck), dtype=jnp.int8)),
        board_size=env.board_size,
        total_board_size=env.total_board_size,
        rules=env.rules,
    )
    return env, jnp.array(0, dtype=jnp.int8), env.done