import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import os
import optax
import TicTacToeV2 as ttt_v2
import TicTacToe as ttt
import functools
from train import SimplePolicy

def load_checkpoint(path: str, params_template, opt_state_template=None):
    """
    Lädt params in params_template und optional opt_state in opt_state_template.
    Rückgabe: (params, opt_state_or_None)
    """
    with open(path + ".params", "rb") as f:
        params_bytes = f.read()
    params = flax.serialization.from_bytes(params_template, params_bytes)
    opt_state = None
    if opt_state_template is not None and os.path.exists(path + ".opt"):
        with open(path + ".opt", "rb") as f:
            opt_bytes = f.read()
        opt_state = flax.serialization.from_bytes(opt_state_template, opt_bytes)
    return params, opt_state

def get_trained_action(policy_apply_fn, params, board):
    """Erhält die beste Aktion vom trainierten Modell."""
    logits = policy_apply_fn(params, board)
    valid_mask = (board.flatten() == 0)
    logits = jnp.where(valid_mask, logits, -jnp.inf)
    return jnp.argmax(logits)

def get_random_action(board, rng_key):
    """Wählt eine zufällige, gültige Aktion."""
    valid_actions = jnp.where(board.flatten() == 0)[0]
    return jax.random.choice(rng_key, valid_actions)

def play_match(game, trained_params, rng_key, trained_player=1):
    """Spielt eine Partie: Trainierter Agent vs. Random Bot."""
    env = game.env_reset(0)
    
    while not env.done:
        rng_key, action_key = jax.random.split(rng_key)
        
        if env.current_player == trained_player:
            action = get_trained_action(SimplePolicy().apply, trained_params, env.board)
            # print("Trained player: ", action)
        else:
            action = get_random_action(env.board, action_key)
            # print("Random player: ", action)
            
        env, _, _ = game.env_step(env, action.astype(jnp.int8))
        # print("Current board:\n", env.board)
        # print(env.done)
        # print(jnp.all(env.board != 0))
    
    # Gibt den Gewinner zurück (1 für trainierten Agent, -1 für Random Bot, 0 für Unentschieden)
    # print(f"Final board:\n", env.board, "\n-----------------------------\n")
    return game.get_winner(env.board) * trained_player

def play_random_match(game, rng_key, games=100):
    """Spielt eine Partie: Random Bot vs. Random Bot."""
    print(f"\nStarte Evaluation Random Bot gegen Random Bot ({games} Partien)...")

    one = 0
    neg_one = 0
    for _ in range(games):
        env = game.env_reset(0)
        while not env.done:
            rng_key, action_key = jax.random.split(rng_key)
            action = get_random_action(env.board, action_key)
            env, _, _ = game.env_step(env, action.astype(jnp.int8))
    
        winner = game.get_winner(env.board)
        if winner == 1:
            one += 1
        elif winner == -1:
            neg_one += 1

    print(f"Random Bot 1: {one/games}, Random Bot -1: {neg_one/games}, Draw: {(games - one - neg_one)/games}")

def evaluate_agent(game, trained_params, num_matches=1000):
    """Evaluiert den trainierten Agenten über viele Partien."""
    print(f"\nStarte Evaluation gegen Random Bot ({num_matches} Partien)...")
    rng_key = jax.random.PRNGKey(1)
    
    wins = 0
    losses = 0
    draws = 0
    
    # Spiele als Spieler 1 (beginnt)
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = play_match(game, trained_params, game_key, trained_player=1)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
            
    # Spiele als Spieler 2
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = play_match(game, trained_params, game_key, trained_player=-1)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
            
    win_rate = wins / num_matches
    loss_rate = losses / num_matches
    draw_rate = draws / num_matches
    
    print("\n--- Evaluationsergebnis ---")
    print(f"Gewinnrate: {win_rate:.2%}")
    print(f"Verlustrate: {loss_rate:.2%}")
    print(f"Unentschieden: {draw_rate:.2%}")
    print("---------------------------\n")


# game = ttt_v2
game = ttt_v2
print(f"{game.__name__} selected for evaluation.")
policy = SimplePolicy()
dummy = jnp.zeros((3,3))
params_template = policy.init(jax.random.PRNGKey(0), dummy)
print("Begin Evaluation")
play_random_match(game, jax.random.PRNGKey(1), games=1000)
name = f"{game.__name__}_cp_1k_e3"
path = "C:\\Users\\marco\\Informatikstudium\\Master\\Masterarbeit\\Exploring-MuZero-on-DOG\\TicTacToe\\Checkpoints\\" + name
params, opt_state = load_checkpoint(path, params_template)
evaluate_agent(game, params, 1000)