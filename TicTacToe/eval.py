import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import os
import optax
import TicTacToeV2 as ttt_v2
import TicTacToe as ttt
import functools
from train import ConvTicTacToeNet, ImprovedTicTacToeNet, SimplePolicy, LargerNN
from mcts import run_mcts, run_gumbel

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

def get_mcts_action(env, board, rng_key, num_simulations):
    """Erhält die beste Aktion vom MCTS-Algorithmus."""
    policy_output = run_mcts(rng_key, env, num_simulations=num_simulations)
    action_weights = policy_output.action_weights.mean(axis=0)
    valid_mask = (board.flatten() == 0)
    action_weights = jnp.where(valid_mask, action_weights, -jnp.inf)
    return jnp.argmax(action_weights)

def get_gumbel_action(env, board, rng_key, num_simulations):
    """Wählt eine Aktion basierend auf Gumbel-Softmax Sampling."""
    policy_output = run_gumbel(rng_key, env, num_simulations=num_simulations)
    action_weights = policy_output.action_weights.mean(axis=0)
    valid_mask = (board.flatten() == 0)
    action_weights = jnp.where(valid_mask, action_weights, -jnp.inf)
    return jnp.argmax(action_weights)

def get_trained_action(policy_apply_fn, params, board):
    """Erhält die beste Aktion vom trainierten Modell."""
    logits = policy_apply_fn(params, board)
    valid_mask = (board.flatten() == 0)
    logits = jnp.where(valid_mask, logits, -jnp.inf)
    return jnp.argmax(logits)

def get_random_action(board, rng_key):
    """Wählt eine zufällige, gültige Aktion."""
    logits = jnp.where(board.flatten() == 0, 0.0, -jnp.inf)
    action = jax.random.categorical(rng_key, logits)
    return action

def play_parallel_match(network, game, batch_size, trained_params, rng_key, trained_player=1):
    """Spielt mehrere Partien parallel: Trainierter Agent vs. Random Bot."""
    envs = jax.vmap(game.env_reset)(jnp.zeros((batch_size, ), dtype=jnp.int8))
    step = 0
    limit = 30
    done_mask = jnp.zeros((batch_size,), dtype=bool)
    
    def cond(a):
        envs, key, step, done_mask = a
        return jnp.logical_and(~jnp.all(done_mask), step < limit)
    
    def step_fn(a):
        envs, key, step, done_mask = a
        key, subkey = jax.random.split(key)
        
        def single_step(env, key, done):
            key, action_key = jax.random.split(key)
            def trained_move():
                return get_trained_action(network.apply, trained_params, env.board)
            def random_move():
                return get_random_action(env.board, action_key)
            def mcts_move():
                return get_mcts_action(network.apply, trained_params, env.board, action_key)
            action = jax.lax.cond(env.current_player == trained_player, trained_move, mcts_move)
            env, _, _ = game.env_step(env, action.astype(jnp.int8))
            return env, key
        
        envs, keys = jax.vmap(single_step)(envs, jax.random.split(subkey, batch_size), done_mask)
        done_mask = done_mask | envs.done
        return envs, key, step + 1, done_mask
    
    leaf, key, final_step, done_mask = jax.lax.while_loop(cond, step_fn, (envs, rng_key, step, done_mask))
    
    def get_winner_result(env):
        return game.get_winner(env.board) * trained_player
    
    winners = jax.vmap(get_winner_result)(leaf)
    return winners


def play_match(network, game, trained_params, rng_key, trained_player=1, num_simulations=0):
    """Spielt eine Partie: Trainierter Agent vs. Random Bot."""
    env = game.env_reset(0)
    step = 0
    limit = 30
    while not env.done and step < limit:
        rng_key, action_key = jax.random.split(rng_key)
        
        if env.current_player == trained_player:
            action = get_trained_action(network.apply, trained_params, env.board)
            # print("Trained player: ", action)
        else:
            if num_simulations == 0:
                action = get_random_action(env.board, action_key)
            else:
                action = get_mcts_action(env, env.board, action_key, num_simulations)
            # print("Random player: ", action)
            
        env, _, _ = game.env_step(env, action.astype(jnp.int8))
        # print("Current board:\n", env.board)
        # print(env.done)
        # print(jnp.all(env.board != 0))
        step += 1
    
    # Gibt den Gewinner zurück (1 für trainierten Agent, -1 für Random Bot, 0 für Unentschieden)
    # print(f"Final board:\n", env.board, "\n-----------------------------\n")
    if step == limit:
        return 0
    return game.get_winner(env.board) * trained_player

def play_random_match(game, rng_key, limit, games=100):
    """Spielt eine Partie: Random Bot vs. Random Bot."""
    print(f"\nStarte Evaluation Random Bot gegen Random Bot ({games} Partien)...")

    one = 0
    neg_one = 0
    
    for _ in range(games):
        env = game.env_reset(0)
        step = 0
        while not env.done and step < limit:
            rng_key, action_key = jax.random.split(rng_key)
            action = get_random_action(env.board, action_key)
            env, _, _ = game.env_step(env, action.astype(jnp.int8))
            step += 1

        winner = game.get_winner(env.board)
        if winner == 1:
            one += 1
        elif winner == -1:
            neg_one += 1

    print(f"Random Bot 1: {one/games}, Random Bot -1: {neg_one/games}, Draw: {(games - one - neg_one)/games}")

def play_mcts_match(game, rng_key, limit, num_simulations, games=100, gumbel=False):
    """Spielt eine Partie: MCTS Bot vs. MCTS Bot."""
    print(f"\nStarte Evaluation MCTS Bot gegen MCTS Bot mit {num_simulations} Simulationen ({games} Partien)...")

    one = 0
    neg_one = 0
    
    for _ in range(games):
        env = game.env_reset(0)
        step = 0
        while not env.done and step < limit:
            rng_key, action_key = jax.random.split(rng_key)
            if gumbel:
                action = get_gumbel_action(env, env.board, action_key, num_simulations)
            else:
                action = get_mcts_action(env, env.board, action_key, num_simulations)
            env, _, _ = game.env_step(env, action.astype(jnp.int8))
            step += 1

        winner = game.get_winner(env.board)
        if winner == 1:
            one += 1
        elif winner == -1:
            neg_one += 1

    print(f"MCTS Bot 1: {one/games}, MCTS Bot -1: {neg_one/games}, Draw: {(games - one - neg_one)/games}")

def evaluate_agent(network, game, trained_params, num_matches=1000, num_simulations=0):
    """Evaluiert den trainierten Agenten über viele Partien."""
    if num_simulations == 0:
        print(f"\nStarte Evaluation gegen Random Bot ({num_matches} Partien)...")
    else:
        print(f"\nStarte Evaluation gegen MCTS Bot mit {num_simulations} Simulationen ({num_matches} Partien)...")
    rng_key = jax.random.PRNGKey(1)
    
    wins = 0
    losses = 0
    draws = 0
    
    # Spiele als Spieler 1 (beginnt)
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = play_match(network, game, trained_params, game_key, trained_player=1, num_simulations=num_simulations)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    
    print(f"Ergebnis als Spieler 1 {num_matches//2} Partien: Wins: {wins}, Losses: {losses}, Draws: {draws}")
    # Spiele als Spieler 2
    wins2 = 0
    losses2 = 0
    draws2 = 0
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = play_match(network, game, trained_params, game_key, trained_player=-1, num_simulations=num_simulations)
        if winner == 1:
            wins2 += 1
        elif winner == -1:
            losses2 += 1
        else:
            draws2 += 1

    print(f"Ergebnis als Spieler 2 {num_matches//2} Partien: Wins: {wins2}, Losses: {losses2}, Draws: {draws2}")

    win_rate = (wins + wins2) / num_matches
    loss_rate = (losses + losses2) / num_matches
    draw_rate = (draws + draws2) / num_matches

    print("\n--- Evaluationsergebnis ---")
    print(f"Gewinnrate: {win_rate:.2%}")
    print(f"Verlustrate: {loss_rate:.2%}")
    print(f"Unentschieden: {draw_rate:.2%}")
    print("---------------------------\n")

game = ttt_v2

policy = ImprovedTicTacToeNet()
dummy = jnp.zeros((3,3))
params_template = policy.init(jax.random.PRNGKey(0), dummy)
# play_random_match(game, jax.random.PRNGKey(1), limit=20, games=100)
name = f"{game.__name__}_imp_net_3000ep_00001lr"   
path = "C:\\Users\\marco\\Informatikstudium\\Master\\Masterarbeit\\Exploring-MuZero-on-DOG\\TicTacToe\\Checkpoints\\" + name
params, opt_state = load_checkpoint(path, params_template)
print(f"Begin Evaluation - {game.__name__} - with random policy")
evaluate_agent(policy, game, params, 1000, num_simulations=0)
num_simulations = 5
print(f"Begin Evaluation - {game.__name__} - with {num_simulations} MCTS simulations")
evaluate_agent(policy, game, params, 1000, num_simulations=num_simulations)
num_simulations = 10
print(f"Begin Evaluation - {game.__name__} - with {num_simulations} MCTS simulations")
evaluate_agent(policy, game, params, 1000, num_simulations=num_simulations)
num_simulations = 30
print(f"Begin Evaluation - {game.__name__} - with {num_simulations} MCTS simulations")
evaluate_agent(policy, game, params, 1000, num_simulations=num_simulations)
# play_mcts_match(game, jax.random.PRNGKey(1), limit=30, num_simulations=5, games=1000, gumbel=True)
# play_mcts_match(game, jax.random.PRNGKey(6), limit=30, num_simulations=10, games=1000, gumbel=True)
# play_mcts_match(game, jax.random.PRNGKey(90), limit=30, num_simulations=30, games=1000, gumbel=True)

def test(game, rng_key, mcts_player=1, num_simulations=0):
    """Spielt eine Partie: Trainierter Agent vs. Random Bot."""
    env = game.env_reset(0)
    step = 0
    limit = 30
    while not env.done and step < limit:
        rng_key, action_key = jax.random.split(rng_key)
        
        if env.current_player == mcts_player:
            action = get_gumbel_action(env, env.board, action_key, num_simulations)
            # print("Trained player: ", action)
        else:
            action = get_random_action(env.board, action_key)
            
        env, _, _ = game.env_step(env, action.astype(jnp.int8))
        # print("Current board:\n", env.board)
        # print(env.done)
        # print(jnp.all(env.board != 0))
        step += 1
    
    # Gibt den Gewinner zurück (1 für trainierten Agent, -1 für Random Bot, 0 für Unentschieden)
    # print(f"Final board:\n", env.board, "\n-----------------------------\n")
    if step == limit:
        return 0
    return game.get_winner(env.board) * mcts_player

def eva_test(game, num_matches=1000, num_simulations=0):
    """Evaluiert den trainierten Agenten über viele Partien."""
    if num_simulations == 0:
        print(f"\nStarte Evaluation gegen Random Bot ({num_matches} Partien)...")
    else:
        print(f"\nStarte Evaluation gegen MCTS Bot mit {num_simulations} Simulationen ({num_matches} Partien)...")
    rng_key = jax.random.PRNGKey(1)
    
    wins = 0
    losses = 0
    draws = 0
    
    # Spiele als Spieler 1 (beginnt)
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = test(game, game_key, mcts_player=1, num_simulations=num_simulations)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    
    print(f"Ergebnis als Spieler 1 {num_matches//2} Partien: Wins: {wins}, Losses: {losses}, Draws: {draws}")
    # Spiele als Spieler 2
    wins2 = 0
    losses2 = 0
    draws2 = 0
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = test(game, game_key, mcts_player=-1, num_simulations=num_simulations)
        if winner == 1:
            wins2 += 1
        elif winner == -1:
            losses2 += 1
        else:
            draws2 += 1

    print(f"Ergebnis als Spieler 2 {num_matches//2} Partien: Wins: {wins2}, Losses: {losses2}, Draws: {draws2}")

    win_rate = (wins + wins2) / num_matches
    loss_rate = (losses + losses2) / num_matches
    draw_rate = (draws + draws2) / num_matches
    print("\n--- Evaluationsergebnis ---")
    print(f"Gewinnrate: {win_rate:.2%}")
    print(f"Verlustrate: {loss_rate:.2%}")
    print(f"Unentschieden: {draw_rate:.2%}")
    print("---------------------------\n")

eva_test(game, 1000, num_simulations=5)
eva_test(game, 1000, num_simulations=10)
eva_test(game, 1000, num_simulations=30)
eva_test(game, 1000, num_simulations=100)
