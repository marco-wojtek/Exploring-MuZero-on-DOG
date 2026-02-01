import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Verhindert, dass JAX den gesamten GPU-Speicher belegt, damit mehrere eval Prozesse laufen können
import chex
import jax
import jax.numpy as jnp
from time import time
from functools import partial
import pickle
import numpy as np
import math
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MADN.deterministic_madn import *
from MuZero.muzero_deterministic_madn import *

def manual_get_winner(board: Board, num_players, goal, rules) -> chex.Array:
    '''
    Bestimmt den Gewinner des Spiels basierend auf dem aktuellen Spielfeld und den Spielregeln.
        Args:
            env: Die aktuelle Spielumgebung
            board: Das aktuelle Spielfeld
        Returns:
            Ein Array, das angibt, welche Spieler gewonnen haben.
    '''
    collect_winners = jax.vmap(is_player_done, in_axes=(None, None, None, 0))
    players_done = collect_winners(num_players, board, goal, jnp.arange(4, dtype=jnp.int8))  # (4,)

    def four_players_case():
        team_0 = players_done[0] & players_done[2]  # Team 0&2 fertig
        team_1 = players_done[1] & players_done[3]  # Team 1&3 fertig
        both = team_0 & team_1  # Beide Teams fertig (unentschieden)
        none = ~(team_0 | team_1)  # Kein Team fertig
        
        return jax.lax.cond(
            both | none,  # Bei Unentschieden oder keinem Gewinner
            lambda: jnp.full(players_done.shape, False, dtype=jnp.bool_),  # [-1, -1]
            lambda: jax.lax.cond(
                team_0,  # Falls Team 0&2 gewonnen hat
                lambda:jnp.array([False, True, False, True], dtype=jnp.bool_),  # [0, 2]
                lambda: jnp.array([True, False, True, False], dtype=jnp.bool_)   # [1, 3]
            )
        )


    return jax.lax.cond(rules['enable_teams'], four_players_case, lambda: players_done)


def env_reset_batched(seed, starting_player):
    return env_reset(
        seed,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=10,
        starting_player=starting_player,
        seed=seed,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=False,
        enable_initial_free_pin=True,
        enable_circular_board=False
    )

# 2. Vektorisierte Funktionen vorbereiten
batch_reset = jax.vmap(env_reset_batched, in_axes=(0, 0))
batch_valid_action = jax.vmap(valid_action)
batch_encode = jax.vmap(encode_board)
batch_env_step = jax.vmap(env_step, in_axes=(0, 0))
batch_map_action = jax.vmap(map_action)

@jax.jit
def multiactor_step(envs, params_list, rng_key):
    """
    Führt einen Schritt für N parallele Spiele aus.
    params_list: Eine Liste der 4 Parameter-Sets für die Spieler.
    """
    # A. Observations
    obs = batch_encode(envs)
    val_actions = batch_valid_action(envs).reshape(envs.board.shape[0], -1)
    dones = envs.done
    current_players = envs.current_player
    branches = [lambda p=p: p for p in params_list]
    def perform_action(env, obs, val_act, done, player_idx, rng_key):
        # Wähle die richtigen Parameter basierend auf dem Index des aktuellen Spielers.
        # jax.lax.switch wird für JIT-kompatibles bedingtes Indexieren verwendet.
        
        params = jax.lax.switch(player_idx, branches)

        def mcts_step():
            obs_batched = obs[None, ...] 
            invalid_actions_batched = (~val_act)[None, ...]
            
            policy_output, root_values = run_muzero_mcts(params, rng_key, obs_batched, invalid_actions=invalid_actions_batched, temperature=0.25)
            
            act = policy_output.action[0]
            action_weights = policy_output.action_weights[0]
            root_value = root_values[0]
            
            mapped_act = map_action(act)
            next_env, reward, next_done = env_step(env, mapped_act)
            return next_env, obs, act, reward, root_value, action_weights, next_done
        
        def no_action_step():
            next_env, reward, next_done = no_step(env)
            dummy_action = jnp.int32(-1)
            dummy_policy = jnp.zeros_like(val_act, dtype=jnp.float32)
            dummy_root_value = 0.0
            return next_env, obs, dummy_action, reward, dummy_root_value, dummy_policy, next_done

        return jax.lax.cond(
            jnp.any(val_act) & (~done),
            mcts_step,
            no_action_step
        )

    # Führe vmap aus. Beachte, dass `params_list` nicht mehr Teil des vmap-Aufrufs ist.
    # Stattdessen übergeben wir `current_players`, um die Auswahl innerhalb von `perform_action` zu treffen.
    next_envs, obs, actions, rewards, root_values, policy_output_action_weights, next_dones = jax.vmap(
        perform_action, in_axes=(0, 0, 0, 0, 0, 0)
    )(envs, obs, val_actions, dones, current_players, jax.random.split(rng_key, envs.board.shape[0]))
    
    rewards = jnp.where(dones, 0.0, rewards)
    final_dones = jnp.logical_or(dones, next_dones)
    
    return next_envs, (obs, actions, rewards, root_values, policy_output_action_weights, final_dones)

def calculate_progress(env: deterministic_MADN, player_idx: int) -> int:
    '''
    Berechnet den Fortschritt eines Spielers im Vergleich zum Ziel.
    Fortschritt ist die durchschnittliche Distanz der restlichen Pins zum Ziel.
    Falls Überspringen im Ziel erlaubt ist, wird der Fortschritt so angepasst dass die mittlere Distanz aller freien Pins zum ersten freien Zielfeld bestimmt wird.
    Falls Überspringen nicht erlaubt ist, wird der Fortschritt als die mittlere Distanz aller Pins entsprechend des Fortschritts bestimmt (e.g. Pin am weitesten bekommt distanz zum hintersten Zielfeld).
    Falls traverse start enabled ist, wird +1 zur Distanz berechnet.
            Args:
                env: Die aktuelle Spielumgebung
    Rückgabe: Gesamte Distanz aller Pins zum Ziel mit Penalty für Home-Pins.
    '''
    board_size = env.board_size
    distance = board_size // env.num_players
    pins = env.pins[player_idx]  # (num_pins,)
    goals = env.goal[player_idx]  # (num_pins,)
    rules = env.rules
    travers_start_enabled = rules['must_traverse_start']


    rotated_pins = jnp.where(
        pins < 0,  # Home
        pins -5, # Pins at home get penalty of 5 since 6 is needed to be freed
        jnp.where(
            pins < board_size,  # Auf dem Board
            (pins - distance * player_idx) % board_size - jnp.int32(travers_start_enabled),
            pins - 4 * player_idx  # Im Ziel
        )
    )
    
    rotated_goals = goals - 4 * player_idx 

    sorted_pins = jnp.sort(rotated_pins)

    distance_matrix = jnp.abs(sorted_pins[:, None] - rotated_goals[None, :])  # (num_pins, num_pins)
    
    def match_iteration(i, carry):
        total_dist, mask = carry
        
        # Finde Minimum unter maskierten Werten
        masked_distances = jnp.where(mask, distance_matrix, jnp.inf)
        flat_idx = jnp.argmin(masked_distances)
        
        row = flat_idx // 4
        col = flat_idx % 4
        
        # Addiere minimale Distanz
        min_dist = distance_matrix[row, col]
        new_total = total_dist + min_dist
        
        # Aktualisiere Maske: Zeile und Spalte blockieren
        new_mask = mask.at[row, :].set(False)
        new_mask = new_mask.at[:, col].set(False)
        
        return new_total, new_mask
    
    # Initialisiere
    initial_mask = jnp.ones((4, 4), dtype=jnp.bool_)
    initial_total = jnp.float32(0.0)
    
    # Führe 4 Iterationen aus (für 4 Pins)
    final_total, _ = jax.lax.fori_loop(
        0, 4,
        match_iteration,
        (initial_total, initial_mask)
    )

    return final_total

@jax.jit
def calculate_player_progress(envs):
    """
    Berechnet den Fortschritt jedes Spielers im Vergleich zum Ziel.
    Rückgabe: Array der Form (num_envs, num_players) mit Fortschrittswerten.
    """
    def player_progress_single(env):
        def single_player_progress(player_idx):
            return calculate_progress(env, player_idx)
        
        return jax.vmap(single_player_progress)(jnp.arange(env.num_players))
    
    return jnp.mean(jax.vmap(player_progress_single)(envs), axis=0)

def play_n_games_for_eval(params_list, rng_key, num_envs=20, starting_player=0):
    """
    Spielt num_envs Spiele parallel und gibt eine Liste von Episoden zurück.
    """
    # 1. Initialisierung der Environments
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs,), 0, 1000000)
    envs = batch_reset(seeds, jnp.full((num_envs,), starting_player))
    
    # Buffer für jedes Environment (Liste von Listen)
    winners = jnp.zeros((num_envs,4), dtype=jnp.int32)
    
    active_mask = np.ones(num_envs, dtype=bool)
    
    step_counter = 0
    MAX_STEPS = 2000 # Sicherheitsabbruch, falls Spiele hängen
    
    # Loop solange noch mindestens ein Spiel läuft
    while np.any(active_mask) and step_counter < MAX_STEPS:
        step_counter += 1
        rng_key, subkey = jax.random.split(rng_key)
        
        # JIT-Step ausführen (läuft auf GPU/TPU für alle Envs gleichzeitig)
        current_players = envs.current_player
        params_for_envs = [params_list[int(player)] for player in current_players]
        envs, data = multiactor_step(envs, tuple(params_for_envs), subkey)
        
        # Daten auf CPU holen für Listen-Operationen
        obs, acts, rews, vals, pols, dones = jax.device_get(data)
        
        # fetch winners if done
        for i in range(num_envs):
            if active_mask[i] and dones[i]:
                active_mask[i] = False
                winner = manual_get_winner(envs.board[i], envs.num_players, envs.goal[i], envs.rules)
                winners = winners.at[i].add(jnp.array(winner, dtype=jnp.int32))
    
    # Get Progress Stats
    progress = calculate_player_progress(envs)
    return jnp.sum(winners, axis=0), progress

def evaluate_agent_parallel(params1, params2, params3, params4, batch_size=20):
    # use random agents if params are None
    env = env_reset(
        0,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=10,
        starting_player=0,
        seed=0,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=False,
        enable_initial_free_pin=True,
        enable_circular_board=False
    )
    enc = encode_board(env)  # z.B. (8, 56)
    agents = []
    for param in [params1, params2, params3, params4]:
        if param is None:
            param = init_muzero_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc.shape)

        agents.append(param)

    winners = jnp.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
    
    average_progress = jnp.array([[0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]])
    
    for i in range(4):
        winners_batch, progress = play_n_games_for_eval(agents, jax.random.PRNGKey(i*12345), num_envs=batch_size, starting_player=i)
        winners = winners.at[i].add(winners_batch)
        average_progress = average_progress.at[i].set(progress)

    print("Final Results:")
    winners = jnp.array(winners)
    print("Total Wins per Player and different Starters:\n", winners)
    print("Total Wins per Player:\n", jnp.sum(winners, axis=0))
    print("Average Final Pin distance per Player and different Starters:\n", average_progress)
    print("Average Final Pin distance per Player:\n", jnp.sum(average_progress, axis=0) / 4)

def play_n_randomly(batch_size=20):
    winners = jnp.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
    
    for i in range(4):
        envs = batch_reset(jax.random.randint(jax.random.PRNGKey(i*54321), (batch_size,), 0, 1000000), jnp.full((batch_size,), i))
        active_mask = np.ones(batch_size, dtype=bool)
        step_counter = 0
        MAX_STEPS = 2000
        dones = envs.done  # Initialen Done-Status speichern
        
        while np.any(active_mask) and step_counter < MAX_STEPS:
            step_counter += 1
            rng_key, subkey = jax.random.split(jax.random.PRNGKey(i*99999 + step_counter))
            valid_actions = batch_valid_action(envs).reshape(envs.board.shape[0], -1)

            def random_step(env, val_actions, done, key):
                def do_step():
                    # Maskiere ungültige Aktionen mit -1e9
                    logits = jnp.where(val_actions, 0.0, -1e9)
                    action = jax.random.categorical(key, logits)
                    mapped_act = map_action(action)
                    next_env, reward, next_done = env_step(env, mapped_act)
                    return next_env, reward, next_done
                def no_step_action():
                    next_env, reward, next_done = no_step(env)
                    return next_env, reward, next_done
                return jax.lax.cond(
                    jnp.any(val_actions) & (~done),
                    do_step,
                    no_step_action
                )
            
            envs, rewards, next_dones = jax.vmap(random_step, in_axes=(0,0,0,0))(envs, valid_actions, dones, jax.random.split(subkey, batch_size))
            final_dones = jnp.logical_or(dones, next_dones)
            
            # Winner-Tracking wie in play_n_games_for_eval
            for j in range(batch_size):
                if active_mask[j] and final_dones[j]:
                    active_mask[j] = False
                    winner = manual_get_winner(envs.board[j], envs.num_players, envs.goal[j], envs.rules)
                    winners = winners.at[i].add(jnp.array(winner, dtype=jnp.int32))
            
            dones = final_dones  # Update dones für nächsten Step

    print("Final Results for Random Agents:")
    print("Total Wins per Player and different Starters:\n", winners)
    print("Total Wins per Player:\n", jnp.sum(winners, axis=0))
    print("Statistics:")
    print("Total win chances in %:", jnp.sum(winners,axis=0) / jnp.sum(winners) * 100)
    print("Chance to win when starting first:", jnp.sum(jnp.diag(winners)) / jnp.sum(winners) * 100)

def test_agent_vs_random(params, num_games, batch_size=100, seed=42):
    '''
    Testet einen einzelnen Agenten gegen Random-Gegner über viele Spiele.
    
    Args:
        params: Parameter des zu testenden Agenten
        num_games: Gesamtanzahl der Spiele
        batch_size: Anzahl paralleler Spiele pro Batch
        seed: Random seed
    
    Returns:
        Anzahl der Siege des Agenten
    '''
    rng_key = jax.random.PRNGKey(seed)
    
    # Initialize dummy params if None
    env = env_reset(
        0,
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=10,
        starting_player=0,
        seed=0,
        enable_teams=False,
        enable_initial_free_pin=True,
        enable_circular_board=False
    )
    enc = encode_board(env)
    
    agent = params if params is not None else init_muzero_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc.shape)
    dummy_agent = init_muzero_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc.shape)
    
    total_wins = 0
    num_batches = (num_games + batch_size - 1) // batch_size  # Aufrunden
    
    for batch_idx in range(num_batches):
        # Berechne Batch-Größe für letzten Batch
        current_batch_size = min(batch_size, num_games - batch_idx * batch_size)
        
        rng_key, subkey = jax.random.split(rng_key)
        seeds = jax.random.randint(subkey, (current_batch_size,), 0, 1000000)
        envs = batch_reset(seeds, jnp.full((current_batch_size,), -1))  # Random starting player
        
        winners = jnp.zeros((current_batch_size, 4), dtype=jnp.int32)
        active_mask = np.ones(current_batch_size, dtype=bool)
        
        step_counter = 0
        MAX_STEPS = 2000
        
        while np.any(active_mask) and step_counter < MAX_STEPS:
            step_counter += 1
            rng_key, subkey = jax.random.split(rng_key)
            
            # Generate RNG keys for this step
            step_keys = jax.random.split(subkey, current_batch_size)
            
            current_players = envs.current_player
            
            # Build params_for_envs with use_mcts flags
            params_for_envs = []
            use_mcts_flags = []
            for i in range(current_batch_size):
                player_idx = int(current_players[i])
                if player_idx == 0:
                    # Trained agent plays at position 0
                    params_for_envs.append(agent)
                    use_mcts_flags.append(True)
                else:
                    # Random agent
                    params_for_envs.append(dummy_agent)
                    use_mcts_flags.append(False)
            
            use_mcts_flags = jnp.array(use_mcts_flags, dtype=jnp.bool_)
            
            envs, data = multiactor_step_with_random_agent_v2(envs, tuple(params_for_envs), use_mcts_flags, step_keys)
            
            obs, acts, rews, vals, pols, dones = jax.device_get(data)
            
            for i in range(current_batch_size):
                if active_mask[i] and dones[i]:
                    active_mask[i] = False
                    winner = manual_get_winner(envs.board[i], envs.num_players, envs.goal[i], envs.rules)
                    winners = winners.at[i].add(jnp.array(winner, dtype=jnp.int32))
        
        # Count wins at position 0 (where the agent plays)
        batch_wins = jnp.sum(winners[:, 0])
        total_wins += int(batch_wins)
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            current_total = (batch_idx + 1) * batch_size
            current_total = min(current_total, num_games)
            print(f"  Progress: {current_total}/{num_games} games, Wins so far: {total_wins} ({total_wins/current_total*100:.1f}%)")
    
    return total_wins

@jax.jit
def multiactor_step_with_random_agent_v2(envs, params, use_mcts, rng_key):
    """
    Führt einen Schritt für N parallele Spiele aus.
    params: Tuple von Parameter-Sets, eines pro Environment.
    use_mcts: Boolean array, True = use MCTS, False = random action
    rng_key: Array of RNG keys, one per environment
    """
    obs = batch_encode(envs)
    val_actions = batch_valid_action(envs).reshape(envs.board.shape[0], -1)
    dones = envs.done

    def perform_action(env, obs, val_act, done, params, use_mcts_flag, rng_key):
        def mcts_step():
            obs_batched = obs[None, ...] 
            invalid_actions_batched = (~val_act)[None, ...]
            
            policy_output, root_values = run_muzero_mcts(params, rng_key, obs_batched, invalid_actions=invalid_actions_batched, temperature=0.0)
            
            act = policy_output.action[0]
            action_weights = policy_output.action_weights[0]
            root_value = root_values[0]
            
            mapped_act = map_action(act)
            next_env, reward, next_done = env_step(env, mapped_act)
            return next_env, obs, act, reward, root_value, action_weights, next_done
        
        def random_step():
            logits = jnp.where(val_act, 0.0, -1e9)
            action = jax.random.categorical(rng_key, logits)
            mapped_act = map_action(action)
            next_env, reward, next_done = env_step(env, mapped_act)
            dummy_policy = jnp.zeros_like(val_act, dtype=jnp.float32)
            dummy_root_value = 0.0
            return next_env, obs, action, reward, dummy_root_value, dummy_policy, next_done
        
        def no_action_step():
            next_env, reward, next_done = no_step(env)
            dummy_action = jnp.int32(-1)
            dummy_policy = jnp.zeros_like(val_act, dtype=jnp.float32)
            dummy_root_value = 0.0
            return next_env, obs, dummy_action, reward, dummy_root_value, dummy_policy, next_done

        return jax.lax.cond(
            jnp.any(val_act) & (~done),
            lambda: jax.lax.cond(
                use_mcts_flag,
                mcts_step,
                random_step
            ),
            no_action_step
        )

    results = []
    for i in range(envs.board.shape[0]):
        result = perform_action(
            jax.tree.map(lambda x: x[i], envs),
            obs[i],
            val_actions[i],
            dones[i],
            params[i],
            use_mcts[i],
            rng_key[i]
        )
        results.append(result)

    next_envs = jax.tree.map(lambda *xs: jnp.stack(xs), *[r[0] for r in results])
    obs = jnp.stack([r[1] for r in results])
    actions = jnp.stack([r[2] for r in results])
    rewards = jnp.stack([r[3] for r in results])
    root_values = jnp.stack([r[4] for r in results])
    policy_output_action_weights = jnp.stack([r[5] for r in results])
    next_dones = jnp.stack([r[6] for r in results])
    
    rewards = jnp.where(dones, 0.0, rewards)
    final_dones = jnp.logical_or(dones, next_dones)
    
    return next_envs, (obs, actions, rewards, root_values, policy_output_action_weights, final_dones)

def compare_agents_statistically(params1, params2, num_games=1000, batch_size=100):
    '''
    Vergleicht zwei Agenten statistisch über viele unabhängige Spiele.
    Jeder spielt separat gegen Random-Agenten.
    '''
    print(f"\n{'='*60}")
    print(f"Statistical Agent Comparison")
    print(f"{'='*60}")
    print(f"Total games per agent: {num_games}")
    print(f"Batch size: {batch_size}")
    
    print(f"\n{'='*60}")
    print(f"Testing Agent 1...")
    print(f"{'='*60}")
    i = np.random.randint(0, 1000000)
    wins1 = test_agent_vs_random(params1, num_games, batch_size, seed=i)
    
    print(f"\n{'='*60}")
    print(f"Testing Agent 2...")
    print(f"{'='*60}")
    wins2 = test_agent_vs_random(params2, num_games, batch_size, seed=i)  # Different seed
    
    winrate1 = wins1 / num_games
    winrate2 = wins2 / num_games
    
    # Statistischer Test (z.B. Z-Test für Proportionen)
    diff = winrate1 - winrate2
    se = np.sqrt((winrate1 * (1 - winrate1) / num_games) + 
                 (winrate2 * (1 - winrate2) / num_games))
    
    # Vermeiden von Division durch 0
    if se > 0:
        z_score = diff / se
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))  # Two-tailed test
    else:
        z_score = 0
        p_value = 1.0
    
    print(f"\n{'='*60}")
    print(f"Statistical Comparison Results")
    print(f"{'='*60}")
    print(f"Agent 1: {wins1}/{num_games} wins = {winrate1*100:.2f}%")
    print(f"Agent 2: {wins2}/{num_games} wins = {winrate2*100:.2f}%")
    print(f"\nDifference: {diff*100:+.2f}% (Agent 1 - Agent 2)")
    print(f"Standard Error: {se*100:.3f}%")
    print(f"Z-score: {z_score:.3f}")
    print(f"P-value: {p_value:.4f}")
    
    print(f"\n{'='*60}")
    if abs(z_score) > 1.96:  # 95% Konfidenzintervall
        print(f"✓ Result: Statistically SIGNIFICANT (p < 0.05)")
        if z_score > 0:
            print(f"  → Agent 1 is significantly BETTER!")
        else:
            print(f"  → Agent 2 is significantly BETTER!")
    else:
        print(f"✗ Result: No significant difference (p >= 0.05)")
        print(f"  → Agents perform similarly")
    print(f"{'='*60}\n")
    
    return winrate1, winrate2

start_time = time()

#play_n_randomly(batch_size=10000)  
params1 = None
params2 = None
params3 = None
params4 = None

# compare_agents_statistically(params1, params2, num_games=100, batch_size=10)
print("5001")
params1 = load_params_from_file("models/params/gumbelmuzero_madn_params_lr0.01_g1500_it100_seed5001.pkl")
# params1 = load_params_from_file("models/params/gumbelmuzero_madn_params_lr0.01_g1500_it70.pkl")
compare_agents_statistically(params1, params2, num_games=150, batch_size=30)

#evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

end_time = time()
print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
