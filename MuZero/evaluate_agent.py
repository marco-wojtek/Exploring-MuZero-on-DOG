import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Verhindert, dass JAX den gesamten GPU-Speicher belegt, damit mehrere eval Prozesse laufen können
import chex
import jax
import jax.numpy as jnp
from time import time
from functools import partial
import pickle
import numpy as np
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
batch_encode = jax.vmap(old_encode_board)
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
            
            policy_output, root_values = run_muzero_mcts(params, rng_key, obs_batched, invalid_actions=invalid_actions_batched)
            
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
    
    # Filtern von None (falls MAX_STEPS erreicht wurde)
    return jnp.sum(winners, axis=0)

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
    enc = old_encode_board(env)  # z.B. (8, 56)
    agents = []
    for param in [params1, params2, params3, params4]:
        if param is None:
            param = init_muzero_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc.shape)

        agents.append(param)

    winners = jnp.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
    
    for i in range(4):
        winners = winners.at[i].add(play_n_games_for_eval(agents, jax.random.PRNGKey(i*12345), num_envs=batch_size, starting_player=i))

    print("Final Results:")
    winners = jnp.array(winners)
    print("Total Wins per Player and different Starters:\n", winners)
    print("Total Wins per Player:\n", jnp.sum(winners, axis=0))

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
    print("Statisctics:")
    print("Total win chances in %:", jnp.sum(winners,axis=0) / jnp.sum(winners) * 100)
    print("Chance to win when starting first:", jnp.sum(jnp.diag(winners)) / jnp.sum(winners) * 100)


#play_n_randomly(batch_size=1000)  
params1 = None
params3 = load_params_from_file("gumbelmuzero_madn_params_lr5e-05_g1500_it40.pkl")
params4 = None
params2 = None
start_time = time()
evaluate_agent_parallel(params1, params2, params3, params4, batch_size=100)
end_time = time()
print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
