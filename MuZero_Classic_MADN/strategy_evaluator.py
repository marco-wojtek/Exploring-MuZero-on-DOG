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
from MADN.classic_madn import *
from MuZero_Classic_MADN.muzero_classic_madn import *

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
                lambda:jnp.array([True, False, True, False], dtype=jnp.bool_),  # [0, 2]
                lambda: jnp.array([False, True, False, True], dtype=jnp.bool_)   # [1, 3]
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
        enable_teams=RULES['enable_teams'],
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        enable_start_on_1=RULES['enable_start_on_1'],
        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
        must_traverse_start=RULES['must_traverse_start'],
    )

# 2. Vektorisierte Funktionen vorbereiten
batch_reset = jax.vmap(env_reset_batched, in_axes=(0, 0))
batch_valid_action = jax.vmap(valid_action)
batch_encode = jax.vmap(encode_board)
batch_env_step = jax.vmap(env_step, in_axes=(0, 0))
batch_map_action = jax.vmap(map_action)

def calculate_progress(env: classic_MADN, player_idx: int) -> int:
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
            board_size + (pins - goals[0])  # Im Ziel
        )
    )
    
    rotated_goals = jnp.arange(board_size, board_size + 4)

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
    
    x = jax.vmap(player_progress_single)(envs)
    return jnp.mean(x, axis=0), x

def evaluate_agent_parallel(params1, params2, params3, params4, batch_size=20, set_seed=None):
    # use random agents if params are None
    env = env_reset_batched(0, 0)  # Dummy-Reset, um die Form der Beobachtungen zu erhalten
    enc = encode_board(env)  # z.B. (8, 56)
    agents = []
    for param in [params1, params2, params3, params4]:
        if param is None:
            param = init_muzero_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc.shape)
            param['type'] = 1
        elif param == 'rule_based_agent':
            param = init_muzero_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 2
        elif param == 'random_agent':
            param = init_muzero_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 3
        else:               
            param['type'] = 0
        agents.append(param)

    winners = jnp.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
    
    average_progress = jnp.array([[0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]])
    
    # for i in range(4):
    #     winners_batch, progress = play_n_games_for_eval(agents, jax.random.PRNGKey(i*12345), num_envs=batch_size, starting_player=i)
    #     winners = winners.at[i].add(winners_batch)
    #     average_progress = average_progress.at[i].set(progress)

    # for i in range(4):
    #     winners_batch, progress = play_n_games_for_eval_jitted(
    #         agents, 
    #         jax.random.PRNGKey(i * 12345),
    #         num_envs=batch_size,
    #         starting_player=i
    #     )
    #     winners = winners.at[i].set(winners_batch)
    #     average_progress = average_progress.at[i].set(progress)
    if set_seed is not None:
        seed = jax.random.PRNGKey(set_seed)
    else:
        seed = jax.random.PRNGKey(np.random.randint(0, 1000000))
    winners_batch, progress = play_n_games_for_eval_jitted(
            agents, 
            seed,
            num_envs=batch_size,
        )
    print(winners_batch.shape)
    winners_split = jnp.array_split(winners_batch, 4, axis=0)
    progress_split = jnp.array_split(progress, 4, axis=0)
    for i in range(4):
        winners = winners.at[i].set(jnp.sum(winners_split[i], axis=0))
        average_progress = average_progress.at[i].set(jnp.mean(progress_split[i], axis=0))
    print("Final Results:")
    print("Total Wins per Player and different Starters:\n", winners)
    print("Total Wins per Player:\n", jnp.sum(winners, axis=0))
    print("Average Final Pin distance per Player and different Starters:\n", average_progress)
    print("Average Final Pin distance per Player:\n", jnp.sum(average_progress, axis=0) / 4)

def play_n_games_for_eval_jitted(params_list, rng_key, num_envs=20, starting_player=0):
    """JIT-compilierte Version wie game_agent"""
    rng_key, subkey = jax.random.split(rng_key)
    # num_total_envs = num_envs * 4  # Wir spielen 4 separate Batches für jeden Startspieler
    seeds = jax.random.randint(subkey, (num_envs * 4,), 0, 1000000)
    # envs = batch_reset(seeds, jnp.full((num_envs,), starting_player))
    envs = batch_reset(seeds, jnp.repeat(jnp.arange(4), num_envs))
    # Konvertiere params_list zu Tuple für JIT
    params_tuple = tuple(params_list)
    
    # ✅ Verwende JAX while_loop
    final_envs, winners = play_eval_loop_jitted(
        envs, params_tuple, subkey, num_envs*4
    )
    
    progress_mean, progress = calculate_player_progress(final_envs)
    return winners, progress

@functools.partial(jax.jit, static_argnames=['num_envs'])
def play_eval_loop_jitted(envs, params_tuple, rng_key, num_envs):
    """Vollständig JIT-compiliert"""
    
    def body_fn(carry):
        envs, winners, dones, step_count, rng_key = carry
        
        # Keys für diesen Step
        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)
        
        # Step-Funktion
        def step_single_env(env, done, key, winner):
            def do_step(env, winner):
                # MCTS für aktuellen Spieler
                current_player = env.current_player
                params = jax.lax.switch(current_player, [
                    lambda: params_tuple[0],
                    lambda: params_tuple[1],
                    lambda: params_tuple[2],
                    lambda: params_tuple[3],
                ])
                env = throw_die(env)
                obs = encode_board(env)[None, ...]
                valid_mask = valid_action(env).flatten()
                invalid_mask = (~valid_mask)[None, :]
                
                def do_mcts():
                    policy_output, _ = run_stochastic_muzero_mcts(
                        params, key, obs, 
                        invalid_actions=invalid_mask,
                        num_simulations=NUM_SIMULATIONS,
                        max_depth=MAX_DEPTH,
                        temperature=TEMPERATURE
                    )
                    action = policy_output.action[0]
                    next_env, reward, next_done = env_step(env, action)
                    return next_env, next_done
                
                def do_random():
                    logits = jnp.where(valid_mask, 0.0, -1e9)
                    action = jax.random.categorical(key, logits)
                    next_env, reward, next_done = env_step(env, action)
                    return next_env, next_done
                
                def do_rule_based():
                    # Rule-based agent for classic MADN: only pin selection (4 actions)
                    # Dice value is given in env.die
                    current_player = env.current_player
                    die_value = env.die  # scalar, e.g., 1-6
                    num_pins = 4
                    current_positions = env.pins[current_player]  # (4,)
                    current_goal = env.goal[current_player]  # (4,)
                    board_size = env.board_size
                    target = env.target[current_player]
                    start_pos = env.start[current_player]
                    
                    # Calculate new position for each pin if it moves by die_value
                    moved_positions = current_positions + die_value  # (4,)
                    fitted_positions = moved_positions % board_size  # (4,)
                    x = moved_positions - target - jnp.int8(env.rules['must_traverse_start'])  # (4,)
                    
                    # Calculate new positions for each pin
                    new_positions = jnp.where(
                        current_positions < 0,
                        start_pos,  # pin in home moves to start
                        jnp.where(
                            current_positions >= board_size,
                            moved_positions,  # pin in goal area moves normally
                            jnp.where(
                                (4 >= x) & (x > 0) & (current_positions <= target),
                                current_goal[x-1],  # entering goal
                                fitted_positions  # pin on board moves normally
                            )
                        )
                    )  # (4,)
                    
                    # Calculate opponent pins
                    all_pins = env.pins
                    opp = jnp.ones_like(all_pins).at[current_player].set(0)
                    pos = jax.lax.cond(
                        env.rules['enable_teams'],
                        lambda opp: opp.at[(current_player + 2) % 4].set(0),
                        lambda opp: opp,
                        operand=opp
                    )
                    opponent_pins = jnp.where(
                        pos == 1,
                        all_pins,
                        -jnp.ones_like(all_pins)
                    ).flatten()
                    
                    pins_in_home = jnp.sum(current_positions < 0)
                    
                    # BASE SCORE: Start with zero
                    base_score = jnp.zeros(num_pins)  # (4,)
                    
                    # BONUS 1: Moving into goal (+5.0)
                    goal_bonus = jnp.where(
                        jnp.isin(new_positions, current_goal) & (current_positions < board_size),
                        5.0,
                        0.0
                    )  # (4,)
                    
                    # BONUS 2: Getting pin out of house
                    out_of_home_weight = jnp.where(pins_in_home >= 2, 3.0, 2.0)
                    out_bonus = jnp.where(
                        (current_positions < 0) & (new_positions == start_pos),
                        out_of_home_weight,
                        0.0
                    )  # (4,)
                    
                    # BONUS 3: Hitting opponent (+2.5)
                    hit_bonus = jnp.where(
                        (new_positions != current_positions) & jnp.isin(new_positions, opponent_pins),
                        2.5,
                        0.0
                    )  # (4,)
                    
                    # TOTAL SCORE
                    policy_scores = base_score + goal_bonus + out_bonus + hit_bonus  # (4,)
                    
                    # Mask illegal pins (valid_mask is (4,) for pin selection)
                    policy_scores = jnp.where(valid_mask, policy_scores, -jnp.inf)
                    
                    # Softmax with temperature
                    temperature = 0.25
                    policy_logits = policy_scores / temperature
                    
                    action = jax.random.categorical(key, policy_logits)
                    next_env, reward, next_done = env_step(env, action)
                    return next_env, next_done

                def do_no_step():
                    next_env, reward, next_done = no_step(env)
                    return next_env, next_done
                
                next_env, next_done = jax.lax.cond(
                    jnp.any(valid_mask),
                    lambda: jax.lax.cond(
                        params['type'] == 3,
                        do_random,
                        lambda: jax.lax.cond(
                            params['type'] == 2,
                            do_rule_based,
                            do_mcts
                        )
                    ),
                    do_no_step
                )
                
                # Winner-Check bei Done
                def update_winner(env, winner):
                    win = manual_get_winner(env.board, env.num_players, env.goal, env.rules)
                    return winner + win.astype(jnp.int32)
                
                new_winner = jax.lax.cond(
                    next_done,
                    lambda: update_winner(next_env, winner),
                    lambda: winner
                )
                
                return next_env, next_done, new_winner
            
            def skip_step(env, winner):
                return env, done, winner
            
            return jax.lax.cond(
                ~done,
                lambda: do_step(env, winner),
                lambda: skip_step(env, winner)
            )
        
        # vmap über alle Envs
        new_envs, new_dones, new_winners = jax.vmap(step_single_env)(
            envs, dones, step_keys, winners
        )
        
        return (new_envs, new_winners, new_dones, step_count + 1, rng_key)
    
    # Initialisierung
    init_dones = envs.done
    init_winners = jnp.zeros((num_envs, 4), dtype=jnp.int32)
    
    def cond_fn(carry):
        _, _, dones, step_count, _ = carry
        return jnp.any(~dones) & (step_count < 2000)
    
    final_envs, final_winners, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (envs, init_winners, init_dones, 0, rng_key)
    )
    
    return final_envs, final_winners


# Rules for evaluation games - can be adjusted to test specific rule variations
RULES = {
    'enable_teams': False,
    'enable_initial_free_pin': True,
    'enable_circular_board': False,
    'enable_friendly_fire': True,
    'enable_start_blocking': False,
    'enable_jump_in_goal_area': True,
    'enable_start_on_1': True,
    'enable_bonus_turn_on_6': True,
    'must_traverse_start': False
}

start_time = time()
NUM_SIMULATIONS = 200
MAX_DEPTH = 100
TEMPERATURE = 0.0
FOLDER = "MuZero_Classic_MADN/models/params/"
print("Games with Temperature =", TEMPERATURE)
print("And set seed for same dice outcomes:", 12345)
# # play_n_randomly(batch_size=1000)  
# print("\nVersus Random Agents:")
# params1 = 'random_agent'  # Random Agent
# params2 = 'rule_based_agent'  # Stochastic MuZero Agent
# params3 = 'random_agent'  # MCTS Agent
# params4 = 'rule_based_agent'  # Stochastic MuZero Agent

# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

# # play_n_randomly(batch_size=1000)  
# print("\nVersus Rule-Based Agents:")
# params1 = 'rule_based_agent'  # Rule-Based Agent
# params2 = 'rule_based_agent'  # Stochastic MuZero Agent
# params3 = 'rule_based_agent'  # MCTS Agent
# params4 = 'rule_based_agent'  # Stochastic MuZero Agent

# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)
FILENAME = f"{FOLDER}stochastic_muzero_madn_params_lr0.005_g1500_it50_seed22"
# play_n_randomly(batch_size=1000)  
print(FILENAME)
# print("\nTrained Stochastic MuZero Agents 12345:")
# params1 = 'random_agent'#load_params_from_file(f"{FILENAME}.pkl")  # Rule-Based Agent
# params2 = 'random_agent'  # Stochastic MuZero Agent
# params3 = load_params_from_file(f"{FILENAME}.pkl")  # MCTS Agent
# params4 =  'random_agent'  # Stochastic MuZero Agent

# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150, set_seed=12345)

print("\nVersus 'random_agent':")
params1 = 'random_agent'#load_params_from_file(f"{FILENAME}.pkl")  # Rule-Based Agent
params2 = 'random_agent'  # Stochastic MuZero Agent
params3 = load_params_from_file(f"{FILENAME}.pkl")  # MCTS Agent
params4 =  'random_agent'  # Stochastic MuZero Agent

evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

print("\nVersus 'rule_based_agent':")
params1 = 'rule_based_agent' #load_params_from_file(f"{FILENAME}.pkl")  # Rule-Based Agent
params2 = 'rule_based_agent'  # Stochastic MuZero Agent
params3 = load_params_from_file(f"{FILENAME}.pkl")  # MCTS Agent
params4 =  'rule_based_agent'  # Stochastic MuZero Agent

evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

print("\nVersus 'none':")
params1 = None#load_params_from_file(f"{FILENAME}.pkl")  # Rule-Based Agent
params2 = None # Stochastic MuZero Agent
params3 = load_params_from_file(f"{FILENAME}.pkl")  # MCTS Agent
params4 =  None  # Stochastic MuZero Agent

evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)
end_time = time()
print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")


#######################################################################################
### STRATEGY EVALUATION - TEST DIFFERENT SETTINGS FOR MCTS/RULE-BASED/Random AGENTS ###
#######################################################################################

# TEST 1: Analysis of action selection in different game states
# The model is given a state with 4 pins in different situations (home, on board, in goal) 
# The opponent pins are also in different situations. 

def test_action_selection_in_different_states():
    '''
    Analysiere die Aktionsauswahl von MCTS, Regelbasiertem und Zufallsagenten in verschiedenen Spielsituationen.
    jnp.array([
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ])
    '''
    # Situation A: Alle Pins können bewegt werden und erreichen unterschiedliche Felder
    # situation A.1: Alle Gegnerischen Pins im Home, außer einer der geschlagen werden kann. Alles mit Zug 6
    sitA1 = jnp.array([
        [-1, 5, 20, 37],
        [-1, 11, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ])
    # situation A.2: Alle Gegnerischen Pins im Home, außer einer der geschlagen werden kann. Alles mit Zug 1
    sitA2 = jnp.array([
        [-1, 5, 20, 39],
        [-1, 6, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ])
    # situation A.3: Alle Gegnerischen Pins im Home, außer einer der geschlagen werden kann. Alles mit Zug 6
    sitA3 = jnp.array([
        [-1, 5, 20, 37],
        [-1, -1, -1, -1],
        [-1, 11, -1, -1],
        [-1, -1, -1, -1],
    ])
    # situation A.4: Alle Gegnerischen Pins im Home, außer einer der geschlagen werden kann. Alles mit Zug 1
    sitA4 = jnp.array([
        [-1, 5, 20, 39],
        [-1, -1, -1, -1],
        [-1, 6, -1, -1],
        [-1, -1, -1, -1],
    ])
    # situation A.5: Alle Gegnerischen Pins im Home, außer einer der geschlagen werden kann. Alles mit Zug 6
    sitA5 = jnp.array([
        [-1, 5, 20, 37],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, 11, -1, -1],
    ])
    # situation A.6: Alle Gegnerischen Pins im Home, außer einer der geschlagen werden kann. Alles mit Zug 1
    sitA6 = jnp.array([
        [-1, 5, 20, 39],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, 6, -1, -1],
    ])

    situations = [sitA1, sitA2, sitA3, sitA4, sitA5, sitA6]
    for i, sit in enumerate(situations):
        print(f"Situation A.{i+1}:")
        print(sit)
        # Hier könnte man die Aktionsauswahl der verschiedenen Agenten analysieren, z.B. durch Aufruf einer Funktion wie get_agent_action(agent_type, sit, die_value) für MCTS, Regelbasiert und Zufall und Vergleich der Ergebnisse.
        pass

def test_with_dice_rule():
    '''
    Spiele Spiele mit würfeln die einer festen Regel folgen (e.g. Immer +1)
    '''
    def throw_set_die(env: classic_MADN) -> classic_MADN:
        '''
        Simuliert einen Würfelwurf und aktualisiert den Zustand der Umgebung mit dem neuen Würfelwert.
            Args:
                env: Die aktuelle Spielumgebung
            Returns:
                Die aktualisierte Spielumgebung mit dem neuen Würfelwert.
        '''
        return env.replace(
            die=(env.die + 1) % 6
        )