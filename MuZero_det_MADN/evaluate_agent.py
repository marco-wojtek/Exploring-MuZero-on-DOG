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
from MuZero_det_MADN.muzero_deterministic_madn import *

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

def evaluate_agent_parallel(params1, params2, params3, params4, type1=None, type2=None, type3=None, type4=None, batch_size=20):
    # use random agents if params are None
    env = env_reset_batched(0, 0)  # Dummy-Reset, um die Form der Beobachtungen zu erhalten
    enc = encode_board(env)  # z.B. (8, 56)
    agents = []
    for param, agent_type in zip([params1, params2, params3, params4], [type1, type2, type3, type4]):
        if param is None:
            param = init_muzero_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc.shape)
            param['type'] = agent_type if agent_type is not None else 0
        elif param == 'rule_based_agent':
            param = init_muzero_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 2
        elif param == 'random_agent':
            param = init_muzero_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 3
        else:               
            param['type'] = agent_type if agent_type is not None else 0
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
    winners_batch, progress = play_n_games_for_eval_jitted(
            agents, 
            jax.random.PRNGKey(np.random.randint(0, 1000000)),
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


def fairness_check(batch_size=20, seed=None):
    """
    Spielt batch_size * 4 Spiele mit ausschließlich Random-Agenten.
    Jeder Startspieler (0-3) beginnt exakt batch_size Spiele.
    Nutzt einen eigenen JIT-Loop der nur random / no_step kennt.
    """
    if seed is None:
        seed = np.random.randint(0, 1_000_000)

    rng_key = jax.random.PRNGKey(seed)
    rng_key, subkey = jax.random.split(rng_key)

    num_total = batch_size * 4
    seeds = jax.random.randint(subkey, (num_total,), 0, 1_000_000)
    starting_players = jnp.repeat(jnp.arange(4), batch_size)
    envs = batch_reset(seeds, starting_players)

    final_envs, winners_flat = _random_only_loop(envs, subkey, num_total)

    winners_split = jnp.array_split(winners_flat, 4, axis=0)
    winners = jnp.stack([jnp.sum(w, axis=0) for w in winners_split])  # (4, 4)

    progress_mean, _ = calculate_player_progress(final_envs)

    total_wins = jnp.sum(winners)
    wins_per_player = jnp.sum(winners, axis=0)
    win_pct = wins_per_player / total_wins * 100

    print("\n" + "=" * 60)
    print("FAIRNESS CHECK – All Random Agents")
    print("=" * 60)
    print(f"Games per starting position: {batch_size}  (total: {num_total})")
    print("\nTotal Wins per Player and different Starters:\n", winners)
    print("\nTotal Wins per Player:\n", wins_per_player)
    print("\nWin % per Player:", win_pct)
    if RULES['enable_teams']:
        team_a = float(win_pct[0] + win_pct[2])
        team_b = float(win_pct[1] + win_pct[3])
        print(f"\nTeam A (0&2): {team_a:.1f}%  |  Team B (1&3): {team_b:.1f}%")
    print("\nMean Final Pin Distance per Player:\n", progress_mean)
    print("=" * 60)

    return winners, progress_mean


@functools.partial(jax.jit, static_argnames=['num_envs'])
def _random_only_loop(envs, rng_key, num_envs):
    """JIT-Loop der ausschließlich Random-Aktionen oder No-Step ausführt."""

    def body_fn(carry):
        envs, winners, dones, step_count, rng_key = carry

        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)

        def step_single_env(env, done, key, winner):
            def do_step(env, winner):
                valid_mask = valid_action(env).flatten()

                def do_random():
                    logits = jnp.where(valid_mask, 0.0, -1e9)
                    action = jax.random.categorical(key, logits)
                    next_env, _, next_done = env_step(env, map_action(action))
                    return next_env, next_done

                def do_no_step():
                    next_env, _, next_done = no_step(env)
                    return next_env, next_done

                next_env, next_done = jax.lax.cond(
                    jnp.any(valid_mask),
                    do_random,
                    do_no_step,
                )

                new_winner = jax.lax.cond(
                    next_done,
                    lambda: winner + manual_get_winner(
                        next_env.board, next_env.num_players,
                        next_env.goal, next_env.rules
                    ).astype(jnp.int32),
                    lambda: winner,
                )
                return next_env, next_done, new_winner

            return jax.lax.cond(
                ~done,
                lambda: do_step(env, winner),
                lambda: (env, done, winner),
            )

        new_envs, new_dones, new_winners = jax.vmap(step_single_env)(
            envs, dones, step_keys, winners
        )
        return (new_envs, new_winners, new_dones, step_count + 1, rng_key)

    def cond_fn(carry):
        _, _, dones, step_count, _ = carry
        return jnp.any(~dones) & (step_count < 2000)

    init_winners = jnp.zeros((num_envs, 4), dtype=jnp.int32)
    final_envs, final_winners, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (envs, init_winners, envs.done, 0, rng_key),
    )
    return final_envs, final_winners


def play_n_randomly(batch_size=20, seed=42):
    winners = jnp.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
    average_game_length = 0
    max_game_length = 0
    games_longer_600 = 0
    for i in range(4):
        envs = batch_reset(jax.random.randint(jax.random.PRNGKey(seed + i*54321), (batch_size,), 0, 1000000), jnp.full((batch_size,), i))
        active_mask = np.ones(batch_size, dtype=bool)
        step_counter = 0
        MAX_STEPS = 2000
        dones = envs.done  # Initialen Done-Status speichern
        
        while np.any(active_mask) and step_counter < MAX_STEPS:
            step_counter += 1
            rng_key, subkey = jax.random.split(jax.random.PRNGKey(seed + i*99999 + step_counter))
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
                    average_game_length += step_counter
                    if step_counter > max_game_length:
                        max_game_length = step_counter
                    if step_counter > 600:
                        games_longer_600 += 1
                    winner = manual_get_winner(envs.board[j], envs.num_players, envs.goal[j], envs.rules)
                    winners = winners.at[i].add(jnp.array(winner, dtype=jnp.int32))
            
            dones = final_dones  # Update dones für nächsten Step

    print("Final Results for Random Agents:")
    print("Total Wins per Player and different Starters:\n", winners)
    print("Total Wins per Player:\n", jnp.sum(winners, axis=0))
    print("Statistics:")
    if RULES['enable_teams']:
        total_win_chance = jnp.sum(winners,axis=0)/ jnp.sum(winners) * 100
        print("Total win chances in % for Team 0&2 and Team 1&3:", ( total_win_chance[0] + total_win_chance[2], total_win_chance[1] + total_win_chance[3]))
    else:
        print("Total win chances in %:", jnp.sum(winners,axis=0) / jnp.sum(winners) * 100)
    print("Chance to win when starting first:", jnp.sum(jnp.diag(winners)) / jnp.sum(winners) * 100)
    progress_mean, progress_all = calculate_player_progress(envs)
    print("Mean Final Pin distance per Player:\n", progress_mean)
    print("Average game length:", average_game_length / (4 * batch_size))
    print("Max game length:", max_game_length)
    print("Games longer than 600 steps:", games_longer_600)

def play_n_games_for_eval_jitted(params_list, rng_key, num_envs=20):
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
                
                obs = encode_board(env)[None, ...]
                valid_mask = valid_action(env).flatten()
                invalid_mask = (~valid_mask)[None, :]
                
                def do_mcts():
                    policy_output, _ = run_muzero_mcts(
                        params, key, obs, 
                        invalid_actions=invalid_mask,
                        num_simulations=NUM_SIMULATIONS,
                        max_depth=MAX_DEPTH,
                        temperature=TEMPERATURE
                    )
                    action = policy_output.action[0]
                    next_env, reward, next_done = env_step(env, map_action(action))
                    return next_env, next_done

                def do_gumbel():
                    policy_output, _ = run_gumbel_muzero_mcts(
                        params, key, obs, 
                        invalid_actions=invalid_mask,
                        num_simulations=NUM_SIMULATIONS,
                        max_depth=MAX_DEPTH,
                        temperature=TEMPERATURE
                    )
                    action = policy_output.action[0]
                    next_env, reward, next_done = env_step(env, map_action(action))
                    return next_env, next_done
                
                def do_random():
                    logits = jnp.where(valid_mask, 0.0, -1e9)
                    action = jax.random.categorical(key, logits)
                    next_env, reward, next_done = env_step(env, map_action(action))
                    return next_env, next_done
                
                def do_rule_based():
                    current_player = env.current_player
                    current_goal = env.goal[current_player] # (num_pins,)
                    current_positions = env.pins[current_player][:,None] # (num_pins, 1)
                    actions = jnp.arange(6)
                    # normal moved positions
                    moved_positions = current_positions + actions  # (num_pins, 6)
                    # fitted moved positions
                    fitted_positions = moved_positions % env.board_size # (num_pins, 6)
                    # steps into goal area
                    x = moved_positions - env.target[current_player] - jnp.int8(env.rules['must_traverse_start']) # (num_pins, 6)

                    # calc which position is correct for each pin x action
                    new_positions = jnp.where( # shape: (num_pins, 6)
                        (current_positions < 0) ,
                        env.start[current_player],  # pins in home can only move to start, shape: (num_pins, 6)
                        jnp.where(
                            current_positions >= env.board_size,
                            moved_positions,  # pins in goal area move normally, shape: (num_pins, 6)
                            jnp.where(
                                (4 >= x) & (x > 0) & (current_positions <= env.target[current_player]),
                                env.goal[current_player, x-1],
                                fitted_positions  # pins on board move normally, shape: (num_pins, 6)
                            )
                        )
                    ) 
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

                    # Count pins in home for early-game strategy
                    pins_in_home = jnp.sum(env.pins[current_player] < 0)
                    
                    # BASE SCORE: Prefer actions that are abundant
                    valid_mask_reshaped = valid_mask.reshape(4, 6)  # (num_pins, 6)
                    action_counts = jnp.sum(valid_mask_reshaped, axis=0)  # (6,)
                    action_abundance = action_counts / jnp.maximum(jnp.sum(action_counts), 1.0)
                    base_score = jnp.repeat(action_abundance, 4)  # (24,)
                    
                    # BONUS 1: Moving into goal (+5.0)
                    goal_bonus = jnp.where(
                        jnp.isin(new_positions, current_goal) & (current_positions < env.board_size),
                        5.0,
                        0.0
                    ).flatten()
                    
                    # BONUS 2: Getting pin out of house
                    out_of_home_weight = jnp.where(pins_in_home >= 2, 3.0, 2.0)
                    out_bonus = jnp.where(
                        (current_positions < 0) & (new_positions == env.start[current_player]),
                        out_of_home_weight,
                        0.0
                    ).flatten()
                    
                    # BONUS 3: Hitting opponent (+2.5)
                    hit_bonus = jnp.where(
                        (new_positions != current_positions) & jnp.isin(new_positions, opponent_pins),
                        2.0,
                        0.0
                    ).flatten()
                    
                    # TOTAL SCORE
                    policy_scores = base_score + goal_bonus + out_bonus + hit_bonus

                    # WICHTIG: Erst mit valid_mask maskieren, damit nur legale Aktionen gewählt werden!
                    policy_scores = jnp.where(valid_mask, policy_scores, -jnp.inf)
                    
                    # Softmax with temperature
                    temperature = 0.25
                    policy_logits = policy_scores / temperature
                    
                    action = jax.random.categorical(key, policy_logits)
                    mapped_act = map_action(action)
                    next_env, reward, next_done = env_step(env, mapped_act)
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
                            lambda: jax.lax.cond(
                                params['type'] == 1,
                                do_gumbel,
                                do_mcts
                            )
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
    'enable_teams': True,
    'enable_initial_free_pin': True,
    'enable_circular_board': False,
    'enable_friendly_fire': False,
    'enable_start_blocking': False,
    'enable_jump_in_goal_area': True,
    'enable_start_on_1': True,
    'enable_bonus_turn_on_6': True,
    'must_traverse_start': False
}

# start_time = time()
NUM_SIMULATIONS = 100
MAX_DEPTH = 50
TEMPERATURE = 0.05
# # play_n_randomly(batch_size=1000)  
params1 = None
params2 = None
params3 = None
params4 = None

# print("Begin longer evaluation session...")

# print("\nEvaluating 2 random agents vs 2 rule based agents:")
FOLDER = "MuZero_det_MADN/models/params/"
TRAINED_AGENTS = [
    f"{FOLDER}gumbelmuzero_madn_params_lr0.005_g1500_it50_seed76.pkl",
    f"{FOLDER}muzero_madn_params_lr0.005_g1500_it100_seed77.pkl",
]
p1 = load_params_from_file(TRAINED_AGENTS[0])
p2 = load_params_from_file(TRAINED_AGENTS[1])
evaluate_agent_parallel(p1, p2, p1, p2, 0,1,0,1, batch_size=250)
# print("First in Teams (0&2 vs 1&3), then all separate:")
# params1 = 'random_agent'
# params2 = 'rule_based_agent'
# params3 = 'random_agent'
# params4 = 'rule_based_agent'
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=200)

# RULES['enable_teams'] = False
# print("\nEvaluating 2 random agents vs 2 rule based agents without teams:")
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=200)
# print("\nEvaluating rule-based agent vs random agent:")
# params1 = 'rule_based_agent'
# params2 = 
# params3 = 'rule_based_agent'
# params4 = 'random_agent'
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)
# TEMPERATURE = 0.10

# FILENAME = "gumbelmuzero_madn_params_lr0.005_g1500_it100_seed71"

# print("\n{} vs random agents:".format(FILENAME))
# params1 = 'random_agent'
# params2 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params3 = 'random_agent'
# params4 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=250)

# print("\n{} vs rule-based agents:".format(FILENAME))
# params1 = 'rule_based_agent'
# params2 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params3 = 'rule_based_agent'
# params4 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=250)

# print("\n{} vs None agents:".format(FILENAME))
# params1 = None
# params2 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params3 = None
# params4 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=250)

# FILENAME2 = "gumbelmuzero_madn_params_lr0.005_g1500_it50_seed71"
# print("\n{} vs {}:".format(FILENAME, FILENAME2))
# params1 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params2 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# params3 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params4 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=250)

# FILENAME2 = "gumbelmuzero_madn_params_lr0.005_g1500_it100_seed69"
# print("\n{} vs {}:".format(FILENAME, FILENAME2))
# params1 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params2 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# params3 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params4 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=250)

# FILENAME2 = "gumbelmuzero_madn_params_lr0.005_g1500_it100_seed65"
# print("\n{} vs {}:".format(FILENAME, FILENAME2))
# params1 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params2 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# params3 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params4 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=250)

# FILENAME2 = "Experiment_33_100"
# print("\n{} vs {}:".format(FILENAME, FILENAME2))
# params1 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params2 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# params3 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME}.pkl')
# params4 = load_params_from_file(f'MuZero_det_MADN/models/params/{FILENAME2}.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=250)
# print("Seed Experiment_40_100 vs rule-based agents TEMP 0.20:")
# TEMPERATURE = 0.20
# params1 = load_params_from_file('MuZero_det_MADN/models/params/Experiment_40_100.pkl')
# params2 = 'rule_based_agent'
# params3 = load_params_from_file('MuZero_det_MADN/models/params/Experiment_40_100.pkl')
# params4 = 'rule_based_agent'
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

# print("Seed Experiment_40_100 vs None agents TEMP 0.20:")
# TEMPERATURE = 0.20
# params1 = load_params_from_file('MuZero_det_MADN/models/params/Experiment_40_100.pkl')
# params2 = None
# params3 = load_params_from_file('MuZero_det_MADN/models/params/Experiment_40_100.pkl')
# params4 = None
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)


# params2 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it150_seed78913.pkl')
# params4 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it150_seed78913.pkl')
# print("\nEvaluating TEAM Gumbel MuZero MADN (150) vs random agents:")
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

# params1 = 'rule_based_agent'
# params3 = 'rule_based_agent'
# params2 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it100_seed78913.pkl')
# params4 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it100_seed78913.pkl')
# print("\nEvaluating TEAM Gumbel MuZero MADN (100) vs rule-based agents:")
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

# params2 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it150_seed78913.pkl')
# params4 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it150_seed78913.pkl')
# print("\nEvaluating TEAM Gumbel MuZero MADN (150) vs rule-based agents:")
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

# print("vs 6115")
# params1 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it160_seed6115.pkl')
# params3 = load_params_from_file('models/params/TEAMgumbelmuzero_madn_params_lr0.01_g1500_it160_seed6115.pkl')
# evaluate_agent_parallel(params1, params2, params3, params4, batch_size=150)

# end_time = time()
# print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")


