import functools
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
from DOG.dog import *
# from MuZero_DOG.muzero_dog import *
from MuZero_DOG.eval_config.classic import run_muzero_mcts as classic_muzero, init_muzero_params as init_classic_params
from MuZero_DOG.eval_config.sample import run_muzero_mcts as sample_muzero, init_muzero_params as init_sample_params
from MuZero_DOG.eval_config.session import run_muzero_mcts as session_muzero, init_muzero_params as init_session_params

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
        distance=16,
        starting_player=starting_player,
        seed=seed,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=RULES['enable_teams'],
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        must_traverse_start=RULES['must_traverse_start'],
        disable_swapping=RULES['disable_swapping'],
        disable_hot_seven=RULES['disable_hot_seven'],
        disable_joker=RULES['disable_joker']
    )

# 2. Vektorisierte Funktionen vorbereiten
batch_reset = jax.vmap(env_reset_batched, in_axes=(0, 0))
batch_valid_action = jax.vmap(valid_actions)
batch_env_step = jax.vmap(env_step, in_axes=(0, 0))
batch_map_action = jax.vmap(map_move_to_action)


def calculate_progress(env: DOG, player_idx: int) -> int:
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
    enc = encode_board(env)  #
    agents = []
    for param, agent_type in zip([params1, params2, params3, params4], [type1, type2, type3, type4]):
        if param is None:
            effective_type = agent_type if agent_type is not None else 4
            rng = jax.random.PRNGKey(np.random.randint(0, 1000000))
            if effective_type == 2:
                param = init_classic_params(rng, enc.shape)
            elif effective_type == 3:
                param = init_sample_params(rng, enc.shape)
            else:  # 4 or unknown → session
                param = init_session_params(rng, enc.shape)
            param['type'] = effective_type
        elif param == 'rule_based_agent':
            param = init_session_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 1
        elif param == 'random_agent':
            param = init_session_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 0
        else:               
            param['type'] = agent_type if agent_type is not None else 1
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

def play_n_games_for_eval_jitted(params_list, rng_key, num_envs=20, starting_player=0):
    """JIT-compilierte Version wie game_agent"""
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs * 4,), 0, 1000000)
    envs = batch_reset(seeds, jnp.repeat(jnp.arange(4), num_envs))
    params_tuple = tuple(params_list)
    # agent_types is static so JAX compiles a separate kernel per unique combination,
    # allowing each player to use a different network architecture without lax.cond
    # tracing incompatible architectures against the wrong params.
    agent_types = tuple(int(p.get('type', 4)) for p in params_list)

    final_envs, winners = play_eval_loop_jitted(
        envs, params_tuple, subkey, num_envs * 4, agent_types
    )

    progress_mean, progress = calculate_player_progress(final_envs)
    return winners, progress

def _build_player_step_fn(agent_type: int):
    """Build a per-player step closure with the MCTS variant resolved at Python/trace time.

    agent_type (static int):
        0 = random
        1 = rule_based
        2 = classic_muzero  (stochastic MuZero, nn.Embed dynamics)
        3 = sample_muzero   (Gumbel + combined recurrent fn)
        4 = session_muzero  (Gumbel + deterministic recurrent fn)
    """
    def _step(params, key, env):
        obs = encode_board(env)[None, ...]
        valid_mask = valid_actions(env).flatten()
        invalid_mask = (~valid_mask)[None, :]

        # MCTS function baked in at trace time — no lax.cond across architectures.
        if agent_type == 2:
            def do_mcts():
                policy_output, _ = classic_muzero(
                    params, key, obs,
                    invalid_actions=invalid_mask,
                    num_simulations=NUM_SIMULATIONS,
                    max_depth=MAX_DEPTH,
                    temperature=TEMPERATURE,
                )
                action = policy_output.action[0]
                next_env, _, next_done = env_step(env, action)
                return next_env, next_done
        elif agent_type == 3:
            def do_mcts():
                policy_output, _ = sample_muzero(
                    params, key, obs,
                    invalid_actions=invalid_mask,
                    num_simulations=NUM_SIMULATIONS,
                    max_depth=MAX_DEPTH,
                    temperature=TEMPERATURE,
                )
                action = policy_output.action[0]
                next_env, _, next_done = env_step(env, action)
                return next_env, next_done
        else:  # type 4 or unknown trained agent → session
            def do_mcts():
                policy_output, _ = session_muzero(
                    params, key, obs,
                    invalid_actions=invalid_mask,
                    num_simulations=NUM_SIMULATIONS,
                    max_depth=MAX_DEPTH,
                    temperature=TEMPERATURE,
                )
                action = policy_output.action[0]
                next_env, _, next_done = env_step(env, action)
                return next_env, next_done

        def do_random():
            logits = jnp.where(valid_mask, 0.0, -1e9)
            action = jax.random.categorical(key, logits)
            next_env, _, next_done = env_step(env, action)
            return next_env, next_done

        def do_rule_based():
            def do_play_phase():
                player_id = env.current_player
                actual_player = jnp.where(
                    env.rules["enable_teams"] & is_player_done(
                        env.num_players, env.board, env.goal, player_id),
                    (player_id + 2) % 4,
                    player_id
                )
                baseline = calculate_progress(env, actual_player)

                def score_action(a):
                    ne, _, _ = env_step(env, a)
                    return baseline - calculate_progress(ne, actual_player)

                return jnp.concatenate([jax.vmap(score_action)(jnp.arange(440)), jnp.full(14, -jnp.inf)])

            def do_swap_phase():
                player_id = env.current_player
                partner   = (player_id + 2) % 4
                hand      = env.hands[player_id].astype(jnp.float32)

                own_progress     = calculate_progress(env, player_id)
                partner_progress = calculate_progress(env, partner)
                own_done         = own_progress <= 0.0
                partner_needs    = partner_progress > 0.0

                base_priority = jnp.array([
                     0.0,  # 0:  Joker        — keep
                     6.0,  # 1:  Swap-pin     — situational
                     8.0,  # 2:  2            — small, easy to pass
                     7.0,  # 3:  3
                     7.0,  # 4:  -4/4         — risky, pass first
                     7.0,  # 5:  5
                     7.0,  # 6:  6
                     0.0,  # 7:  hot-7        — very strong, keep
                     6.0,  # 8:  8
                     6.0,  # 9:  9
                     5.0,  # 10: 10
                     0.0,  # 11: 1/11         — exit-home card, keep
                     5.0,  # 12: 12
                     2.0,  # 13: 13           — exit-home card, keep
                ], dtype=jnp.float32)
                starting_bonus = jnp.array([
                    15.0,  # 0:  Joker
                     0.0,  # 1:  Swap-pin
                     0.0,  # 2:  2
                     0.0,  # 3:  3
                     0.0,  # 4:  -4/4
                     0.0,  # 5:  5
                     0.0,  # 6:  6
                     0.0,  # 7:  hot-7
                     0.0,  # 8:  8
                     0.0,  # 9:  9
                     0.0,  # 10: 10
                    15.0,  # 11: 1/11
                     0.0,  # 12: 12
                    15.0,  # 13: 13
                ], dtype=jnp.float32)
                pass_starting_bonus = jnp.where(own_done & partner_needs, starting_bonus, 0.0)
                is_special = jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=jnp.float32)
                double_bonus = jnp.where((hand >= 2.0) & (is_special > 0), 3.0, 0.0)
                priority    = base_priority + pass_starting_bonus + double_bonus
                in_hand     = hand > 0.0
                swap_scores = jnp.where(in_hand, priority, -jnp.inf)
                return jnp.concatenate([jnp.full(440, -jnp.inf), swap_scores])

            scores = jax.lax.cond(env.phase == 0, do_play_phase, do_swap_phase)
            action = jnp.argmax(jnp.where(valid_mask, scores, -jnp.inf))
            next_env, _, next_done = env_step(env, action)
            return next_env, next_done

        def do_no_step():
            next_env, _, next_done = no_step(env)
            return next_env, next_done

        # Dispatch random/rule_based at JAX level (same architecture — safe).
        # MCTS variant is already baked in via do_mcts() above (Python-level).
        if agent_type == 0:
            def _action(env):
                return jax.lax.cond(jnp.any(valid_mask), do_random, do_no_step)
        elif agent_type == 1:
            def _action(env):
                return jax.lax.cond(jnp.any(valid_mask), do_rule_based, do_no_step)
        else:
            def _action(env):
                return jax.lax.cond(jnp.any(valid_mask), do_mcts, do_no_step)

        return _action(env)

    return _step


@functools.partial(jax.jit, static_argnames=['num_envs', 'agent_types'])
def play_eval_loop_jitted(envs, params_tuple, rng_key, num_envs, agent_types):
    """Vollständig JIT-compiliert.

    agent_types is a static tuple of ints, one per player position.
    JAX compiles a fresh version for each unique combination, so players
    with different network architectures never get traced against each other.
    """
    # Build per-player step functions at Python/trace time.
    step_fns = [_build_player_step_fn(agent_types[i]) for i in range(4)]

    def body_fn(carry):
        envs, winners, dones, step_count, rng_key = carry

        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)

        def step_single_env(env, done, key, winner):
            def do_step(env, winner):
                current_player = env.current_player
                # Switch selects the right pre-built step fn; all have identical
                # output structure so lax.switch is safe here.
                # Use default-arg binding to capture each i's value at definition
                # time — without it, all lambdas would close over the same i=3.
                next_env, next_done = jax.lax.switch(
                    current_player,
                    [lambda fn=step_fns[i], par=params_tuple[i]: fn(par, key, env)
                     for i in range(4)],
                )

                def update_winner(env, winner):
                    win = manual_get_winner(env.board, env.num_players, env.goal, env.rules)
                    return winner + win.astype(jnp.int32)

                new_winner = jax.lax.cond(
                    next_done,
                    lambda: update_winner(next_env, winner),
                    lambda: winner,
                )
                return next_env, next_done, new_winner

            def skip_step(env, winner):
                return env, done, winner

            return jax.lax.cond(
                ~done,
                lambda: do_step(env, winner),
                lambda: skip_step(env, winner),
            )

        new_envs, new_dones, new_winners = jax.vmap(step_single_env)(
            envs, dones, step_keys, winners
        )

        return (new_envs, new_winners, new_dones, step_count + 1, rng_key)

    init_dones = envs.done
    init_winners = jnp.zeros((num_envs, 4), dtype=jnp.int32)

    def cond_fn(carry):
        _, _, dones, step_count, _ = carry
        return jnp.any(~dones) & (step_count < 2000)

    final_envs, final_winners, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (envs, init_winners, init_dones, 0, rng_key),
    )

    return final_envs, final_winners

def fairness_check(batch_size=20, seed=None):
    """
    Plays batch_size * 4 games with exclusively random agents.
    Each starting player (0-3) starts exactly batch_size games.
    Uses a dedicated JIT loop that only knows random / no_step.
    """
    if seed is None:
        seed = np.random.randint(0, 1_000_000)

    rng_key = jax.random.PRNGKey(seed)
    rng_key, subkey = jax.random.split(rng_key)

    num_total = batch_size * 4
    seeds = jax.random.randint(subkey, (num_total,), 0, 1_000_000)
    starting_players = jnp.repeat(jnp.arange(4), batch_size)
    envs = batch_reset(seeds, starting_players)

    final_envs, winners_flat, pscs_flat = _random_only_loop(envs, subkey, num_total)

    winners_split = jnp.array_split(winners_flat, 4, axis=0)
    winners = jnp.stack([jnp.sum(w, axis=0) for w in winners_split])  # (4, 4)

    progress_mean, _ = calculate_player_progress(final_envs)

    total_wins = jnp.sum(winners)
    wins_per_player = jnp.sum(winners, axis=0)
    win_pct = wins_per_player / jnp.maximum(total_wins, 1) * 100

    # Step counts per player (summed over all games)
    steps_per_player = jnp.sum(pscs_flat, axis=0)  # (4,)
    total_steps = jnp.sum(steps_per_player)
    steps_pct = steps_per_player / jnp.maximum(total_steps, 1) * 100

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
    print("\nTotal Steps per Player:", steps_per_player)
    print("Steps % per Player:", steps_pct)
    if RULES['enable_teams']:
        steps_a = float(steps_pct[0] + steps_pct[2])
        steps_b = float(steps_pct[1] + steps_pct[3])
        print(f"Steps Team A (0&2): {steps_a:.2f}%  |  Steps Team B (1&3): {steps_b:.2f}%")
    print("\nMean Final Pin Distance per Player:\n", progress_mean)
    print("=" * 60)

    return winners, progress_mean


@functools.partial(jax.jit, static_argnames=['num_envs'])
def _random_only_loop(envs, rng_key, num_envs):
    """JIT loop that only executes random actions or no_step."""

    def body_fn(carry):
        envs, winners, dones, step_count, rng_key, player_step_counts = carry

        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)

        def step_single_env(env, done, key, winner, psc):
            def do_step(env, winner, psc):
                valid_mask = valid_actions(env).flatten()

                def do_random():
                    logits = jnp.where(valid_mask, 0.0, -1e9)
                    action = jax.random.categorical(key, logits)
                    next_env, _, next_done = env_step(env, action)
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
                # Count the step for the player who just acted
                new_psc = psc + (jnp.arange(4) == env.current_player).astype(jnp.int32)
                return next_env, next_done, new_winner, new_psc

            return jax.lax.cond(
                ~done,
                lambda: do_step(env, winner, psc),
                lambda: (env, done, winner, psc),
            )

        new_envs, new_dones, new_winners, new_pscs = jax.vmap(step_single_env)(
            envs, dones, step_keys, winners, player_step_counts
        )
        return (new_envs, new_winners, new_dones, step_count + 1, rng_key, new_pscs)

    def cond_fn(carry):
        _, _, dones, step_count, _, _ = carry
        return jnp.any(~dones) & (step_count < 2000)

    init_winners = jnp.zeros((num_envs, 4), dtype=jnp.int32)
    init_player_step_counts = jnp.zeros((num_envs, 4), dtype=jnp.int32)
    final_envs, final_winners, _, _, _, final_pscs = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (envs, init_winners, envs.done, 0, rng_key, init_player_step_counts),
    )
    return final_envs, final_winners, final_pscs


# Rules for evaluation games - can be adjusted to test specific rule variations
RULES = {
    'enable_teams': True, # DOG-standard is True
    'enable_initial_free_pin': False, # DOG-standard is False
    'enable_circular_board': False, # DOG-standard is True
    'enable_friendly_fire': True, # DOG-standard is True
    'enable_start_blocking': True, # DOG-standard is True
    'enable_jump_in_goal_area': False, # DOG-standard is False
    'must_traverse_start': True, # DOG-standard is True
    'disable_swapping': False, # DOG-standard is False
    'disable_hot_seven': False, # DOG-standard is False
    'disable_joker': False, # DOG-standard is False
}
NUM_SIMULATIONS = 50
MAX_DEPTH = 25
TEMPERATURE = 0.0

