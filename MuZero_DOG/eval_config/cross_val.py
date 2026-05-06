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
    enc_with_belief = encode_board_with_belief(env)  #
    agents = []
    for param, agent_type in zip([params1, params2, params3, params4], [type1, type2, type3, type4]):
        if param is None:
            inferred_type = 1 if agent_type is None else agent_type
            if inferred_type == 4:
                param = init_session_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc_with_belief.shape)
            else:
                param = init_session_params(jax.random.PRNGKey(np.random.randint(0, 1000000)), enc.shape)
            param['type'] = inferred_type
        elif param == 'rule_based_agent':
            param = init_session_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 2
        elif param == 'random_agent':
            param = init_session_params(jax.random.PRNGKey(0), enc.shape)
            param['type'] = 3
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

def _make_obs_fn(use_belief: bool):
    """Return a Python-level observation function (not traced by JAX)."""
    if use_belief:
        return encode_board_with_belief
    return encode_board


def play_n_games_for_eval_jitted(params_list, rng_key, num_envs=20):
    """JIT-compilierte Version wie game_agent.
    
    Each player's observation function is determined at Python level from
    params['type']: type 4 uses encode_board_with_belief; all others use encode_board.
    Because JAX traces both branches of lax.cond, agents with different network
    shapes CANNOT share a single jitted loop. Instead we group players by their
    observation type and run one loop per unique combination of agent types.
    """
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs * 4,), 0, 1000000)
    envs = batch_reset(seeds, jnp.repeat(jnp.arange(4), num_envs))
    params_tuple = tuple(params_list)

    # Determine per-player obs type at Python level (static for JIT)
    obs_types = tuple(int(p.get('type', 1)) for p in params_list)

    final_envs, winners = play_eval_loop_jitted(
        envs, params_tuple, subkey, num_envs * 4, obs_types
    )

    progress_mean, progress = calculate_player_progress(final_envs)
    return winners, progress


def _build_step_fn(obs_type: int):
    """Return a closure that takes (params, key, env) → (next_env, next_done).
    
    obs_type is resolved at Python/tracing time so each returned function has a
    fixed observation shape and can be safely traced by JAX.
    """
    use_belief = (obs_type == 4)

    def _step(params, key, env):
        if use_belief:
            obs = encode_board_with_belief(env)[None, ...]
        else:
            obs = encode_board(env)[None, ...]
        valid_mask = valid_actions(env).flatten()
        invalid_mask = (~valid_mask)[None, :]

        def do_mcts():
            policy_output, _ = session_muzero(
                params, key, obs,
                invalid_actions=invalid_mask,
                num_simulations=NUM_SIMULATIONS,
                max_depth=MAX_DEPTH,
                temperature=TEMPERATURE
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

        agent_type = params['type']
        return jax.lax.cond(
            jnp.any(valid_mask),
            lambda: jax.lax.cond(
                agent_type == 3,
                do_random,
                lambda: jax.lax.cond(
                    agent_type == 2,
                    do_rule_based,
                    do_mcts,  # type 1 or 4 — obs shape already fixed above
                )
            ),
            do_no_step,
        )

    return _step


@functools.partial(jax.jit, static_argnames=['num_envs', 'obs_types'])
def play_eval_loop_jitted(envs, params_tuple, rng_key, num_envs, obs_types):
    """Vollständig JIT-compiliert.
    
    obs_types is a static tuple of ints (one per player) so JAX compiles a
    separate version for each unique combination of agent observation types.
    This lets players with different encoding shapes co-exist safely.
    """
    # Build per-player step functions at trace time (static on obs_types)
    step_fns = [_build_step_fn(obs_types[i]) for i in range(4)]

    def body_fn(carry):
        envs, winners, dones, step_count, rng_key = carry

        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)

        def step_single_env(env, done, key, winner):
            def do_step(env, winner):
                current_player = env.current_player
                # Select the right per-player step function at trace time via switch
                next_env, next_done = jax.lax.switch(
                    current_player,
                    [lambda: step_fns[i](params_tuple[i], key, env) for i in range(4)],
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

def collect_random_game_stats(num_envs=50, max_steps=2000, seed=42):
    """
    Plays fully random games and collects per-step stats:
      - number of valid actions
      - hand_size of the current player
      - phase (0=play, 1=swap)
      - current player index

    All stats are filtered to real steps (has_valid=True) only.
    Prints a full breakdown at the end.
    """
    rng_key = jax.random.PRNGKey(seed)
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs,), 0, 1_000_000)
    envs = jax.vmap(lambda s: env_reset_batched(s, 0))(seeds)

    # Per-step stat accumulators (numpy lists, appended each outer step)
    all_n_valid   = []  # int
    all_hand_size = []  # int
    all_phase     = []  # 0 or 1
    all_player    = []  # 0..3

    # No-step event accumulators (active env, but has_valid=False → no legal move)
    no_step_hand_size = []
    no_step_phase     = []
    no_step_player    = []

    # Action frequency counter — shape (454,), accumulated across all steps
    action_freq = np.zeros(454, dtype=np.int64)

    # Low-valid-action samples: capture env state when n_valid <= 2
    # Sampled lazily (at most MAX_LOW_SAMPLES total, reservoir-sampled)
    MAX_LOW_SAMPLES = 50
    low_samples = []     # list of dicts, filled during the loop
    low_sample_count = 0  # total low-valid steps seen (for reservoir sampling)

    @jax.jit
    def random_step_all(envs, rng_key):
        """One random step for every env in parallel. Returns (next_envs, stats, dones)."""
        def step_one(env, key):
            valid_mask = valid_actions(env).flatten()   # (454,)
            has_valid  = jnp.any(valid_mask)

            def do_random(env):
                logits   = jnp.where(valid_mask, 0.0, -1e9)
                action   = jax.random.categorical(key, logits)
                next_env, _, done = env_step(env, action)
                return next_env, done, action.astype(jnp.int32)

            def do_no_step(env):
                next_env, _, done = no_step(env)
                return next_env, done, jnp.int32(-1)

            next_env, done, action_taken = jax.lax.cond(has_valid, do_random, do_no_step, env)

            n_valid   = jnp.sum(valid_mask).astype(jnp.int32)
            hand_size = env.hand_size.astype(jnp.int32)
            phase     = env.phase.astype(jnp.int32)
            player    = env.current_player.astype(jnp.int32)
            return next_env, done, n_valid, hand_size, phase, player, has_valid, action_taken

        keys = jax.random.split(rng_key, num_envs)
        next_envs, dones, n_valids, hand_sizes, phases, players, has_valids, actions_taken = jax.vmap(step_one)(envs, keys)
        return next_envs, dones, n_valids, hand_sizes, phases, players, has_valids, actions_taken

    active = np.ones(num_envs, dtype=bool)
    for step in range(max_steps):
        if not np.any(active):
            break
        rng_key, subkey = jax.random.split(rng_key)
        prev_envs = envs  # snapshot BEFORE step — used for low-valid sampling
        envs, dones, n_valids, hand_sizes, phases, players, has_valids, actions_taken = random_step_all(envs, subkey)

        # device → host once per outer step
        dones_np        = np.array(dones,         dtype=bool)
        n_valids_np     = np.array(n_valids,       dtype=np.int32)
        hand_sizes_np   = np.array(hand_sizes,     dtype=np.int32)
        phases_np       = np.array(phases,         dtype=np.int32)
        players_np      = np.array(players,        dtype=np.int32)
        has_valids_np   = np.array(has_valids,     dtype=bool)
        actions_taken_np = np.array(actions_taken, dtype=np.int32)

        # Only record real (non-forced-skip) steps in still-active envs
        record_mask = active & has_valids_np
        if np.any(record_mask):
            all_n_valid.extend(n_valids_np[record_mask].tolist())
            all_hand_size.extend(hand_sizes_np[record_mask].tolist())
            all_phase.extend(phases_np[record_mask].tolist())
            all_player.extend(players_np[record_mask].tolist())
            # Accumulate action frequencies for recorded steps
            valid_actions_taken = actions_taken_np[record_mask]
            np.add.at(action_freq, valid_actions_taken, 1)

        # ── Low-valid-action sampling ─────────────────────────────────
        # For steps with n_valid <= 2, capture env state for diagnostic printing.
        # Uses reservoir sampling so we get a uniform sample over all such steps.
        low_mask = active & has_valids_np & (n_valids_np <= 2)
        low_indices = np.where(low_mask)[0]
        for env_i in low_indices:
            low_sample_count += 1
            # Reservoir sampling: keep with prob MAX_LOW_SAMPLES / count
            if len(low_samples) < MAX_LOW_SAMPLES:
                slot = len(low_samples)
                do_insert = True
            else:
                slot = int(np.random.randint(0, low_sample_count))
                do_insert = slot < MAX_LOW_SAMPLES
            if do_insert:
                # Extract single env from the pre-step batch (consistent with phases_np, n_valids_np)
                env_single = jax.tree_util.tree_map(lambda x: x[env_i], prev_envs)
                cp = int(np.array(env_single.current_player))
                hand = np.array(env_single.hands[cp], dtype=np.int32)
                pins = np.array(env_single.pins, dtype=np.int32)
                phase = int(phases_np[env_i])
                hs = int(hand_sizes_np[env_i])
                nv = int(n_valids_np[env_i])
                # Decode which specific action indices are valid
                vmask = np.array(valid_actions(env_single).flatten(), dtype=bool)
                valid_idxs = np.where(vmask)[0].tolist()
                sample = {
                    'n_valid': nv, 'phase': phase, 'hand_size': hs,
                    'player': cp, 'hand': hand.tolist(), 'pins': pins.tolist(),
                    'valid_action_indices': valid_idxs,
                }
                if len(low_samples) < MAX_LOW_SAMPLES:
                    low_samples.append(sample)
                else:
                    low_samples[slot] = sample
        no_step_mask = active & ~has_valids_np
        if np.any(no_step_mask):
            no_step_hand_size.extend(hand_sizes_np[no_step_mask].tolist())
            no_step_phase.extend(phases_np[no_step_mask].tolist())
            no_step_player.extend(players_np[no_step_mask].tolist())

        active &= ~dones_np

    # ── Analysis ────────────────────────────────────────────────
    all_n_valid   = np.array(all_n_valid,   dtype=np.int32)
    all_hand_size = np.array(all_hand_size, dtype=np.int32)
    all_phase     = np.array(all_phase,     dtype=np.int32)
    all_player    = np.array(all_player,    dtype=np.int32)
    N = len(all_n_valid)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RANDOM GAME STATS  ({num_envs} games, {N:,} real steps)")
    print(sep)

    print(f"\n--- Valid Actions per Step ---")
    print(f"  mean   : {all_n_valid.mean():.2f}")
    print(f"  median : {int(np.median(all_n_valid))}")
    print(f"  min    : {all_n_valid.min()}")
    print(f"  max    : {all_n_valid.max()}")
    buckets = [(1,1,'=1'), (2,3,'2-3'), (4,6,'4-6'), (7,10,'7-10'),
               (11,20,'11-20'), (21,50,'21-50'), (51,100,'51-100'), (101,454,'101+')]
    for lo, hi, lbl in buckets:
        cnt = int(np.sum((all_n_valid >= lo) & (all_n_valid <= hi)))
        if cnt > 0:
            print(f"    {lbl:>8} : {cnt:>8,}  ({100*cnt/N:.1f}%)")

    print(f"\n--- Valid Actions by Hand Size ---")
    for hs in sorted(np.unique(all_hand_size)):
        mask = all_hand_size == hs
        sub  = all_n_valid[mask]
        print(f"  hand_size={hs} : {mask.sum():>7,} steps | "
              f"mean={sub.mean():.2f}  median={int(np.median(sub))}  "
              f"min={sub.min()}  max={sub.max()}")

    print(f"\n--- Valid Actions by Phase ---")
    for ph, label in [(0, 'play'), (1, 'swap')]:
        mask = all_phase == ph
        if mask.sum() == 0:
            continue
        sub = all_n_valid[mask]
        print(f"  phase={ph} ({label}) : {mask.sum():>7,} steps | "
              f"mean={sub.mean():.2f}  median={int(np.median(sub))}  "
              f"min={sub.min()}  max={sub.max()}")

    print(f"\n--- Step Distribution by Phase ---")
    for ph, label in [(0, 'play'), (1, 'swap')]:
        cnt = int(np.sum(all_phase == ph))
        print(f"  phase={ph} ({label}) : {cnt:>8,}  ({100*cnt/N:.1f}%)")

    print(f"\n--- Step Distribution by Player ---")
    for p in range(4):
        cnt = int(np.sum(all_player == p))
        print(f"  player {p} : {cnt:>8,}  ({100*cnt/N:.1f}%)")

    print(f"\n--- Hand Size Distribution ---")
    for hs in sorted(np.unique(all_hand_size)):
        cnt = int(np.sum(all_hand_size == hs))
        print(f"  hand_size={hs} : {cnt:>8,}  ({100*cnt/N:.1f}%)")

    # ── Action frequency breakdown ───────────────────────────────
    # Action space layout (440 play + 14 swap-phase = 454 total):
    #   [0:220]    joker copies of all play actions
    #   [0:48]     joker × swap-pin moves
    #   [48:168]   joker × hot-7 splits (120 combos)
    #   [168:212]  joker × normal moves (11 cards × 4 pins)
    #   [212:220]  joker × neg-4 moves (4 pins × 2)  ← actually last 4 of first 220
    #   [220:268]  swap-pin moves (48)
    #   [268:388]  hot-7 splits (120)
    #   [388:432]  normal card moves (11 cards × 4 pins)
    #   [432:440]  neg-4 moves (4 pins, counted ×2 here: 8 total across joker/real)
    #   [440:454]  swap-phase actions (14)
    segments = [
        (0,   220, "joker copies (0-219)"),
        (0,    48, "  joker × swap-pin (0-47)"),
        (48,  168, "  joker × hot-7     (48-167)"),
        (168, 216, "  joker × normal    (168-215)"),
        (216, 220, "  joker × neg-4     (216-219)"),
        (220, 440, "real play actions (220-439)"),
        (220, 268, "  real swap-pin   (220-267)"),
        (268, 388, "  real hot-7       (268-387)"),
        (388, 436, "  real normal      (388-435)"),
        (436, 440, "  real neg-4       (436-439)"),
        (440, 454, "swap-phase actions (440-453)"),
    ]
    print(f"\n--- Action Frequency by Segment ---")
    print(f"  (total recorded steps: {N:,})")
    for lo, hi, label in segments:
        cnt    = int(action_freq[lo:hi].sum())
        nonzero = int(np.sum(action_freq[lo:hi] > 0))
        indent = "  " if label.startswith("  ") else ""
        print(f"  {indent}{label:<38} : {cnt:>8,}  ({100*cnt/N:5.1f}%)  "
              f"[{nonzero}/{hi-lo} actions ever used]")
    print(f"\n--- Zero-frequency actions ---")
    never_used = int(np.sum(action_freq == 0))
    print(f"  {never_used}/454 actions never selected in {num_envs} random games")

    # ── Low-valid-action deep-dive ───────────────────────────────
    # Card index → human name (deck layout: 0=joker,1=swap,2..13=cards 2-13 except 7→hot7)
    CARD_NAMES = {0:'Joker', 1:'Swap', 2:'2', 3:'3', 4:'4(-4)', 5:'5',
                  6:'6', 7:'7(hot)', 8:'8', 9:'9', 10:'10', 11:'1/11', 12:'12', 13:'13'}
    # Action segment labels for quick human-readable decode
    def action_label(idx):
        if idx < 0:   return 'none'
        if idx < 48:  return f'J×swap #{idx}'
        if idx < 168: return f'J×hot7 dist#{idx-48}'
        if idx < 216: return f'J×normal #{idx-168}'
        if idx < 220: return f'J×neg4 pin{idx-216}'
        if idx < 268: return f'swap #{idx-220}'
        if idx < 388: return f'hot7 dist#{idx-268}'
        if idx < 436: return f'normal #{idx-388}'
        if idx < 440: return f'neg4 pin{idx-436}'
        return f'swap-phase card{idx-440}'

    print(f"\n--- Low-Valid-Action Samples (n_valid<=2, {len(low_samples)} of {low_sample_count} sampled) ---")
    print(f"  hand layout: [Joker,Swap,2,3,4(-4),5,6,7(hot),8,9,10,1/11,12,13]")
    print(f"  pins layout: (player, pin_slot) — value=-1 means home, 0..63 circular board, 64+ goal")
    print()
    for s in low_samples:
        hand = s['hand']
        cards_held = [CARD_NAMES[i] for i, cnt in enumerate(hand) if cnt > 0]
        pins_str = '  '.join(
            f"P{pi}:{s['pins'][pi]}" for pi in range(len(s['pins']))
        )
        acts_str = '  '.join(action_label(a) for a in s['valid_action_indices'])
        print(f"  n={s['n_valid']}  phase={'play' if s['phase']==0 else 'swap'}  "
              f"hs={s['hand_size']}  player={s['player']}")
        print(f"    hand      : {hand}  →  cards: {cards_held}")
        print(f"    pins      : {pins_str}")
        print(f"    valid acts: [{acts_str}]")
        print()

    # ── No-step events ───────────────────────────────────────────
    no_step_hand_size_np = np.array(no_step_hand_size, dtype=np.int32)
    no_step_phase_np     = np.array(no_step_phase,     dtype=np.int32)
    no_step_player_np    = np.array(no_step_player,    dtype=np.int32)
    N_no = len(no_step_phase_np)
    total_active = N + N_no
    print(f"\n--- No-Step Events (game active, no legal moves) ---")
    print(f"  Total no-steps : {N_no:,}  ({100*N_no/total_active:.2f}% of all active steps)")
    if N_no > 0:
        print(f"  By phase:")
        for ph, label in [(0, 'play'), (1, 'swap')]:
            cnt = int(np.sum(no_step_phase_np == ph))
            if cnt > 0:
                print(f"    phase={ph} ({label}) : {cnt:>7,}  ({100*cnt/N_no:.1f}%)")
        print(f"  By hand size:")
        for hs in sorted(np.unique(no_step_hand_size_np)):
            cnt = int(np.sum(no_step_hand_size_np == hs))
            print(f"    hand_size={hs} : {cnt:>7,}  ({100*cnt/N_no:.1f}%)")
        print(f"  By player:")
        for p in range(4):
            cnt = int(np.sum(no_step_player_np == p))
            print(f"    player {p} : {cnt:>7,}  ({100*cnt/N_no:.1f}%)")

    print(sep)
    return (all_n_valid, all_hand_size, all_phase, all_player, action_freq,
            no_step_hand_size_np, no_step_phase_np, no_step_player_np)

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
# start_time = time()
NUM_SIMULATIONS = 50
MAX_DEPTH = 25
TEMPERATURE = 0.0
