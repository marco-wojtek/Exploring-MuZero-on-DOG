import chex
import jax
import jax.numpy as jnp
from flax import struct
import sys, os
from time import time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *
from MuZero_DOG.muzero_dog_session import *

# RULES = {
#     'enable_teams': True, # DOG-standard is True
#     'enable_initial_free_pin': True, # DOG-standard is False
#     'enable_circular_board': False, # DOG-standard is True
#     'enable_friendly_fire': True, # DOG-standard is True
#     'enable_start_blocking': True, # DOG-standard is True
#     'enable_jump_in_goal_area': False, # DOG-standard is False
#     'must_traverse_start': False, # DOG-standard is True
#     'disable_swapping': False, # DOG-standard is False
#     'disable_hot_seven': False, # DOG-standard is False
#     'disable_joker': False, # DOG-standard is False
# }
RULES = {
    'enable_teams': True, # DOG-standard is True
    'enable_initial_free_pin': False, # DOG-standard is False
    'enable_circular_board': True, # DOG-standard is True
    'enable_friendly_fire': True, # DOG-standard is True
    'enable_start_blocking': True, # DOG-standard is True
    'enable_jump_in_goal_area': False, # DOG-standard is False
    'must_traverse_start': True, # DOG-standard is True
    'disable_swapping': False, # DOG-standard is False
    'disable_hot_seven': False, # DOG-standard is False
    'disable_joker': False, # DOG-standard is False
}

def env_reset_batched(seed):
    return env_reset(
        0,  # <- Das wird an '_' übergeben
        num_players=4,
        layout=jnp.array([True, True, True, True], dtype=jnp.bool_),
        distance=16,
        starting_player=0,
        seed=seed,  # <- Das ist das eigentliche Seed-Keyword-Argument
        enable_teams=RULES['enable_teams'],
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        must_traverse_start=RULES['must_traverse_start'],
        disable_swapping=RULES['disable_swapping'],
        disable_hot_seven=RULES['disable_hot_seven'],
        disable_joker=RULES['disable_joker']
    )

# 2. Vektorisierte Funktionen vorbereiten
# NOTE: batch_reset cannot be jax.jit-wrapped because env_reset uses boolean array
# indexing ([layout]) whose output shape depends on concrete values — incompatible
# with JAX abstract tracing. Plain vmap is correct here.
batch_reset = jax.vmap(env_reset_batched)
batch_valid_action = jax.vmap(valid_actions)
batch_env_step = jax.vmap(env_step, in_axes=(0, 0))

@functools.partial(jax.jit, static_argnames=['num_envs', 'input_shape', 'num_simulations', 'max_depth', 'max_steps', 'temp'])
def play_batch_of_games_jitted(envs, num_envs, input_shape, params, rng_key, num_simulations, max_depth, max_steps, temp):
    """MCTS parallel + Early Exit + XLA optimiert
    Verwende play_batch_of_games_jitted, wenn du viele Spiele parallel simulieren möchtest, insbesondere für Training oder Datengewinnung.
    """
    def body_fn(carry):
        envs_state, buffers, dones, step_count, rng_key = carry
        
        # Neue Keys für diesen Step generieren
        rng_key, *step_keys = jax.random.split(rng_key, num_envs + 1)
        step_keys = jnp.array(step_keys)

        # ✅ PARALLEL: vmap über alle aktiven Envs
        def step_single_env(env, buffer, done, key):
            def do_active_step(env, buffer):
                # 1. WÜRFELN (automatisch in der Environment)
                key1, key2 = jax.random.split(key)
                
                obs = encode_board_with_belief(env)[None, ...] # TODO:
                valid_mask = valid_actions(env).flatten()
                invalid_mask = (~valid_mask)[None, :]
                has_valid = jnp.any(valid_mask)

                current_player_before = env.current_player
                current_team_before = jax.lax.cond(
                    env.rules['enable_teams'],
                    lambda: jnp.int8(current_player_before % 2),
                    lambda: jnp.int8(-1)
                )
                
                # 3. Unterscheidung: MCTS oder no_step
                def do_mcts(env):
                    # Stochastic MuZero MCTS
                    policy_output, root_value = run_muzero_mcts(
                        params, key2, obs, invalid_actions=invalid_mask, num_simulations=num_simulations, max_depth=max_depth, temperature=temp
                    )
                    # Action ist ein Index (0-454)
                    action = policy_output.action[0]
                    next_env, reward, next_done = env_step(env, action)

                    # Spieler NACH dem Zug (wichtig für Reward- und Discount-Targets!)
                    next_player = next_env.current_player
                    next_team = jax.lax.cond(
                        env.rules['enable_teams'],
                        lambda: jnp.int8(next_player % 2),
                        lambda: jnp.int8(-1)
                    )

                    # Reward Target: Klasse 0=-1, Klasse 1=0, Klasse 2=+1
                    reward_target = jnp.where(
                        next_done & (reward > 0), 2,
                        jnp.where(next_done & (reward < 0), 0, 1)
                    )

                    # Discount Target: Klasse 0=-1, Klasse 1=0, Klasse 2=+1
                    discount_target = jnp.where(
                        next_done, 1,  # Terminal → Klasse 1 (discount=0)
                        jax.lax.cond(
                            env.rules['enable_teams'],
                            lambda: jnp.where(current_team_before == next_team, 2, 0),
                            lambda: jnp.where(current_player_before == next_player, 2, 0)
                        )
                    )

                    # Detect if a deal happened: only a play-phase→swap-phase transition
                    # (env_step_play_phase calls distribute_cards which sets phase=1).
                    # Using hand-sum comparison fires a false positive on the 4th swap:
                    # execute_team_swap re-adds 4 cards after 3 were removed (net +3).
                    deal_happened = (next_env.phase == jnp.int8(1)) & (env.phase == jnp.int8(0))

                    return (next_env, obs[0].astype(jnp.float16),
                            action.astype(jnp.int16), reward,
                            root_value[0].astype(jnp.float16),
                            policy_output.action_weights[0].astype(jnp.float16),
                            next_done, jnp.bool_(True),
                            deal_happened.astype(jnp.bool_),
                            discount_target.astype(jnp.int8),
                            reward_target.astype(jnp.int8))
                
                def do_skip(env):
                    # Keine validen Actions → no_step (discards all cards; may trigger distribute_cards)
                    next_env, reward, next_done = no_step(env)
                    dummy_obs = jnp.zeros_like(obs[0], dtype=jnp.float16)
                    # Must detect deal here too: no_step calls distribute_cards (phase 0→1) when
                    # all hands become empty after discarding. Hand-sum comparison is unreliable
                    # (same false-positive risk), so use the same phase-transition check.
                    deal_happened = (next_env.phase == jnp.int8(1)) & (env.phase == jnp.int8(0))
                    return next_env, dummy_obs, jnp.int16(-1), reward, jnp.float16(0.0), jnp.zeros(454, dtype=jnp.float16), next_done, jnp.bool_(False), deal_happened.astype(jnp.bool_), jnp.int8(1), jnp.int8(1)
                
                # Wähle zwischen MCTS und no_step
                next_env, step_obs, action, reward, value, policy, next_done, mask, deal_happened, discount_target, reward_target = jax.lax.cond(
                    has_valid,
                    do_mcts,
                    do_skip,
                    env
                )
                
                # Buffer Update
                idx = buffer['idx']
                current_player = env.current_player
                team = jax.lax.cond(env.rules['enable_teams'], lambda: jnp.int8(current_player%2), lambda: jnp.int8(-1))
                
                new_buffer = {
                    'obs': buffer['obs'].at[idx].set(step_obs.astype(jnp.float16)),
                    'act': buffer['act'].at[idx].set(action.astype(jnp.int16)),
                    'rew': buffer['rew'].at[idx].set(reward_target.astype(jnp.int8)),
                    'val': buffer['val'].at[idx].set(value.astype(jnp.float16)),
                    'pol': buffer['pol'].at[idx].set(policy.astype(jnp.float16)),
                    'mask': buffer['mask'].at[idx].set(mask.astype(jnp.bool_)),
                    'deal_happened': buffer['deal_happened'].at[idx].set(deal_happened.astype(jnp.bool_)),
                    'player': buffer['player'].at[idx].set(current_player),
                    'team': buffer['team'].at[idx].set(team),
                    'discount': buffer['discount'].at[idx].set(discount_target.astype(jnp.int8)),
                    'idx': idx + 1
                }
                return next_env, new_buffer, next_done
            
            def do_skip_step(env, buffer):
                # Game ist fertig, nichts tun
                return env, buffer, done
            
            return jax.lax.cond(~done, do_active_step, do_skip_step, env, buffer)
        
        # ✅ HIER: vmap über alle Envs gleichzeitig!
        new_envs, new_buffers, new_dones = jax.vmap(step_single_env)(
            envs_state, buffers, dones, step_keys
        )
        
        return (new_envs, new_buffers, new_dones, step_count + 1, rng_key)
    
    # Initialisierung
    # Session based: Anstatt erst beim Replay Buffer in Spielsessions zu teilen kann das bereits hier geschehen. 
    # Dazu sind kleinere aber mehr Buffers nötig
    # Ein Idx trackt 
    init_buffers = {
        # float16 saves ~358 MB vs float32; repr_net does x.astype(float32) at first line
        'obs': jnp.zeros((num_envs, max_steps, *input_shape), dtype=jnp.float16),
        'act': jnp.zeros((num_envs, max_steps), dtype=jnp.int16),    # range 0-997 < 32767
        'rew': jnp.zeros((num_envs, max_steps), dtype=jnp.int8),     # class labels 0/1/2
        'val': jnp.zeros((num_envs, max_steps), dtype=jnp.float16),  # root value in [-1,1]
        'pol': jnp.zeros((num_envs, max_steps, 454), dtype=jnp.float16),
        'mask': jnp.zeros((num_envs, max_steps), dtype=jnp.bool_),   # binary 0/1
        'deal_happened': jnp.zeros((num_envs, max_steps), dtype=jnp.bool_),   # binary 0/1
        'player': jnp.zeros((num_envs, max_steps), dtype=jnp.int8),  # 0..3
        'team': jnp.full((num_envs, max_steps), -1, dtype=jnp.int8), # -1/0/1
        'discount': jnp.zeros((num_envs, max_steps), dtype=jnp.int8),# class labels 0/1/2
        'idx': jnp.zeros(num_envs, dtype=jnp.int32),
    }
    init_dones = jnp.zeros(num_envs, dtype=jnp.bool_)
    
    def cond_fn(carry):
        _, _, dones, step_count, _ = carry
        # Stoppe wenn ALLE done ODER max_steps erreicht
        return jnp.any(~dones) & (step_count < max_steps)
    
    final_envs, final_buffers, final_dones, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (envs, init_buffers, init_dones, 0, rng_key)
    )
    
    return final_buffers

def play_n_games_v3(params, rng_key, input_shape, num_envs, num_simulation, max_depth, max_steps, temp):
    """Bester Ansatz: Alles in JAX, aber mit bedingter Ausführung"""
    rng_key, subkey = jax.random.split(rng_key)
    seeds = jax.random.randint(subkey, (num_envs,), 0, 1000000)
    envs = batch_reset(seeds)
    
    all_buffers = play_batch_of_games_jitted(envs, num_envs, input_shape, params, subkey, num_simulation, max_depth, max_steps, temp)
    return all_buffers
