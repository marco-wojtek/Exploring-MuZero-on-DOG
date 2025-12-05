import functools
from classic_madn import *
import jax
import mctx
from visualize_madn import *
import time, sys, os


def simulate_game(env, key):
    rng_key = jax.random.PRNGKey(key)
    steps = 0
    while not env.done:
        print("Current player: ", env.current_player)
        print("Current board:\n", env.board)
        print("Current pins:\n", env.pins)
        rng_key, subkey = jax.random.split(rng_key)
        env = throw_die(env, subkey)
        print("Die throw:\n", env.die)
        valid_actions = valid_action(env)
        print("Valid actions:\n", valid_actions)
        if not jnp.any(valid_actions):
            env, reward, done = no_step(env)
            print("No valid actions available. Skipping turn.")
        else:
            valid_action_indices = jnp.argwhere(valid_actions)
            N = valid_action_indices.shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            idx = jax.random.randint(subkey, (), 0, N)
            chosen = valid_action_indices[idx][0]
            print("Chosen action: ", chosen)
            env, reward, done = env_step(env, chosen)
        print("Reward: ", reward)
        print("Done: ", done)
        print("-"*30)
        steps += 1
    print("Game over after ", steps, " steps.")
    print("Final board:\n", env.board)
    print("Final pins:\n", env.pins)

@functools.partial(jax.jit, static_argnums=(2,))
def run_mcts_search(env: classic_MADN, rng_key: chex.PRNGKey, num_simulations: int = 100):
    """
    Führt MCTS Suche mit stochastic MuZero aus
    """
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    # MCTS Policy mit chance function
    mcts_policy = mctx.stochastic_muzero_policy(
        params=None,  # Keine gelernten Parameter in diesem Beispiel
        rng_key=key1,
        root=jax.vmap(root_fn, (None, 0))(env,jax.random.split(key2, batch_size)),
        decision_recurrent_fn=jax.vmap(recurrent_fn, (None,None,0,0)),
        chance_recurrent_fn=jax.vmap(recurrent_chance_fn, (None,None,0,0)),
        num_simulations=num_simulations,
        invalid_actions=~valid_action(env)[None, :],
        max_depth=500,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1)
        )
    
    return mcts_policy

def madn_mcts_example():
    rng_key = jax.random.PRNGKey(1)
    env = env_reset(0, num_players=2, distance=10, enable_circular_board=False)
    pins = jnp.array([[3,2,6,18],
                      [17, 4, 19, 0]])
    env.pins = pins
    env.board = set_pins_on_board(env.board, pins)    
    print(env.board)    
    # Würfel werfen
    rng_key, subkey = jax.random.split(rng_key)
    env = throw_die(env, subkey)
    print("Die throw:\n", env.die)
    valid_actions = valid_action(env)
    print("Valid actions:\n", valid_actions)    

    
    # MCTS Suche
    rng_key, subkey = jax.random.split(rng_key)
    policy_output = run_mcts_search(env, subkey, num_simulations=500)
    
    # Beste Aktion wählen
    action = jnp.argmax(policy_output.action_weights)
    
    return action, policy_output.action_weights

# simulate_game(env_reset(0, num_players=2, distance=10, enable_dice_rethrow=True, enable_initial_free_pin=True), key=42)

# print(madn_mcts_example())
def step(env, key):
    rng_key = jax.random.PRNGKey(key)
    steps = 0
    print("Current player: ", env.current_player)
    print(matrix_to_string(board_to_matrix(env)))
    rng_key, subkey = jax.random.split(rng_key)
    env = throw_die(env, subkey)
    print("Die throw:\n", env.die)
    valid_actions = valid_action(env)
    print("Valid actions:\n", valid_actions)
    if not jnp.any(valid_actions):
        env, reward, done = no_step(env)
        print("No valid actions available. Skipping turn.")
    else:
        valid_action_indices = jnp.argwhere(valid_actions)
        N = valid_action_indices.shape[0]
        rng_key, subkey = jax.random.split(rng_key)
        idx = jax.random.randint(subkey, (), 0, N)
        chosen = valid_action_indices[idx][0]
        print("Chosen action: ", chosen)
        env, reward, done = env_step(env, chosen)
    return env


def get_trajectory(env, key):
    rng_key = jax.random.PRNGKey(key)
    steps = 0
    trajectory = [board_to_matrix(env)]
    dices = []
    player = []
    action = []
    while not env.done:
        player.append(env.current_player)
        rng_key, subkey = jax.random.split(rng_key)
        env = throw_die(env, subkey)
        dices.append(env.die)
        valid_actions = valid_action(env)
        if not jnp.any(valid_actions):
            action.append(-1)
            env, reward, done = no_step(env)
        else:
            valid_action_indices = jnp.argwhere(valid_actions)
            N = valid_action_indices.shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            idx = jax.random.randint(subkey, (), 0, N)
            chosen = valid_action_indices[idx][0]
            action.append(int(chosen))
            env, reward, done = env_step(env, chosen)
        steps += 1
        trajectory.append(board_to_mat(env))
    return env, player, action, trajectory, dices

def get_game(env, key):
    rng_key = jax.random.PRNGKey(key)
    steps = 0
    dices = []
    player = []
    action = []
    pins = [env.pins]
    while not env.done:
        player.append(env.current_player)
        rng_key, subkey = jax.random.split(rng_key)
        env = throw_die(env, subkey)
        dices.append(env.die)
        valid_actions = valid_action(env)
        if not jnp.any(valid_actions):
            action.append(-1)
            env, reward, done = no_step(env)
        else:
            valid_action_indices = jnp.argwhere(valid_actions)
            N = valid_action_indices.shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            idx = jax.random.randint(subkey, (), 0, N)
            chosen = valid_action_indices[idx][0]
            action.append(int(chosen))
            env, reward, done = env_step(env, chosen)
        steps += 1
        pins.append(env.pins)
    # add final pins
    pins.append(env.pins)
    return env, player, action, pins, dices

def save_games(num_games=100, filename="madn_trajectories.txt", show_progress=True):
    with open(filename, "w") as f:
        for game_idx in range(num_games):
            env = env_reset(0, num_players=4, distance=10, 
                            layout=jnp.array([True, True, True, True]),
                            enable_circular_board=False, 
                            enable_dice_rethrow=True, 
                            enable_initial_free_pin=True,
                            enable_bonus_turn_on_6=True,
                            enable_friendly_fire=False,
                            enable_jump_in_goal_area=True,
                            enable_start_blocking=False,
                            enable_teams=True,
                            enable_start_on_1=True
                            )
            env, players, actions, pins, dices = get_game(env, key=game_idx)
            f.write(f"--- Game {game_idx} ---\n")
            for i in range(len(players)):
                f.write(f"Player: {players[i]}, Dice: {int(dices[i])}, Action: {actions[i]}\n")
                f.write(f"Pins: {pins[i]}\n")
            f.write(f"Final Pins: {pins[-1]}\n")
            f.write("---\n\n")
            if (game_idx + 1) % 10 == 0 or game_idx == num_games - 1:
                print(f"Progress: {game_idx + 1}/{num_games} games")


if __name__ == "__main__":
    save_games(num_games=10, filename="team_simulation.txt", show_progress=True)

def simulate_games_fast(num_games=100):
    ''' Simuliert ein Spiel ohne Ausgabe '''
    winners = jnp.full((num_games,), -1)
    def simulate_single_game(game_idx, winners):
        env = env_reset(game_idx, num_players=4, distance=10, 
                        layout=jnp.array([True, True, True, True]),
                        enable_circular_board=False, 
                        enable_dice_rethrow=False, 
                        enable_initial_free_pin=True,
                        enable_bonus_turn_on_6=True,
                        enable_friendly_fire=False,
                        enable_jump_in_goal_area=True,
                        enable_start_blocking=False,
                        enable_teams=False,
                        enable_start_on_1=True
                        )
        steps = 0
        rng_key = jax.random.PRNGKey(game_idx)
        while not env.done:
            rng_key, subkey = jax.random.split(rng_key)
            env = throw_die(env, subkey)
            valid_actions = valid_action(env)
            if not jnp.any(valid_actions):
                env, reward, done = no_step(env)
            else:
                valid_action_indices = jnp.argwhere(valid_actions)
                N = valid_action_indices.shape[0]
                rng_key, subkey = jax.random.split(rng_key)
                idx = jax.random.randint(subkey, (), 0, N)
                chosen = valid_action_indices[idx][0]
                env, reward, done = env_step(env, chosen)
            steps += 1
        winner = jnp.where(get_winner(env))[0][0]
        winners = winners.at[game_idx].set(winner)
        return winners
    jax.lax.fori_loop(0, num_games, lambda game_idx, winners: simulate_single_game(game_idx, winners), winners)
    return winners