import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import os
import optax
import TicTacToeV2 as ttt_v2
import TicTacToe as ttt
import functools

# 1. Ein einfaches Policy-Netzwerk, das Aktionen basierend auf dem Zustand vorschlägt
class SimplePolicy(nn.Module):
    num_actions: int = 9

    @nn.compact
    def __call__(self, x):
        # x ist der Zustand des Spielfelds (3, 3)
        x = x.flatten()  # Umwandeln in einen Vektor der Größe (9,)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_actions)(x)
        return x  # Gibt Logits für jede Aktion zurück

# 2. Funktion zum Spielen einer kompletten Partie, um Trainingsdaten zu sammeln
def play_game(game, policy_apply_fn, params, rng_key):
    """Spielt eine Partie (nicht-jittbar) und sammelt die Trajektorie.

    Wir verwenden eine Python-Schleife statt einer jittbaren Funktion, um
    appends in Listen zu erlauben. Das ist in Ordnung für ein einfaches Demo-Training.
    """
    env = game.env_reset(0)
    max_steps = 14
    states, actions, players = jnp.zeros((max_steps, 3, 3)), jnp.zeros((max_steps), dtype=jnp.int8), jnp.zeros((max_steps), dtype=jnp.float32)
    key = rng_key

    def cond(a):
        env, _, _, _, _, step = a
        return ~env.done & (step < max_steps)
    def body(carry):
        env, key, states, actions, players, step = carry
        # Aktion vom Policy-Netzwerk erhalten
        logits = policy_apply_fn(params, env.board)

        # Ungültige Züge maskieren
        valid_mask = (env.board.flatten() == 0)
        logits = jnp.where(valid_mask, logits, -jnp.inf)

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits).astype(jnp.int8)

        states = states.at[step].set(env.board)
        actions = actions.at[step].set(action)
        players = players.at[step].set(env.current_player)

        env, reward, done = game.env_step(env, action)
        step = step + 1
        return env, key, states, actions, players, step
    
    initial_carry = (env, rng_key, states, actions, players, jnp.int8(0))
    final_env, _, final_states, final_actions, final_players, final_steps = jax.lax.while_loop(cond, body, initial_carry)
    # Verwende das finale Spielergebnis als Target für alle Zeitpunkte

    step_valid = jnp.any(final_states.reshape(max_steps, -1) != 0, axis=1)
    num_steps = jnp.sum(step_valid)

    final_outcome = game.get_winner(final_env.board)  # 1, -1 oder 0

    final_outcome = jnp.where(final_outcome == 0, -0.2, final_outcome)

    returns = final_outcome * final_players

    return {
        'states': final_states,
        'actions': final_actions,
        'returns': returns,
        'num_steps': num_steps
    }

# 3. Loss-Funktion und Trainingsschritt
def train_step(params, opt_state, optimizer, trajectory):
    """Berechnet den Verlust, die Gradienten und aktualisiert die Modellparameter."""

    num_steps = trajectory.get('num_steps', len(trajectory['states']))
    
    # Nur gültige Schritte verwenden
    states = trajectory['states'][:num_steps]
    actions = trajectory['actions'][:num_steps]
    returns = trajectory['returns'][:num_steps]
    
    if float(jnp.sum(returns)) == 0.0:
        return params, opt_state, 0.0
    
    def loss_fn(p, states, actions, returns):
        # Wende das Netzwerk auf den gesamten Batch von Zuständen an, indem wir vmap nutzen
        logits = jax.vmap(SimplePolicy().apply, in_axes=(None, 0))(p, states)
        log_probs = jax.nn.log_softmax(logits)
        
        # Log-Wahrscheinlichkeit der ausgeführten Aktion auswählen
        action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
        
        # Advantage (baseline) und Normalisierung
        adv = returns - jnp.mean(returns)
        adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

        policy_loss = -jnp.mean(action_log_probs * adv)

        # Entropiebonus zur Förderung der Exploration
        probs = jnp.exp(log_probs)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=1))
        entropy_coef = 0.01

        loss = policy_loss - entropy_coef * entropy
        return loss

    # Gradienten berechnen
    loss, grads = jax.value_and_grad(loss_fn)(params, states, actions, returns)
    
    # Parameter aktualisieren
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss

# 4. Haupt-Trainingsschleife
def main_training_loop(game, num_episodes=10000, learning_rate=0.001):
    rng_key = jax.random.PRNGKey(42)
    
    # Netzwerk und Optimizer initialisieren
    policy_net = SimplePolicy()
    dummy_board = jnp.zeros((3, 3))
    params = policy_net.init(rng_key, dummy_board)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    print("Starte einfaches Training für TicTacToeV2...")
    
    for episode in range(num_episodes):
        rng_key, game_key = jax.random.split(rng_key)
        
        # Eine Partie spielen, um Daten zu sammeln
        trajectory = play_game(game, policy_net.apply, params, game_key)
        
        # Trainingsschritt ausführen
        params, opt_state, loss = train_step(params, opt_state, optimizer, trajectory)

        if episode % (num_episodes // 10) == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")
            
    print("Training abgeschlossen!")
    return params

def save_checkpoint(path: str, params, opt_state=None):
    """Speichert params (+ optional optimizer state) als Bytes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".params", "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    if opt_state is not None:
        with open(path + ".opt", "wb") as f:
            f.write(flax.serialization.to_bytes(opt_state))


if __name__ == "__main__":
    game = ttt
    # game = ttt_v2
    trained_params = main_training_loop(game, num_episodes=1000, learning_rate=0.001)
    name = f"{game.__name__}_cp_1k_e3"
    path = "C:\\Users\\marco\\Informatikstudium\\Master\\Masterarbeit\\Exploring-MuZero-on-DOG\\TicTacToe\\Checkpoints\\" + name

    save_checkpoint(path, trained_params)
    print("Trainierte Parameter wurden erstellt.")