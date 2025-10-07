import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import os
import optax
from TicTacToeV2 import env_reset, env_step, get_winner
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
def play_game(policy_apply_fn, params, rng_key):
    """Spielt eine Partie (nicht-jittbar) und sammelt die Trajektorie.

    Wir verwenden eine Python-Schleife statt einer jittbaren Funktion, um
    appends in Listen zu erlauben. Das ist in Ordnung für ein einfaches Demo-Training.
    """
    env = env_reset(0)

    states, actions, rewards, players = [], [], [], []
    key = rng_key

    for _ in range(9):
        # Aktion vom Policy-Netzwerk erhalten
        logits = policy_apply_fn(params, env.board)

        # Ungültige Züge maskieren
        valid_mask = (env.board.flatten() == 0)
        logits = jnp.where(valid_mask, logits, -jnp.inf)

        key, subkey = jax.random.split(key)
        action = int(jax.random.categorical(subkey, logits))

        states.append(env.board)
        actions.append(action)
        players.append(int(env.current_player))

        env, reward, done = env_step(env, action)
        rewards.append(int(reward))

        if done:
            break

    # Verwende das finale Spielergebnis als Target für alle Zeitpunkte
    final_outcome = int(get_winner(env.board))  # 1, -1 oder 0

    if final_outcome == 0:
        final_outcome = -0.2 # penalty für unentschieden

    players_array = jnp.array(players, dtype=jnp.int32)
    returns = final_outcome * players_array

    # Staple Zustände in ein Array (shape: steps x 3 x 3)
    if len(states) == 0:
        # Keine Schritte gemacht (sollte nicht passieren), gib leere Trajektorie zurück
        return {'states': jnp.zeros((0, 3, 3)), 'actions': jnp.zeros((0,), dtype=jnp.int32), 'returns': jnp.zeros((0,), dtype=jnp.float32)}

    states_array = jnp.stack([jnp.asarray(s) for s in states])

    return {
        'states': states_array,
        'actions': jnp.array(actions, dtype=jnp.int32),
        'returns': returns.astype(jnp.float32),
    }

# 3. Loss-Funktion und Trainingsschritt
def train_step(params, opt_state, optimizer, trajectory):
    """Berechnet den Verlust, die Gradienten und aktualisiert die Modellparameter."""

    if float(jnp.sum(trajectory['returns'])) == 0.0:
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
    loss, grads = jax.value_and_grad(loss_fn)(params, trajectory['states'], trajectory['actions'], trajectory['returns'])
    
    # Parameter aktualisieren
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss

# 4. Haupt-Trainingsschleife
def main_training_loop(num_episodes=10000, learning_rate=0.001):
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
        trajectory = play_game(policy_net.apply, params, game_key)
        
        # Trainingsschritt ausführen
        params, opt_state, loss = train_step(params, opt_state, optimizer, trajectory)

        if episode % (num_episodes // 10) == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")
            
    print("Training abgeschlossen!")
    return params

# 5. Evaluationsphase: Trainiertes Modell gegen einen Random Bot testen
def get_trained_action(policy_apply_fn, params, board):
    """Erhält die beste Aktion vom trainierten Modell."""
    logits = policy_apply_fn(params, board)
    valid_mask = (board.flatten() == 0)
    logits = jnp.where(valid_mask, logits, -jnp.inf)
    return jnp.argmax(logits)

def get_random_action(board, rng_key):
    """Wählt eine zufällige, gültige Aktion."""
    valid_actions = jnp.where(board.flatten() == 0)[0]
    return jax.random.choice(rng_key, valid_actions)

def play_match(trained_params, rng_key, trained_player=1):
    """Spielt eine Partie: Trainierter Agent vs. Random Bot."""
    env = env_reset(0)
    
    while not env.done:
        rng_key, action_key = jax.random.split(rng_key)
        
        if env.current_player == trained_player:
            action = get_trained_action(SimplePolicy().apply, trained_params, env.board)
        else:
            action = get_random_action(env.board, action_key)
            
        env, _, _ = env_step(env, action.astype(jnp.int8))
    
    # Gibt den Gewinner zurück (1 für trainierten Agent, -1 für Random Bot, 0 für Unentschieden)
    return get_winner(env.board) * trained_player

def evaluate_agent(trained_params, num_matches=1000):
    """Evaluiert den trainierten Agenten über viele Partien."""
    print(f"\nStarte Evaluation gegen Random Bot ({num_matches} Partien)...")
    rng_key = jax.random.PRNGKey(123)
    
    wins = 0
    losses = 0
    draws = 0
    
    # Spiele als Spieler 1 (beginnt)
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = play_match(trained_params, game_key, trained_player=1)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
            
    # Spiele als Spieler 2
    for i in range(num_matches // 2):
        rng_key, game_key = jax.random.split(rng_key)
        winner = play_match(trained_params, game_key, trained_player=-1)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
            
    win_rate = wins / num_matches
    loss_rate = losses / num_matches
    draw_rate = draws / num_matches
    
    print("\n--- Evaluationsergebnis ---")
    print(f"Gewinnrate: {win_rate:.2%}")
    print(f"Verlustrate: {loss_rate:.2%}")
    print(f"Unentschieden: {draw_rate:.2%}")
    print("---------------------------\n")

def save_checkpoint(path: str, params, opt_state=None):
    """Speichert params (+ optional optimizer state) als Bytes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".params", "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    if opt_state is not None:
        with open(path + ".opt", "wb") as f:
            f.write(flax.serialization.to_bytes(opt_state))

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

if __name__ == "__main__":
    # trained_params = main_training_loop(num_episodes=10000, learning_rate=0.0001)
    # save_checkpoint("D:\Informatikstudium\Master\Masterarbeit\Exploring-MuZero-on-DOG\TicTacToe\Checkpoints\cp_10k_0e3", trained_params)
    # print("Trainierte Parameter wurden erstellt.")
    
    # Starte die Evaluation nach dem Training
    policy = SimplePolicy()
    dummy = jnp.zeros((3,3))
    params_template = policy.init(jax.random.PRNGKey(0), dummy)
    print("Begin Evaluation")
    params, opt_state = load_checkpoint("D:\\Informatikstudium\\Master\\Masterarbeit\\Exploring-MuZero-on-DOG\\TicTacToe\\Checkpoints\\cp_10k_0e3", params_template)
    evaluate_agent(params, 1000)
