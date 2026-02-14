import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.features)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        x = nn.LayerNorm()(x)
        # Skip Connection: Addiere Input zum Output
        return nn.relu(residual + x)

class RepresentationNetwork(nn.Module):
    latent_dim: int = 256  # Größer als 64 für MADN
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, 14, 56)
        # 1. Sicherstellen, dass es Float ist
        x = x.astype(jnp.float32)
        
        # 2. Channel-Dimension hinzufügen für Conv2D
        # Shape wird zu: (Batch, 14, 56, 1)
        x = jnp.transpose(x, (0, 2, 1))
        
        # 3. Convolutional Layers (Feature Extraction auf dem Board)
        x = nn.Conv(features=32, kernel_size=(3,), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(3,), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=(5,), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Global Pooling: Aggregiere über die räumliche Dimension
        # Wir wollen einen 1D latenten Vektor
        x_mean = jnp.mean(x, axis=1)  # (Batch, 128)
        x_max = jnp.max(x, axis=1)    # (Batch, 128)
        x = jnp.concatenate([x_mean, x_max], axis=-1)  # (Batch, 256)
        
        # 5. Projektion auf Latent Dim
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Residual Blocks zur weiteren Verarbeitung
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # Normalisierung des Latent States
        x = nn.Dense(self.latent_dim)(x)
        min_val = jnp.min(x, axis=0, keepdims=True)
        max_val = jnp.max(x, axis=0, keepdims=True)
        x = (x - min_val) / (max_val - min_val + 1e-8)
        return x

class DynamicsNetwork(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 6
    num_actions: int = 24 # 4 Pins * 6 Würfelaugen
    
    @nn.compact
    def __call__(self, latent_state, action):
        # 1. Action Encoding
        # Wir wandeln die Integer-Action in einen One-Hot Vektor um
        action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions)
        
        # 2. Konkatenation: State + Action
        x = jnp.concatenate([latent_state, action_one_hot], axis=-1)
        
        # 3. Verarbeitung durch ResBlocks
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # --- HEADS (Ausgaben) ---
        
        # A. Next Latent State
        next_latent = nn.Dense(self.latent_dim)(x)
        # ✅ PAPER: Min-Max Scaling statt Sigmoid
        # Min-Max über Batch-Dimension
        min_val = jnp.min(next_latent, axis=0, keepdims=True)
        max_val = jnp.max(next_latent, axis=0, keepdims=True)
        next_latent = (next_latent - min_val) / (max_val - min_val + 1e-8)
        
        # B. Reward Prediction
        # MADN Rewards sind oft sparse (0 oder 1). 
        # Linearer Output ist flexibel, tanh ist gut wenn Rewards [-1, 1] sind.
        reward = nn.Dense(1)(x) 
        
        # C. Discount Prediction (WICHTIG!)
        # Sagt vorher, ob das Spiel weitergeht.
        # Ausgabe sind Logits. Positive Werte = Weiter, Negative Werte = Ende.
        discount_logits = nn.Dense(1)(x)
        
        return next_latent, reward, discount_logits

class PredictionNetwork(nn.Module):
    latent_dim: int = 256
    num_actions: int = 24
    num_res_blocks: int = 6
    
    @nn.compact
    def __call__(self, latent_state):
        x = latent_state
        
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # --- HEADS ---
        
        # A. Policy (Welche Aktion ist gut?)
        policy_logits = nn.Dense(self.num_actions)(x)
        
        # B. Value (Wie gut ist der Zustand für den aktuellen Spieler?)
        value = nn.Dense(1)(x)
        value = nn.tanh(value)
        
        return policy_logits, value

class PredictionNetwork2(nn.Module):
    latent_dim: int = 256
    num_actions: int = 24
    num_res_blocks: int = 6
    num_head_layers: int = 2  # Neue Parameter für Head-Tiefe
    
    @nn.compact
    def __call__(self, latent_state):
        # Gemeinsamer Trunk
        x = latent_state
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
        
        # --- POLICY HEAD (separate Verarbeitung) ---
        policy = x
        for _ in range(self.num_head_layers):
            policy = nn.Dense(self.latent_dim // 2)(policy)  # Reduzierte Dim
            policy = nn.LayerNorm()(policy)
            policy = nn.relu(policy)
        policy_logits = nn.Dense(self.num_actions)(policy)
        
        # --- VALUE HEAD (separate Verarbeitung) ---
        value = x
        for _ in range(self.num_head_layers):
            value = nn.Dense(self.latent_dim // 4)(value)  # Noch kleiner für Value
            value = nn.LayerNorm()(value)
            value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        
        return policy_logits, value

class StochasticDynamicsNetwork(nn.Module):
    latent_dim: int = 64
    num_chance_outcomes: int = 6

    # Teil 1: Action Dynamics
    def action_dynamics(self, latent_state, action):
        action_one_hot = jax.nn.one_hot(action, num_classes=4)
        x = jnp.concatenate([latent_state, action_one_hot], axis=-1)
        
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        afterstate = nn.Dense(self.latent_dim)(x)
        reward = nn.Dense(1)(x)
        chance_logits = nn.Dense(self.num_chance_outcomes)(x)
        
        # NEU: Discount Head
        # Wir geben Logits aus. Ein hoher Wert bedeutet "Spiel geht weiter" (Discount ~ 1),
        # ein niedriger Wert bedeutet "Spiel vorbei" (Discount ~ 0).
        discount_logits = nn.Dense(1)(x) 
        
        return afterstate, reward, chance_logits, discount_logits

    # Teil 2: Was passiert, wenn eine bestimmte Zahl gewürfelt wird?
    def chance_dynamics(self, afterstate, chance_outcome):
        # Input: Afterstate + Welcher Würfelwert ist gefallen?
        # chance_outcome ist ein Index (z.B. 0 für "1", 5 für "6")
        chance_one_hot = jax.nn.one_hot(chance_outcome, num_classes=self.num_chance_outcomes)
        
        # HIER passiert die Magie: Wir füttern den Würfelwert in das Netz.
        # Das Netz lernt: "Wenn Afterstate X ist und eine 6 gewürfelt wird, ist der neue State Y"
        x = jnp.concatenate([afterstate, chance_one_hot], axis=-1)
        
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Output: Der nächste echte State (in dem der nächste Spieler dran ist)
        next_latent_state = nn.Dense(self.latent_dim)(x)
        
        return next_latent_state

repr_net = RepresentationNetwork()
dynamics_net = StochasticDynamicsNetwork()
pred_net = PredictionNetwork2()

def decision_recurrent_fn(params, rng_key, action, embedding):
    # Aufruf des Netzwerks
    afterstate, reward, chance_logits, discount_logits = dynamics_net.apply(params['dynamics'], embedding, action, method=dynamics_net.action_dynamics)
    
    # Umwandlung Logits -> Wahrscheinlichkeit (0 bis 1)
    # Wenn das Netz sicher ist, dass es weitergeht, ist das nahe 1.
    # Wenn das Netz denkt, das Spiel ist aus, ist das nahe 0.
    discount = jax.nn.sigmoid(discount_logits)
    
    return mctx.DecisionRecurrentFnOutput(
        reward=reward,
        discount=discount, # <--- Gelernt, nicht hardcoded!
        afterstate=afterstate,
        chance_logits=chance_logits
    )

def chance_recurrent_fn(params, rng_key, chance_outcome, afterstate):
    # chance_outcome wird von mctx basierend auf chance_logits ausgewählt (oder alle durchprobiert)
    
    # Rufe Teil 2 des Netzwerks auf
    next_embedding = dynamics_net.apply(params['dynamics'], afterstate, chance_outcome, method=dynamics_net.chance_dynamics)
    
    # Jetzt brauchen wir Policy und Value für den NEUEN State
    # Das macht das Prediction Network
    prior_logits, value = pred_net.apply(params['prediction'], next_embedding)
    
    return mctx.ChanceRecurrentFnOutput(
        value=value,
        prior_logits=prior_logits,
        embedding=next_embedding
    )

def root_inference_fn(params, observation):
    embedding = repr_net.apply(params['representation'], observation)
    prior_logits, value = pred_net.apply(params['prediction'], embedding)
    # value: (Batch, 1) -> (Batch,)
    value = value.squeeze(-1)
    return mctx.RootFnOutput(
        embedding=embedding,
        prior_logits=prior_logits,
        value=value
    )

@functools.partial(jax.jit, static_argnames=['num_simulations', 'max_depth', 'temperature'])
def run_stochastic_muzero_mcts(params, rng_key, observations, invalid_actions=None, num_simulations=50, max_depth=25, temperature=1.0):
    """
    Führt Stochastic MuZero MCTS auf einem Environment aus.
    WICHTIG: Das Environment muss bereits gewürfelt haben (env.die gesetzt sein)!
    
    Args:
        params: MuZero Netzwerk-Parameter (representation, dynamics, prediction)
                ODER None für Ground Truth MCTS ohne gelerntes Netzwerk
        rng_key: JAX PRNG Key
        observation: Beobachtung (Observation) des Environments (NACH dem Würfeln!)
        invalid_actions: Optional Maske für ungültige Aktionen
        num_simulations: Anzahl der MCTS Simulationen
        max_depth: Maximale Suchtiefe
        temperature: Temperatur für die Aktionsauswahl (Softmax-Temperatur)
    Returns:
        policy_output: mctx PolicyOutput mit action, action_weights, etc.
        root_value: Der geschätzte Wert des Root-States
    """
    key1, key2 = jax.random.split(rng_key)
    
    root_output = root_inference_fn(params, observations)
    
    # MCTS Policy mit chance function
    policy_output = mctx.stochastic_muzero_policy(
        params=params,
        rng_key=key2,
        root=root_output,
        decision_recurrent_fn=decision_recurrent_fn,
        chance_recurrent_fn=chance_recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        temperature=temperature
    )
    
    # Root value aus dem Policy Output extrahieren
    root_value = policy_output.search_tree.node_values[0]  # Root node value
    
    return policy_output, root_value

def init_muzero_params(rng_key, input_shape):
    """
    Initialisiert die Parameter für alle drei MuZero-Netzwerke.
    
    Args:
        rng_key: JAX PRNG Key
        input_shape: Shape der Observation (z.B. (Features, BoardSize) oder flach)
                     Beispiel für MADN: (180,) wenn linear encoded.
    
    Returns:
        Ein Dictionary mit den Parametern:
        {
            'representation': params_repr,
            'dynamics': params_dyn,
            'prediction': params_pred
        }
    """
    key_repr, key_dyn, key_pred = jax.random.split(rng_key, 3)
    
    # 1. Representation Network
    # Input: Observation (Batch-Dimension hinzufügen für init)
    dummy_obs = jnp.ones((1, *input_shape))
    params_repr = repr_net.init(key_repr, dummy_obs)
    
    # Um die Output-Shape des Representation Networks zu bekommen,
    # führen wir einmal apply aus (oder wissen es aus der Config).
    # Hier holen wir uns den latent state, um Dynamics/Prediction zu initialisieren.
    dummy_latent = repr_net.apply(params_repr, dummy_obs)
    
    # 2. Dynamics Network
    # Input: Latent State + Action (Integer)
    dummy_action = jnp.array([0]) # Batch size 1, Action 0
    params_dyn = dynamics_net.init(key_dyn, dummy_latent, dummy_action)
    
    # 3. Prediction Network
    # Input: Latent State
    params_pred = pred_net.init(key_pred, dummy_latent)
    
    return {
        'representation': params_repr,
        'dynamics': params_dyn,
        'prediction': params_pred
    }

def load_params_from_file(param_file):
    """Lädt die MuZero-Parameter aus einer Datei."""
    import pickle
    with open(param_file, 'rb') as f:
        params = pickle.load(f)
    return params
