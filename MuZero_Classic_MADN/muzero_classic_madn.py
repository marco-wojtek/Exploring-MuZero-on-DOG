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

class RepresentationNetwork2(nn.Module):
    latent_dim: int = 256  # Größer als 64 für MADN
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, 14, 56)
        # 1. Sicherstellen, dass es Float ist
        x = x.astype(jnp.float32)

        # trenne spatial und global informationen:
        spacial = x[:, :6, :]  # (Batch, 6, 56)
        global_f = x[:, 6:, 0]  # (Batch, 5, 56) -> (Batch, 28)
        
        # 2. Channel-Dimension hinzufügen für Conv2D
        # Shape wird zu: (Batch, 6, 56, 1)
        spacial = jnp.transpose(spacial, (0, 2, 1))
        
        # 3. Convolutional Layers (Feature Extraction auf dem Board)
        spacial = nn.Conv(features=32, kernel_size=(3,), padding='SAME')(spacial)
        spacial = nn.LayerNorm()(spacial)
        spacial = nn.relu(spacial)
        
        spacial = nn.Conv(features=64, kernel_size=(3,), padding='SAME')(spacial)
        spacial = nn.LayerNorm()(spacial)
        spacial = nn.relu(spacial)
        
        spacial = nn.Conv(features=64, kernel_size=(5,), padding='SAME')(spacial)
        spacial = nn.LayerNorm()(spacial)
        spacial = nn.relu(spacial)
        
        # spatial flatten
        spatial_flat = spacial.reshape(spacial.shape[0], -1)  # (Batch, 6*56*64)
        
        # 5. Projektion auf Latent Dim
        spatial_flat = nn.Dense(self.latent_dim)(spatial_flat)
        spatial_flat = nn.LayerNorm()(spatial_flat)
        spatial_flat = nn.relu(spatial_flat)
        
        # === GLOBAL STREAM ===
        # home_positions: wie viele Pins pro Spieler im Haus
        # action_channels: wie viele Aktionen pro Spieler verfügbar
        global_f = nn.Dense(64)(global_f)   # (Batch, 64)
        global_f = nn.LayerNorm()(global_f)
        global_f = nn.relu(global_f)
        
        global_f = nn.Dense(64)(global_f)   # (Batch, 64)
        global_f = nn.LayerNorm()(global_f)
        global_f = nn.relu(global_f)

        # === KOMBINIERE BEIDE STREAMS ===
        combined = jnp.concatenate([spatial_flat, global_f], axis=-1)

        # Projektion auf Latent Dim
        x = nn.Dense(self.latent_dim)(combined)  # (Batch, 256)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        # Residual Blocks zur weiteren Verarbeitung
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # Normalisierung des Latent States
        x = nn.Dense(self.latent_dim)(x)
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        x = (x - min_val) / (max_val - min_val + 1e-8)
        return x
    
class PredictionNetwork(nn.Module):
    latent_dim: int = 256
    num_actions: int = 4
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
    num_actions: int = 4
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

class PredictionNetwork4(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 4
    
    @nn.compact
    def __call__(self, latent_state):
        # LayerNorm am Eingang: [0,1] → N(0,1)
        # EINE Normalisierung für RepNet UND DynNet Latents!
        x = nn.LayerNorm()(latent_state)
        
        # Shared Trunk: 2 ResBlocks (mehr Kapazität als PredNet2)
        # Skip Connection funktioniert: N(0,1) + N(0,1) ✅
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
        
        # --- POLICY HEAD ---
        policy = nn.Dense(self.latent_dim)(x)
        policy = nn.LayerNorm()(policy)
        policy = nn.relu(policy)
        policy = nn.Dense(self.latent_dim // 2)(policy)
        policy = nn.LayerNorm()(policy)
        policy = nn.relu(policy)
        policy_logits = nn.Dense(self.num_actions)(policy)
        
        # --- VALUE HEAD ---
        value = nn.Dense(self.latent_dim // 2)(x)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)
        value = nn.Dense(self.latent_dim // 4)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        
        return policy_logits, value
    
class StochasticDynamicsNetwork(nn.Module):
    latent_dim: int = 256
    num_chance_outcomes: int = 6
    num_res_blocks: int = 4  # Weniger als deterministic (6), aber mehr Kapazität als vorher (0)

    @nn.compact
    def __call__(self, latent_state, action, chance_outcome=None):
        """
        Default call für Initialisierung.
        Ruft BEIDE Methoden auf um alle Parameter zu initialisieren.
        """
        # Teil 1: Action Dynamics
        afterstate, reward, chance_logits, discount_logits = self.action_dynamics(latent_state, action)
        
        # Teil 2: Chance Dynamics (nur zur Initialisierung, falls chance_outcome gegeben)
        if chance_outcome is not None:
            next_state = self.chance_dynamics(afterstate, chance_outcome)
            return afterstate, reward, chance_logits, discount_logits, next_state
        
        return afterstate, reward, chance_logits, discount_logits
    
    # Teil 1: Action Dynamics (Player wählt Action)
    @nn.compact
    def action_dynamics(self, latent_state, action):
        """
        Verarbeitet: latent_state + action → afterstate
        WICHTIG: Ähnliche Komplexität wie deterministisches Dynamics Network!
        """
        action_one_hot = jax.nn.one_hot(action, num_classes=4)
        x = jnp.concatenate([latent_state, action_one_hot], axis=-1)
        
        # Initial projection mit LayerNorm für Stabilität
        x = nn.Dense(self.latent_dim, name='action_dense1')(x)
        x = nn.LayerNorm(name='action_norm1')(x)
        x = nn.relu(x)
        
        # ResBlocks für tiefere Verarbeitung
        for i in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
        
        # --- HEADS (Outputs) ---
        # A. Afterstate (latenter Zustand nach Action, vor Würfel)
        afterstate = nn.Dense(self.latent_dim, name='action_afterstate')(x)
        # Min-Max Normalisierung wie bei deterministischem MuZero
        min_val = jnp.min(afterstate, axis=0, keepdims=True)
        max_val = jnp.max(afterstate, axis=0, keepdims=True)
        afterstate = (afterstate - min_val) / (max_val - min_val + 1e-8)
        
        # B. Reward (sparse in MADN)
        reward = nn.Dense(1, name='action_reward')(x)
        
        # C. Chance Logits (Vorhersage der Würfelverteilung)
        chance_logits = nn.Dense(self.num_chance_outcomes, name='action_chance_logits')(x)
        
        # D. Discount (Spiel-Ende Prädiktion)
        discount_logits = nn.Dense(1, name='action_discount')(x) 
        
        return afterstate, reward, chance_logits, discount_logits

    # Teil 2: Chance Dynamics (Würfel wird gewürfelt)
    @nn.compact
    def chance_dynamics(self, afterstate, chance_outcome):
        """
        Verarbeitet: afterstate + chance_outcome → next_state
        Auch hier brauchen wir ausreichend Kapazität!
        """
        chance_one_hot = jax.nn.one_hot(chance_outcome, num_classes=self.num_chance_outcomes)
        x = jnp.concatenate([afterstate, chance_one_hot], axis=-1)
        
        # Initial projection
        x = nn.Dense(self.latent_dim, name='chance_dense1')(x)
        x = nn.LayerNorm(name='chance_norm1')(x)
        x = nn.relu(x)
        
        for i in range(self.num_res_blocks):  
            x = ResBlock(self.latent_dim)(x)
        
        # Output: Der nächste echte State
        next_latent_state = nn.Dense(self.latent_dim, name='chance_next_state')(x)
        # Min-Max Normalisierung
        min_val = jnp.min(next_latent_state, axis=0, keepdims=True)
        max_val = jnp.max(next_latent_state, axis=0, keepdims=True)
        next_latent_state = (next_latent_state - min_val) / (max_val - min_val + 1e-8)
        
        return next_latent_state

class StochasticDynamicsNetwork4(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 4       # Classic MADN: 4 Pins
    num_chance_outcomes: int = 6  # Würfel 1-6

    @nn.compact
    def __call__(self, latent_state, action, chance_outcome=None):
        """Init-Call: durchläuft beide Pfade um alle Parameter zu erstellen."""
        afterstate, reward_logits, chance_logits, discount_logits = self.action_dynamics(latent_state, action)
        if chance_outcome is not None:
            next_state = self.chance_dynamics(afterstate, chance_outcome)
            return afterstate, reward_logits, chance_logits, discount_logits, next_state
        return afterstate, reward_logits, chance_logits, discount_logits

    @nn.compact
    def action_dynamics(self, latent_state, action):
        """State + Action → Afterstate + Reward/Discount/Chance-Logits"""
        # 1. Action Embedding
        action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embed = nn.Dense(64, name='act_embed')(action_one_hot)
        action_embed = nn.relu(action_embed)

        # 2. FiLM Conditioning
        latent_normed = nn.LayerNorm(name='act_input_ln')(latent_state)
        scale = nn.Dense(self.latent_dim, name='act_film_scale')(action_embed)
        shift = nn.Dense(self.latent_dim, name='act_film_shift')(action_embed)
        x = latent_normed * (1 + scale) + shift

        # 3. Hauptverarbeitung
        x = nn.Dense(self.latent_dim, name='act_dense1')(x)
        x = nn.LayerNorm(name='act_ln1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, name='act_dense2')(x)
        x = nn.LayerNorm(name='act_ln2')(x)
        x = nn.relu(x)
        for i in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)

        # 4. Residual + Min-Max
        x = nn.Dense(self.latent_dim, name='act_proj')(x)
        x = latent_state + x
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        afterstate = (x - min_val) / (max_val - min_val + 1e-8)

        # 5. Reward Head: 3 Klassen {-1, 0, +1}
        reward_input = jnp.concatenate([afterstate, action_one_hot], axis=-1)
        reward_logits = nn.Dense(64, name='reward_dense')(reward_input)
        reward_logits = nn.relu(reward_logits)
        reward_logits = nn.Dense(3, name='reward_head')(reward_logits)

        # 6. Discount Head: 3 Klassen {-1, 0, +1}
        discount_logits = nn.Dense(32, name='discount_dense')(latent_state)
        discount_logits = nn.LayerNorm(name='discount_ln')(discount_logits)
        discount_logits = nn.relu(discount_logits)
        discount_logits = nn.Dense(3, name='discount_head')(discount_logits)

        # 7. Chance Logits: Vorhersage der Würfelverteilung
        chance_logits = nn.Dense(self.num_chance_outcomes, name='chance_head')(afterstate)

        return afterstate, reward_logits, chance_logits, discount_logits

    @nn.compact
    def chance_dynamics(self, afterstate, chance_outcome):
        """Afterstate + Würfel → Next State"""
        # 1. Chance Embedding
        chance_one_hot = jax.nn.one_hot(chance_outcome, num_classes=self.num_chance_outcomes)
        chance_embed = nn.Dense(64, name='chance_embed')(chance_one_hot)
        chance_embed = nn.relu(chance_embed)

        # 2. FiLM Conditioning (gleiche Struktur wie action_dynamics)
        afterstate_normed = nn.LayerNorm(name='chance_input_ln')(afterstate)
        scale = nn.Dense(self.latent_dim, name='chance_film_scale')(chance_embed)
        shift = nn.Dense(self.latent_dim, name='chance_film_shift')(chance_embed)
        x = afterstate_normed * (1 + scale) + shift

        # 3. Hauptverarbeitung
        x = nn.Dense(self.latent_dim, name='chance_dense1')(x)
        x = nn.LayerNorm(name='chance_ln1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, name='chance_dense2')(x)
        x = nn.LayerNorm(name='chance_ln2')(x)
        x = nn.relu(x)
        for i in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)

        # 4. Residual + Min-Max
        x = nn.Dense(self.latent_dim, name='chance_proj')(x)
        x = afterstate + x  # Skip zum Afterstate
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        next_state = (x - min_val) / (max_val - min_val + 1e-8)

        return next_state
    
repr_net = RepresentationNetwork2()
dynamics_net = StochasticDynamicsNetwork4()
pred_net = PredictionNetwork4()

def decision_recurrent_fn(params, rng_key, action, embedding):
    afterstate, reward_logits, chance_logits, discount_logits = dynamics_net.apply(
        params['dynamics'], embedding, action, method=dynamics_net.action_dynamics
    )
    # Reward/Discount → Scalar
    support = jnp.array([-1.0, 0.0, 1.0])
    reward = jnp.sum(jax.nn.softmax(reward_logits) * support, axis=-1)
    discount = jnp.sum(jax.nn.softmax(discount_logits) * support, axis=-1)
    
    # Reward + Discount an Afterstate anhängen (werden in chance_recurrent_fn extrahiert)
    afterstate_with_info = jnp.concatenate([afterstate, reward[:, None], discount[:, None]], axis=-1)
    
    _, afterstate_value = pred_net.apply(params['prediction'], afterstate)
    afterstate_value = afterstate_value.squeeze(-1)
    
    return mctx.DecisionRecurrentFnOutput(
        chance_logits=chance_logits,
        afterstate_value=afterstate_value,
    ), afterstate_with_info  # ← 258-dim statt 256

def chance_recurrent_fn(params, rng_key, chance_outcome, afterstate_with_info):
    # Extrahiere Reward/Discount
    afterstate = afterstate_with_info[:, :-2]  # (Batch, 256)
    reward = afterstate_with_info[:, -2]       # (Batch,)
    discount = afterstate_with_info[:, -1]     # (Batch,)
    
    next_embedding = dynamics_net.apply(
        params['dynamics'], afterstate, chance_outcome, method=dynamics_net.chance_dynamics
    )
    prior_logits, value = pred_net.apply(params['prediction'], next_embedding)
    value = value.squeeze(-1)
    
    return mctx.ChanceRecurrentFnOutput(
        action_logits=prior_logits,
        value=value,
        reward=reward,      # ← aus Action-Phase
        discount=discount,  # ← aus Action-Phase
    ), next_embedding

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
def run_stochastic_muzero_mcts(params, rng_key, observations, invalid_actions, num_simulations, max_depth, temperature):
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
        #qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=0.1),
        temperature=temperature
    )
    
    # ✅ KORREKT laut MuZero Paper: Verwende MCTS-verfeinerten Value als Target
    # 
    # WARUM MCTS-Value?
    # - MCTS macht Lookahead und findet bessere Werte als das rohe Netzwerk
    # - Das Netzwerk soll lernen, direkt zu sehen, was MCTS durch Suche findet
    # - Dies ist "Knowledge Distillation" vom langsamen aber genauen MCTS zum schnellen Netzwerk
    # 
    # WICHTIG: Hat höhere Varianz bei wenigen Simulationen!
    # Lösung: Höhere Loss-Gewichtung für Value (50-100× statt 10×)
    root_value = policy_output.search_tree.node_values[0] # MCTS-verfeinerter Value
    root_value = jnp.clip(root_value, -1.0, 1.0)
    # Alternative (stabiler aber schlechteres Signal):
    # root_value = root_output.value  # Raw network value 
    
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
    
    # 2. Dynamics Network (Stochastic hat 2 Methoden!)
    # WICHTIG: Wir müssen __call__ mit chance_outcome aufrufen, 
    # damit BEIDE Methoden initialisiert werden
    dummy_action = jnp.array([0])  # Batch size 1, Action 0
    dummy_chance = jnp.array([0])  # Batch size 1, Chance outcome 0
    
    # Initialisiere mit __call__ und chance_outcome, um beide Pfade zu durchlaufen
    params_dyn = dynamics_net.init(key_dyn, dummy_latent, dummy_action, dummy_chance)
    
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
