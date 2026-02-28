import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MADN.deterministic_madn import *
# Ein einfacher Residual Block für MLPs
# Hilft dem Gradientenfluss bei tieferen Netzen
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
        # Shape wird zu: (Batch, 34, 56, 1)
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
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
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
        global_f = x[:, 6:, 0]  # (Batch, 28)
        
        # 2. Channel-Dimension hinzufügen für Conv2D
        # Shape wird zu: (Batch, 34, 56, 1)
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

class RepresentationNetwork3(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)

        # === SPATIAL STREAM ===
        spacial = x[:, :6, :]  # (Batch, 6, 56)
        global_f = x[:, 6:, 0]  # (Batch, 28)
        
        spacial = jnp.transpose(spacial, (0, 2, 1))
        
        spacial = nn.Conv(features=32, kernel_size=(3,), padding='SAME')(spacial)
        spacial = nn.LayerNorm()(spacial)
        spacial = nn.relu(spacial)
        
        spacial = nn.Conv(features=64, kernel_size=(3,), padding='SAME')(spacial)
        spacial = nn.LayerNorm()(spacial)
        spacial = nn.relu(spacial)
        
        spacial = nn.Conv(features=64, kernel_size=(5,), padding='SAME')(spacial)
        spacial = nn.LayerNorm()(spacial)
        spacial = nn.relu(spacial)
        
        spatial_flat = spacial.reshape(spacial.shape[0], -1)
        
        spatial_flat = nn.Dense(self.latent_dim)(spatial_flat)
        spatial_flat = nn.LayerNorm()(spatial_flat)
        spatial_flat = nn.relu(spatial_flat)
        
        # === GLOBAL STREAM ===
        global_f = nn.Dense(64)(global_f)
        global_f = nn.LayerNorm()(global_f)
        global_f = nn.relu(global_f)
        
        global_f = nn.Dense(64)(global_f)
        global_f = nn.LayerNorm()(global_f)
        global_f = nn.relu(global_f)

        # === KOMBINIERE ===
        combined = jnp.concatenate([spatial_flat, global_f], axis=-1)

        x = nn.Dense(self.latent_dim)(combined)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # ✅ LayerNorm statt Min-Max!
        # Output ≈ N(0, 1) mit lernbarem γ, β
        # → PredNet3 ResBlocks funktionieren direkt
        # → Keine Information durch Ausreißer-Komprimierung verloren
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x) # TEST 4
        
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
        min_val = jnp.min(next_latent, axis=-1, keepdims=True)
        max_val = jnp.max(next_latent, axis=-1, keepdims=True)
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
    
class DynamicsNetwork2(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2  # Reduziert: weniger Gradient nötig
    num_actions: int = 24
    
    @nn.compact
    def __call__(self, latent_state, action):
        # 1. Action Encoding - Separates Embedding statt raw One-Hot
        action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embed = nn.Dense(64)(action_one_hot)   # (Batch, 64)
        action_embed = nn.relu(action_embed)
        
        # 2. LayerNorm am Eingang: [0,1] → N(0,1)
        #    Damit Dense Layers korrekt skaliert sind
        latent_normed = nn.LayerNorm()(latent_state)  # (Batch, 256)
        
        # 3. Konditionierung: Action moduliert den Latent State
        #    Statt einfacher Concat → FiLM-artige Konditionierung
        #    (Feature-wise Linear Modulation)
        scale = nn.Dense(self.latent_dim)(action_embed)   # (Batch, 256)
        shift = nn.Dense(self.latent_dim)(action_embed)   # (Batch, 256)
        x = latent_normed * (1 + scale) + shift           # (Batch, 256)
        
        # 4. Hauptverarbeitung
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # 5. ResBlocks - weniger, weil Gradient durch stop_gradient(0.5) abnimmt
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
        
        # 6. Residual Connection zum ORIGINAL Latent State
        #    Kernidee: next_state = current_state + delta
        #    Die meisten Züge ändern nur wenig am Gesamtzustand!
        #    Das hilft dem Netzwerk weil es nur die DIFFERENZ lernen muss
        x = nn.Dense(self.latent_dim)(x)
        x = latent_state + x  # ← Globaler Skip zum Input!
        
        # 7. Min-Max Normalisierung (konsistent mit RepNet Output)
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        next_latent = (x - min_val) / (max_val - min_val + 1e-8)
        
        # --- Unused Heads ---
        reward = jnp.zeros_like(latent_state[:, :1])
        discount_logits = jnp.zeros_like(latent_state[:, :1])
        
        return next_latent, reward, discount_logits
    
class DynamicsNetwork3(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 24
    
    @nn.compact
    def __call__(self, latent_state, action):
        # 1. Action Embedding
        action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embed = nn.Dense(64)(action_one_hot)
        action_embed = nn.relu(action_embed)
        
        # 2. FiLM Konditionierung
        #    latent_state ist bereits ≈ N(0,1) von RepNet3/DynNet3
        #    → Kein extra LayerNorm nötig!
        scale = nn.Dense(self.latent_dim)(action_embed)
        shift = nn.Dense(self.latent_dim)(action_embed)
        x = latent_state * (1 + scale) + shift
        
        # 3. Hauptverarbeitung
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # 4. ResBlocks
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
        
        # 5. Globaler Skip + LayerNorm
        x = nn.Dense(self.latent_dim)(x)
        x = latent_state + x  # next_state = current_state + delta
        
        # ✅ LayerNorm statt Min-Max!
        # Konsistent mit RepNet3 Output
        # TEST 4
        next_latent = nn.LayerNorm()(x)
        
        # TEST 5
        #next_latent = x
        
        # --- Unused Heads ---
        reward = jnp.zeros_like(latent_state[:, :1])
        discount_logits = jnp.zeros_like(latent_state[:, :1])
        
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

class PredictionNetwork3(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2 
    num_actions: int = 24
    
    @nn.compact
    def __call__(self, latent_state):
        # Latent ist bereits N(0,1) wenn RepNet/DynNet LayerNorm am Output nutzen
        # For TEST4
        x = latent_state
        
        # For TEST5
        #x = nn.LayerNorm()(latent_state)
        # Shared Trunk mit wenigen ResBlocks
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
        # → Skip Connection funktioniert: N(0,1) + N(0,1) ✅
        
        # Policy Head
        policy = nn.Dense(self.latent_dim)(x)
        policy = nn.LayerNorm()(policy)
        policy = nn.relu(policy)
        policy_logits = nn.Dense(self.num_actions)(policy)
        
        # Value Head - gradueller Dimensionsabbau
        value = nn.Dense(self.latent_dim // 2)(x)    # 256→128
        value = nn.LayerNorm()(value)
        value = nn.relu(value)
        value = nn.Dense(self.latent_dim // 4)(value) # 128→64
        value = nn.relu(value)
        value = nn.Dense(1)(value)                     # 64→1
        # Output ≈ N(0, 0.18) → tanh nicht saturiert ✅
        value = nn.tanh(value)
        
        return policy_logits, value
    
# Test <=3
class PredictionNetwork2(nn.Module):
    latent_dim: int = 256
    num_head_layers: int = 2
    num_actions: int = 24
    
    @nn.compact
    def __call__(self, latent_state):
        x = nn.LayerNorm()(latent_state)
        
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
        # ✅ Kleine Initialisierung → Output nahe 0 → tanh nicht saturiert!
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        
        return policy_logits, value

class PredictionNetwork4(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 24
    
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
# LEGACY:
# class PredictionNetwork2(nn.Module):
#     latent_dim: int = 256
#     num_actions: int = 24
#     num_res_blocks: int = 6
#     num_head_layers: int = 2  # Neue Parameter für Head-Tiefe
    
#     @nn.compact
#     def __call__(self, latent_state):
#         # Gemeinsamer Trunk
#         x = latent_state
#         for _ in range(self.num_res_blocks):
#             x = ResBlock(self.latent_dim)(x)
        
#         # --- POLICY HEAD (separate Verarbeitung) ---
#         policy = x
#         for _ in range(self.num_head_layers):
#             policy = nn.Dense(self.latent_dim // 2)(policy)  # Reduzierte Dim
#             policy = nn.LayerNorm()(policy)
#             policy = nn.relu(policy)
#         policy_logits = nn.Dense(self.num_actions)(policy)
        
#         # --- VALUE HEAD (separate Verarbeitung) ---
#         value = x
#         for _ in range(self.num_head_layers):
#             value = nn.Dense(self.latent_dim // 4)(value)  # Noch kleiner für Value
#             value = nn.LayerNorm()(value)
#             value = nn.relu(value)
#         value = nn.Dense(1)(value)
#         value = nn.tanh(value)
        
#         return policy_logits, value

repr_net = RepresentationNetwork2()
dynamics_net = DynamicsNetwork2()
pred_net = PredictionNetwork4()

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

def recurrent_inference_fn(params, rng_key, action, embedding):
    ## Vereinfachte Darstellung:
    # Q(s, a) = reward + discount * Value(next_state)
    next_embedding, reward, discount_logits = dynamics_net.apply(params['dynamics'], embedding, action)
    discount = jax.nn.sigmoid(discount_logits)
    # discount wird nicht gelernt also muss es auf 1 gesetzt werden
    discount = -jnp.ones_like(discount.squeeze(-1))  # (Batch, 1) -> (Batch,) # Positive or negative 
    # discount = discount.squeeze(-1)
    prior_logits, value = pred_net.apply(params['prediction'], next_embedding)
    # reward, value: (Batch, 1) -> (Batch,)
    # reward to 0 setzen, da wir in MADN nur sparse Rewards haben und das Dynamics-Netzwerk oft 0 vorhersagt.
    reward = jnp.zeros_like(reward.squeeze(-1))  # (Batch, 1) -> (Batch,)
    # reward = reward.squeeze(-1)
    value = value.squeeze(-1)
    recurrent_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value
    )
    return recurrent_output, next_embedding

@functools.partial(jax.jit, static_argnames=['num_simulations', 'max_depth', 'temperature'])
def run_muzero_mcts(params, rng_key, observations, invalid_actions, num_simulations, max_depth, temperature):
    key1, key2 = jax.random.split(rng_key)

    # 1. Root-Knoten berechnen (Inference)
    root_output = root_inference_fn(params, observations)

    #dirichlet_fraction = temperature * 0.2

    # 2. MCTS ausführen
    policy_output = mctx.gumbel_muzero_policy(
        params=params,               # Wird an recurrent_fn weitergereicht
        rng_key=key2,
        root=root_output,            # Startpunkt der Suche
        recurrent_fn=recurrent_inference_fn, # Funktion für Schritte im latenten Raum
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1), # Wichtig für MuZero Value-Skalierung
        gumbel_scale=temperature
    )
    # policy_output = mctx.muzero_policy(
    #    params=params,               # Wird an recurrent_fn weitergereicht
    #    rng_key=key2,
    #    root=root_output,            # Startpunkt der Suche
    #    recurrent_fn=recurrent_inference_fn, # Funktion für Schritte im latenten Raum
    #     num_simulations=num_simulations,
    #     max_depth=max_depth,
    #     invalid_actions=invalid_actions,
    #     qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1), # Wichtig für MuZero Value-Skalierung
    #     dirichlet_fraction=0.25,     # Exploration Noise
    #     dirichlet_alpha=0.3,
    #     temperature=temperature
    # )
    
    # Der Root-Value ist der geschätzte Wert des aktuellen Zustands (für den aktuellen Spieler) nach der MCTS-Suche.
    root_value = policy_output.search_tree.summary().value
    # clip root_value auf [-1, 1], da unsere Value-Head-Ausgabe auch in diesem Bereich liegt
    # root_value = jnp.clip(root_value, -1.0, 1.0)
    # root_value = root_output.value
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
