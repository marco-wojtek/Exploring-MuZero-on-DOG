import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MADN.deterministic_madn import *

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
        return nn.relu(residual + x)

class RepresentationNetwork2(nn.Module):
    latent_dim: int = 256 
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, 14, 56)
        x = x.astype(jnp.float32)

        spacial = x[:, :6, :]  # (Batch, 6, 56)
        global_f = x[:, 6:, 0]  # (Batch, 28)
        
        spacial = jnp.transpose(spacial, (0, 2, 1))
        
        # Convolutional Layers (Feature Extraction auf dem Board)
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
   
class DynamicsNetwork4(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2  # Reduziert: weniger Gradient nötig
    num_actions: int = 24
    
    @nn.compact
    def __call__(self, latent_state, action):
        # Action Encoding - Separates Embedding statt raw One-Hot
        action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embed = nn.Dense(64)(action_one_hot)   # (Batch, 64)
        action_embed = nn.relu(action_embed)
        
        latent_normed = nn.LayerNorm()(latent_state)  # (Batch, 256)
        
        # FiLM-ähnliche Modulation: Latent State wird durch Action-Embedding skaliert und verschoben
        scale = nn.Dense(self.latent_dim)(action_embed)   # (Batch, 256)
        shift = nn.Dense(self.latent_dim)(action_embed)   # (Batch, 256)
        x = latent_normed * (1 + scale) + shift           # (Batch, 256)
        
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # ResBlocks - weniger, weil Gradient durch stop_gradient(0.5) abnimmt
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
        
        x = nn.Dense(self.latent_dim)(x)
        x = latent_state + x  
        
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        next_latent = (x - min_val) / (max_val - min_val + 1e-8)
        
        # --- Reward Head: 3 Klassen {-1, 0, +1} ---
        reward_input = jnp.concatenate([
            next_latent, 
            action_one_hot
        ], axis=-1)
        reward_logits = nn.Dense(64)(reward_input)
        reward_logits = nn.relu(reward_logits)
        reward_logits = nn.Dense(3, name='reward_head')(reward_logits)  
        
        # --- Discount Head: 2 Klassen {0=Terminal, 1=Non-Terminal} ---
        discount_input = jnp.concatenate([
            next_latent,
            action_one_hot
        ], axis=-1)
        discount_logits = nn.Dense(64)(discount_input)
        discount_logits = nn.relu(discount_logits)
        discount_logits = nn.Dense(2, name='discount_head')(discount_logits)

        # --- Depth Delta Head: {0=gleicher Spieler (6er), 1=Spielerwechsel} ---
        depth_delta_logit = nn.Dense(1, name='depth_delta_head')(action_one_hot)  # (B, 1)

        return next_latent, reward_logits, discount_logits, depth_delta_logit

class PredictionNetwork4(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 24
    
    @nn.compact
    def __call__(self, latent_state):
        x = nn.LayerNorm()(latent_state)

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
        value = nn.Dense(4)(value) # Multi-Value Head
        # values[:, 0] = Wert für aktuellen Spieler
        # values[:, 1] = Wert für nächsten Spieler relativ
        value = nn.tanh(value)
        
        return policy_logits, value

repr_net = RepresentationNetwork2()
dynamics_net = DynamicsNetwork4()
pred_net = PredictionNetwork4()

def root_inference_fn(params, observation):
    embedding = repr_net.apply(params['representation'], observation)  # (B, 256)
    prior_logits, values = pred_net.apply(params['prediction'], embedding)  # values: (B, 4)
    # Depth=0 am Root: aktueller Spieler ist der Root-Spieler
    # values[:, 0] = Value für den Root-Spieler (dessen MCTS-Perspektive)
    value = values[:, 0]  # (B,)
    # Depth-Scalar an Embedding anhängen (0 = Root-Spieler ist am Zug)
    depth_init = jnp.zeros((embedding.shape[0], 1))
    embedding_with_depth = jnp.concatenate([embedding, depth_init], axis=-1)  # (B, 257)
    return mctx.RootFnOutput(
        embedding=embedding_with_depth,
        prior_logits=prior_logits,
        value=value
    )

def recurrent_inference_fn(params, rng_key, action, embedding):
    # Embedding = [latent (256), depth_counter (1)]
    # depth_counter: 0=Root-Spieler am Zug, 1=Spieler+1, 2=Spieler+2, 3=Spieler+3
    latent = embedding[:, :256]   # (B, 256)
    depth = embedding[:, 256:]    # (B, 1), Werte 0.0/1.0/2.0/3.0

    next_latent, reward_logits, discount_logits, depth_delta_logit = dynamics_net.apply(params['dynamics'], latent, action)

    # Depth Delta: Netz lernt ob Spieler wechselt (1) oder gleich bleibt (0 = 6er Bonus-Zug)
    # sigmoid → soft [0, 1], rundet effektiv zu 0 oder 1 sobald trainiert
    depth_delta = jax.nn.sigmoid(depth_delta_logit)  # (B, 1)
    next_depth = (depth + depth_delta) % 4.0         # (B, 1)
    next_embedding = jnp.concatenate([next_latent, next_depth], axis=-1)  # (B, 257)

    prior_logits, values = pred_net.apply(params['prediction'], next_latent)
    # values: (B, 4) = [v_aktueller_Spieler, v_nächster, v_übernächster, v_+3]
    # Root-Spieler-Index: Root ist (4 - next_depth) Schritte VOR dem aktuellen Spieler
    # Bei next_depth=1: Root ist 3 Schritte weg → values[:, 3]
    # Bei next_depth=2: Root ist 2 Schritte weg → values[:, 2]
    # Bei next_depth=0: Root ist aktueller Spieler → values[:, 0]
    next_depth_int = jnp.round(next_depth).astype(jnp.int32).squeeze(-1) % 4  # (B,)
    old_depth_int = jnp.round(depth).astype(jnp.int32).squeeze(-1) % 4  # (B,) 
    root_idx = (4 - next_depth_int) % 4  # (B,)
    value = values[jnp.arange(values.shape[0]), root_idx]  # (B,) - Root-Spieler-Value

    # Reward: Categorical {-1, 0, +1} aus Sicht des AKTUELLEN Spielers
    #   FFA (Free-for-All): wenn Gegner gewinnt, verliert Root → Vorzeichen flip
    #   Bei next_depth==0 (Root am Zug): kein Flip nötig
    support_reward = jnp.array([-1.0, 0.0, 1.0])
    reward_probs = jax.nn.softmax(reward_logits, axis=-1)
    reward_current = jnp.sum(reward_probs * support_reward, axis=-1)  # (B,)
    is_root_turn = (old_depth_int == 0)  # Root-Spieler ist am Zug
    reward = jnp.where(is_root_turn, reward_current, -reward_current)  # (B,)

    # Discount: Binary {0.0=Terminal, 1.0=Non-Terminal}
    support_disc = jnp.array([0.0, 1.0])
    discount_probs = jax.nn.softmax(discount_logits, axis=-1)
    discount = jnp.sum(discount_probs * support_disc, axis=-1)  # (B,)

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
    # policy_output = mctx.gumbel_muzero_policy(
    #     params=params,               # Wird an recurrent_fn weitergereicht
    #     rng_key=key2,
    #     root=root_output,            # Startpunkt der Suche
    #     recurrent_fn=recurrent_inference_fn, # Funktion für Schritte im latenten Raum
    #     num_simulations=num_simulations,
    #     max_depth=max_depth,
    #     invalid_actions=invalid_actions,
    #     qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=0.5),
    #     gumbel_scale=temperature,    
    # )
    policy_output = mctx.muzero_policy(
       params=params,               # Wird an recurrent_fn weitergereicht
       rng_key=key2,
       root=root_output,            # Startpunkt der Suche
       recurrent_fn=recurrent_inference_fn, # Funktion für Schritte im latenten Raum
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        qtransform=mctx.qtransform_by_parent_and_siblings, 
        dirichlet_fraction=0.25,     # Exploration Noise
        dirichlet_alpha=0.3,
        temperature=temperature
    )
    
    # Der Root-Value ist der geschätzte Wert des aktuellen Zustands (für den aktuellen Spieler) nach der MCTS-Suche.
    root_value = policy_output.search_tree.summary().value
    # clip root_value auf [-1, 1], da unsere Value-Head-Ausgabe auch in diesem Bereich liegt
    # root_value = jnp.clip(root_value, -1.0, 1.0)
    # root_value = root_output.value
    return policy_output, root_value

@functools.partial(jax.jit, static_argnames=['num_simulations', 'max_depth', 'temperature'])
def run_gumbel_muzero_mcts(params, rng_key, observations, invalid_actions, num_simulations, max_depth, temperature):
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
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=0.5),
        gumbel_scale=temperature,    
    )
    # policy_output = mctx.muzero_policy(
    #    params=params,               # Wird an recurrent_fn weitergereicht
    #    rng_key=key2,
    #    root=root_output,            # Startpunkt der Suche
    #    recurrent_fn=recurrent_inference_fn, # Funktion für Schritte im latenten Raum
    #     num_simulations=num_simulations,
    #     max_depth=max_depth,
    #     invalid_actions=invalid_actions,
    #     qtransform=mctx.qtransform_by_parent_and_siblings, 
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
    
    # Representation Network
    # Input: Observation (Batch-Dimension hinzufügen für init)
    dummy_obs = jnp.ones((1, *input_shape))
    params_repr = repr_net.init(key_repr, dummy_obs)
    
    dummy_latent = repr_net.apply(params_repr, dummy_obs)
    
    # Dynamics Network
    # Input: Latent State + Action (Integer)
    dummy_action = jnp.array([0]) # Batch size 1, Action 0
    params_dyn = dynamics_net.init(key_dyn, dummy_latent, dummy_action)
    
    # Prediction Network
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
