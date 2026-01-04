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
    latent_dim: int = 128  # Größer als 64 für MADN
    num_res_blocks: int = 2

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, 14, 56)
        # 1. Sicherstellen, dass es Float ist
        x = x.astype(jnp.float32)
        
        # 2. Channel-Dimension hinzufügen für Conv2D
        # Shape wird zu: (Batch, 14, 56, 1)
        x = x[..., None]
        
        # 3. Convolutional Layers (Feature Extraction auf dem Board)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        # 4. Flatten für Dense Layers
        # Wir behalten die Batch-Dimension (0) und flachen den Rest
        x = x.reshape((x.shape[0], -1))
        
        # 5. Projektion auf Latent Dim
        x = nn.Dense(self.latent_dim)(x)
        x = nn.relu(x)
        
        # Residual Blocks zur weiteren Verarbeitung
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # Normalisierung des Latent States
        x = nn.Dense(self.latent_dim)(x)
        x = nn.sigmoid(x)
        return x

class DynamicsNetwork(nn.Module):
    latent_dim: int = 128
    num_res_blocks: int = 2
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
        x = nn.relu(x)
        
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # --- HEADS (Ausgaben) ---
        
        # A. Next Latent State
        next_latent = nn.Dense(self.latent_dim)(x)
        next_latent = nn.sigmoid(next_latent) # Gleiche Skalierung wie Representation!
        
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
    latent_dim: int = 128
    num_actions: int = 24
    num_res_blocks: int = 2
    
    @nn.compact
    def __call__(self, latent_state):
        x = latent_state
        
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        # --- HEADS ---
        
        # A. Policy (Welche Aktion ist gut?)
        policy_logits = nn.Dense(self.num_actions)(x)
        
        # B. Value (Wie gut ist der Zustand für den aktuellen Spieler?)
        # Tanh für [-1 (Verlieren), 1 (Gewinnen)]
        value = nn.Dense(1)(x)
        value = nn.tanh(value)
        
        return policy_logits, value
    
repr_net = RepresentationNetwork()
dynamics_net = DynamicsNetwork()
pred_net = PredictionNetwork()

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
    next_embedding, reward, discount_logits = dynamics_net.apply(params['dynamics'], embedding, action)
    discount = jax.nn.sigmoid(discount_logits)
    discount = discount.squeeze(-1)  # (Batch, 1) -> (Batch,)
    prior_logits, value = pred_net.apply(params['prediction'], next_embedding)
    # reward, value: (Batch, 1) -> (Batch,)
    reward = reward.squeeze(-1)
    value = value.squeeze(-1)
    recurrent_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value
    )
    return recurrent_output, next_embedding

@jax.jit
def run_muzero_mcts(params, rng_key, observations, invalid_actions=None):
    batch_size = observations.shape[0]
    key1, key2 = jax.random.split(rng_key)

    # 1. Root-Knoten berechnen (Inference)
    root_output = root_inference_fn(params, observations)

    # 2. MCTS ausführen
    policy_output = mctx.muzero_policy(
        params=params,               # Wird an recurrent_fn weitergereicht
        rng_key=key2,
        root=root_output,            # Startpunkt der Suche
        recurrent_fn=recurrent_inference_fn, # Funktion für Schritte im latenten Raum
        num_simulations=100,
        max_depth=50,
        invalid_actions=invalid_actions,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1), # Wichtig für MuZero Value-Skalierung
        dirichlet_fraction=0.25,     # Exploration Noise
        dirichlet_alpha=0.3
    )
    
    # Wir geben zusätzlich den rohen Value des Root-Knotens zurück (vom Netzwerk geschätzt)
    return policy_output, root_output.value

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