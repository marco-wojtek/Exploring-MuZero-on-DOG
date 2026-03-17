import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *

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
    latent_dim: int = 256
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)

        # === SPATIAL STREAM === 
        spacial = x[:, :6, :]  # (Batch, 6, 56) !! ANPASSEN !!
        global_f = x[:, 6:, 0]  # (Batch, 28) !! ANPASSEN !!
        
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
    pass

class PredictionNetwork(nn.Module):
    pass

repr_net = RepresentationNetwork()
dynamics_net = DynamicsNetwork()
pred_net = PredictionNetwork()

def root_inference_fn(params, observation):
    pass

def recurrent_inference_fn(params, rng_key, action, embedding):
    pass

@functools.partial(jax.jit, static_argnames=['num_simulations', 'max_depth', 'temperature'])
def run_muzero_mcts(params, rng_key, observations, invalid_actions, num_simulations, max_depth, temperature):
    """
    Runs the MuZero MCTS algorithm for a batch of observations.
    Args:
        params: Model parameters
        rng_key: JAX random key
        observations: Batch of observations (shape: [batch_size, observation_dim])
        invalid_actions: Batch of invalid action masks (shape: [batch_size, num_actions])
        num_simulations: Number of MCTS simulations to run
        max_depth: Maximum depth for MCTS
        temperature: Temperature parameter for action selection
    Returns:
        A batch of action probabilities (shape: [batch_size, num_actions])
    """
    key1, key2 = jax.random.split(rng_key)

    # 1. Root-Knoten berechnen (Inference)
    root_output = root_inference_fn(params, observations)

    # 2. MCTS ausführen
    policy_output = mctx.gumbel_muzero_policy(
        params=params,               # Wird an recurrent_fn weitergereicht
        rng_key=key2,
        root=root_output,            # Startpunkt der Suche
        recurrent_fn=recurrent_inference_fn, # Funktion für Schritte im latenten Raum
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        # qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1), # Wichtig für MuZero Value-Skalierung
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=0.5),
        gumbel_scale=temperature,  
    )

    root_value = policy_output.search_tree.summary().value

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