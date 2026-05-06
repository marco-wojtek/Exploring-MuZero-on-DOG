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
        return nn.relu(residual + x)

class RepresentationNetwork2(nn.Module):
    latent_dim: int = 256  # Größer als 64 für MADN
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, 14, 56)
        x = x.astype(jnp.float32)

        # trenne spatial und global informationen:
        spacial = x[:, :6, :]  # (Batch, 6, 56)
        global_f = x[:, 6:, 0]  # (Batch, 5, 56) -> (Batch, 28)
        
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
        
        # Projektion auf Latent Dim
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

        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        x = nn.Dense(self.latent_dim)(x)
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        x = (x - min_val) / (max_val - min_val + 1e-8)
        return x
    
class PredictionNetwork4(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 4
    
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
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        
        return policy_logits, value
    
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
        # Action Embedding
        action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embed = nn.Dense(64, name='act_embed')(action_one_hot)
        action_embed = nn.relu(action_embed)

        # FiLM Conditioning
        latent_normed = nn.LayerNorm(name='act_input_ln')(latent_state)
        scale = nn.Dense(self.latent_dim, name='act_film_scale')(action_embed)
        shift = nn.Dense(self.latent_dim, name='act_film_shift')(action_embed)
        x = latent_normed * (1 + scale) + shift

        x = nn.Dense(self.latent_dim, name='act_dense1')(x)
        x = nn.LayerNorm(name='act_ln1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, name='act_dense2')(x)
        x = nn.LayerNorm(name='act_ln2')(x)
        x = nn.relu(x)
        for i in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)

        # Residual + Min-Max
        x = nn.Dense(self.latent_dim, name='act_proj')(x)
        x = latent_state + x
        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        afterstate = (x - min_val) / (max_val - min_val + 1e-8)

        # Reward Head: 3 Klassen {-1, 0, +1}
        reward_input = jnp.concatenate([afterstate, action_one_hot], axis=-1)
        reward_logits = nn.Dense(64, name='reward_dense')(reward_input)
        reward_logits = nn.relu(reward_logits)
        reward_logits = nn.Dense(3, name='reward_head')(reward_logits)

        # Discount Head: 3 Klassen {-1, 0, +1}
        discount_logits = nn.Dense(32, name='discount_dense')(latent_state)
        discount_logits = nn.LayerNorm(name='discount_ln')(discount_logits)
        discount_logits = nn.relu(discount_logits)
        discount_logits = nn.Dense(3, name='discount_head')(discount_logits)

        # Chance Logits: Vorhersage der Würfelverteilung
        chance_logits = nn.Dense(self.num_chance_outcomes, name='chance_head')(afterstate)

        return afterstate, reward_logits, chance_logits, discount_logits

    @nn.compact
    def chance_dynamics(self, afterstate, chance_outcome):
        """Afterstate + Würfel → Next State"""
        # Chance Embedding
        chance_one_hot = jax.nn.one_hot(chance_outcome, num_classes=self.num_chance_outcomes)
        chance_embed = nn.Dense(64, name='chance_embed')(chance_one_hot)
        chance_embed = nn.relu(chance_embed)

        # FiLM Conditioning (gleiche Struktur wie action_dynamics)
        afterstate_normed = nn.LayerNorm(name='chance_input_ln')(afterstate)
        scale = nn.Dense(self.latent_dim, name='chance_film_scale')(chance_embed)
        shift = nn.Dense(self.latent_dim, name='chance_film_shift')(chance_embed)
        x = afterstate_normed * (1 + scale) + shift

        x = nn.Dense(self.latent_dim, name='chance_dense1')(x)
        x = nn.LayerNorm(name='chance_ln1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, name='chance_dense2')(x)
        x = nn.LayerNorm(name='chance_ln2')(x)
        x = nn.relu(x)
        for i in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)

        # Residual + Min-Max
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
        reward=reward,      
        discount=discount,  
    ), next_embedding

def root_inference_fn(params, observation):
    embedding = repr_net.apply(params['representation'], observation)
    prior_logits, value = pred_net.apply(params['prediction'], embedding)
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
        # qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=0.1),
        qtransform=mctx.qtransform_by_parent_and_siblings,
        temperature=temperature
    )

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
    
    # representation Network
    # Input: Observation (Batch-Dimension hinzufügen für init)
    dummy_obs = jnp.ones((1, *input_shape))
    params_repr = repr_net.init(key_repr, dummy_obs)

    dummy_latent = repr_net.apply(params_repr, dummy_obs)
    
    # Dynamics Network (Stochastic hat 2 Methoden!)
    dummy_action = jnp.array([0]) 
    dummy_chance = jnp.array([0])  

    params_dyn = dynamics_net.init(key_dyn, dummy_latent, dummy_action, dummy_chance)
    
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
