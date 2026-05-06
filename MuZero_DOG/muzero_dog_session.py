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
    
class RepresentationNetwork(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)

        # === SPATIAL STREAM === 
        spacial = x[:, :6, :] 
        global_f = x[:, 6:, 0]  
        
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

        # === COMBINE ===
        combined = jnp.concatenate([spatial_flat, global_f], axis=-1)

        x = nn.Dense(self.latent_dim)(combined)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x) 
        
        return x
    
class DynamicsNetwork(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 454

    @nn.compact
    def __call__(self, latent_state, action):
        # Action Encoding 
        action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embed = nn.Dense(64)(action_one_hot)   # (Batch, 64)
        action_embed = nn.relu(action_embed)

        # LayerNorm at input
        latent_normed = nn.LayerNorm()(latent_state)  # (Batch, 256)

        # FiLM conditioning: action modulates latent state
        scale = nn.Dense(self.latent_dim)(action_embed)
        shift = nn.Dense(self.latent_dim)(action_embed)
        x = latent_normed * (1 + scale) + shift        # (Batch, 256)

        # Main processing
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)

        x = nn.Dense(self.latent_dim)(x)
        x = latent_state + x  # global skip

        min_val = jnp.min(x, axis=-1, keepdims=True)
        max_val = jnp.max(x, axis=-1, keepdims=True)
        next_latent = (x - min_val) / (max_val - min_val + 1e-8)

        # --- Reward Head ---
        # Reward (winning) depends on board/goal state AND the action taken.
        reward_input = jnp.concatenate([next_latent, action_embed], axis=-1)  # (Batch, 320)
        reward_logits = nn.Dense(64)(reward_input)
        reward_logits = nn.relu(reward_logits)
        reward_logits = nn.Dense(3, name='reward_head')(reward_logits)

        # --- Discount Head ---
        # Session-end (discount=0) is determined by hand size reaching 0 —
        discount_logits = nn.Dense(64)(next_latent)   # state only, no action
        discount_logits = nn.relu(discount_logits)
        discount_logits = nn.Dense(3, name='discount_head')(discount_logits)

        return next_latent, reward_logits, discount_logits

class PredictionNetwork(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 454  # Pins x Karten
    
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

repr_net = RepresentationNetwork()
dynamics_net = DynamicsNetwork()
pred_net = PredictionNetwork()

def recurrent_fn(params, rng_key, action, embedding):
    """Single deterministic transition: no chance nodes, one NN call per simulation."""
    next_embedding, reward_logits, discount_logits = dynamics_net.apply(
        params['dynamics'], embedding, action
    )
    prior_logits, value = pred_net.apply(params['prediction'], next_embedding)
    value = value.squeeze(-1)

    support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
    reward   = jnp.sum(jax.nn.softmax(reward_logits)   * support, axis=-1)
    discount = jnp.sum(jax.nn.softmax(discount_logits) * support, axis=-1)

    return mctx.RecurrentFnOutput(
        prior_logits=prior_logits,
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
        value=value,
    )


@functools.partial(jax.jit, static_argnames=['num_simulations', 'max_depth', 'temperature'])
def run_muzero_mcts(params, rng_key, observations, invalid_actions,
                    num_simulations, max_depth, temperature):
    """Gumbel MuZero with deterministic recurrent_fn (no chance nodes)."""
    _, key2 = jax.random.split(rng_key)
    root_output = root_inference_fn(params, observations)
    # policy_output = mctx.muzero_policy(
    #     params=params,
    #     rng_key=key2,
    #     root=root_output,
    #     recurrent_fn=combined_recurrent_fn,
    #     num_simulations=num_simulations,
    #     invalid_actions=invalid_actions,
    #     max_depth=max_depth,
    #     qtransform=mctx.qtransform_by_parent_and_siblings,
    #     temperature=temperature,
    # )
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=key2,
        root=root_output,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
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
    
    dummy_obs = jnp.ones((1, *input_shape))
    params_repr = repr_net.init(key_repr, dummy_obs)
    
    dummy_latent = repr_net.apply(params_repr, dummy_obs)
    
    dummy_action = jnp.array([0])  
    params_dyn = dynamics_net.init(key_dyn, dummy_latent, dummy_action)

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