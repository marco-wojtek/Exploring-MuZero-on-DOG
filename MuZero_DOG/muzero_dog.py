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

        # === KOMBINIERE ===
        combined = jnp.concatenate([spatial_flat, global_f], axis=-1)

        x = nn.Dense(self.latent_dim)(combined)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.latent_dim)(x)
            
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x) # TEST 4
        
        return x
    
class DynamicsNetwork(nn.Module):
    latent_dim: int = 256
    num_res_blocks: int = 2
    num_actions: int = 454       # Classic MADN: 4 Pins 
    num_chance_outcomes: int = 128  # 2^7: joker|swap|13|7|1-11|4-card|normal-fwd

    @nn.compact
    def __call__(self, latent_state, action, chance_outcome=None):
        """Init-Call: durchläuft beide Pfade um alle Parameter zu erstellen."""
        afterstate, afterstate_value, reward_logits, chance_logits, discount_logits = self.action_dynamics(latent_state, action)
        if chance_outcome is not None:
            next_state = self.chance_dynamics(afterstate, chance_outcome)
            return afterstate, afterstate_value, reward_logits, chance_logits, discount_logits, next_state
        return afterstate, afterstate_value, reward_logits, chance_logits, discount_logits

    @nn.compact
    def action_dynamics(self, latent_state, action):
        """State + Action → Afterstate + Reward/Discount/Chance-Logits"""
        # action_one_hot = jax.nn.one_hot(action, num_classes=self.num_actions, dtype=jnp.float32)
        # action_embed = nn.Dense(64, name='act_embed')(action_one_hot)
        # action_embed = nn.relu(action_embed)
        action_embed = nn.Embed(num_embeddings=self.num_actions, features=64, name='act_embed')(action)

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

        x = nn.Dense(self.latent_dim, name='act_proj')(x)
        x = latent_state + x
        afterstate = nn.LayerNorm(name='afterstate_ln')(x)

        # 5. Reward Head: 3 Klassen {-1, 0, +1}
        reward_input = jnp.concatenate([afterstate, action_embed], axis=-1)  # (B, 256+64) instead of (B, 1254)
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

        afterstate_value = nn.Dense(64, name='afterstate_val_dense')(afterstate)
        afterstate_value = nn.relu(afterstate_value)
        afterstate_value = nn.Dense(1, name='afterstate_val_head')(afterstate_value)
        afterstate_value = nn.tanh(afterstate_value).squeeze(-1)

        return afterstate, afterstate_value, reward_logits, chance_logits, discount_logits

    @nn.compact
    def chance_dynamics(self, afterstate, chance_outcome):
        """Afterstate + Card Category Bits → Next State"""
        # 1. Unpack 7 bits from outcome integer (B,) → (B, 7)
        # Unlike one_hot(128), similar outcomes that differ by one card type
        # share gradient information through shared weight columns.
        # Dense(7→64) has 448 params vs one_hot Dense(128→64) = 8192 params.
        chance_bits = ((chance_outcome[:, None] >> jnp.arange(7, dtype=jnp.int32)) & 1).astype(jnp.float32)
        chance_embed = nn.Dense(64, name='chance_embed')(chance_bits)
        chance_embed = nn.relu(chance_embed)

        # FiLM Conditioning
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

        x = nn.Dense(self.latent_dim, name='chance_proj')(x)
        x = afterstate + x  
        next_state = nn.LayerNorm(name='next_state_ln')(x)

        return next_state

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

def decision_recurrent_fn(params, rng_key, action, embedding):
    afterstate, afterstate_value, reward_logits, chance_logits, discount_logits = dynamics_net.apply(
        params['dynamics'], embedding, action, method=dynamics_net.action_dynamics
    )

    support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
    reward = jnp.sum(jax.nn.softmax(reward_logits) * support, axis=-1)
    discount = jnp.sum(jax.nn.softmax(discount_logits) * support, axis=-1)

    afterstate_with_info = jnp.concatenate([afterstate, reward[:, None], discount[:, None]], axis=-1)
    
    return mctx.DecisionRecurrentFnOutput(
        chance_logits=chance_logits,
        afterstate_value=afterstate_value,
    ), afterstate_with_info 

def chance_recurrent_fn(params, rng_key, chance_outcome, afterstate_with_info):
    afterstate = afterstate_with_info[:, :-2] 
    reward = afterstate_with_info[:, -2]      
    discount = afterstate_with_info[:, -1]     
    
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

    root_output = root_inference_fn(params, observations)

    policy_output = mctx.stochastic_muzero_policy(
        params=params,
        rng_key=key2,
        root=root_output,
        decision_recurrent_fn=decision_recurrent_fn,
        chance_recurrent_fn=chance_recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        qtransform=mctx.qtransform_by_parent_and_siblings,
        temperature=temperature
    )

    root_value = policy_output.search_tree.summary().value

    return policy_output, root_value

def combined_recurrent_fn(params, rng_key, action, embedding):
    """1 NN forward pass per simulation vs 4 for stochastic_muzero_policy.
    
    stochastic_muzero_policy uses jnp.where (not lax.cond) → BOTH
    decision_recurrent_fn AND chance_recurrent_fn always execute.
    Here we collapse the decision+chance step into one call.
    
    The chance outcome is sampled from chance_logits so different simulations
    explore different card futures → trees stay broad → while_loop exits early.
    """
    afterstate, afterstate_value, reward_logits, chance_logits, discount_logits = dynamics_net.apply(
        params['dynamics'], embedding, action, method=dynamics_net.action_dynamics
    )
    chance_outcome = jax.random.categorical(rng_key, chance_logits)  

    next_embedding = dynamics_net.apply(
        params['dynamics'], afterstate, chance_outcome, method=dynamics_net.chance_dynamics
    )
    prior_logits, value = pred_net.apply(params['prediction'], next_embedding)
    value = value.squeeze(-1)

    support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
    reward   = jnp.sum(jax.nn.softmax(reward_logits)   * support, axis=-1)
    discount = jnp.sum(jax.nn.softmax(discount_logits) * support, axis=-1)

    return mctx.RecurrentFnOutput(
        prior_logits=prior_logits, value=value,
        reward=reward, discount=discount,
    ), next_embedding


@functools.partial(jax.jit, static_argnames=['num_simulations', 'max_depth', 'temperature'])
def run_muzero_mcts_fast(params, rng_key, observations, invalid_actions,
                         num_simulations, max_depth, temperature):
    """muzero_policy + combined_recurrent_fn: 4x fewer NN calls than stochastic variant."""
    key1, key2 = jax.random.split(rng_key)
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
        recurrent_fn=combined_recurrent_fn, 
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
    dummy_chance = jnp.array([0])
    params_dyn = dynamics_net.init(key_dyn, dummy_latent, dummy_action, dummy_chance)
    
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