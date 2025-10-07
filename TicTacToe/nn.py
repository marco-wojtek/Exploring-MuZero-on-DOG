import flax.linen as nn
import jax.numpy as jnp

class RepresentationNetwork(nn.Module):
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        # x shape: (3, 3) für TicTacToe board
        x = x.flatten()  # (9,)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x

class DynamicsNetwork(nn.Module):
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, latent_state, action):
        # action als one-hot encoding
        action_encoded = jnp.eye(9)[action]  # 9 Aktionen für TicTacToe
        x = jnp.concatenate([latent_state, action_encoded])
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Reward prediction
        reward = nn.Dense(1)(x)
        reward = nn.tanh(reward)  # [-1, 1] für TicTacToe
        
        # Next latent state
        next_latent = nn.Dense(self.latent_dim)(x)
        
        return reward, next_latent

class PredictionNetwork(nn.Module):
    num_actions: int = 9
    
    @nn.compact
    def __call__(self, latent_state):
        x = nn.Dense(128)(latent_state)
        x = nn.relu(x)
        
        # Policy logits
        policy_logits = nn.Dense(self.num_actions)(x)
        
        # Value
        value = nn.Dense(1)(x)
        value = nn.tanh(value)  # [-1, 1] für TicTacToe
        
        return policy_logits, value