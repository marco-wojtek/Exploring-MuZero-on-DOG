import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx

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
    
def decision_recurrent_fn(params, rng_key, action, embedding):
    # Aufruf des Netzwerks
    afterstate, reward, chance_logits, discount_logits = params['dynamics'].action_dynamics(embedding, action)
    
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
    next_embedding = params['dynamics'].chance_dynamics(afterstate, chance_outcome)
    
    # Jetzt brauchen wir Policy und Value für den NEUEN State
    # Das macht das Prediction Network
    prior_logits, value = params['prediction'](next_embedding)
    
    return mctx.ChanceRecurrentFnOutput(
        value=value,
        prior_logits=prior_logits,
        embedding=next_embedding
    )