# Notes for building MuZero for MADN variants

## Network input
- In AZ/MZ (chess) the input is 8x8x119 encoding all positions current colour and more
    - one 8x8 plane shows where white pawns are, ...
    - same for all black pieces 
    - one plane completly one or zero to show current player
    - one plane with a value at each position for steps
    - 8 history states (necessary for chess)
- For MADN:
    - board has size with d (distance between starts), p (num_players): d\*p + 4p (4p for goal area)
    - need planes for positions for all pins (num_players*board_size)
    - need plane for current player
    - maybe current dice
    - maybe history
    - maybe information about used cards? -> if another player has no action and discards the hand, those cards stay hidden, but any card played by an opponent was seen and thus can be remembered

    - state encoding necessary

## Nework output
- In AZ/MZ 8x8x73 (4672) possible actions
    - 8x8 for position of pins to move
    - 73 actions:
        - moving 1-7 in one of 8 directions (56)
        - horse moves (8)
        - 9 special cases (castle, ...)
    - large action space = a lot of masking but clear structure and easy to expand
- For MADN:
    - take 6 actions (for dice) or 13 (for cards)
        - difficult to implement special cards -> 1/11 card, swap card
    - AZ/MZ approach board focus:
        - board_size x 13 normal forward move actions 
        - board_size x 1 for walking 4 backwards
        - maybe an action for start move
        - board_size² for swap actions
        - all actions * 2 for alternative choice with joker
    - hierarchical actions
        - select action/card first, then pin
    - adapted AZ/MZ approach pin focus:
        - for each pin all actions
        - 4 x X
        - Total number of cards: 14 (8 normal, 6 special)
        - possible actions:
            - move 1-13 spaces, includes 1 or 11 with the special card, exact 7 with 7 card and 4 with +-4 card
            - move -4 
            - move out of home to start: 1|11, 13, Joker
            - swap pin with pin in location x: board_size(without goal area) -1 (current position of pin) +1 (No possible swap) options
            - 7 card:
                - distribute 7 moves over any number of pins
                - 120 possible distributions
                - can be done independent of pin if always a dist over ALL pins must be selected
                - if selection by distribution -> one action less for normal moves
            - ? card can do any card
        - TOTAL: 4* (2*(12 + 1  + board_size) + 3) + 2 * 120 = 868 for distance 16, (868 - 4\*2\*4\*(16-10))676 for distance 10
        - or 4 x 157 + 2*120
        - 157 = (each pin)4* ((Joker and normal)2* (12 (moves 1 to 13 without 7) + 1 (moving -4)) + 3(move out home))
        - =>4x  {(1 + joker_enabled) * [num_normal_cards (-1 if hot_7 enabled) + neg_4_enabled] + 2 (1|11 and 13 start move) + joker_enabled}  + (1 + joker_enabled) * (hot_7_enabled)
    - Encoder and Decoder for Action <-> Index necessary
        - actions are jnp.array([pin, move]), for hot 7 we have [a,b,c,d] as moves; for a,b,,c,d > 0 a step call is made, where the active player doesn't change
        - first get "action" or "hot 7" by action_idx % 157  
        - then filter joker or no joker
        - filter swap or not
        - filter move start or not
        - perform normal step

# Imperfect Information
- Train perfect information agent and build upon that to further train for hidden information
- learning with dropout
- res blocks, experience replay buffer, L2 regularization

# possible parameters for setting rule in the transition from dMADN to DOG
- enable_dice (disables cards and action set) ONLY IF NOT SEPERATE GAMES
- enable_cards (disables dice and action set) ONLY IF NOT SEPERATE GAMES
- enable_special_cards (1|11 and 13 must always be enabled for start move)
    - joker_enabled
    - hot_7_enabled
    - swap_enabled
    - neg_4_enabled
- enable_start_block XXXXXXXXXXXXX
- enable_teams 
- enable_jumping_in_goal XXXXXXXXXXXXXXXXXXXXXX
- enable_initial_free_pin XXXXXXXXXXXXXXX
- enable_extra_move_on_6 (only dMADN and MADN) XXXXXXXXXXXXXXXXXXXX
- enable_start_on_1_and_6 (dMADN, MADN) XXXXXXXXXXXXXXXXX
- enable_rethrow (up to 3 throws if all pins in start, MADN only) XXXXXXXXXXXXXXXXX
- ~enable_forced_start (only MADN)~
- ~enable_forced_attack~
- enable_friendly_fire (on team member pins) XXXXXXXXXXXXXXXXXXXXXXXX
- enable_circular_board XXXXXXXXXXXXXXXXX

# stochastic MuZero (mctx)
Die policy beginnt immer mit einem decision Node also nach einem Würfelwurf. 

Decision_Node --action--> Afterstate (vor Würfelwurf) --Würfelwurf--> Chance_Node ---Baue Kanten mit mögl. actions--->Decision_Node

# Zugzwang
Wichtig für MuZero:
1. Dynamics-Modell f muss no-step Übergänge kennen (kommen in Trainingsdaten vor)
2. policy g sollte für Zugzwang keinen output lernen
    a) no-step als eigenes Action-label
    b) vor policy aufruf, automatisch weiterleiten und kein branching dort einfügen
3. policy kann verzerrt lernen (lernen verlangsamt) wenn es in Zugzwang-Zuständen die policy lernt

# DOG Regeln
- Figuren auf dem eigenen Startfeld dürfen nicht getauscht werden
- Startfeld blockiert alle Figuren; d.h. man kommt nicht ins eigene Ziel wenn eine Figur dort steht

# Action overview
Für distance 10, 2 oder 4 Spieler Immer [a, b[:
- Joker:
  - Swap: 0 - 192(=4*40 + 8 oder 16) | 0 - 224
  - Hot 7: 192 - 312 | 224 - 344
  - Normal: 312 - 360 | 344 - 392
  - -4: 360 - 364 | 392 - 396
- Swap: 364 - 556 | 392 - 616
- Hot 7: 556 - 676 | 616 - 736
- Normal: 676 - 724 | 736 - 784
- -4: 724 - 728 | 784 - 788

# Sample und Training prozess

Datenfluss vom Sampling bis zur Loss-Berechnung
1. Sampling (vec_replay_buffer.py)
Für jede Episode im Batch (batch_size=128):

Schritt 1: Wähle zufällige Episode + zufälligen Startpunkt t
```
ep_indices = [ep₁, ep₂, ..., ep₁₂₈]  # 128 zufällige Episoden
t_starts = [t₁, t₂, ..., t₁₂₈]       # 128 zufällige Startpunkte
```

Schritt 2: Extrahiere K=6 aufeinanderfolgende Timesteps (t, t+1, t+2, t+3, t+4, t+5):

Root Observation: observations[t] → Shape: (batch_size, 14, 56)
Actions: [a_t, a_{t+1}, a_{t+2}, a_{t+3}, a_{t+4}] → Shape: (batch_size, K-1=5)
Rewards: [r_t, r_{t+1}, r_{t+2}, r_{t+3}, r_{t+4}] → Shape: (batch_size, 5)
Policies: [π_t, π_{t+1}, ..., π_{t+5}] → Shape: (batch_size, K=6, 24)
Values: [v_t, v_{t+1}, ..., v_{t+5}] → Shape: (batch_size, 6)
Target Values: [z_t, z_{t+1}, ..., z_{t+5}] → Shape: (batch_size, 6)
Masks: [m_t, m_{t+1}, ..., m_{t+5}] → Shape: (batch_size, 6)
2. Loss-Berechnung (train.py)
Die Loss-Funktion iteriert über K+1=6 Timesteps mit jax.lax.scan:

```
Iteration 0 (Root, k=0):
├─ Latent State: repr_net(observations[t])
├─ Prediction: pred_net(latent) → (policy_logits, value)
├─ Policy Loss: CE(policy_logits, target_policies[0])  ← π_t
├─ Value Loss: MSE(value, target_values[0])            ← z_t
├─ Dynamics: latent ← dynamics_net(latent, actions[0]) ← a_t
└─ step_loss = policy_loss + value_loss

Iteration 1 (k=1):
├─ Prediction: pred_net(latent) → (policy_logits, value)
├─ Policy Loss: CE(policy_logits, target_policies[1])  ← π_{t+1}
├─ Value Loss: MSE(value, target_values[1])            ← z_{t+1}
├─ Dynamics: latent ← dynamics_net(latent, actions[1]) ← a_{t+1}
└─ step_loss = policy_loss + value_loss

... (Iteration 2, 3, 4 analog)

Iteration 5 (k=5):
├─ Prediction: pred_net(latent) → (policy_logits, value)
├─ Policy Loss: CE(policy_logits, target_policies[5])  ← π_{t+5}
├─ Value Loss: MSE(value, target_values[5])            ← z_{t+5}
├─ NO Dynamics (k=5 ist letzter Step)
└─ step_loss = policy_loss + value_loss

Total Loss = Σ(step_loss für k=0..5)  ← SUMME über alle 6 Steps 
```

3. Wichtige Details
Target Values Berechnung:

```
# Für jeden der K=6 Timesteps:
if (episode_length - seq_index) >= K:
    # Noch >= K Steps übrig → Bootstrap mit Value nach K Steps
    target_values[k] = γ^K * value[t+k+K]  # (mit Perspektiven-Flip!)
else:
    # < K Steps bis Ende → Bootstrap mit finalem Reward z
    target_values[k] = z  # (aus Perspektive von Spieler bei t+k)
```

Beispiel für Episode mit t=10, length=20:

k=0 (t=10): 10 Steps bis Ende → Bootstrap: value[t+6] (mit Flip)
k=1 (t=11): 9 Steps bis Ende → Bootstrap: value[t+7] (mit Flip)
k=2 (t=12): 8 Steps bis Ende → Bootstrap: value[t+8] (mit Flip)
k=3 (t=13): 7 Steps bis Ende → Bootstrap: value[t+9] (mit Flip)
k=4 (t=14): 6 Steps bis Ende → Bootstrap: value[t+10] (mit Flip)
k=5 (t=15): 5 Steps bis Ende → Bootstrap: z (finaler Reward)