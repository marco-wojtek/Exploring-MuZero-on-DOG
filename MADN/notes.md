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
- enable_start_block
- enable_teams
- enable_jumping_in_goal
- enable_initial_free_pin XXXXXXXXXXXXXXX
- enable_extra_move_on_6 (only dMADN and MADN) XXXXXXXXXXXXXXXXXXXX
- enable_start_on_1_and_6 (dMADN, MADN) XXXXXXXXXXXXXXXXX
- enable_rethrow (up to 3 throws if all pins in start, MADN only) XXXXXXXXXXXXXXXXX
- enable_forced_start (only MADN)
- enable_forced_attack
- enable_friendly_fire (on team member pins)
- enable_circular_board XXXXXXXXXXXXXXXXX