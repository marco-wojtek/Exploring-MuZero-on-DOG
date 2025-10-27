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
        - in X:
            - base actions -> move pin 1-13 steps
            - board_size -> swap figure to pos y
            - start with 1;11 , 13 or ?
            - go 4 backwards

    - Encoder and Decoder for Action <-> Index necessary

# Imperfect Information
- Train perfect information agent and build upon that to further train for hidden information
- learning with dropout
- res blocks, experience replay buffer, L2 regularization