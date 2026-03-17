import numpy as np
import pygame
import jax.numpy as jnp
import jax
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MADN.deterministic_madn import *
from MuZero_det_MADN.muzero_deterministic_madn import run_muzero_mcts, load_params_from_file
from utils.visualize import board_to_mat, matrix_to_string
from pygames.pygame_utils import *
ENABLE_PIN_IMAGES = True

# # Farben wie im Bild
# BACKGROUND_COLOR = (250, 235, 180)  # Cremefarbener Hintergrund
# FIELD_WHITE = (255, 255, 255)       # Weiße Felder
# FIELD_OUTLINE = (50, 50, 50)        # Umrandung
# BLAU = (0, 120, 220)
# ROT = (220, 50, 50)
# GELB = (220, 200, 0)
# GRUEN = (40, 160, 40)
# COLORS = {
#     (-1, 9): FIELD_WHITE,  # Leeres Feld (weiß)
#     (-1, 1): FIELD_WHITE,  # Leeres Feld (weiß)
#     (0, 7): (150, 150, 150),   # Deaktiviertes Feld (grau)
#     (1, 0): (0, 50, 150),     # Blau Zielfeld
#     (1, 1): BLAU,     # Blaue Figur
#     (1, 2): BLAU,     # Blaue Figur
#     (1, 3): BLAU,     # Blaue Figur
#     (1, 4): BLAU,     # Blaue Figur
#     (2, 0): (180, 50, 50),     # Rot Zielfeld
#     (2, 1): ROT,     # Rote Figur
#     (2, 2): ROT,     # Rote Figur
#     (2, 3): ROT,     # Rote Figur
#     (2, 4): ROT,     # Rote Figur
#     (3, 0): (200, 180, 0),     # Gelb Zielfeld
#     (3, 1): GELB,     # Gelbe Figur
#     (3, 2): GELB,     # Gelbe Figur
#     (3, 3): GELB,     # Gelbe Figur
#     (3, 4): GELB,     # Gelbe Figur
#     (4, 0): (40, 140, 40),     # Grün Zielfeld
#     (4, 1): GRUEN,     # Grüne Figur
#     (4, 2): GRUEN,     # Grüne Figur
#     (4, 3): GRUEN,     # Grüne Figur
#     (4, 4): GRUEN,     # Grüne Figur
# }

# Load images
dice_images = {}
for i in range(1, 7):
    dice_images[i] = pygame.image.load(f"images/dice/dice_{i}.png")

pin_images = {}
for color in ['blue', 'red', 'yellow', 'green']:
    pin_images[color] = pygame.image.load(f"images/pins/pin_{color}.png")

def create_dice_buttons(screen_width, screen_height):
    buttons = []
    button_width = 30
    button_height = 30
    spacing = 5
    center_x = screen_width // 2
    center_y = screen_height // 2
    y = center_y + 40 + 20
    total_width = 6 * button_width + 5 * spacing
    start_x = center_x - total_width // 2

    for i in range(1, 7):
        x = start_x + (i-1) * (button_width + spacing)
        button = Button(x, y, button_width, button_height, str(i), image=dice_images[i])
        buttons.append(button)
    return buttons

def create_player_functions(player_types, params_path=None):
    """
    Erstellt Lambda-Funktionen für jeden Spieler.
    
    Args:
        player_types: Liste mit 0 (Human), 1 (COM/MCTS), oder 2 (Random) für jeden Spieler
        params_path: Pfad zu den MuZero-Parametern
    
    Returns:
        Liste von Funktionen (obs, invalid_actions) -> action oder None für Human
    """
    players = []
    
    # Lade MuZero-Parameter einmalig (nur wenn COM-Spieler vorhanden)
    if 1 in player_types:
        if params_path is None:
            params_path = "models/params/Experiment_33_100.pkl"  # Anpassen!
        params = load_params_from_file(params_path)
        rng = jax.random.PRNGKey(np.random.randint(0, 100000))
    
    for player_type in player_types:
        if player_type == 0:
            # Human-Spieler (gibt None zurück, wird manuell gesteuert)
            players.append(lambda obs, invalid_actions: None)
        elif player_type == 1:
            # COM-Spieler mit MuZero MCTS
            def com_player(obs, invalid_actions, p=params):
                policy_output, _ = run_muzero_mcts(
                    params=p,
                    rng_key=jax.random.PRNGKey(np.random.randint(0, 100000)),
                    observations=obs[None, ...],
                    invalid_actions=invalid_actions[None,:],
                    num_simulations=100,  # Anpassen nach Bedarf
                    max_depth=50,
                    temperature=0.0
                )
                # print(policy_output)
                return policy_output.action[0]  # Rückgabe der Aktion als Integer
            players.append(com_player)
        else:  # player_type == 2
            # Random-Spieler (wählt zufällig gültige Aktion)
            def random_player(obs, invalid_actions):
                logits = jnp.where(invalid_actions, -1e9, 0)  # Ungültige Aktionen mit sehr niedrigem Logit bestrafen
                key = jax.random.PRNGKey(np.random.randint(0, 100000))
                action = jax.random.categorical(key, logits)
                return action
            players.append(random_player)
    
    return players

def create_board_surface(matrix, scale):
    """
    Erstellt ein Surface mit dem statischen Spielfeld (Hintergrund + Felder).
    Wird nur einmal aufgerufen.
    """
    h, w = matrix.shape
    surface = pygame.Surface((w * scale, h * scale))
    surface.fill(BACKGROUND_COLOR)
    
    rect = pygame.Rect(0, 0, scale - 6, scale - 6)
    
    for y in range(h):
        for x in range(w):
            v = int(matrix[y, x])
            rect.topleft = (x * scale + 3, y * scale + 3)
            
            color_key = int(v // 10)
            fig_key = int(v % 10)
            
            if color_key > 0 and fig_key > 0:
                fig_key = 0  # Nur Felder zeichnen, keine Figuren
            color = COLORS.get((color_key, fig_key), None)
            if color:
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, FIELD_OUTLINE, rect, 2)
    
    return surface

def draw_pins(screen, matrix, scale):
    """
    Zeichnet nur die Spieler-Pins (dynamisch, wird jeden Frame aufgerufen).
    """
    h, w = matrix.shape
    radius = scale // 2 - 6

    for y in range(h):
        for x in range(w):
            v = int(matrix[y, x])
            
            color_key = int(v // 10)
            fig_key = int(v % 10)
            color = COLORS.get((color_key, fig_key), None)
            
            if color and color_key > 0 and fig_key >= 1:
                if ENABLE_PIN_IMAGES:
                    pin_image = pin_images.get(['blue', 'red', 'yellow', 'green'][color_key - 1])
                    pin_image = pygame.transform.smoothscale(pin_image, (radius * 2, radius * 2))
                    pin_rect = pin_image.get_rect(center=(x * scale + scale // 2, y * scale + scale // 2))
                    screen.blit(pin_image, pin_rect)
                else:
                    center = (x * scale + scale // 2, y * scale + scale // 2)
                    pygame.draw.circle(screen, color, center, radius)
                    pygame.draw.circle(screen, FIELD_OUTLINE, center, radius, 2)
                    
                    # Glanzeffekt (kleiner weißer Kreis oben links)
                    highlight_pos = (center[0] - radius // 3, center[1] - radius // 3)
                    pygame.draw.circle(screen, (255, 255, 255), highlight_pos, radius // 4)

def draw_ui(screen, font, current_player, action_space):
    """
    Zeichnet UI-Elemente (Spieler, Würfel) in die Mitte.
    """
    # Hintergrund für UI
    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2
    
    # Box für UI
    box_rect = pygame.Rect(center_x - 80, center_y - 40, 160, 80)
    pygame.draw.rect(screen, BACKGROUND_COLOR, box_rect)
    pygame.draw.rect(screen, FIELD_OUTLINE, box_rect, 2)
    
    player_color = COLORS.get((current_player + 1, 1), (0, 0, 0))
    player_text = font.render(f"Spieler {current_player + 1}", True, player_color)
    dice_text = font.render(f"Actions: {action_space[current_player]}", True, (0, 0, 0))
    
    screen.blit(player_text, (center_x - player_text.get_width() // 2, center_y - 30))
    screen.blit(dice_text, (center_x - dice_text.get_width() // 2, center_y + 5))

    # perfekt information -> zeichne darunter die möglichen Aktionen für die anderen Spieler in deren entsprechender Farbe
    #gezeichnet unter den Buttons für die Würfelauswahl, damit es nicht zu überladen aussieht
    for i, actions in enumerate(action_space):
        if i != current_player:
            x = i +1 if i < current_player else i  # Aktueller Spieler überspringen
            other_color = COLORS.get((i + 1, 1), (0, 0, 0))
            other_text = font.render(f"Spieler {i + 1} Actions: {actions}", True, other_color)
            screen.blit(other_text, (center_x - other_text.get_width() // 2, center_y + 70 + x * 20))

def create_game_over_buttons(screen_width, screen_height):
    """
    Erstellt die drei Buttons für den Game-Over-Screen.
    """
    center_x = screen_width // 2
    center_y = screen_height // 2
    
    button_width = 200
    button_height = 40
    button_spacing = 10
    
    quit_button = Button(
        center_x - button_width // 2,
        center_y + 60,
        button_width, button_height,
        "Beenden",
        color=(220, 50, 50),
        text_color=(255, 255, 255)
    )
    
    restart_button = Button(
        center_x - button_width // 2,
        center_y + 60 + button_height + button_spacing,
        button_width, button_height,
        "Neu Start (gleiche Teams)",
        color=(50, 150, 220),
        text_color=(255, 255, 255)
    )
    
    menu_button = Button(
        center_x - button_width // 2,
        center_y + 60 + 2 * (button_height + button_spacing),
        button_width, button_height,
        "Zurück zum Menü",
        color=(220, 200, 0),
        text_color=(0, 0, 0)
    )
    
    return quit_button, restart_button, menu_button

def draw_game_over_screen(screen, font, winner_player, teams_enabled, quit_button, restart_button, menu_button):
    """
    Zeichnet den Game-Over-Bildschirm mit Gewinner und Buttons.
    """
    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2
    
    # Hintergrund-Box
    box_width = 400
    box_height = 300
    box_rect = pygame.Rect(center_x - box_width // 2, center_y - box_height // 2, box_width, box_height)
    pygame.draw.rect(screen, BACKGROUND_COLOR, box_rect)
    pygame.draw.rect(screen, FIELD_OUTLINE, box_rect, 4)
    
    # Gewinner-Text
    title_font = pygame.font.SysFont("Arial", 32, bold=True)
    winner_color = COLORS.get((winner_player + 1, 1), (0, 0, 0))
    
    title_text = title_font.render("Spiel beendet!", True, (0, 0, 0))
    if teams_enabled:
        winner_text = title_font.render(f"Gewinner: Team {((winner_player // 2) + 1)}", True, winner_color)
    else:
        winner_text = title_font.render(f"Gewinner: Spieler {winner_player + 1}", True, winner_color)
    
    screen.blit(title_text, (center_x - title_text.get_width() // 2, center_y - 100))
    screen.blit(winner_text, (center_x - winner_text.get_width() // 2, center_y - 50))
    
    # Buttons zeichnen
    quit_button.draw(screen)
    restart_button.draw(screen)
    menu_button.draw(screen)

RULES = {
    'enable_teams': True,
    'enable_initial_free_pin': True,
    'enable_circular_board': False,
    'enable_friendly_fire': False,
    'enable_start_blocking': False,
    'enable_jump_in_goal_area': True,
    'enable_start_on_1': True,
    'enable_bonus_turn_on_6': True,
    'must_traverse_start': False
}

def main():
    pygame.init()
    scale = 50  # Größe pro Feld
    
    # Spielkonfiguration
    layout = jnp.array([True, True, True, True])  # Alle 4 Spieler aktiv
    env = env_reset(
        0,
        num_players=4,
        distance=10, 
        layout=layout,
        seed=23,
        enable_initial_free_pin=RULES['enable_initial_free_pin'], 
        enable_teams=RULES['enable_teams'], 
        enable_circular_board=RULES['enable_circular_board'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        enable_start_on_1=RULES['enable_start_on_1'],
        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
        must_traverse_start=RULES['must_traverse_start']
        )
    _ = env_step(env, jnp.array([1,1])) # Step jit compilation
    matrix = board_to_mat(env, layout)
    h, w = matrix.shape
    
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("Mensch ärgere Dich nicht (Deterministisch)")
    clock = pygame.time.Clock()

    # === SPIELER-AUSWAHL MENÜ ===
    menu = PlayerSelectionMenu(w * scale, h * scale)
    player_types = None
    
    selecting_players = True
    while selecting_players:
        mouse_pos = pygame.mouse.get_pos()
        menu.update_hover(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                result = menu.handle_click(mouse_pos)
                if result is not None:
                    player_types = result
                    selecting_players = False
        
        screen.fill(BACKGROUND_COLOR)
        menu.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    
    # === SPIELER-FUNKTIONEN ERSTELLEN ===
    player_functions = create_player_functions(player_types, params_path=None)
    if jnp.any(jnp.array(player_types) == 1):
        print("MuZero-Parameter geladen. COM-Spieler werden mit MCTS agieren. Kompiliere run functions...")
        # get any idx with mcts function
        mcts_idx = jnp.where(jnp.array(player_types) == 1)[0][0]
        # Dummy-Observation und invalid_actions für Kompilierung
        dummy_obs = encode_board(env)
        dummy_invalid = valid_action(env).flatten()
        # Kompiliere die Funktion für diesen Spieler
        a =player_functions[mcts_idx](dummy_obs, dummy_invalid)
        print("Kompilierung abgeschlossen.")

    # Statisches Board-Surface erstellen (nur einmal!)
    board_surface = create_board_surface(matrix, scale)
    font = pygame.font.SysFont("Arial", 20)
    dice_buttons = create_dice_buttons(w * scale, h * scale)
    quit_button, restart_button, menu_button = create_game_over_buttons(w * scale, h * scale)

    game_phase = 'ROLL'
    running = True
    action = None
    winner_player = None
    pause_time = 500 if jnp.any(jnp.array(player_types) == 0) else 150  # Schneller, wenn nur COM-Spieler
    while running:

        mouse_pos = pygame.mouse.get_pos()
        current_player_idx = int(env.current_player)
        is_human = player_types[current_player_idx] == 0
        valid_actions = valid_action(env).flatten()
        invalid_actions = ~valid_actions
        
        # If player cannot move, skip turn but ndisplay info for 2 seconds
        if jnp.sum(valid_actions) == 0:
            print(f"Spieler {current_player_idx + 1} hat keine gültigen Züge und muss aussetzen!")
            env, _, done = no_step(env) # no-op Aktion
            matrix = board_to_mat(env, layout)
            game_phase = 'ROLL'
            continue
        # === COM-SPIELER AUTOMATISCHER ZUG ===
        if not is_human and game_phase == 'ROLL':
            obs = encode_board(env)
            
            com_action = player_functions[current_player_idx](obs, invalid_actions)
            
            if com_action is not None:
                pin_idx, dice_value = map_action(com_action)
                print(f"COM Spieler {current_player_idx + 1} wählt Pin {pin_idx} mit Würfel {dice_value}")
                
                env, _, done = env_step(env, jnp.array([pin_idx, dice_value]))
                matrix = board_to_mat(env, layout)
                
                if done:
                    winner_player = int(jnp.argwhere(get_winner(env, env.board))[0][0])
                    print(f"Spiel vorbei! Gewinner ist Spieler {winner_player + 1}")
                    game_phase = 'GAME_OVER'
                
                pygame.time.wait(pause_time)  # Kurze Pause für Visualisierung

        # === GAME OVER HANDLING ===
        if game_phase == 'GAME_OVER':
            quit_button.update_hover(mouse_pos)
            restart_button.update_hover(mouse_pos)
            menu_button.update_hover(mouse_pos)
        
        # === HUMAN-SPIELER STEUERUNG ===
        if is_human and game_phase == 'ROLL':
            for button in dice_buttons:
                button.update_hover(mouse_pos)
                
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # === GAME OVER BUTTON HANDLING ===
            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'GAME_OVER':
                if quit_button.is_clicked(mouse_pos):
                    running = False
                elif restart_button.is_clicked(mouse_pos):
                    # Spiel neu starten mit gleichen Spielertypen
                    env = env_reset(
                        0,
                        num_players=4,
                        distance=10, 
                        layout=layout,
                        seed=np.random.randint(0, 100000),
                        enable_initial_free_pin=RULES['enable_initial_free_pin'], 
                        enable_teams=RULES['enable_teams'], 
                        enable_circular_board=RULES['enable_circular_board'],
                        enable_friendly_fire=RULES['enable_friendly_fire'],
                        enable_start_blocking=RULES['enable_start_blocking'],
                        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
                        enable_start_on_1=RULES['enable_start_on_1'],
                        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
                        must_traverse_start=RULES['must_traverse_start']
                    )
                    matrix = board_to_mat(env, layout)
                    board_surface = create_board_surface(matrix, scale)
                    game_phase = 'ROLL'
                    winner_player = None
                elif menu_button.is_clicked(mouse_pos):
                    # Zurück zum Spieler-Auswahlmenü
                    selecting_players = True
                    while selecting_players:
                        mouse_pos = pygame.mouse.get_pos()
                        menu.update_hover(mouse_pos)
                        
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return
                            
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                result = menu.handle_click(mouse_pos)
                                if result is not None:
                                    player_types = result
                                    selecting_players = False
                        
                        screen.fill(BACKGROUND_COLOR)
                        menu.draw(screen)
                        pygame.display.flip()
                        clock.tick(60)
                    
                    # Spieler-Funktionen neu erstellen
                    player_functions = create_player_functions(player_types, params_path=None)
                    if jnp.any(jnp.array(player_types) == 1):
                        print("MuZero-Parameter geladen. COM-Spieler werden mit MCTS agieren. Kompiliere run functions...")
                        mcts_idx = jnp.where(jnp.array(player_types) == 1)[0][0]
                        dummy_obs = encode_board(env)
                        dummy_invalid = valid_action(env).flatten()
                        a = player_functions[mcts_idx](dummy_obs, dummy_invalid)
                        print("Kompilierung abgeschlossen.")
                    
                    # Spiel neu starten
                    env = env_reset(
                        0,
                        num_players=4,
                        distance=10, 
                        layout=layout,
                        seed=np.random.randint(0, 100000),
                        enable_initial_free_pin=RULES['enable_initial_free_pin'], 
                        enable_teams=RULES['enable_teams'], 
                        enable_circular_board=RULES['enable_circular_board'],
                        enable_friendly_fire=RULES['enable_friendly_fire'],
                        enable_start_blocking=RULES['enable_start_blocking'],
                        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
                        enable_start_on_1=RULES['enable_start_on_1'],
                        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
                        must_traverse_start=RULES['must_traverse_start']
                    )
                    matrix = board_to_mat(env, layout)
                    board_surface = create_board_surface(matrix, scale)
                    game_phase = 'ROLL'
                    winner_player = None
                    pause_time = 500 if jnp.any(jnp.array(player_types) == 0) else 50

            if is_human and event.type == pygame.KEYDOWN and game_phase == 'ROLL':
                if pygame.K_1 <= event.key <= pygame.K_6:
                    action = event.key - pygame.K_0
                print(f"Spieler {int(env.current_player) + 1} wählt eine {action}")
                game_phase = 'MOVE'

            # Button-Klicks für Würfel
            if is_human and event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'ROLL':
                for i, button in enumerate(dice_buttons):
                    if button.is_clicked(mouse_pos):
                        action = i + 1
                        print(f"Spieler {int(env.current_player) + 1} wählt eine {action}")
                        game_phase = 'MOVE'

            # 2. Pin auswählen per Mausklick
            if is_human and event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'MOVE':
                mouse_x, mouse_y = event.pos
                grid_x = mouse_x // scale
                grid_y = mouse_y // scale
                
                if 0 <= grid_y < h and 0 <= grid_x < w:
                    clicked_player_id = (int(matrix[grid_y, grid_x]) // 10 )- 1 
                    clicked_player_pin = (int(matrix[grid_y, grid_x]) % 10 )- 1
                    player_id = env.current_player
                    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
                    if clicked_player_id == current_player:
                        print(f"Spieler {int(env.current_player) + 1} zieht mit Zug {action} und Pin {clicked_player_pin}")
                        
                        env, _, done = env_step(env, jnp.array([clicked_player_pin, action]))
                        matrix = board_to_mat(env, layout)
                        game_phase = 'ROLL'
                        
                        if done:
                            winner_player = int(jnp.argwhere(get_winner(env, env.board))[0][0])
                            print(f"Spiel vorbei! Gewinner ist Spieler {winner_player + 1}")
                            game_phase = 'GAME_OVER'
                    else:
                        print("Das ist nicht dein Pin!")

        # === ZEICHNEN ===
        screen.blit(board_surface, (0, 0))
        draw_pins(screen, matrix, scale)
        
        if game_phase == 'GAME_OVER':
            # Game-Over-Bildschirm zeichnen
            draw_game_over_screen(screen, font, winner_player, env.rules["enable_teams"], quit_button, restart_button, menu_button)
        else:
            # Normales UI zeichnen
            draw_ui(screen, font, int(env.current_player), env.action_set)
            
            # Würfel-Buttons zeichnen (nur in ROLL-Phase)
            if is_human and game_phase == 'ROLL':
                for button in dice_buttons:
                    button.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
