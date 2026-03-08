import pygame
import jax.numpy as jnp
import jax
import numpy as np
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MADN.classic_madn import *
from MuZero.muzero_classic_madn import run_stochastic_muzero_mcts, load_params_from_file
from utils.visualize import board_to_mat, matrix_to_string
from pygames.pygame_utils import *

ENABLE_PIN_IMAGES = True

# Farben werden aus pygame_utils importiert

# Load images
dice_images = {}
for i in range(1, 7):
    dice_images[i] = pygame.image.load(f"images/dice/dice_{i}.png")

pin_images = {}
for color in ['blue', 'red', 'yellow', 'green']:
    pin_images[color] = pygame.image.load(f"images/pins/pin_{color}.png")
DICE_ANIMATION_ORDER = [1, 4, 3, 6, 5, 2]

def create_player_functions(player_types, params_path=None):
    """
    Erstellt Lambda-Funktionen für jeden Spieler.
    
    Args:
        player_types: Liste mit 0 (Human), 1 (COM/MCTS), oder 2 (Random) für jeden Spieler
        params_path: Pfad zu den MuZero-Parametern
    
    Returns:
        Liste von Funktionen (env, obs, invalid_actions) -> action oder None für Human
    """
    players = []
    
    # Lade MuZero-Parameter einmalig (nur wenn COM-Spieler vorhanden)
    if 1 in player_types:
        if params_path is None:
            params_path = "models/params/gumbelmuzero_classic_params_lr0.001_g1500_it150_seed42.pkl"
        params = load_params_from_file(params_path)
    
    for player_type in player_types:
        if player_type == 0:
            # Human-Spieler (gibt None zurück, wird manuell gesteuert)
            players.append(lambda env, obs, invalid_actions: None)
        elif player_type == 1:
            # COM-Spieler mit MuZero MCTS
            def com_player(env, obs, invalid_actions, p=params):
                policy_output, _ = run_stochastic_muzero_mcts(
                    params=p,
                    rng_key=jax.random.PRNGKey(np.random.randint(0, 100000)),
                    observations=obs[None, ...],
                    invalid_actions=invalid_actions[None,:],
                    num_simulations=100,
                    max_depth=50,
                    temperature=1.0
                )
                return int(policy_output.action[0])
            players.append(com_player)
        else:  # player_type == 2
            # Random-Spieler (wählt zufällig gültigen Pin)
            def random_player(env, obs, invalid_actions):
                valid_pins = ~invalid_actions
                valid_indices = jnp.where(valid_pins)[0]
                if len(valid_indices) > 0:
                    key = jax.random.PRNGKey(np.random.randint(0, 100000))
                    idx = jax.random.choice(key, valid_indices)
                    return int(idx)
                return 0
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

def draw_ui(screen, font, current_player, is_human, game_phase, dice_roll, show_dice_animation=False, anim_start_time=None):
    """
    Zeichnet UI-Elemente (Spieler, Würfel) in die Mitte.
    """
    # Hintergrund für UI
    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2
    
    # Box für UI
    box_rect = pygame.Rect(center_x - 75, center_y - 40, 150, 100)
    pygame.draw.rect(screen, BACKGROUND_COLOR, box_rect)
    pygame.draw.rect(screen, FIELD_OUTLINE, box_rect, 2)
    
    player_color = COLORS.get((current_player + 1, 1), (0, 0, 0))
    player_text = font.render(f"Spieler {current_player + 1}", True, player_color)
    
    if show_dice_animation and anim_start_time is not None:
        # 10 fps Animation
        elapsed = pygame.time.get_ticks() - anim_start_time
        dice_img = dice_images[DICE_ANIMATION_ORDER[(elapsed // 50) % len(DICE_ANIMATION_ORDER)]]
        img_rect = dice_img.get_rect(center=(center_x, center_y + 25))
        screen.blit(dice_img, img_rect)
    elif dice_roll in dice_images:
        dice_img = dice_images[dice_roll]
        img_rect = dice_img.get_rect(center=(center_x, center_y + 25))
        screen.blit(dice_img, img_rect)
    if is_human and game_phase == "ROLL":
        no_dice_text = font.render("Press Space to roll", True, (0, 0, 0))
        screen.blit(no_dice_text, (center_x - no_dice_text.get_width() // 2, center_y + 60))
    
    screen.blit(player_text, (center_x - player_text.get_width() // 2, center_y - 30))

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
    'enable_teams': False,
    'enable_initial_free_pin': True,
    'enable_circular_board': False,
    'enable_start_blocking': False,
    'enable_jump_in_goal_area': True,
    'enable_friendly_fire': False,
    'enable_start_on_1': True,
    'enable_bonus_turn_on_6': True,
    'enable_dice_rethrow': True,
    'must_traverse_start': False
}

def main():
    pygame.init()
    scale = 50  # Größe pro Feld
    
    # Spielkonfiguration
    layout = jnp.array([True, True, True, True])  # Alle 4 Spieler aktiv
    env = env_reset(
        0, 
        seed=23, 
        num_players=4, 
        distance=10, 
        enable_initial_free_pin=RULES['enable_initial_free_pin'],
        enable_teams=RULES['enable_teams'],
        enable_dice_rethrow=RULES['enable_dice_rethrow'],
        enable_circular_board=RULES['enable_circular_board'],
        enable_start_blocking=RULES['enable_start_blocking'],
        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
        enable_friendly_fire=RULES['enable_friendly_fire'],
        enable_start_on_1=RULES['enable_start_on_1'],
        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
        must_traverse_start=RULES['must_traverse_start'],
        layout=layout
    )
    _ = env_step(env, jnp.array(2))  # Dummy-Schritt für JIT-Kompilierung
    _ = throw_die(env)  # Initialen Würfelwurf
    matrix = board_to_mat(env, layout)
    h, w = matrix.shape
    
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("Mensch ärgere Dich nicht (Klassisch)")
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
    player_functions = create_player_functions(player_types, params_path="models/params/stochastic_muzero_madn_lr0.01_g1500_it75_seed10001.pkl")
    if jnp.any(jnp.array(player_types) == 1):
        print("MuZero-Parameter geladen. COM-Spieler werden mit MCTS agieren. Kompiliere run functions...")
        mcts_idx = jnp.where(jnp.array(player_types) == 1)[0][0]
        dummy_obs = encode_board(env)
        dummy_invalid = valid_action(env)
        a = player_functions[mcts_idx](env, dummy_obs, dummy_invalid)
        print("Kompilierung abgeschlossen.")
    # Statisches Board-Surface erstellen (nur einmal!)
    board_surface = create_board_surface(matrix, scale)
    font = pygame.font.SysFont("Arial", 20)
    quit_button, restart_button, menu_button = create_game_over_buttons(w * scale, h * scale)
    
    game_phase = 'ROLL'
    running = True
    dice_anim_start = None
    winner_player = None
    pending_roll = False
    last_dice_roll = 0  # Speichert die letzte gewürfelte Zahl
    pause_time = 500 if jnp.any(jnp.array(player_types) == 0) else 150

    while running:
        mouse_pos = pygame.mouse.get_pos()
        current_player_idx = int(env.current_player)
        is_human = player_types[current_player_idx] == 0
        
        # === COM-SPIELER AUTOMATISCHER ZUG ===
        if not is_human and game_phase == 'ROLL':
            # COM würfelt automatisch
            dice_anim_start = pygame.time.get_ticks()
            game_phase = 'ANIMATE_DICE'
            pending_roll = True
        
        if not is_human and game_phase == 'MOVE':
            # COM wählt Pin mit MuZero MCTS
            valid_actions_mask = valid_action(env)
            
            # Prüfe ob Spieler ziehen kann
            if jnp.sum(valid_actions_mask) == 0:
                print(f"Spieler {current_player_idx + 1} hat keine gültigen Züge und muss aussetzen!")
                # Würfel neu werfen wenn enabled
                last_dice_roll = int(env.die)
                env, _, _ = no_step(env)  # Kein Zug, aber Zustand aktualisieren (z.B. für Rethrow)
                env = throw_die(env)
                game_phase = 'ROLL'
                pygame.time.wait(pause_time)
            else:
                obs = encode_board(env)
                com_action = player_functions[current_player_idx](env, obs, ~valid_actions_mask)
                
                if com_action is not None:
                    print(f"COM Spieler {current_player_idx + 1} zieht mit Würfel {env.die} und Pin {com_action}")
                    
                    last_dice_roll = int(env.die)  # Speichere Würfelzahl vor dem Zug
                    env, _, done = env_step(env, jnp.array(com_action))
                    matrix = board_to_mat(env, layout)
                    
                    if done:
                        winner_idx = get_winner(env, env.board)
                        winner_player = int(jnp.argmax(winner_idx))
                        print(f"Spiel vorbei! Gewinner ist Spieler {winner_player + 1}")
                        game_phase = 'GAME_OVER'
                    else:
                        env = throw_die(env)
                        game_phase = 'ROLL'
                    
                    pygame.time.wait(pause_time)
        
        # === GAME OVER HANDLING ===
        if game_phase == 'GAME_OVER':
            quit_button.update_hover(mouse_pos)
            restart_button.update_hover(mouse_pos)
            menu_button.update_hover(mouse_pos)
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
                        seed=np.random.randint(0, 100000),
                        num_players=4,
                        distance=10,
                        enable_initial_free_pin=RULES['enable_initial_free_pin'],
                        enable_teams=RULES['enable_teams'],
                        enable_dice_rethrow=RULES['enable_dice_rethrow'],
                        enable_circular_board=RULES['enable_circular_board'],
                        enable_start_blocking=RULES['enable_start_blocking'],
                        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
                        enable_friendly_fire=RULES['enable_friendly_fire'],
                        enable_start_on_1=RULES['enable_start_on_1'],
                        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
                        must_traverse_start=RULES['must_traverse_start'],
                        layout=layout
                    )
                    env = throw_die(env)
                    matrix = board_to_mat(env, layout)
                    board_surface = create_board_surface(matrix, scale)
                    game_phase = 'ROLL'
                    winner_player = None
                    dice_anim_start = None
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
                    player_functions = create_player_functions(player_types, params_path="models/params/gumbelmuzero_classic_params_lr0.001_g1500_it150_seed42.pkl")
                    if jnp.any(jnp.array(player_types) == 1):
                        print("MuZero-Parameter geladen. COM-Spieler werden mit MCTS agieren. Kompiliere run functions...")
                        mcts_idx = jnp.where(jnp.array(player_types) == 1)[0][0]
                        dummy_obs = encode_board(env)
                        dummy_invalid = valid_action(env)
                        a = player_functions[mcts_idx](env, dummy_obs, dummy_invalid)
                        print("Kompilierung abgeschlossen.")
                    
                    # Spiel neu starten
                    env = env_reset(
                        0,
                        seed=np.random.randint(0, 100000),
                        num_players=4,
                        distance=10,
                        enable_initial_free_pin=RULES['enable_initial_free_pin'],
                        enable_teams=RULES['enable_teams'],
                        enable_dice_rethrow=RULES['enable_dice_rethrow'],
                        enable_circular_board=RULES['enable_circular_board'],
                        enable_start_blocking=RULES['enable_start_blocking'],
                        enable_jump_in_goal_area=RULES['enable_jump_in_goal_area'],
                        enable_friendly_fire=RULES['enable_friendly_fire'],
                        enable_start_on_1=RULES['enable_start_on_1'],
                        enable_bonus_turn_on_6=RULES['enable_bonus_turn_on_6'],
                        must_traverse_start=RULES['must_traverse_start'],
                        layout=layout
                    )
                    env = throw_die(env)
                    matrix = board_to_mat(env, layout)
                    board_surface = create_board_surface(matrix, scale)
                    game_phase = 'ROLL'
                    winner_player = None
                    dice_anim_start = None
                    last_dice_roll = 0
                    pause_time = 500 if jnp.any(jnp.array(player_types) == 0) else 150

            # 1. Würfeln per Leertaste (nur für Human)
            if is_human and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and game_phase == 'ROLL':
                dice_anim_start = pygame.time.get_ticks()
                game_phase = 'ANIMATE_DICE'
                pending_roll = True

            # 2. Pin auswählen per Mausklick (nur für Human)
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
                        print(f"Spieler {int(env.current_player) + 1} zieht mit Würfel {env.die} und Pin {clicked_player_pin}")
                        
                        last_dice_roll = int(env.die)  # Speichere Würfelzahl vor dem Zug
                        env, _, done = env_step(env, jnp.array(clicked_player_pin))
                        matrix = board_to_mat(env, layout)
                        
                        if done:
                            winner_idx = get_winner(env, env.board)
                            winner_player = int(jnp.argmax(winner_idx))
                            print(f"Spiel vorbei! Gewinner ist Spieler {winner_player + 1}")
                            game_phase = 'GAME_OVER'
                        else:
                            env = throw_die(env)
                            game_phase = 'ROLL'
                    else:
                        print("Das ist nicht dein Pin!")

        if game_phase == 'ANIMATE_DICE':
            if pygame.time.get_ticks() - dice_anim_start >= 300:
                if pending_roll:
                    env = throw_die(env)
                    print(f"Spieler {int(env.current_player) + 1} würfelt eine {env.die}")
                    pending_roll = False
                game_phase = 'MOVE'
        
        # === ZEICHNEN ===
        screen.blit(board_surface, (0, 0))
        draw_pins(screen, matrix, scale)
        
        if game_phase == 'GAME_OVER':
            # Game-Over-Bildschirm zeichnen
            draw_game_over_screen(screen, font, winner_player, RULES['enable_teams'], quit_button, restart_button, menu_button)
        else:
            # Normales UI zeichnen
            show_anim = (game_phase == 'ANIMATE_DICE')
            # In ROLL-Phase zeige letzte Würfelzahl, sonst aktuelle
            display_dice = last_dice_roll if game_phase == 'ROLL' else int(env.die)
            draw_ui(screen, font, int(env.current_player), is_human, game_phase, display_dice, show_dice_animation=show_anim, anim_start_time=dice_anim_start)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
