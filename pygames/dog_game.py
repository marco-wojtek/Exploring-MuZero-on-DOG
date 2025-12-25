import pygame
import jax.numpy as jnp
import jax
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *
from utils.visualize import board_to_mat, matrix_to_string

# Farben wie im Bild
BACKGROUND_COLOR = (250, 235, 180)  # Cremefarbener Hintergrund
FIELD_WHITE = (255, 255, 255)       # Weiße Felder
FIELD_OUTLINE = (50, 50, 50)        # Umrandung
BLAU = (0, 120, 220)
ROT = (220, 50, 50)
GELB = (220, 200, 0)
GRUEN = (40, 160, 40)
COLORS = {
    (-1, 9): FIELD_WHITE,  # Leeres Feld (weiß)
    (-1, 1): FIELD_WHITE,  # Leeres Feld (weiß)
    (0, 7): (150, 150, 150),   # Deaktiviertes Feld (grau)
    (1, 0): (0, 50, 150),     # Blau Zielfeld
    (1, 1): BLAU,     # Blaue Figur
    (1, 2): BLAU,     # Blaue Figur
    (1, 3): BLAU,     # Blaue Figur
    (1, 4): BLAU,     # Blaue Figur
    (2, 0): (180, 50, 50),     # Rot Zielfeld
    (2, 1): ROT,     # Rote Figur
    (2, 2): ROT,     # Rote Figur
    (2, 3): ROT,     # Rote Figur
    (2, 4): ROT,     # Rote Figur
    (3, 0): (200, 180, 0),     # Gelb Zielfeld
    (3, 1): GELB,     # Gelbe Figur
    (3, 2): GELB,     # Gelbe Figur
    (3, 3): GELB,     # Gelbe Figur
    (3, 4): GELB,     # Gelbe Figur
    (4, 0): (40, 140, 40),     # Grün Zielfeld
    (4, 1): GRUEN,     # Grüne Figur
    (4, 2): GRUEN,     # Grüne Figur
    (4, 3): GRUEN,     # Grüne Figur
    (4, 4): GRUEN,     # Grüne Figur
}

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
                center = (x * scale + scale // 2, y * scale + scale // 2)
                pygame.draw.circle(screen, color, center, radius)
                pygame.draw.circle(screen, FIELD_OUTLINE, center, radius, 2)
                
                # Glanzeffekt (kleiner weißer Kreis oben links)
                highlight_pos = (center[0] - radius // 3, center[1] - radius // 3)
                pygame.draw.circle(screen, (255, 255, 255), highlight_pos, radius // 4)

def draw_ui(screen, font, current_player, dice_roll):
    """
    Zeichnet UI-Elemente (Spieler, Würfel) in die Mitte.
    """
    # Hintergrund für UI
    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2
    
    # Box für UI
    box_rect = pygame.Rect(center_x - 60, center_y - 40, 120, 80)
    pygame.draw.rect(screen, BACKGROUND_COLOR, box_rect)
    pygame.draw.rect(screen, FIELD_OUTLINE, box_rect, 2)
    
    player_color = COLORS.get((current_player + 1, 1), (0, 0, 0))
    player_text = font.render(f"Spieler {current_player + 1}", True, player_color)
    dice_text = font.render(f"Würfel: {dice_roll if dice_roll > 0 else '-'}", True, (0, 0, 0))
    
    screen.blit(player_text, (center_x - player_text.get_width() // 2, center_y - 30))
    screen.blit(dice_text, (center_x - dice_text.get_width() // 2, center_y + 5))

def main():
    pygame.init()
    scale = 30  # Größe pro Feld
    
    # Spielkonfiguration
    layout = jnp.array([True, True, True, True])  # Alle 4 Spieler aktiv
    env = env_reset(0, num_players=4, distance=16, enable_initial_free_pin=True, layout=layout)
    
    matrix = board_to_mat(env, layout)
    print(matrix)
    h, w = matrix.shape
    
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("Mensch ärgere Dich nicht")
    clock = pygame.time.Clock()
    # Statisches Board-Surface erstellen (nur einmal!)
    board_surface = create_board_surface(matrix, scale)
    # UI-Elemente
    font = pygame.font.SysFont("Arial", 20)
    game_phase = 'ROLL'
    rng_key = jax.random.PRNGKey(42)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 1. Würfeln per Leertaste
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and game_phase == 'ROLL':
                rng_key, subkey = jax.random.split(rng_key)
                print(f"Spieler {int(env.current_player) + 1} würfelt eine {env.die}")
                game_phase = 'MOVE'

            # 2. Pin auswählen per Mausklick
            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'MOVE':
                mouse_x, mouse_y = event.pos
                grid_x = mouse_x // scale
                grid_y = mouse_y // scale
                
                if 0 <= grid_y < h and 0 <= grid_x < w:
                    clicked_player_id = (int(matrix[grid_y, grid_x]) // 10 )- 1 
                    clicked_player_pin = (int(matrix[grid_y, grid_x]) % 10 )- 1
                    if clicked_player_id == env.current_player:
                        print(f"Spieler {int(env.current_player) + 1} zieht mit Würfel {env.die}")
                        
                        env, _, done = env_step(env, jnp.array(clicked_player_pin))
                        matrix = board_to_mat(env, layout)
                        game_phase = 'ROLL'
                        
                        if done:
                            print(f"Spiel vorbei! Gewinner ist Spieler {get_winner(env, env.board)}")
                            running = False
                    else:
                        print("Das ist nicht dein Pin!")

        # --- Zeichnen ---
        # 1. Statisches Board (nur kopieren, nicht neu zeichnen)
        screen.blit(board_surface, (0, 0))
        
        # 2. Dynamische Pins
        draw_pins(screen, matrix, scale)
        
        # 3. UI
        draw_ui(screen, font, int(env.current_player), env.die)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
