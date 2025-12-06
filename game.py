import pygame
import jax.numpy as jnp
import jax
from MADN.classic_madn import *
from MADN.visualize_madn import board_to_mat

def draw_board(screen, matrix, scale=60):
    color_map = {
        -1: (240,240,240), 8: (200,200,200), 9: (255,255,255),
        0: (50,120,255), 1: (255,60,60), 2: (255,200,40), 3: (60,200,60),
        10: (0,80,150), 11: (140,0,0), 12: (140,100,0), 13: (0,100,0)
    }
    h, w = matrix.shape
    for y in range(h):
        for x in range(w):
            v = int(matrix[y,x])
            c = color_map.get(v, (0,0,0))
            pygame.draw.rect(screen, c, (x*scale, y*scale, scale, scale))
            # Optional: Pin-Symbole/Text
            if v in [0,1,2,3]:
                font = pygame.font.SysFont(None, scale//2)
                text = font.render(str(v+1), True, (0,0,0)) # Zeige Spieler 1-4 an
                screen.blit(text, (x*scale+scale//4, y*scale+scale//4))

def draw_ui(screen, font, current_player, dice_roll):
    player_text = font.render(f"Player: {current_player}", True, (0, 0, 0))
    dice_text = font.render(f"Dice: {dice_roll if dice_roll > 0 else '-'}", True, (0, 0, 0))
    screen.blit(player_text, (10, 10))
    screen.blit(dice_text, (10, 40))

def main():
    pygame.init()
    scale = 60
    layout = jnp.array([True, False, True, False])
    env = env_reset(0, num_players=2, distance=10, enable_initial_free_pin=True, layout=layout)
    matrix = board_to_mat(env, layout)
    h, w = matrix.shape
    screen = pygame.display.set_mode((w*scale, h*scale))
    running = True
    
    # UI-Elemente
    font = pygame.font.SysFont(None, 36)
    game_phase = 'ROLL' # Phasen: 'ROLL', 'MOVE'
    rng_key = jax.random.PRNGKey(42)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- Spiel-Logik ---
            # 1. Würfeln per Leertaste
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and game_phase == 'ROLL':
                rng_key, subkey = jax.random.split(rng_key)
                env = throw_die(env, subkey)
                print(f"Player {env.current_player} rolled a {env.die}")
                game_phase = 'MOVE'
                # Hier könntest du prüfen, ob überhaupt ein Zug möglich ist.
                # Wenn nicht, Phase direkt wieder auf 'ROLL' setzen und Spieler wechseln.

            # 2. Pin auswählen per Mausklick
            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'MOVE':
                mouse_x, mouse_y = event.pos
                grid_x = mouse_x // scale
                grid_y = mouse_y // scale
                
                clicked_player_id = int(matrix[grid_y, grid_x])
                
                # Prüfen, ob der angeklickte Pin dem aktuellen Spieler gehört
                if clicked_player_id == env.current_player:
                    # Finde den Index des Pins
                    pin_index = -1
                    # Diese Logik ist ein Workaround, da wir die Pin-Position aus der Matrix ableiten
                    # Eine bessere Methode wäre, eine Map von Matrix-Koordinaten zu Pin-Indizes zu haben.
                    # Hier nehmen wir an, wir finden den ersten Pin des Spielers an dieser Position.
                    
                    # Finde den Pin, der bewegt werden soll.
                    # Dies ist eine Vereinfachung. Eine robuste Lösung braucht eine Map von
                    # (grid_x, grid_y) zu einer eindeutigen Pin-ID.
                    # Für den Moment versuchen wir, den Zug auszuführen und schauen, ob er klappt.
                    
                    # Hier müsste die Logik hin, um den richtigen Pin-Index (0-3) zu finden.
                    # Da dies komplex ist, simulieren wir einfach einen Zug mit einem beliebigen Pin
                    # des Spielers, um die Funktionalität zu zeigen.
                    
                    # Vereinfachung: Wir nehmen an, die Aktion ist der Würfelwurf.
                    # Die `env_step`-Logik muss die Pin-Auswahl intern handhaben.
                    # Für eine echte interaktive Steuerung müsstest du `env_step` anpassen,
                    # um einen spezifischen Pin-Index zu akzeptieren.
                    
                    print(f"Attempting to move for player {env.current_player} with dice {env.die}")
                    
                    # Führe den Zug aus. `env_step` muss die Logik für die Pin-Auswahl enthalten.
                    # Da die aktuelle `env_step` nur eine Aktion (den Würfelwurf) nimmt,
                    # übergeben wir diesen.
                    env, _, done = env_step(env, jnp.array(0))
                    
                    # Spielzustand aktualisieren
                    matrix = board_to_mat(env, layout)
                    game_phase = 'ROLL'
                    
                    if done:
                        print(f"Game Over! Winner is Player {get_winner(env)}")
                        running = False
                else:
                    print("Not your pin!")

        # --- Zeichnen ---
        screen.fill((255,255,255))
        draw_board(screen, matrix, scale)
        draw_ui(screen, font, int(env.current_player), env.die)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()