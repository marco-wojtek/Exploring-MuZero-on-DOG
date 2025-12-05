import pygame
import jax.numpy as jnp
from MADN.classic_madn import *
from MADN.visualize_madn import board_to_matrix

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
                text = font.render(str(v), True, (0,0,0))
                screen.blit(text, (x*scale+scale//4, y*scale+scale//4))

def main():
    pygame.init()
    scale = 60
    env = env_reset(0, num_players=4, distance=10, enable_initial_free_pin=True)
    matrix = board_to_matrix(env)
    h, w = matrix.shape
    screen = pygame.display.set_mode((w*scale, h*scale))
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Hier kannst du Maus/Tastatur abfragen und Aktionen ausführen
            # z.B. Pin auswählen, würfeln, Zug machen
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                grid_x = mouse_x // scale
                grid_y = mouse_y // scale
                if matrix[grid_y, grid_x] in [0, 1, 2, 3]:
                    print(f"Pin {matrix[grid_y, grid_x]} auf Feld ({grid_y}, {grid_x}) angeklickt")
                    # Hier kannst du die Pin-Auswahl weiterverarbeiten

        screen.fill((255,255,255))
        draw_board(screen, matrix, scale)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()