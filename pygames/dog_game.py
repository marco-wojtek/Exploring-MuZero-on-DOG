import pygame
import jax.numpy as jnp
import jax
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *
from utils.visualize import board_to_mat, matrix_to_string
# import warnings
# warnings.filterwarnings("error", category=FutureWarning)

ENABLE_PIN_IMAGES = True

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

pin_images = {}
for color in ['blue', 'red', 'yellow', 'green']:
    pin_images[color] = pygame.image.load(f"images/pins/pin_{color}.png")

card_images = {}
for i in range(0, 14):
    card_images[i] = pygame.image.load(f"images/cards/card_{i}.png")

card_4_images = {}
card_4_images[0] = pygame.image.load(f"images/cards/card_pos4.png")
card_4_images[1] = pygame.image.load(f"images/cards/card_neg4.png")

card_1_11_images = {}
card_1_11_images[0] = pygame.image.load(f"images/cards/card_11_1.png")
card_1_11_images[1] = pygame.image.load(f"images/cards/card_11_11.png")

class Button:
    def __init__(self, x, y, width, height, text, color=(200, 200, 200), text_color=(0, 0, 0), image=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.SysFont("Arial", 16)
        self.hovered = False
        self.image = image  # Bild für den Button
        
    def draw(self, screen):
        color = (180, 180, 180) if self.hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        
        if self.image:
            # Bild mittig im Button zeichnen, ggf. skalieren
            img = pygame.transform.smoothscale(self.image, (self.rect.width - 6, self.rect.height - 6))
            img_rect = img.get_rect(center=self.rect.center)
            screen.blit(img, img_rect)
        else:
            text_surface = self.font.render(self.text, True, self.text_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)
    
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

class Hot7Selector:
    def __init__(self, pins, max_steps=7):
        self.pins = [int(p) for p in pins]
        self.selected_steps = {pin: 0 for pin in self.pins}
        self.max_steps = max_steps
        self.plus_buttons = {}
        self.minus_buttons = {}
        self.confirm_button = None
        # Kleinere Fonts für das kompakte Fenster
        self.font = pygame.font.SysFont("Arial", 14)
        self.title_font = pygame.font.SysFont("Arial", 16, bold=True)

    def draw(self, screen, center_x, center_y):
        # Layout-Konstanten
        width = 200
        row_height = 30
        header_height = 35
        footer_height = 45
        height = header_height + len(self.pins) * row_height + footer_height
        
        # Popup Rechteck berechnen
        popup_rect = pygame.Rect(0, 0, width, height)
        popup_rect.center = (center_x, center_y)
        
        # Schatten (einfach versetztes Rechteck)
        shadow_rect = popup_rect.copy()
        shadow_rect.move_ip(3, 3)
        pygame.draw.rect(screen, (100, 100, 100), shadow_rect)
        
        # Hintergrund
        pygame.draw.rect(screen, (250, 250, 245), popup_rect)
        pygame.draw.rect(screen, (0, 0, 0), popup_rect, 2)
        
        current_sum = sum(self.selected_steps.values())
        remaining = self.max_steps - current_sum
        
        # Header Text
        title = f"Verteile 7 (Rest: {remaining})"
        text_surf = self.title_font.render(title, True, (0,0,0))
        text_rect = text_surf.get_rect(center=(popup_rect.centerx, popup_rect.top + 18))
        screen.blit(text_surf, text_rect)
        
        # Zeilen für Pins
        start_y = popup_rect.top + header_height
        btn_size = 22
        
        for i, pin in enumerate(self.pins):
            y = start_y + i * row_height
            # Label
            label = self.font.render(f"Pin {pin+1}: {self.selected_steps[pin]}", True, (0,0,0))
            screen.blit(label, (popup_rect.left + 15, y + 4))
            
            # Buttons (Rechtsbündig)
            # Minus
            minus_rect = pygame.Rect(popup_rect.right - 65, y, btn_size, btn_size)
            minus_active = self.selected_steps[pin] > 0
            self._draw_btn(screen, minus_rect, "-", minus_active)
            self.minus_buttons[pin] = minus_rect
            
            # Plus
            plus_rect = pygame.Rect(popup_rect.right - 35, y, btn_size, btn_size)
            plus_active = remaining > 0
            self._draw_btn(screen, plus_rect, "+", plus_active)
            self.plus_buttons[pin] = plus_rect

        # Confirm Button (nur wenn Summe == 7)
        if current_sum == self.max_steps:
            confirm_rect = pygame.Rect(0, 0, 80, 28)
            confirm_rect.centerx = popup_rect.centerx
            confirm_rect.bottom = popup_rect.bottom - 10
            
            pygame.draw.rect(screen, (100, 200, 100), confirm_rect)
            pygame.draw.rect(screen, (0, 0, 0), confirm_rect, 1)
            
            conf_surf = self.font.render("OK", True, (0,0,0))
            conf_text_rect = conf_surf.get_rect(center=confirm_rect.center)
            screen.blit(conf_surf, conf_text_rect)
            
            self.confirm_button = confirm_rect
        else:
            self.confirm_button = None

    def _draw_btn(self, screen, rect, text, active):
        color = (220, 220, 220) if active else (240, 240, 240)
        text_color = (0, 0, 0) if active else (180, 180, 180)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)
        
        surf = self.font.render(text, True, text_color)
        text_rect = surf.get_rect(center=rect.center)
        screen.blit(surf, text_rect)

    def handle_click(self, pos):
        """Returns True if confirmed, False otherwise"""
        current_sum = sum(self.selected_steps.values())
        
        # Check Confirm
        if self.confirm_button and self.confirm_button.collidepoint(pos):
            return True

        for pin in self.pins:
            if self.plus_buttons.get(pin) and self.plus_buttons[pin].collidepoint(pos):
                if current_sum < self.max_steps:
                    self.selected_steps[pin] += 1
                    return False
            
            if self.minus_buttons.get(pin) and self.minus_buttons[pin].collidepoint(pos):
                if self.selected_steps[pin] > 0:
                    self.selected_steps[pin] -= 1
                    return False
        return False

    def reset(self):
        for pin in self.pins:
            self.selected_steps[pin] = 0

def create_all_card_buttons(screen_width, screen_height):
    """Erstellt Würfel-Buttons für Aktionen 1-6 unter dem UI im Zentrum"""
    buttons = []
    button_size = 35
    spacing = 5
    
    # Zentrum berechnen
    center_x = screen_width // 2
    center_y = screen_height // 2
    
    # Buttons unter dem UI positionieren (UI-Box + Abstand + Button-Höhe)
    y = center_y + 40 + 20  # UI-Box ist 80px hoch (center_y ± 40), +20px Abstand
    
    # Horizontale Zentrierung der 6 Buttons
    total_width = 7 * button_size + 5 * spacing
    start_x = center_x - (total_width // 2)
    
    # erste 7 Buttons erstellen
    for i in range(0, 7):
        x = start_x + (i) * (button_size + spacing)
        button = Button(x, y, button_size, button_size, str(i), image=card_images[i])
        buttons.append(button)
    # weitere 7 buttons darunter
    y += button_size + spacing
    for i in range(7, 14):
        x = start_x + (i-7) * (button_size + spacing)
        button = Button(x, y, button_size, button_size, str(i), image=card_images[i])
        buttons.append(button)
    return buttons

def create_pos_neg_4_buttons(screen_width, screen_height):
    """Erstellt Würfel-Buttons für Aktionen 1-6 unter dem UI im Zentrum"""
    buttons = []
    button_size = 40
    spacing = 5
    
    # Zentrum berechnen
    center_x = screen_width // 2
    center_y = screen_height // 2
    
    # Buttons unter dem UI positionieren (UI-Box + Abstand + Button-Höhe)
    y = center_y + 40 + 20  # UI-Box ist 80px hoch (center_y ± 40), +20px Abstand
    
    # Horizontale Zentrierung der 2 Buttons
    total_width = 2 * button_size + 1 * spacing
    start_x = center_x - total_width // 2
    
    # 2 Buttons erstellen
    for i in range(0, 2):
        x = start_x + (i-1) * (button_size + spacing)
        button = Button(x, y, button_size, button_size, str((-1)**i * 4), image=card_4_images[i])
        buttons.append(button)
    return buttons

def create_1_or_11_buttons(screen_width, screen_height):
    """Erstellt Würfel-Buttons für Aktionen 1-6 unter dem UI im Zentrum"""
    buttons = []
    button_size = 40
    spacing = 5
    
    # Zentrum berechnen
    center_x = screen_width // 2
    center_y = screen_height // 2
    
    # Buttons unter dem UI positionieren (UI-Box + Abstand + Button-Höhe)
    y = center_y + 40 + 20  # UI-Box ist 80px hoch (center_y ± 40), +20px Abstand
    
    # Horizontale Zentrierung der 2 Buttons
    total_width = 2 * button_size + 1 * spacing
    start_x = center_x - total_width // 2
    
    # 2 Buttons erstellen
    for i in range(0, 2):
        x = start_x + (i-1) * (button_size + spacing)
        button = Button(x, y, button_size, button_size, str(1 if i == 0 else 11), image=card_1_11_images[i])
        buttons.append(button)
    return buttons

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
                
                id = v - (color_key * 10)
                if ENABLE_PIN_IMAGES:
                    # center slighly shifted in y relative to total window scale h
                    center = (x * scale + scale // 2, y * scale + scale // 2 + scale // 10)
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

                # Pin-Nummer in die Mitte zeichnen
                font = pygame.font.SysFont("Arial", 16)
                text_surface = font.render(str(id), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=center)
                screen.blit(text_surface, text_rect)

def draw_ui(screen, font, current_player, hands, game_phase):
    """
    Zeichnet UI-Elemente (Spieler, Würfel) in die Mitte.
    """
    # Hintergrund für UI
    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2
    
    # Box für UI
    # box_rect = pygame.Rect(center_x - 80, center_y - 40, 160, 80)
    # pygame.draw.rect(screen, BACKGROUND_COLOR, box_rect)
    # pygame.draw.rect(screen, FIELD_OUTLINE, box_rect, 2)
    
    player_color = COLORS.get((current_player + 1, 1), (0, 0, 0))
    game_phase_text = font.render(f"Phase: {game_phase}", True, (0, 0, 0))
    player_text = font.render(f"Spieler {current_player + 1}", True, player_color)
    dice_text = font.render(f"Karten: {hands[current_player]}", True, (0, 0, 0))
    
    screen.blit(game_phase_text, (center_x - game_phase_text.get_width() // 2, center_y - 55))
    screen.blit(player_text, (center_x - player_text.get_width() // 2, center_y - 30))
    screen.blit(dice_text, (center_x - dice_text.get_width() // 2, center_y + 5))

def main():
    pygame.init()
    scale = 50  # Größe pro Feld
    
    # Spielkonfiguration
    layout = jnp.array([True, True, True, True])  # Alle 4 Spieler aktiv
    env = env_reset(0, seed=42, num_players=4, distance=10, enable_initial_free_pin=True, layout=layout, enable_teams=True)
    # env = env.replace(phase=jnp.int8(0))  # Setze Phase auf Play
    # env = env.replace(hands=jnp.array([[0,0,0,0,0,2,0,0,0,0,0,0,0,0],
    #                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))
    _ = env_step(env, jnp.array(2))
    print("Phase:", env.phase)
    matrix = board_to_mat(env, layout)
    action_space = get_play_action_size(env)
    print(matrix)
    h, w = matrix.shape
    
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("DOG")
    clock = pygame.time.Clock()
    # Statisches Board-Surface erstellen (nur einmal!)
    board_surface = create_board_surface(matrix, scale)
    # UI-Elemente
    font = pygame.font.SysFont("Arial", 20)
    # game phasen: 'CARD' zum Karten/Würfel auswählen, 'MOVE' zum Pin bewegen, 'HOT7' zum Hot7 Auswahl, 'SWAP' zum Tauschen, 'JOKER' zum Joker Auswahl
    game_phase = 'CARD' if env.phase == 0 else 'CARD_EXCHANGE'
    joker_clicked = False
    running = True

    # Karten-Buttons erstellen
    card_buttons = create_all_card_buttons(w * scale, h * scale)
    posneg4_buttons = create_pos_neg_4_buttons(w * scale, h * scale)
    oneor11_buttons = create_1_or_11_buttons(w * scale, h * scale)
    hot7_buttons = Hot7Selector(pins=[0,1,2,3])
    hot7_buttons.reset()

    selected_action = jnp.zeros(6, dtype=jnp.int32)

    while running:

        mouse_pos = pygame.mouse.get_pos()
        
        # Hover-Effekte aktualisieren
        if game_phase == 'CARD':
            for button in card_buttons:
                button.update_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game_phase == 'CARD' and jnp.sum(env.hands[env.current_player]) == 0:
                print(f"Spieler {int(env.current_player) + 1} hat keine Karten mehr, überspringe Zug.")
                env, r, done = no_step(env) # Aktion -1 zum Überspringen
                matrix = board_to_mat(env, layout)
                if done:
                    print(f"Spiel vorbei! Gewinner ist Spieler {jnp.argwhere(get_winner(env, env.board))[0][0]+1}")
                    running = False
                    
            # Button-Klicks für Karten
            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'CARD':
                for i, button in enumerate(card_buttons):
                    player_hand = env.hands[env.current_player]
                    if button.is_clicked(mouse_pos) and ((player_hand[i] > 0) or joker_clicked):
                        action = i
                        print(f"Spieler {int(env.current_player) + 1} wählt eine {action}")
                        if action == 7:
                            game_phase = 'HOT7'
                            player_id = env.current_player
                        elif action == 0: # Joker nutzung erwartet neue Kartenauswahl
                            game_phase = 'CARD'
                            selected_action = selected_action.at[0].set(1)
                            joker_clicked = True
                        elif action == 1:
                            game_phase = 'SWAP'
                            selected_action = selected_action.at[1].set(1)
                        elif action == 11:
                            game_phase = '1OR11'
                        elif action == 4:
                            game_phase = 'POSNEG4'
                        else:
                            target_idx = action
                            game_phase = 'MOVE'

            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'HOT7':
                confirmed = hot7_buttons.handle_click(mouse_pos)
                if confirmed:
                    print(f"Spieler {int(env.current_player) + 1} wählt Hot7 mit Verteilung: {hot7_buttons.selected_steps}")
                    for pin in hot7_buttons.pins:
                        selected_action = selected_action.at[pin+2].set(hot7_buttons.selected_steps[pin])
                    print(map_move_to_action(env, selected_action))
                    env, r, done = env_step(env, map_move_to_action(env, selected_action))
                    print(f"Reward: {r}")
                    matrix = board_to_mat(env, layout)
                    selected_action = jnp.zeros(6, dtype=jnp.int32)
                    hot7_buttons.reset()
                    game_phase = 'CARD'
                    
                    if done:
                        print(f"Spiel vorbei! Gewinner ist Spieler {jnp.argwhere(get_winner(env, env.board))[0][0]+1}")
                        running = False

            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'POSNEG4':
                for i, button in enumerate(posneg4_buttons):
                    if button.is_clicked(mouse_pos):
                        action = -4 if i == 1 else 4
                        print(f"Spieler {int(env.current_player) + 1} wählt eine {action}")
                        target_idx = action
                        game_phase = 'MOVE'

            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == '1OR11':
                for i, button in enumerate(oneor11_buttons):
                    if button.is_clicked(mouse_pos):
                        action = 1 if i == 0 else 11
                        print(f"Spieler {int(env.current_player) + 1} wählt eine {action}")
                        target_idx = action
                        game_phase = 'MOVE'

            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'SWAP':
                mouse_x, mouse_y = event.pos
                grid_x = mouse_x // scale
                grid_y = mouse_y // scale
                
                if 0 <= grid_y < h and 0 <= grid_x < w:
                    clicked_player_id = (int(matrix[grid_y, grid_x]) // 10 )- 1 
                    clicked_player_pin = (int(matrix[grid_y, grid_x]) % 10 )- 1
                    player_id = env.current_player
                    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
                    
                    target_idx = env.pins[clicked_player_id, clicked_player_pin]
                    
                    print(f"Spieler {int(env.current_player) + 1} tauscht mit Spieler {clicked_player_id + 1} Pin auf Position {target_idx}")
                    if clicked_player_id != current_player and target_idx >= 0 and target_idx < env.board_size:
                        game_phase = 'MOVE'
                    else:
                        print("Kein gültiger Tauschpartner!")

            # 2. Pin auswählen per Mausklick
            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'MOVE':
                mouse_x, mouse_y = event.pos
                grid_x = mouse_x // scale
                grid_y = mouse_y // scale
                
                if 0 <= grid_y < h and 0 <= grid_x < w:
                    clicked_player_id = (int(matrix[grid_y, grid_x]) // 10 )- 1 
                    clicked_player_pin = (int(matrix[grid_y, grid_x]) % 10 )- 1
                    player_id = env.current_player
                    current_player = jnp.where(env.rules["enable_teams"] & is_player_done(env.num_players, env.board, env.goal, player_id), (player_id + 2)%4, player_id)
                    if clicked_player_id == current_player:
                        selected_action = selected_action.at[clicked_player_pin+2].set(target_idx)
                        env, _, done = env_step(env, map_move_to_action(env, selected_action))
                        matrix = board_to_mat(env, layout)
                        selected_action = jnp.zeros(6, dtype=jnp.int32)
                        game_phase = 'CARD' if env.phase == 0 else 'CARD_EXCHANGE'
                        joker_clicked = False
                        
                        if done:
                            print(f"Spiel vorbei! Gewinner ist Spieler {jnp.argwhere(get_winner(env, env.board))[0][0]+1}")
                            running = False
                    else:
                        print("Das ist nicht dein Pin!")

            if event.type == pygame.MOUSEBUTTONDOWN and game_phase == 'CARD_EXCHANGE':
                for i, button in enumerate(card_buttons):
                    if button.is_clicked(mouse_pos) and env.hands[env.current_player, i] > 0:
                        action = i + get_play_action_size(env) 
                        env, _, done = env_step(env, jnp.array(action))
                        matrix = board_to_mat(env, layout)
                        selected_action = jnp.zeros(6, dtype=jnp.int32)
                        game_phase = 'CARD' if env.phase == 0 else 'CARD_EXCHANGE'

        # --- Zeichnen ---
        # 1. Statisches Board (nur kopieren, nicht neu zeichnen)
        screen.blit(board_surface, (0, 0))
        
        # 2. Dynamische Pins
        draw_pins(screen, matrix, scale)
        
        # 3. UI
        draw_ui(screen, font, int(env.current_player), env.hands, game_phase)

        # Würfel-Buttons zeichnen (nur in CARD-Phase)
        if game_phase == 'CARD':
            for button in card_buttons:
                button.draw(screen)
        if game_phase == 'HOT7':
            hot7_buttons.draw(screen, center_x=screen.get_width()//2, center_y=screen.get_height()//2)
        if game_phase == 'POSNEG4':
            for button in posneg4_buttons:
                button.draw(screen)
        if game_phase == '1OR11':
            for button in oneor11_buttons:
                button.draw(screen)
        if game_phase == 'CARD_EXCHANGE':
            for button in card_buttons:
                button.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
