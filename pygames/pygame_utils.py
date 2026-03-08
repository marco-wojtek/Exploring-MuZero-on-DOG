import pygame
import jax.numpy as jnp
import jax
import sys, os

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

class PlayerSelectionMenu:
    def __init__(self, screen_width, screen_height):
        self.width = 400
        self.height = 350
        self.x = (screen_width - self.width) // 2
        self.y = (screen_height - self.height) // 2
        self.font = pygame.font.SysFont("Arial", 20)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Spielertypen: 0 = Human, 1 = COM (MCTS), 2 = Random
        self.player_types = [0, 0, 0, 0]
        
        # Buttons für jeden Spieler
        self.buttons = []
        button_width = 100
        button_height = 40
        start_y = self.y + 80
        
        for i in range(4):
            x = self.x + 40
            y = start_y + i * 60
            button = Button(x, y, button_width, button_height, "Human", COLORS.get((i+1, 1), (200, 200, 200)))
            self.buttons.append(button)
        
        # Start-Button
        self.start_button = Button(
            self.x + self.width // 2 - 60,
            self.y + self.height - 40,
            120, 40, "Start Game", (50, 200, 50)
        )
    
    def draw(self, screen):
        # Hintergrund
        pygame.draw.rect(screen, BACKGROUND_COLOR, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, FIELD_OUTLINE, (self.x, self.y, self.width, self.height), 3)
        
        # Titel
        title = self.title_font.render("Spieler-Auswahl", True, (0, 0, 0))
        screen.blit(title, (self.x + self.width // 2 - title.get_width() // 2, self.y + 20))
        
        # Spieler-Buttons
        for i, button in enumerate(self.buttons):
            type_names = ["Human", "COM", "Random"]
            button.text = type_names[self.player_types[i]]
            button.draw(screen)
            
            # Spieler-Label
            label = self.font.render(f"Spieler {i+1}:", True, (0, 0, 0))
            screen.blit(label, (button.rect.x + 110, button.rect.y + 10))
        
        # Start-Button
        self.start_button.draw(screen)
    
    def update_hover(self, pos):
        for button in self.buttons:
            button.update_hover(pos)
        self.start_button.update_hover(pos)
    
    def handle_click(self, pos):
        for i, button in enumerate(self.buttons):
            if button.is_clicked(pos):
                # Toggle zwischen Human (0) -> COM (1) -> Random (2) -> Human (0)
                self.player_types[i] = (self.player_types[i] + 1) % 3
                return None
        
        if self.start_button.is_clicked(pos):
            return self.player_types
        
        return None