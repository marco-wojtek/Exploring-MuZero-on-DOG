import matplotlib.pyplot as plt
import numpy as np

# Farben und Reihenfolge
COLORS = ['grey', 'red', 'blue', 'green', 'yellow']
BOARD_SIZE = 40
GOAL_SIZE = 4
NUM_PLAYERS = 4
BACKGROUND_COLOR = (250/255, 235/255, 180/255)

# Board-Positionen (wie im Haupttool)
def get_board_positions():
    # Dünner Rahmen um das Plus (Zielfelder), 40 Felder als Ring
    pos = []
    # Oben: von links nach rechts
    for x in range(-1, 2, 1):
        pos.append((x, 5))

    # Oben rechts
    for y in range(4, 0, -1):
        pos.append((1, y))

    # Mitte oben rechts
    for x in range(2, 6, 1):
        pos.append((x, 1))

    # Rechts
    for x in range(0, -2, -1):
        pos.append((5, x))

    # Mitte unten rechts
    for x in range(4, 0, -1):
        pos.append((x, -1))

    # Unten rechts
    for x in range(-2, -6, -1):
        pos.append((1, x))

    # Unten
    for x in range(0, -2, -1):
        pos.append((x, -5))

    # Unten links
    for x in range(-4, 0, 1):
        pos.append((-1, x))

    # Mitte unten links
    for x in range(-2, -6, -1):
        pos.append((x, -1))

    # Links
    for x in range(0, 2, 1):
        pos.append((-5, x))

    # Mitte oben links
    for x in range(-4, 0, 1):
        pos.append((x, 1))

    # Oben links
    for y in range(2, 5, 1):
        pos.append((-1, y))


    return pos  # 40 Felder

# Ziel-Positionen (wie im Haupttool)
def get_goal_positions():
    goals = []

    # Rot (nach links)
    for i in range(1, 5):
        goals.append((-i, 0))
    # Blau (nach oben)
    for i in range(1, 5):
        goals.append((0, i))
    # Grün (nach rechts)
    for i in range(1, 5):
        goals.append((i, 0))
    # Gelb (nach unten)
    for i in range(1, 5):
        goals.append((0, -i))
    return goals

# Haus-Positionen (Startfelder, wie im Haupttool)
def get_start_positions():
    starts = []
    # Rot
    starts += [(-5, 5), (-5, 4), (-4, 5), (-4, 4)]
    # Blau
    starts += [(5, 5), (5, 4), (4, 5), (4, 4)]
    # Grün
    starts += [(5, -5), (5, -4), (4, -5), (4, -4)]
    # Gelb
    starts += [(-5, -5), (-5, -4), (-4, -5), (-4, -4)]
    
    return starts

def interactive_board():
    positions = get_board_positions()
    goal_positions = get_goal_positions()
    start_positions = get_start_positions()
    color_indices = [0]*len(positions)  # Start: alles grau
    goal_filled = [False]*len(goal_positions)
    start_filled = [False]*len(start_positions)
    special_indices = [32, 2, 12, 22]
    special_colors = ['red', 'blue', 'green', 'yellow']
    special_state = [0]*4  # 0=default (Zielfeld-Optik), 1-4=rot,blau,gelb,grün
    fig, ax = plt.subplots(figsize=(12,12))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    # Listen für die Marker immer leeren, bevor sie neu befüllt werden
    scatters = []
    goal_scatters = []
    start_scatters = []
    goal_colors = ['red']*4 + ['blue']*4 + ['green']*4 + ['yellow']*4
    start_colors = ['red']*4 + ['blue']*4 + ['green']*4 + ['yellow']*4
    
    def draw():
        ax.clear()
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        scatters.clear()
        goal_scatters.clear()
        start_scatters.clear()
        # Normale Felder und Spezialfelder
        for i, (x, y) in enumerate(positions):
            if i in special_indices:
                idx = special_indices.index(i)
                if special_state[idx] == 0:
                    s = ax.scatter(x, y, s=2000, c='white', edgecolors=special_colors[idx], linewidths=2, zorder=2, picker=True)
                else:
                    c = special_colors[special_state[idx]-1]
                    s = ax.scatter(x, y, s=2000, c=c, edgecolors='k', zorder=2, picker=True)
                scatters.append(s)
            else:
                c = COLORS[color_indices[i]]
                s = ax.scatter(x, y, s=2000, c=c, edgecolors='k', zorder=2, picker=True)
                scatters.append(s)
        # Zielfelder
        for i, (x, y) in enumerate(goal_positions):
            edge_c = goal_colors[i]
            s = ax.scatter(x, y, s=2000, c='white', edgecolors=edge_c, linewidths=2, zorder=2, picker=True)
            goal_scatters.append(s)
            if goal_filled[i]:
                ax.scatter(x, y, s=1200, c=edge_c, edgecolors='k', zorder=3)
        # Hausfelder
        for i, (x, y) in enumerate(start_positions):
            edge_c = start_colors[i]
            s = ax.scatter(x, y, s=2000, c='white', edgecolors=edge_c, linewidths=2, zorder=2, picker=True)
            start_scatters.append(s)
            if start_filled[i]:
                ax.scatter(x, y, s=1200, c=edge_c, edgecolors='k', zorder=3)
        ax.set_aspect('equal')
        ax.axis('off')
        # Nach dem Zeichnen: schwarzen Rahmen um das Bild
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4)
            spine.set_color('black')
        plt.tight_layout()
        fig.canvas.draw()
    def on_pick(event):
        # Normale Felder und Spezialfelder
        for i in range(len(positions)):
            s = scatters[i]
            if i in special_indices:
                idx = special_indices.index(i)
                cont, _ = s.contains(event.mouseevent)
                if cont:
                    special_state[idx] = (special_state[idx] + 1) % 5  # 0=default, 1-4=rot,blau,gelb,grün
                    draw()
                    return
            else:
                cont, _ = s.contains(event.mouseevent)
                if cont:
                    color_indices[i] = (color_indices[i] + 1) % len(COLORS)
                    draw()
                    return
        # Zielfelder
        for i, s in enumerate(goal_scatters):
            cont, _ = s.contains(event.mouseevent)
            if cont:
                goal_filled[i] = not goal_filled[i]
                draw()
                return
        # Hausfelder
        for i, s in enumerate(start_scatters):
            cont, _ = s.contains(event.mouseevent)
            if cont:
                start_filled[i] = not start_filled[i]
                draw()
                return
    draw()
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == '__main__':
    interactive_board()
