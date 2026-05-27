"""
Mouse Maze Q-Learning
=====================
A reinforcement learning demo where a mouse learns to navigate a maze
to find cheese using Q-learning. The cheese can be moved mid-training
and the agent will adapt.

Controls:
  SPACE       - Train 100 episodes
  R           - Run the mouse along its learned policy
  Click cell  - Move the cheese (resets Q-table)
  ESC         - Quit

Dependencies:
  pip install pygame numpy
"""

import pygame
import numpy as np
import random
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROWS, COLS = 12, 12
CELL = 50               # Pixel size of each maze cell
FPS  = 60

# Q-learning hyperparameters (can be tuned at runtime via sliders — see UI)
ALPHA   = 0.5           # Learning rate
GAMMA   = 0.9           # Discount factor
EPSILON = 0.3           # Exploration rate
MAX_STEPS = 300         # Max steps per episode before giving up

# Colours (R, G, B)
COL_BG       = (245, 243, 238)
COL_WALL     = (60,  58,  54)
COL_GRID     = (220, 218, 210)
COL_PATH     = (55,  130, 220, 120)   # RGBA — drawn on a surface with alpha
COL_QHEAT    = (55,  130, 220)        # Base colour for Q-value heatmap
COL_ARROW    = (100, 100, 100)
COL_TEXT_FG  = (40,  40,  38)
COL_TEXT_MUT = (120, 118, 110)
COL_PANEL    = (235, 232, 226)
COL_BTN      = (200, 198, 192)
COL_BTN_HV   = (180, 178, 172)
COL_BTN_ACT  = (80,  140, 220)

# ---------------------------------------------------------------------------
# Maze layout
# ---------------------------------------------------------------------------

# 1 = wall, 0 = open.  Outer border is always wall.
MAZE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,1,1,1,0,0,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,1,1,1,1],
    [1,0,1,1,0,1,1,0,1,0,0,1],
    [1,0,1,1,1,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,0,0,1,0,1],
    [1,0,1,0,0,0,0,1,0,0,0,1],
    [1,0,1,0,1,1,0,0,0,1,1,1],
    [1,0,0,0,0,1,0,0,1,0,0,1],
    [1,1,1,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.int8)

ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]   # up, down, left, right
ACTION_SYMBOLS = ['↑','↓','←','→']

# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

class QAgent:
    def __init__(self, rows: int, cols: int, n_actions: int = 4):
        self.rows = rows
        self.cols  = cols
        self.n_actions = n_actions
        # Q-table: shape (rows, cols, actions)
        self.q = np.zeros((rows, cols, n_actions), dtype=np.float32)
        self.episodes = 0
        self.alpha  = ALPHA
        self.gamma  = GAMMA
        self.epsilon = EPSILON

    def reset(self):
        self.q[:] = 0
        self.episodes = 0

    def choose_action(self, r: int, c: int, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q[r, c]))

    def update(self, r, c, a, reward, nr, nc, done):
        """Q-learning Bellman update."""
        future = 0.0 if done else float(np.max(self.q[nr, nc]))
        td_error = reward + self.gamma * future - self.q[r, c, a]
        self.q[r, c, a] += self.alpha * td_error

    def train_episode(self, maze: np.ndarray, cheese: tuple) -> int:
        """
        Run one full training episode from (1,1) to cheese.
        Returns number of steps taken.
        """
        r, c = 1, 1
        for step in range(MAX_STEPS):
            a = self.choose_action(r, c)
            dr, dc = ACTIONS[a]
            nr, nc = r + dr, c + dc

            # Hit a wall → penalise, stay put
            if maze[nr, nc] == 1:
                self.update(r, c, a, -2, r, c, False)
                continue

            done   = (nr == cheese[0] and nc == cheese[1])
            reward = 100.0 if done else -1.0
            self.update(r, c, a, reward, nr, nc, done)
            r, c = nr, nc

            if done:
                self.episodes += 1
                return step + 1

        self.episodes += 1
        return MAX_STEPS

    def greedy_path(self, maze: np.ndarray, cheese: tuple) -> list[tuple]:
        """Follow the greedy policy from (1,1). Returns list of (r,c) positions."""
        r, c = 1, 1
        path = [(r, c)]
        visited = {(r, c)}
        for _ in range(MAX_STEPS):
            a = self.choose_action(r, c, greedy=True)
            dr, dc = ACTIONS[a]
            nr, nc = r + dr, c + dc
            if maze[nr, nc] == 1 or (nr, nc) in visited:
                break
            r, c = nr, nc
            path.append((r, c))
            visited.add((r, c))
            if (r, c) == cheese:
                break
        return path

    @property
    def confidence(self) -> float:
        """
        Mean 'certainty' across open cells: how dominant is the best action?
        Returns a value in [0, 1].
        """
        total, count = 0.0, 0
        for r in range(self.rows):
            for c in range(self.cols):
                if MAZE[r, c] == 1:
                    continue
                vals = self.q[r, c]
                s = np.sum(np.abs(vals))
                if s > 0:
                    total += np.max(vals) / s
                    count += 1
        return total / count if count else 0.0

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def draw_maze(surface, agent: QAgent, cheese: tuple, mouse: tuple,
              path: list[tuple] | None, anim_idx: int):
    surface.fill(COL_BG)
    w = COLS * CELL
    h = ROWS * CELL

    # Q-value heatmap
    if agent.episodes > 0:
        max_q = np.max(agent.q)
        if max_q > 0:
            heat_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            for r in range(ROWS):
                for c in range(COLS):
                    if MAZE[r, c] == 1:
                        continue
                    best = float(np.max(agent.q[r, c]))
                    if best > 0:
                        alpha = min(int(best / max_q * 110), 110)
                        pygame.draw.rect(heat_surf, (*COL_QHEAT, alpha),
                                         (c*CELL, r*CELL, CELL, CELL))
            surface.blit(heat_surf, (0, 0))

    # Animated path
    if path and anim_idx > 0:
        pts = [(pc*CELL+CELL//2, pr*CELL+CELL//2)
               for pr, pc in path[:anim_idx+1]]
        if len(pts) >= 2:
            path_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.lines(path_surf, (55, 130, 220, 140), False, pts, 3)
            surface.blit(path_surf, (0, 0))

    # Walls + grid
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c*CELL, r*CELL, CELL, CELL)
            if MAZE[r, c] == 1:
                pygame.draw.rect(surface, COL_WALL, rect)
            else:
                pygame.draw.rect(surface, COL_GRID, rect, 1)

    # Policy arrows (after some training)
    if agent.episodes >= 20:
        font_sm = pygame.font.SysFont('Arial', 14)
        for r in range(1, ROWS-1):
            for c in range(1, COLS-1):
                if MAZE[r, c] == 1 or (r, c) == cheese:
                    continue
                a = int(np.argmax(agent.q[r, c]))
                if agent.q[r, c, a] < 1:
                    continue
                sym = ACTION_SYMBOLS[a]
                img = font_sm.render(sym, True, (*COL_ARROW, 160))
                surface.blit(img, img.get_rect(
                    center=(c*CELL+CELL//2, r*CELL+CELL//2)))

    # Cheese emoji substitute (yellow square + hole)
    cr, cc = cheese
    cx, cy = cc*CELL + CELL//2, cr*CELL + CELL//2
    pygame.draw.rect(surface, (240, 200, 50),
                     (cx-14, cy-10, 28, 20), border_radius=4)
    pygame.draw.circle(surface, (200, 160, 30), (cx-5, cy), 4)
    pygame.draw.circle(surface, (200, 160, 30), (cx+6, cy+2), 3)

    # Mouse (triangle body + ears)
    mr, mc = mouse
    mx, my = mc*CELL+CELL//2, mr*CELL+CELL//2
    pygame.draw.circle(surface, (180, 170, 165), (mx, my), 12)
    pygame.draw.circle(surface, (160, 150, 145), (mx-8, my-10), 6)
    pygame.draw.circle(surface, (160, 150, 145), (mx+8, my-10), 6)
    pygame.draw.circle(surface, (230, 100, 120), (mx, my-2), 3)  # nose


def draw_panel(surface, agent: QAgent, offset_x: int, panel_w: int,
               win_h: int, last_steps: int | None, font, font_sm):
    """Right-side info panel."""
    pygame.draw.rect(surface, COL_PANEL,
                     (offset_x, 0, panel_w, win_h))

    y = 20
    def text(msg, colour=COL_TEXT_FG, big=False):
        nonlocal y
        f = font if big else font_sm
        img = f.render(msg, True, colour)
        surface.blit(img, (offset_x + 16, y))
        y += img.get_height() + 6

    text("Mouse Maze RL", big=True)
    y += 4
    text(f"Episodes trained: {agent.episodes}")
    text(f"Last path length: {last_steps if last_steps else '—'}")
    text(f"Policy confidence: {agent.confidence*100:.0f}%")
    y += 8
    text("Controls", COL_TEXT_MUT)
    text("SPACE  → train 100 eps")
    text("R      → run mouse")
    text("Click  → move cheese")
    text("ESC    → quit")
    y += 8
    text("Hyperparameters", COL_TEXT_MUT)
    text(f"α (learning rate): {agent.alpha:.2f}")
    text(f"γ (discount):      {agent.gamma:.2f}")
    text(f"ε (exploration):   {agent.epsilon:.2f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    maze_w = COLS * CELL
    maze_h = ROWS * CELL
    panel_w = 230
    win_w  = maze_w + panel_w
    screen = pygame.display.set_mode((win_w, maze_h))
    pygame.display.set_caption("Mouse Maze — Q-Learning")
    clock  = pygame.time.Clock()

    font    = pygame.font.SysFont('Arial', 18, bold=True)
    font_sm = pygame.font.SysFont('Arial', 14)

    agent  = QAgent(ROWS, COLS)
    cheese = (9, 9)
    mouse  = (1, 1)

    # Animation state
    path: list[tuple] | None = None
    anim_idx   = 0
    anim_timer = 0
    ANIM_DELAY = 80   # ms per step

    last_steps: int | None = None
    training   = False
    train_batch = 0

    def start_training(n: int):
        nonlocal training, train_batch
        training    = True
        train_batch = n

    def run_mouse():
        nonlocal path, anim_idx, anim_timer, mouse, last_steps
        mouse = (1, 1)
        path = agent.greedy_path(MAZE, cheese)
        last_steps = len(path) - 1
        anim_idx = 0
        anim_timer = 0

    while True:
        dt = clock.tick(FPS)

        # ---- Events --------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    start_training(100)
                if event.key == pygame.K_r:
                    run_mouse()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx_px, my_px = event.pos
                if mx_px < maze_w:   # click within the maze
                    cc = mx_px // CELL
                    cr = my_px // CELL
                    if 0 <= cr < ROWS and 0 <= cc < COLS and MAZE[cr, cc] == 0:
                        if (cr, cc) != (1, 1):
                            cheese = (cr, cc)
                            agent.reset()
                            mouse = (1, 1)
                            path = None

        # ---- Training tick -------------------------------------------------
        if training and train_batch > 0:
            # Do 10 episodes per frame to keep UI responsive
            for _ in range(min(10, train_batch)):
                agent.train_episode(MAZE, cheese)
            train_batch -= 10
            if train_batch <= 0:
                training = False

        # ---- Animation tick ------------------------------------------------
        if path and anim_idx < len(path) - 1:
            anim_timer += dt
            if anim_timer >= ANIM_DELAY:
                anim_timer = 0
                anim_idx  += 1
                mouse = path[anim_idx]

        # ---- Draw ----------------------------------------------------------
        draw_maze(screen, agent, cheese, mouse, path, anim_idx)
        draw_panel(screen, agent, maze_w, panel_w, maze_h,
                   last_steps, font, font_sm)
        pygame.display.flip()


if __name__ == '__main__':
    main()
