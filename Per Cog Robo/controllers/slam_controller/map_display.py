"""
map_display.py - Fast pygame SLAM map.

Grid layout:  grid[row][col], row=0 is Y=0 (south), col=0 is X=0 (west)
Display:      col → screen X (left=west, right=east)
              row → screen Y flipped (bottom=south, top=north)

surfarray needs (W, H) = (cols, rows) with:
  axis-0 = screen X = col = world X
  axis-1 = screen Y = (grid_h-1-row) = flipped world Y
"""
import math
import numpy as np

try:
    import pygame
    import pygame.surfarray
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


class MapDisplay:
    def __init__(self, world_w, world_h, cell_px=10):
        self.world_w = world_w
        self.world_h = world_h
        self.cell_px = cell_px
        self.enabled = PYGAME_OK
        self._init   = False
        self._last   = 0
        self._gap    = 150      # ms between frames (~6 fps)
        self.traj    = []
        if self.enabled:
            pygame.init()

    def _setup(self, gw, gh):
        self.gw     = gw
        self.gh     = gh
        self.vw     = gw * self.cell_px
        self.vh     = gh * self.cell_px
        self.screen = pygame.display.set_mode((self.vw, self.vh + 36))
        pygame.display.set_caption("SLAM Map — E-puck Milestone 2")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("monospace", 12)
        self.surf   = pygame.Surface((gw, gh))
        self._init  = True

    def _w2s(self, wx, wy):
        """World (x,y) → screen pixel. West=left East=right South=bottom North=top."""
        sx = int(wx / self.world_w * self.vw)
        sy = self.vh - int(wy / self.world_h * self.vh)
        return (max(0, min(self.vw-1, sx)),
                max(0, min(self.vh-1, sy)))

    def update(self, occ_grid, est_pose, true_pose, landmarks, cov, t):
        if not self.enabled: return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.enabled = False; return
        now = pygame.time.get_ticks()
        if now - self._last < self._gap: return
        self._last = now
        if not self._init:
            self._setup(occ_grid.grid_w, occ_grid.grid_h)

        # ── Build texture ──────────────────────────────────────────
        # get_rgb_array() → (H, W, 3):  H=rows=Y-axis,  W=cols=X-axis
        rgb = occ_grid.get_rgb_array()   # shape (H, W, 3)

        # We want screen pixel (sx, sy) to show grid cell (col, row) where:
        #   sx = col  (X increases right)
        #   sy = (gh-1-row)  (Y increases upward → flip rows)
        #
        # surfarray.blit_array expects array[sx, sy] → needs shape (W, H, 3)
        # Step 1: flip rows so row=0 (south) goes to bottom of screen
        rgb_flip = np.flipud(rgb)                       # (H, W, 3), row0=north at top
        # Step 2: transpose to (W, H, 3) for surfarray (axis0=screen-X=col)
        rgb_t    = np.ascontiguousarray(
                       np.transpose(rgb_flip, (1, 0, 2)))  # (W, H, 3)

        pygame.surfarray.blit_array(self.surf, rgb_t)
        scaled = pygame.transform.scale(self.surf, (self.vw, self.vh))
        self.screen.fill((20, 20, 20))
        self.screen.blit(scaled, (0, 0))

        # ── Robot trajectory ───────────────────────────────────────
        rx, ry, rt = est_pose
        self.traj.append((rx, ry))
        if len(self.traj) > 600: self.traj = self.traj[-600:]
        if len(self.traj) > 1:
            pygame.draw.lines(self.screen, (0, 180, 80), False,
                              [self._w2s(p[0], p[1]) for p in self.traj], 2)

        # Robot dot + heading
        sx, sy = self._w2s(rx, ry)
        pygame.draw.circle(self.screen, (255, 60, 60), (sx, sy), 6)
        hx = rx + 0.1 * math.cos(rt)
        hy = ry + 0.1 * math.sin(rt)
        pygame.draw.line(self.screen, (255, 220, 0),
                         (sx, sy), self._w2s(hx, hy), 2)

        # ── Info bar ───────────────────────────────────────────────
        info = (f"t={t:.0f}s  pos({rx:.2f},{ry:.2f})  "
                f"hdg={math.degrees(rt):.0f}°  "
                f"[white=free  black=wall  gray=unknown]")
        self.screen.blit(
            self.font.render(info, True, (200, 200, 200)),
            (4, self.vh + 4))
        pygame.display.flip()

    def close(self):
        if self.enabled: pygame.quit()
