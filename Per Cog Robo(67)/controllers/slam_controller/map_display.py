"""
map_display.py - SLAM map + live camera feed in one window.

Left:  occupancy grid map
Right: live camera feed with colour detection overlays
"""
import math
import numpy as np

try:
    import pygame
    import pygame.surfarray
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


# ── Colour detection (pure numpy, no CV libraries) ────────────────────────────

def _rgb_to_hsv(rgb):
    """Vectorised RGB(H,W,3 float32 0-1) → H(0-360), S, V."""
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    mx  = np.maximum(np.maximum(r, g), b)
    mn  = np.minimum(np.minimum(r, g), b)
    d   = mx - mn
    v   = mx
    s   = np.where(mx > 0, d / mx, 0.0)
    h   = np.zeros_like(r)
    mr  = (mx == r) & (d > 0)
    mg  = (mx == g) & (d > 0)
    mb  = (mx == b) & (d > 0)
    h[mr] = (60 * ((g[mr] - b[mr]) / d[mr])) % 360
    h[mg] = (60 * ((b[mg] - r[mg]) / d[mg]) + 120)
    h[mb] = (60 * ((r[mb] - g[mb]) / d[mb]) + 240)
    return h, s, v


def detect_color(rgb_u8, h_min, h_max, s_min=0.45, v_min=0.25):
    """
    Find pixels matching hue range. rgb_u8: (H,W,3) uint8.
    Handles wraparound (e.g. red: h_min=340, h_max=15).
    Returns (x1,y1,x2,y2,cx,cy,count) or None.
    """
    f = rgb_u8.astype(np.float32) / 255.0
    h, s, v = _rgb_to_hsv(f)
    sv = (s >= s_min) & (v >= v_min)
    if h_max >= h_min:
        hm = (h >= h_min) & (h <= h_max)
    else:
        hm = (h >= h_min) | (h <= h_max)
    mask = sv & hm
    ys, xs = np.where(mask)
    if len(xs) < 12:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2, (x1+x2)//2, (y1+y2)//2, len(xs))


# ── Display ───────────────────────────────────────────────────────────────────

CAM_SCALE = 3


class MapDisplay:
    def __init__(self, world_w, world_h, cell_px=4, cam_w=160, cam_h=120):
        self.world_w  = world_w
        self.world_h  = world_h
        self.cell_px  = cell_px
        self.cam_w    = cam_w
        self.cam_h    = cam_h
        self.enabled  = PYGAME_OK
        self._init    = False
        self._last    = 0
        self._gap     = 120      # ~8 fps
        self.traj     = []
        self._cam_rgb = None
        self._cam_dets = []

        if self.enabled:
            pygame.init()

    def _setup(self, gw, gh):
        self.gw     = gw
        self.gh     = gh
        self.map_w  = gw * self.cell_px
        self.map_h  = gh * self.cell_px
        self.cam_dw = self.cam_w * CAM_SCALE
        self.cam_dh = self.cam_h * CAM_SCALE
        self.panel_h = max(self.map_h, self.cam_dh)
        self.total_w = self.map_w + 8 + self.cam_dw
        self.total_h = self.panel_h + 36

        self.screen   = pygame.display.set_mode((self.total_w, self.total_h))
        pygame.display.set_caption("SLAM — Map + Camera  (Milestone 2)")
        self.clock    = pygame.time.Clock()
        self.font     = pygame.font.SysFont("monospace", 11)
        self.font_b   = pygame.font.SysFont("monospace", 12, bold=True)
        self.map_surf = pygame.Surface((gw, gh))
        self.cam_surf = pygame.Surface((self.cam_w, self.cam_h))
        self._init    = True

    def _w2s(self, wx, wy):
        sx = int(wx / self.world_w * self.map_w)
        sy = self.map_h - int(wy / self.world_h * self.map_h)
        return (max(0, min(self.map_w-1, sx)),
                max(0, min(self.map_h-1, sy)))

    # ── Camera ────────────────────────────────────────────────────────────────

    def update_camera(self, camera_device):
        """Read WeBots camera, run colour detection. Call every timestep."""
        if not self.enabled or camera_device is None:
            return
        img = camera_device.getImage()
        if not img:
            return
        w = camera_device.getWidth()
        h = camera_device.getHeight()

        arr = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
        rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])  # BGRA→RGB
        self._cam_rgb = rgb

        dets = []
        # Goal: emissive YELLOW → H≈50-70
        g = detect_color(rgb, h_min=50, h_max=70, s_min=0.5, v_min=0.5)
        if g: dets.append(("GOAL", (255, 255, 0), g))

        # Ball 1: emissive MAGENTA → H≈285-315
        bm = detect_color(rgb, h_min=285, h_max=315, s_min=0.5, v_min=0.4)
        if bm: dets.append(("BALL-M", (255, 0, 255), bm))

        # Ball 2: emissive CYAN → H≈175-195
        bc = detect_color(rgb, h_min=175, h_max=195, s_min=0.5, v_min=0.4)
        if bc: dets.append(("BALL-C", (0, 255, 255), bc))

        self._cam_dets = dets

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, occ_grid, est_pose, true_pose, landmarks, cov, sim_time):
        if not self.enabled:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.enabled = False; return

        now = pygame.time.get_ticks()
        if now - self._last < self._gap:
            return
        self._last = now

        if not self._init:
            self._setup(occ_grid.grid_w, occ_grid.grid_h)

        self.screen.fill((18, 18, 18))

        # ── Occupancy map ─────────────────────────────────────────────────────
        rgb_t = np.ascontiguousarray(
                    np.transpose(np.flipud(occ_grid.get_rgb_array()), (1, 0, 2)))
        pygame.surfarray.blit_array(self.map_surf, rgb_t)
        self.screen.blit(
            pygame.transform.scale(self.map_surf, (self.map_w, self.map_h)), (0, 0))
        pygame.draw.rect(self.screen, (60,60,60), (0, 0, self.map_w, self.map_h), 1)
        self.screen.blit(self.font_b.render("OCCUPANCY MAP", True, (180,180,180)), (4,4))

        # Robot trajectory
        rx, ry, rt = est_pose
        self.traj.append((rx, ry))
        if len(self.traj) > 800: self.traj = self.traj[-800:]
        if len(self.traj) > 1:
            pygame.draw.lines(self.screen, (0,180,80), False,
                              [self._w2s(p[0],p[1]) for p in self.traj], 2)
        sx, sy = self._w2s(rx, ry)
        pygame.draw.circle(self.screen, (255,60,60), (sx,sy), 5)
        hx, hy = self._w2s(rx+0.15*math.cos(rt), ry+0.15*math.sin(rt))
        pygame.draw.line(self.screen, (255,220,0), (sx,sy), (hx,hy), 2)

        # ── Divider ───────────────────────────────────────────────────────────
        div_x = self.map_w + 4
        pygame.draw.line(self.screen, (50,50,50), (div_x,0), (div_x,self.panel_h), 2)
        cam_x = div_x + 4

        # ── Camera panel ──────────────────────────────────────────────────────
        self.screen.blit(self.font_b.render("CAMERA FEED", True, (180,180,180)),
                         (cam_x+4, 4))

        if self._cam_rgb is not None:
            cam_t = np.ascontiguousarray(np.transpose(self._cam_rgb, (1,0,2)))
            pygame.surfarray.blit_array(self.cam_surf, cam_t)
            self.screen.blit(
                pygame.transform.scale(self.cam_surf, (self.cam_dw, self.cam_dh)),
                (cam_x, 0))

            for label, color, det in self._cam_dets:
                x1,y1,x2,y2,cx,cy,cnt = det
                pygame.draw.rect(self.screen, color,
                    (cam_x + x1*CAM_SCALE, y1*CAM_SCALE,
                     (x2-x1)*CAM_SCALE, (y2-y1)*CAM_SCALE), 2)
                self.screen.blit(self.font_b.render(label, True, color),
                    (cam_x + x1*CAM_SCALE, max(0, y1*CAM_SCALE - 14)))

            if self._cam_dets:
                txt = "DETECTED: " + ", ".join(d[0] for d in self._cam_dets)
                col = (0, 255, 100)
            else:
                txt = "No objects detected"
                col = (120, 120, 120)
            self.screen.blit(self.font.render(txt, True, col),
                             (cam_x, self.cam_dh + 4))
        else:
            pygame.draw.rect(self.screen, (30,30,30),
                             (cam_x, 0, self.cam_dw, self.cam_dh))
            self.screen.blit(self.font.render("Camera initialising...", True, (100,100,100)),
                             (cam_x+10, self.cam_dh//2))

        # ── Info bar ──────────────────────────────────────────────────────────
        info = (f"t={sim_time:.0f}s  pos({rx:.2f},{ry:.2f})  "
                f"hdg={math.degrees(rt):.0f}°  "
                f"[white=free  black=wall  gray=unknown]")
        self.screen.blit(self.font.render(info, True, (180,180,180)),
                         (4, self.panel_h+4))
        pygame.display.flip()

    def close(self):
        if self.enabled:
            pygame.quit()
