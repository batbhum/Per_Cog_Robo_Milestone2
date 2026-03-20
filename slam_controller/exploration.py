"""
exploration.py - A* goal-directed navigation for E-puck in WeBots (ENU).

AStarExplorer follows waypoints from AStarPlanner using a proportional
heading controller and escapes wall traps via a 4-phase recovery sequence.

Proportional control
---------------------
  heading_error = normalize(waypoint_bearing - robot_heading)
  omega = Kp * heading_error          (clamped to ±turn_speed)
  v     = max_v * (1 - |err|/π)²     (zero when mis-aligned > 60°)

Obstacle detection — directional, not blind
--------------------------------------------
  The old code blocked on *any* LiDAR hit in the forward cone.  That caused
  constant false triggers when the robot was legally driving along a wall.

  New rule: a hit at local angle `a` only counts if `a` is within
  WAYPOINT_ALIGN_DEG of the bearing to the *current waypoint*.  A wall to
  the side the robot is passing does not trigger a replan.

Grace period — the core fix for infinite spinning
--------------------------------------------------
  After recovery completes AND after set_waypoints() loads a fresh path,
  obstacle detection is suppressed for GRACE_DURATION seconds.  This gives
  the robot time to physically drive away from the wall before the detector
  is live again.  Without this, the robot replans, faces the same wall on
  the new path, and blocks immediately.

Recovery state machine  (triggered after RECOVERY_THRESHOLD replans in
                          RECOVERY_WINDOW s, OR on A* path-not-found)
----------------------------------------------------------------------
  REVERSE  (1.2 s) — drive straight back at 70 % speed
      ↓
  ESCAPE   (1.0 s) — arc away from the wall using the clearest LiDAR side
      ↓
  ROTATE   (1.0 s) — spin toward the most open direction
      ↓
  GRACE    (1.5 s) — drive forward freely; obstacle check suppressed
      ↓
  NONE     — set need_replan=True, normal control resumes

No external libraries — only math.
"""
import math
import random

# ── Recovery tunables ─────────────────────────────────────────────────────────
_RECOVERY_THRESHOLD  = 1      # replans within window before recovery kicks in
_RECOVERY_WINDOW     = 3.0    # seconds sliding window for replan counter
_REVERSE_DURATION    = 7.5    # s  — back away from the wall
_ESCAPE_DURATION     = 1.8    # s  — arc sideways away from wall
_ROTATE_DURATION     = 1.5    # s  — turn toward open space
_GRACE_DURATION      = 2.5    # s  — drive freely after recovery/replan (no obs check)

# ── Obstacle-check tunables ───────────────────────────────────────────────────
_WAYPOINT_ALIGN_DEG  = 45     # only block if obstacle is within this angle
                               # of the bearing toward the current waypoint


def _norm(a):
    """Wrap angle to [-pi, pi]."""
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


class AStarExplorer:
    """
    Waypoint-following controller with LiDAR-based replanning and wall recovery.

    Parameters
    ----------
    forward_speed      : max linear speed   [m/s]
    turn_speed         : max angular speed  [rad/s]
    obstacle_threshold : LiDAR range below which a hit counts as a blockage [m]
    waypoint_radius    : distance to waypoint counted as "reached"           [m]
    Kp_heading         : proportional gain, heading error → omega
    front_arc_deg      : half-angle of raw forward cone (pre-directional filter)
    """

    def __init__(self,
                 forward_speed=6.28,
                 turn_speed=1.2,
                 obstacle_threshold=0.12,
                 waypoint_radius=0.18,
                 Kp_heading=2.2,
                 front_arc_deg=35):

        self.forward_speed      = forward_speed
        self.turn_speed         = turn_speed
        self.obstacle_threshold = obstacle_threshold
        self.waypoint_radius    = waypoint_radius
        self.Kp_heading         = Kp_heading
        self.front_arc          = math.radians(front_arc_deg)

        self._waypoints  = []
        self._wp_idx     = 0
        self.need_replan = True   # True on init → plan immediately

        # ── Recovery / grace state ────────────────────────────────────────────
        # states: 'NONE' | 'REVERSE' | 'ESCAPE' | 'ROTATE' | 'GRACE'
        self._state         = 'NONE'
        self._state_start   = 0.0
        self._recovery_omega = 0.0   # spin direction chosen at recovery entry
        self._escape_v      = 0.0    # linear component during ESCAPE arc
        self._escape_omega  = 0.0    # angular component during ESCAPE arc
        self._grace_until   = 0.0    # sim_time after which obs-check is live

        self._replan_times  = []     # sliding window of recent replan timestamps
        self._last_scan     = []     # cached for use inside recovery phases

    # ── Waypoint management ───────────────────────────────────────────────────

    def set_waypoints(self, waypoints, sim_time=0.0):
        """
        Load a fresh list of world-coordinate waypoints.
        Call from slam_controller whenever A* produces a new path.

        Starts a grace period so the robot can begin moving before
        obstacle detection re-activates.
        """
        self._wp_idx     = 0
        self._state      = 'NONE'

        if waypoints:
            self._waypoints  = list(waypoints)
            self.need_replan = False
            self._replan_times.clear()
            # Grace: suppress obstacle check while robot drives off the wall
            self._grace_until = sim_time + _GRACE_DURATION
            print(f"[Explorer] Path loaded: {len(waypoints)} waypoints  "
                  f"first=({waypoints[0][0]:.2f},{waypoints[0][1]:.2f})  "
                  f"grace until t={self._grace_until:.1f}s")
        else:
            self._waypoints  = []
            print("[Explorer] A* returned no path — forcing recovery.")
            self._start_recovery(sim_time, forced=True)

    def has_waypoints(self):
        return bool(self._waypoints) and self._wp_idx < len(self._waypoints)

    def current_waypoint(self):
        return self._waypoints[self._wp_idx] if self.has_waypoints() else None

    # ── Main control ──────────────────────────────────────────────────────────

    def compute_control(self, robot_x, robot_y, robot_theta, scan_ranges,
                        sim_time=0.0):
        """
        Compute (v, omega) toward the current waypoint.

        Parameters
        ----------
        robot_x, robot_y : world position  [m]
        robot_theta       : heading         [rad]
        scan_ranges       : [(local_angle, range), …] from LiDAR
        sim_time          : simulation time [s]

        Returns
        -------
        (v, omega)
        """
        self._last_scan = scan_ranges   # cache for recovery phases

        # ── Recovery / grace state machine (highest priority) ─────────────────
        if self._state != 'NONE':
            return self._do_recovery(sim_time, scan_ranges)

        # ── No plan yet: slow spin to build the map ────────────────────────────
        if not self.has_waypoints():
            return 0.0, self.turn_speed * 0.5

        # ── Advance past waypoints already reached ────────────────────────────
        self._advance_waypoint(robot_x, robot_y)
        if not self.has_waypoints():
            print("[Explorer] Goal reached — requesting replan.")
            self.need_replan = True
            return 0.0, 0.0

        # ── Directional obstacle check (skipped during grace period) ──────────
        wp_x, wp_y = self._waypoints[self._wp_idx]
        in_grace = sim_time < self._grace_until

        if not in_grace:
            if self._waypoint_blocked(scan_ranges, robot_x, robot_y,
                                      robot_theta, wp_x, wp_y):
                if self._record_replan(sim_time):
                    self._start_recovery(sim_time)
                    return self._do_recovery(sim_time, scan_ranges)
                else:
                    print("[Explorer] Obstacle on path — requesting replan.")
                    self.need_replan = True
                    # Short grace so the new plan has time to load & move
                    self._grace_until = sim_time + 0.5
                    return 0.0, 0.0

        # ── Proportional heading control ──────────────────────────────────────
        dx = wp_x - robot_x
        dy = wp_y - robot_y
        target_bearing = math.atan2(dy, dx)
        heading_error  = _norm(target_bearing - robot_theta)

        omega = self.Kp_heading * heading_error
        omega = max(-self.turn_speed, min(self.turn_speed, omega))

        align = max(0.0, 1.0 - (abs(heading_error) / math.pi) ** 2)
        v = self.forward_speed * align
        if abs(heading_error) > math.radians(60):
            v = 0.0

        return v, omega

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _advance_waypoint(self, rx, ry):
        while self.has_waypoints():
            wx, wy = self._waypoints[self._wp_idx]
            if math.sqrt((wx-rx)**2 + (wy-ry)**2) <= self.waypoint_radius:
                self._wp_idx += 1
                if self.has_waypoints():
                    nw = self._waypoints[self._wp_idx]
                    print(f"[Explorer] WP {self._wp_idx} → ({nw[0]:.2f},{nw[1]:.2f})")
            else:
                break

    def _waypoint_blocked(self, scan_ranges, rx, ry, rtheta, wp_x, wp_y):
        """
        Return True only if a close LiDAR hit lies in the direction of the
        current waypoint — not for walls the robot is merely passing alongside.

        A ray at local angle `a` blocks only when BOTH:
          (a) its range < obstacle_threshold
          (b) `a` is within WAYPOINT_ALIGN_DEG of the bearing to the waypoint
              (converted to the robot's local frame)
        """
        if not scan_ranges:
            return False

        wp_bearing_world = math.atan2(wp_y - ry, wp_x - rx)
        wp_bearing_local = _norm(wp_bearing_world - rtheta)
        align_rad = math.radians(_WAYPOINT_ALIGN_DEG)

        return any(
            r < self.obstacle_threshold
            and abs(_norm(a - wp_bearing_local)) < align_rad
            for a, r in scan_ranges
            if abs(a) < self.front_arc
        )

    # ── Recovery helpers ──────────────────────────────────────────────────────

    def _record_replan(self, sim_time):
        cutoff = sim_time - _RECOVERY_WINDOW
        self._replan_times = [t for t in self._replan_times if t > cutoff]
        self._replan_times.append(sim_time)
        return len(self._replan_times) >= _RECOVERY_THRESHOLD

    def _start_recovery(self, sim_time, forced=False):
        reason = ("forced (A* failed)" if forced else
                  f"{len(self._replan_times)} replans / {_RECOVERY_WINDOW:.1f}s")
        print(f"[Explorer] *** RECOVERY *** ({reason})")
        self._state        = 'REVERSE'
        self._state_start  = sim_time
        self.need_replan   = False
        self._replan_times.clear()

    def _do_recovery(self, sim_time, scan_ranges=None):
        """
        4-phase recovery:  REVERSE → ESCAPE → ROTATE → GRACE → NONE

        REVERSE: straight backward — physically separates from wall.
        ESCAPE:  arc away using the clearest LiDAR side — breaks corner traps.
        ROTATE:  spin toward the most open direction (chosen from live scan).
        GRACE:   drive straight forward; obstacle check suppressed.
        """
        elapsed = sim_time - self._state_start
        sc = scan_ranges if scan_ranges is not None else self._last_scan

        # ── REVERSE ───────────────────────────────────────────────────────────
        if self._state == 'REVERSE':
            if elapsed < _REVERSE_DURATION:
                return -self.forward_speed * 1, 0.0
            # Choose escape arc direction from current scan
            left_min  = self._sector_min(sc, lo= math.radians(15),
                                             hi= math.radians(90))
            right_min = self._sector_min(sc, lo=-math.radians(90),
                                             hi=-math.radians(15))
            # Arc away from the closer side
            if left_min <= right_min:
                self._escape_omega = -self.turn_speed * 0.6   # arc right
            else:
                self._escape_omega =  self.turn_speed * 0.6   # arc left
            self._escape_v    =  self.forward_speed * 0.4
            self._state       = 'ESCAPE'
            self._state_start = sim_time
            print(f"[Explorer] Recovery → ESCAPE "
                  f"({'right' if self._escape_omega < 0 else 'left'})")
            return self._escape_v, self._escape_omega

        # ── ESCAPE ────────────────────────────────────────────────────────────
        if self._state == 'ESCAPE':
            if elapsed < _ESCAPE_DURATION:
                return self._escape_v, self._escape_omega
            # Pick rotation toward most open direction
            self._recovery_omega = self._best_spin_direction(sc)
            self._state       = 'ROTATE'
            self._state_start = sim_time
            print(f"[Explorer] Recovery → ROTATE "
                  f"({'CW' if self._recovery_omega < 0 else 'CCW'})")
            return 0.0, self._recovery_omega

        # ── ROTATE ────────────────────────────────────────────────────────────
        if self._state == 'ROTATE':
            if elapsed < _ROTATE_DURATION:
                return 0.0, self._recovery_omega
            self._state       = 'GRACE'
            self._state_start = sim_time
            self._grace_until = sim_time + _GRACE_DURATION
            print("[Explorer] Recovery → GRACE (driving free)")
            return self.forward_speed * 0.6, 0.0

        # ── GRACE ─────────────────────────────────────────────────────────────
        if self._state == 'GRACE':
            if elapsed < _GRACE_DURATION:
                return self.forward_speed * 0.6, 0.0
            print("[Explorer] Recovery complete — requesting plan.")
            self._state      = 'NONE'
            self.need_replan = True
            return 0.0, 0.0

        # Fallback
        self._state      = 'NONE'
        self.need_replan = True
        return 0.0, 0.0

    @staticmethod
    def _sector_min(scan_ranges, lo, hi):
        """Minimum range of rays whose local angle is in [lo, hi]."""
        vals = [r for a, r in scan_ranges if lo <= a <= hi]
        return min(vals) if vals else 9.0

    @staticmethod
    def _best_spin_direction(scan_ranges):
        """
        Return +turn_speed or -turn_speed toward whichever hemisphere
        (left vs right) has more open space (higher mean range).
        """
        left  = [r for a, r in scan_ranges if  math.radians(10) < a < math.pi]
        right = [r for a, r in scan_ranges if -math.pi < a < -math.radians(10)]
        mean_l = sum(left)  / len(left)  if left  else 0.0
        mean_r = sum(right) / len(right) if right else 0.0
        # spin toward the more open side
        return 1.2 if mean_l >= mean_r else -1.2

    # ── API shim ──────────────────────────────────────────────────────────────

    def update(self, omega, dt):
        """Kept for slam_controller API compatibility — no state needed."""
        pass

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self):
        if self._state != 'NONE':
            return f"RECOVERY/{self._state}"
        if not self.has_waypoints():
            return "IDLE/waiting"
        wp = self._waypoints[self._wp_idx]
        grace = "*GRACE* " if True else ""
        return (f"WP {self._wp_idx+1}/{len(self._waypoints)} "
                f"→ ({wp[0]:.2f},{wp[1]:.2f})  "
                f"replan={self.need_replan}  "
                f"recent_replans={len(self._replan_times)}")
