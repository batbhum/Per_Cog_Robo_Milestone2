"""
exploration.py - Wall-following random walk for E-puck in WeBots (ENU).

Fixes vs previous version:
  - Turn angle increased: 120° min, up to 200° — ensures robot actually
    clears the wall it just hit
  - After turning, scans forward arc to verify path is clear before driving;
    if still blocked, keeps turning (no more wall hits)
  - DRIVING_OUT time increased to 0.8s so robot moves well away from wall
  - Obstacle threshold lowered to 0.10m (E-puck radius ~0.037m + margin)
"""
import math
import random


class RandomWalkExplorer:
    def __init__(self, forward_speed=0.3, turn_speed=1.2, obstacle_threshold=0.10):
        self.forward_speed      = forward_speed
        self.turn_speed         = turn_speed
        self.obstacle_threshold = obstacle_threshold

        self.state           = "MOVING"
        self.turn_remaining  = 0.0
        self.drive_out_timer = 0.0
        self.DRIVE_OUT_TIME  = 0.8   # seconds — enough to clear any wall

    # ----------------------------------------------------------------
    def compute_control(self, scan_ranges):
        if self.state == "TURNING":
            return self._do_turn(scan_ranges)

        if self.state == "DRIVING_OUT":
            return self.forward_speed, 0.0

        # Check narrow front arc ±25°
        front_blocked = any(
            r < self.obstacle_threshold
            for a, r in scan_ranges
            if abs(a) < math.radians(25)
        )

        if front_blocked:
            self._pick_turn(scan_ranges)
            return self._do_turn(scan_ranges)

        return self.forward_speed, 0.0

    # ----------------------------------------------------------------
    def _pick_turn(self, scan_ranges):
        """Choose turn direction and magnitude toward the clearer side."""
        # Measure clearance in left and right hemispheres
        # CW convention: positive angle = clockwise = right side of robot
        right_clear = min((r for a, r in scan_ranges
                           if  math.radians(20) < a < math.pi), default=1.5)
        left_clear  = min((r for a, r in scan_ranges
                           if -math.pi < a < -math.radians(20)), default=1.5)

        # Turn between 120° and 200° (enough to fully reverse away from wall)
        turn_mag = random.uniform(math.radians(120), math.radians(200))
        self.turn_remaining = turn_mag if left_clear >= right_clear else -turn_mag
        self.state = "TURNING"

    # ----------------------------------------------------------------
    def _do_turn(self, scan_ranges=None):
        if abs(self.turn_remaining) < math.radians(3):
            # Turn complete — check if forward is actually clear now
            if scan_ranges is not None:
                still_blocked = any(
                    r < self.obstacle_threshold
                    for a, r in scan_ranges
                    if abs(a) < math.radians(25)
                )
                if still_blocked:
                    # Pick a new turn without resetting to MOVING
                    self._pick_turn(scan_ranges)
                    omega = self.turn_speed if self.turn_remaining > 0 else -self.turn_speed
                    return 0.0, omega

            self.state           = "DRIVING_OUT"
            self.drive_out_timer = self.DRIVE_OUT_TIME
            return self.forward_speed, 0.0

        omega = self.turn_speed if self.turn_remaining > 0 else -self.turn_speed
        return 0.0, omega

    # ----------------------------------------------------------------
    def update(self, omega, dt):
        if self.state == "TURNING":
            delta = abs(omega * dt)
            if self.turn_remaining > 0:
                self.turn_remaining -= delta
            else:
                self.turn_remaining += delta

        elif self.state == "DRIVING_OUT":
            self.drive_out_timer -= dt
            if self.drive_out_timer <= 0:
                self.state = "MOVING"
