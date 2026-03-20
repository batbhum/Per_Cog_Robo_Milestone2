"""
slam_controller.py - WeBots R2025a E-puck SLAM + A* Navigation (Milestone 2+)

ENU: X=East, Y=North, Z=Up
Compass raw (0,1,0) → heading=0° → robot faces East (+X). Confirmed.
LiDAR raw: ray[0]=0° points North. Offset=-π/2 corrects to East.

This version:
  - Uses GPS pose directly for mapping and navigation (no EKF drift).
  - Replaces RandomWalkExplorer with AStarExplorer (goal-directed).
  - Calls AStarPlanner whenever a new path is needed (start-up or blockage).
  - EKF is retained only for the uncertainty ellipse display.

Navigation flow
---------------
  1. At start (and whenever explorer.need_replan is True), run A* from the
     robot's current grid cell to TARGET grid cell.
  2. Feed the resulting waypoint list to explorer.set_waypoints().
  3. explorer.compute_control() steers the robot along the path.
  4. If the LiDAR detects a newly discovered wall blocking the path,
     explorer.need_replan is set True → go back to step 1 on the next loop.
"""
from controller import Robot
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import normalize_angle
from occupancy_grid import OccupancyGrid
from exploration   import AStarExplorer
from path_planning import AStarPlanner
from map_display   import MapDisplay

# ── CONFIG ────────────────────────────────────────────────────────────────────
WORLD_W      = 6.0
WORLD_H      = 6.0
GRID_RES     = 0.05       # metres per cell → 120×120 grid for 6 m arena
MAX_LIDAR    = 3.0
UPDATE_EVERY = 3           # update occupancy grid every N timesteps
CELL_PX      = 4           # 120 * 4 = 480 px map window

# Arena is centred at world origin: GPS gives -3..+3, grid needs 0..6
GPS_OFFSET_X = 3.0
GPS_OFFSET_Y = 3.0

WHEEL_RADIUS = 0.0205
WHEEL_BASE   = 0.052
MAX_SPEED    = 6.28

# ── NAVIGATION TARGET ─────────────────────────────────────────────────────────
# World coordinates in metres (after GPS offset is applied).
# (0,0) = SW corner, (6,6) = NE corner for a 6 m arena.
# Change these to point the robot at any goal in the arena.
TARGET_X = 5.0    # metres East  (e.g. far NE corner)
TARGET_Y = 5.0    # metres North

# How many grid cells must be mapped before we try A* for the first time.
# Prevents planning on a completely empty grid.
MIN_CELLS_BEFORE_PLAN = 50

# Minimum sim-time gap between replanning attempts [seconds].
# Prevents hammering A* every timestep when stuck.
REPLAN_COOLDOWN = 1.5

# ─────────────────────────────────────────────────────────────────────────────


def try_motor(robot, *names):
    for n in names:
        d = robot.getDevice(n)
        if d:
            print(f"  Motor: '{n}'")
            return d
    return None


def compass_heading(cv):
    """ENU: atan2(East, North) gives heading relative to East (+X axis)."""
    return math.atan2(cv[0], cv[1])


def mapped_cells(grid):
    """Return number of cells that have been observed (|log-odds| > 0.05)."""
    import numpy as np
    return int((abs(grid.grid) > 0.05).sum())


def main():
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())
    dt       = timestep / 1000.0

    print("=" * 55)
    print("  E-puck SLAM + A* Navigation — Milestone 2+")
    print(f"  Grid : {int(WORLD_W/GRID_RES)}×{int(WORLD_H/GRID_RES)}  res={GRID_RES} m")
    print(f"  Goal : world ({TARGET_X:.2f}, {TARGET_Y:.2f})")
    print("=" * 55)

    # ── Motors ────────────────────────────────────────────────────────────────
    lm = try_motor(robot, "left wheel motor",  "left_wheel_motor")
    rm = try_motor(robot, "right wheel motor", "right_wheel_motor")
    if not lm or not rm:
        print("ERROR: motors not found"); return
    lm.setPosition(float('inf')); lm.setVelocity(0)
    rm.setPosition(float('inf')); rm.setVelocity(0)

    # ── Sensors ───────────────────────────────────────────────────────────────
    lidar = robot.getDevice("lidar")
    gps   = robot.getDevice("gps")
    comp  = robot.getDevice("compass")
    if not lidar or not gps or not comp:
        print("ERROR: sensor missing"); return
    lidar.enable(timestep); lidar.enablePointCloud()
    gps.enable(timestep)
    comp.enable(timestep)

    camera = robot.getDevice("slam_camera")
    if camera:
        camera.enable(timestep)
        print(f"  Camera: {camera.getWidth()}×{camera.getHeight()} px")
    else:
        print("  WARNING: camera device not found")

    # Warm-up step
    robot.step(timestep)

    gv = gps.getValues(); cv = comp.getValues()
    start_x = gv[0] + GPS_OFFSET_X
    start_y = gv[1] + GPS_OFFSET_Y
    print(f"  GPS start  : ({gv[0]:.3f}, {gv[1]:.3f})  → grid ({start_x:.2f},{start_y:.2f})")
    print(f"  Compass raw: ({cv[0]:.3f}, {cv[1]:.3f}, {cv[2]:.3f})")
    print(f"  Heading    : {math.degrees(compass_heading(cv)):.1f}°")

    # ── Core objects ──────────────────────────────────────────────────────────
    grid     = OccupancyGrid(WORLD_W, WORLD_H, resolution=GRID_RES)
    planner  = AStarPlanner(grid_resolution=GRID_RES)
    explorer = AStarExplorer(
        forward_speed=0.28,
        turn_speed=1.2,
        obstacle_threshold=0.12,
        waypoint_radius=0.18,
        Kp_heading=2.2,
        front_arc_deg=30,
    )
    display  = MapDisplay(WORLD_W, WORLD_H, cell_px=CELL_PX)

    sim_time       = 0.0
    step_cnt       = 0
    last_replan_t  = -REPLAN_COOLDOWN   # allow immediate first plan

    # ── Main loop ─────────────────────────────────────────────────────────────
    while robot.step(timestep) != -1:
        sim_time += dt
        step_cnt += 1

        # ── GPS pose (ground truth, used for everything) ───────────────────
        gv  = gps.getValues()
        cv  = comp.getValues()
        rx  = gv[0] + GPS_OFFSET_X
        ry  = gv[1] + GPS_OFFSET_Y
        rt  = compass_heading(cv)

        # ── LiDAR scan ────────────────────────────────────────────────────
        raw   = lidar.getRangeImage()
        fov   = lidar.getFov()
        nrays = lidar.getHorizontalResolution()
        scan  = []
        if raw:
            inc   = fov / nrays
            start = -fov / 2.0
            for i in range(nrays):
                r = raw[i]
                if math.isinf(r) or math.isnan(r) or r > MAX_LIDAR:
                    r = MAX_LIDAR
                scan.append((start + i * inc, r))

        # ── Camera feed ───────────────────────────────────────────────────
        if camera:
            display.update_camera(camera)

        # ── Occupancy grid update ─────────────────────────────────────────
        if step_cnt % UPDATE_EVERY == 0 and scan:
            grid.update(rx, ry, rt, scan, MAX_LIDAR)

        # ── A* replanning ─────────────────────────────────────────────────
        needs_plan = (
            explorer.need_replan
            and (sim_time - last_replan_t) >= REPLAN_COOLDOWN
            and mapped_cells(grid) >= MIN_CELLS_BEFORE_PLAN
        )

        if needs_plan:
            print(f"[SLAM] Replanning at t={sim_time:.1f}s  "
                  f"pos=({rx:.2f},{ry:.2f})  goal=({TARGET_X},{TARGET_Y})")
            waypoints = planner.plan(
                grid.grid,
                start_world=(rx, ry),
                goal_world=(TARGET_X, TARGET_Y),
            )
            explorer.set_waypoints(waypoints, sim_time=sim_time)
            last_replan_t = sim_time

            if not waypoints:
                # A* failed (goal unreachable right now) — keep spinning
                print("[SLAM] No path found — will retry after cooldown.")

        # ── Explorer: compute wheel commands ──────────────────────────────
        v_cmd, w_cmd = explorer.compute_control(rx, ry, rt, scan, sim_time)
        explorer.update(w_cmd, dt)

        # ── Convert (v, omega) → individual wheel speeds ──────────────────
        vl_raw = (v_cmd - w_cmd * WHEEL_BASE / 2) / WHEEL_RADIUS
        vr_raw = (v_cmd + w_cmd * WHEEL_BASE / 2) / WHEEL_RADIUS
        vl = max(-MAX_SPEED, min(MAX_SPEED, vl_raw))
        vr = max(-MAX_SPEED, min(MAX_SPEED, vr_raw))
        lm.setVelocity(vl)
        rm.setVelocity(vr)

        # ── Display update ────────────────────────────────────────────────
        if step_cnt % UPDATE_EVERY == 0 and display.enabled:
            display.update(
                grid,
                (rx, ry, rt),   # estimated pose (= GPS ground truth here)
                (rx, ry, rt),   # true pose (same — no EKF drift)
                [],             # landmarks (clean display)
                [[0.001,0,0],[0,0.001,0],[0,0,0.001]],
                sim_time,
            )

        # ── Periodic console log ──────────────────────────────────────────
        if step_cnt % 150 == 0:
            dist_to_goal = math.sqrt((TARGET_X - rx)**2 + (TARGET_Y - ry)**2)
            print(f"t={sim_time:.1f}s | GPS({rx:.2f},{ry:.2f}) "
                  f"hdg={math.degrees(rt):.0f}°  "
                  f"dist_goal={dist_to_goal:.2f}m  "
                  f"status={explorer.status()}")

        # ── Goal reached? ─────────────────────────────────────────────────
        dist_to_goal = math.sqrt((TARGET_X - rx)**2 + (TARGET_Y - ry)**2)
        if dist_to_goal < 0.20:
            print(f"\n[SLAM] *** GOAL REACHED at t={sim_time:.1f}s! ***")
            lm.setVelocity(0); rm.setVelocity(0)
            # Keep display alive so we can inspect the final map
            while robot.step(timestep) != -1:
                display.update(
                    grid,
                    (rx, ry, rt),
                    (rx, ry, rt),
                    [],
                    [[0.001,0,0],[0,0.001,0],[0,0,0.001]],
                    sim_time,
                )
                for e in __import__('pygame').event.get():
                    if e.type == __import__('pygame').QUIT:
                        display.close(); return

    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
