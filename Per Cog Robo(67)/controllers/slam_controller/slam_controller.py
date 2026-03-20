"""
slam_controller.py - WeBots R2025a E-puck SLAM (Milestone 2)

ENU: X=East, Y=North, Z=Up
Compass raw (0,1,0) → heading=0° → robot faces East (+X). Confirmed.
LiDAR raw: ray[0]=0° points North. Offset=-π/2 corrects to East.

This version uses GPS pose directly for everything (no EKF drift).
EKF is only used to display uncertainty ellipse — it does NOT affect the map.
"""
from controller import Robot
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import normalize_angle
from occupancy_grid import OccupancyGrid
from exploration import RandomWalkExplorer
from map_display import MapDisplay

# ── CONFIG ───────────────────────────────────────────────────
WORLD_W      = 6.0
WORLD_H      = 6.0
GRID_RES     = 0.05     # 120x120 grid for 6m arena
MAX_LIDAR    = 3.0
UPDATE_EVERY = 3        # update grid every N timesteps (keep it frequent)
CELL_PX      = 4        # 120*4 = 480px window

# Arena centered at origin: GPS gives -3..+3, grid needs 0..6
GPS_OFFSET_X = 3.0
GPS_OFFSET_Y = 3.0

WHEEL_RADIUS = 0.0205
WHEEL_BASE   = 0.052
MAX_SPEED    = 6.28
# ─────────────────────────────────────────────────────────────


def try_motor(robot, *names):
    for n in names:
        d = robot.getDevice(n)
        if d:
            print(f"  Motor: '{n}'")
            return d
    return None


def compass_heading(cv):
    # ENU: compass points North (+Y). Heading = atan2(East, North) = atan2(cv[0], cv[1])
    return math.atan2(cv[0], cv[1])


def main():
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())
    dt       = timestep / 1000.0

    print("=" * 50)
    print("  E-puck SLAM — Milestone 2  (GPS-direct mode)")
    print(f"  Grid: {int(WORLD_W/GRID_RES)}x{int(WORLD_H/GRID_RES)}  res={GRID_RES}m")
    print("=" * 50)

    lm = try_motor(robot, "left wheel motor",  "left_wheel_motor")
    rm = try_motor(robot, "right wheel motor", "right_wheel_motor")
    if not lm or not rm:
        print("ERROR: motors not found"); return
    lm.setPosition(float('inf')); lm.setVelocity(0)
    rm.setPosition(float('inf')); rm.setVelocity(0)

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
        print(f"  Camera: {camera.getWidth()}x{camera.getHeight()} px")
    else:
        print("  WARNING: camera device not found")

    robot.step(timestep)

    gv = gps.getValues(); cv = comp.getValues()
    print(f"  GPS start:   ({gv[0]:.3f}, {gv[1]:.3f})  → grid ({gv[0]+GPS_OFFSET_X:.2f},{gv[1]+GPS_OFFSET_Y:.2f})")
    print(f"  Compass raw: ({cv[0]:.3f}, {cv[1]:.3f}, {cv[2]:.3f})")
    print(f"  Heading:     {math.degrees(compass_heading(cv)):.1f}°")

    grid     = OccupancyGrid(WORLD_W, WORLD_H, resolution=GRID_RES)
    explorer = RandomWalkExplorer(forward_speed=0.3, turn_speed=1.2,
                                  obstacle_threshold=0.10)
    display  = MapDisplay(WORLD_W, WORLD_H, cell_px=CELL_PX)

    sim_time = 0.0
    step_cnt = 0

    while robot.step(timestep) != -1:
        sim_time += dt
        step_cnt += 1

        # ── GPS pose (ground truth, used for everything) ──────────
        gv  = gps.getValues()
        cv  = comp.getValues()
        rx  = gv[0] + GPS_OFFSET_X   # East (offset: arena centered at origin)
        ry  = gv[1] + GPS_OFFSET_Y   # North
        rt  = compass_heading(cv)    # heading in world frame

        # ── LiDAR scan (raw angles, no offset yet) ────────────────
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
                scan.append((start + i * inc, r))   # raw angle, no offset

        # ── Camera update (every step — fast, just reads image) ────
        if 'camera' in dir() and camera:
            display.update_camera(camera)

        # ── Explorer (uses raw angles — raw 0° = forward = East) ──
        v_cmd, w_cmd = explorer.compute_control(scan)
        explorer.update(w_cmd, dt)

        vl = max(-MAX_SPEED, min(MAX_SPEED,
                 (v_cmd - w_cmd * WHEEL_BASE / 2) / WHEEL_RADIUS))
        vr = max(-MAX_SPEED, min(MAX_SPEED,
                 (v_cmd + w_cmd * WHEEL_BASE / 2) / WHEEL_RADIUS))
        lm.setVelocity(vl)
        rm.setVelocity(vr)

        # ── Occupancy grid update (offset applied here) ───────────
        if step_cnt % UPDATE_EVERY == 0 and scan:
            grid.update(rx, ry, rt, scan, MAX_LIDAR)

            display.update(grid,
                           (rx, ry, rt),   # use GPS as both "EKF" and true pose
                           (rx, ry, rt),
                           [],             # no landmarks shown (clean display)
                           [[0.001,0,0],[0,0.001,0],[0,0,0.001]],
                           sim_time)

        if step_cnt % 150 == 0:
            print(f"t={sim_time:.1f}s | GPS({rx:.2f},{ry:.2f}) "
                  f"heading={math.degrees(rt):.0f}°")

    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
