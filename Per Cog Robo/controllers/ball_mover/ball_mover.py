"""
ball_mover.py - Bouncing ball controller (ENU / Z-up coordinate system).

In ENU: X=East, Y=North, Z=Up
Balls move in the X-Y plane. Z stays constant (= ball radius = 0.03).
Arena spans X=[0,2], Y=[0,2].
"""
from controller import Supervisor
import sys

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    try:
        vx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.22
        vy = float(sys.argv[2]) if len(sys.argv) > 2 else 0.17
    except Exception:
        vx, vy = 0.22, 0.17

    self_node   = robot.getSelf()
    trans_field = self_node.getField("translation")

    pos = trans_field.getSFVec3f()
    x, y, z = pos[0], pos[1], pos[2]

    BALL_R   = 0.03
    WALL_MIN = 0.02 + BALL_R
    WALL_MAX = 1.98 - BALL_R

    print(f"[ball_mover] ENU start ({x:.2f}, {y:.2f}), v=({vx:.2f}, {vy:.2f})")

    while robot.step(timestep) != -1:
        x += vx * dt
        y += vy * dt

        # Bounce off arena walls (X and Y axes)
        if x < WALL_MIN: x = WALL_MIN; vx =  abs(vx)
        if x > WALL_MAX: x = WALL_MAX; vx = -abs(vx)
        if y < WALL_MIN: y = WALL_MIN; vy =  abs(vy)
        if y > WALL_MAX: y = WALL_MAX; vy = -abs(vy)

        trans_field.setSFVec3f([x, y, z])

if __name__ == "__main__":
    main()
