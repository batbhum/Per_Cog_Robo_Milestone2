# Perception of Cognitive Robots: Milestone 2 🤖🗺️

**An autonomous, from-scratch SLAM, Path Planning, and Computer Vision pipeline for the Webots E-puck robot.**

This repository contains the Milestone 2 submission for the Perception of Cognitive Robots course. The core philosophy of this project is **algorithmic purity**: no external robotics, path-planning, or computer vision libraries (such as ROS, OpenCV, Scikit-Image, or GTSAM) were used for the mathematical logic. Everything from ray-tracing and state estimation to color space conversion was built entirely from scratch using standard Python `math` and `numpy`.

---

## 🌟 Key Features & System Architecture

### 1. Probabilistic Mapping & SLAM
* **Log-Odds Occupancy Grid:** Calculates Bayesian log-odds probabilities to build a robust 2D map, filtering out sensor noise over time.
* **Custom Ray-Tracing:** Implements Bresenham's Line Algorithm from scratch to trace LiDAR rays and accurately mark "free space" between the robot and detected obstacles.
* **Extended Kalman Filter (EKF) Fundamentals:** Engineered matrix math (Jacobians, covariance updates, Mahalanobis distance) to track probabilistic landmarks and manage pose uncertainty.

### 2. Autonomous Navigation (A* & Exploration)
* **A* (A-Star) Global Path Planning:** A purely mathematical grid planner that calculates the optimal route to the goal. It intelligently treats "unknown" (unmapped) space as walkable, driving the robot to actively explore the maze.
* **C-Space Inflation:** Solves the "1-Pixel Robot Fallacy" by artificially inflating mapped walls by 2 grid cells. This forces the A* algorithm to route the physical E-puck safely down the center of hallways rather than scraping corners.
* **Reactive Emergency Override:** A low-level hardware safety loop that constantly monitors raw LiDAR. If the robot gets snagged on a tight corner (breaching a 5cm threshold), it temporarily bypasses A*, forcing a 0.6-second blind reverse to safely un-stick itself before requesting a new path.

### 3. Computer Vision & Feature Extraction
* **Split-and-Merge Algorithm:** Recursively processes and clusters raw LiDAR point clouds to extract straight line segments and definitively identify geometric landmarks.
* **Dynamic Object Tracking (Discrepancy Detection):** Compares live LiDAR hits against the established static occupancy grid. Objects detected in known "free space" are flagged as dynamic, tracked in real-time, and excluded from permanently staining the map.
* **Vectorized Color Space Conversion:** Converts raw RGB camera feeds to HSV space to isolate and track specific emissive objects (e.g., the yellow goal and moving balls) using custom thresholding.

### 4. Custom PyGame UI Dashboard
* **Dual-Panel Live Rendering:** A synchronized UI that translates the robot's Webots ENU coordinates to a 2D graphical display.
* Plots the real-time occupancy grid, the robot's A* trajectory, moving object clusters (rendered in red), and an active camera feed with bounding boxes over detected targets.

---

## 📂 File Structure

* `slam_controller.py` - The main execution loop. Initializes hardware, grabs sensor data, and integrates the mapping and exploration modules.
* `occupancy_grid.py` - Handles the log-odds grid mathematics and Bresenham ray-tracing.
* `exploration.py` - Manages the state machine for autonomous movement and wall-following/obstacle avoidance.
* `ekf_slam.py` - Contains the Extended Kalman Filter prediction and update steps for landmark-based SLAM.
* `landmark_extraction.py` - Processes LiDAR scans using the Split-and-Merge algorithm.
* `camera_display.py` / `map_display.py` - Renders the PyGame UI, occupancy maps, and custom HSV object detection bounding boxes.
* `utils.py` - Core mathematical helper functions (angle normalization, Euclidean distance, algebraic line intersections).

---

## 🚀 How to Run the Simulation

**1. Prerequisites:**
You will need Webots (R2025a or compatible) installed on your machine. You also need to install the two required Python libraries for math and UI rendering:
```bash
pip install numpy pygame
```
2. Launching:

Open the .wbt world file located in the worlds directory using Webots.

Ensure the E-puck robot's controller is set to slam_controller.

Hit the Play button in Webots. The PyGame dashboard will launch automatically, and the robot will begin its autonomous mapping and exploration!

👥 Team Members
Panapon (6638114021)

Kittibhum (6638018521)

Nuntis (6638103121)

Course: 2147331.i Perception of Cognitive Robots - Chulalongkorn University
<img width="1160" height="1148" alt="Screenshot 2026-03-20 233308" src="https://github.com/user-attachments/assets/c33ca6b7-8f6d-49ac-ba30-61792461c0d1" />
<img width="1662" height="1219" alt="Screenshot 2026-03-20 233250" src="https://github.com/user-attachments/assets/debd1f04-f8f8-4dab-859c-3136575c72f0" />
<img width="1281" height="236" alt="Screenshot 2026-03-20 233231" src="https://github.com/user-attachments/assets/4b63f134-9f39-4e02-bff3-4f5473c3bcac" />
<img width="1415" height="820" alt="Screenshot 2026-03-20 233224" src="https://github.com/user-attachments/assets/5a11cd19-3d31-46df-8a94-9efe3ec9c0fd" />
<img width="2559" height="1520" alt="Screenshot 2026-03-20 233218" src="https://github.com/user-attachments/assets/75344ba4-88e2-48f4-a51d-374b78a29c35" />
<img width="2559" height="1532" alt="Screenshot 2026-03-20 233147" src="https://github.com/user-attachments/assets/ad0e839c-c0ec-41c4-8cc0-a97ed700450f" />
<img width="1447" height="813" alt="Screenshot 2026-03-20 221231" src="https://github.com/user-attachments/assets/941d214f-3c89-4615-9219-cf00c3c30837" />
<img width="959" height="950" alt="Screenshot 2026-03-20 220437" src="https://github.com/user-attachments/assets/6ba0c376-5c3f-4ad1-9236-64cbed24cb9f" />
<img width="324" height="672" alt="Screenshot 2026-03-20 233328" src="https://github.com/user-attachments/assets/70035596-6518-49e5-9f40-f5ce127d6909" />
