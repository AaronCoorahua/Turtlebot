# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Autonomous visual navigation for a TurtleBot4 robot that follows a circuit reacting to B&W printed signs:
- **STOP sign** (prohibition circle) — stop for 5 seconds, then continue
- **DER/IZQ signs** (curved arrows in circle) — wait going straight until the lateral LIDAR clears, then turn 90°
- **QR codes** — detect and register up to 3 unique checkpoint codes

## Running the Project

```bash
# Activate virtualenv (always required)
source venv/bin/activate.fish   # fish shell
# or
source venv/bin/activate        # bash/zsh

# Receive telemetry (camera + LIDAR) from robot
python recibidor_datos.py

# Keyboard teleop control (Linux only, uses termios)
python control_linux.py

# Main autonomous controller (to be created)
python main.py
```

## Network & Protocol

**Robot:** `192.168.0.103` | Lab WiFi: `Lab_Computech_5G`  
**SSH access (on turtlebot4 network):** `ssh ubuntu@10.42.0.1` — password: `turtlebot4`  
**ROS_DOMAIN_ID:** `4`

Two separate UDP channels:
- **Port 6000** — robot → client: telemetry (SCAN and IMG messages), requires handshake
- **Port 5007** — client → robot: velocity commands as `struct.pack("ff", linear, angular)`

**Handshake protocol** (`recibidor_datos.py`): client sends `HELLO <domain_id> <pairing_code>`, robot replies `ACK <domain_id> <robot_name>`. Constants: `PAIRING_CODE = "ROBOT_A_42"`, `EXPECTED_ROBOT_NAME = "turtlebot4_lite_1"`.

**Telemetry message formats:**
- `SCAN <domain_id> <robot_name> <sec> <nsec> <angle_min> <angle_inc> <n> r1 ... rn`
- `IMG <domain_id> <robot_name> <sec> <nsec> <base64_jpeg>`

## Architecture

```
recibidor_datos.py   ← UDP port 6000 ← Robot (camera + LIDAR)
control_linux.py     → UDP port 5007 → Robot (cmd_vel: linear, angular)
vision.py            ← HSV-based color detection (traffic lights, arrows)
vision_signs.py      ← ORB template matching for B&W printed signs (DER/IZQ/STOP)
obstacle.py          ← LIDAR corridor analysis + ScanData helpers
rl_agent.py          ← PPO online learning agent (Stable-Baselines3)
main.py              ← entry point: FSM, decision hierarchy, UDP send
config.yaml          ← all tunable parameters
```

**Constraint:** `recibidor_datos.py` and `control_linux.py` are base components — do not modify their internal structure.

## FSM Decision Hierarchy (`main.py`)

Priority order in `_decide()` (highest first):

1. **OBSTACLE** state tick — backup + turn recovery
2. `obs_report.is_front` — frontal collision imminent → start recovery
3. **STOPPED** state tick — stop sign timer or red light timer expires → MOVING
4. **PENDING_DER / PENDING_IZQ / TURNING_DER / TURNING_IZQ** — sign-based turn logic (`_tick_sign_turn`)
5. Red traffic light detected → STOPPED
6. STOP sign detected (`vision_signs`) → STOPPED for 5s
7. DER/IZQ sign detected → PENDING_DER / PENDING_IZQ
8. Lateral obstacle soft correction
9. RL agent or arrow fallback

## Sign Detection (`vision_signs.py`)

`SignDetector` uses a two-stage pipeline:
1. `HoughCircles` finds the circular sign border in the frame → crops + resizes to 200×200
2. ORB descriptor matching (ratio test 0.75) against templates loaded from `fotos/`

Template files: `fotos/der.jpeg`, `fotos/izq.jpeg`, `fotos/stop1.jpeg`, `fotos/stop2.jpeg`

Tune `signs.orb_min_matches` in `config.yaml` if getting false positives (raise) or misses (lower).

## LIDAR Gate for Turns

In `PENDING_DER` / `PENDING_IZQ`, the robot goes straight while `ScanData.sector_min_reliable(±90°)` returns a distance below `signs.clear_threshold` (0.50 m). When the side opens up (returns `None` or exceeds threshold), it transitions to `TURNING_DER` / `TURNING_IZQ` and spins for `signs.turn_duration_90` seconds (≈ π/2 ÷ angular_speed).

## Config (`config.yaml`)

All constants go here — never hardcode in logic files:
- `signs.stop_duration`, `signs.clear_threshold`, `signs.turn_duration_90`, `signs.orb_min_matches`
- `obstacle.side_cone_deg`, `obstacle.min_valid_readings`, `obstacle.min_range`
- `movement.linear_speed`, `movement.angular_speed`
- `robot.ip`, `robot.data_port`, `robot.control_port`

## Environment

- Python 3.14, venv at `venv/`
- Installed: `opencv-python`, `numpy`, `PyQt5`
- Still needed: `stable-baselines3`, `gymnasium`, `pyzbar`, `pyyaml`
- Force X11 backend on Wayland: `os.environ["QT_QPA_PLATFORM"] = "xcb"` (already in `recibidor_datos.py`)

## Robot Setup (one-time)

```bash
ros2 launch turtlebot4_bringup lite.launch.py   # main bringup
ros2 launch turtlebot4_bringup rplidar.launch.py # LIDAR
ros2 launch turtlebot4_bringup oakd.launch.py    # camera
```
