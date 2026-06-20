"""
Observation building, action definitions, and reward computation for the RL agent.

This module is intentionally NOT a full gymnasium.Env subclass — the robot runs
in real-time and we manage the RL loop manually in rl_agent.py.

Observación de 8 sectores LIDAR panorámicos:
  Índice  Sector    Descripción
  ------  --------  ----------------------------
    0       0°      Frente
    1      45°      Frente-izquierda (diagonal)
    2      90°      Izquierda
    3     135°      Atrás-izquierda
    4     180°      Atrás
    5    -135°      Atrás-derecha
    6     -90°      Derecha
    7     -45°      Frente-derecha (diagonal)

Los diagonales son críticos: el robot a alta velocidad clips esquinas que el
sector frontal estrecho no detectaría a tiempo.
"""
import math
import numpy as np
import cv2
from enum import IntEnum
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from vision import TrafficState, ArrowDir
from obstacle import ObstacleReport, ScanData


# ── Action space ──────────────────────────────────────────────────────────────

class RLAction(IntEnum):
    FORWARD       = 0
    GENTLE_LEFT   = 1
    GENTLE_RIGHT  = 2
    SHARP_LEFT    = 3
    SHARP_RIGHT   = 4

    @property
    def label(self) -> str:
        return self.name


def action_to_vel(action: "RLAction", cfg: dict) -> tuple[float, float]:
    """Returns (linear m/s, angular rad/s) for the given action."""
    v, w = cfg["movement"]["rl_actions"][int(action)]
    return float(v), float(w)


# ── Observation ───────────────────────────────────────────────────────────────

_PANORAMIC_SECTORS = [0, 45, 90, 135, 180, -135, -90, -45]  # grados
_SECTOR_HALF_DEG   = 20   # ± para cada sector


def obs_space(img_size: int) -> spaces.Dict:
    return spaces.Dict({
        "image":   spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8),
        "lidar":   spaces.Box(0.0, 1.0, (8,),  dtype=np.float32),  # 8 sectores panorámicos
        "traffic": spaces.Box(0.0, 1.0, (3,),  dtype=np.float32),
        "arrow":   spaces.Box(0.0, 1.0, (4,),  dtype=np.float32),
    })


def act_space() -> spaces.Discrete:
    return spaces.Discrete(len(RLAction))


_TRAFFIC_IDX = {TrafficState.RED: 0, TrafficState.GREEN: 1, TrafficState.UNKNOWN: 2}
_ARROW_IDX   = {ArrowDir.LEFT: 0, ArrowDir.RIGHT: 1,
                ArrowDir.STRAIGHT: 2, ArrowDir.UNKNOWN: 3}

_LIDAR_MAX_D = 2.0   # distancia máxima (normalización)
_LIDAR_MIN_R = 0.10  # ignorar reflecciones propias


def build_obs(frame: np.ndarray, obs_report: ObstacleReport,
              traffic: TrafficState, arrow: ArrowDir,
              img_size: int, scan: Optional[ScanData] = None) -> dict:
    img = cv2.resize(frame, (img_size, img_size))

    if scan is not None:
        # Panorámica de 8 sectores: cobertura completa del entorno del robot
        lidar = np.array([
            scan.sector_min_any(s, _SECTOR_HALF_DEG, _LIDAR_MIN_R, _LIDAR_MAX_D) / _LIDAR_MAX_D
            for s in _PANORAMIC_SECTORS
        ], dtype=np.float32)
    else:
        # Fallback cuando no hay scan: expandir ObstacleReport a 8 valores
        f = min(obs_report.front_dist or _LIDAR_MAX_D, _LIDAR_MAX_D) / _LIDAR_MAX_D
        l = min(obs_report.left_dist  or _LIDAR_MAX_D, _LIDAR_MAX_D) / _LIDAR_MAX_D
        r = min(obs_report.right_dist or _LIDAR_MAX_D, _LIDAR_MAX_D) / _LIDAR_MAX_D
        fl = min(f, l * 0.7 + f * 0.3)   # estimación diagonal frontal-izq
        fr = min(f, r * 0.7 + f * 0.3)   # estimación diagonal frontal-der
        # [F, FL, L, RL, Rear, RR, R, FR]
        lidar = np.array([f, fl, l, 1.0, 1.0, 1.0, r, fr], dtype=np.float32)

    traffic_oh = np.zeros(3, dtype=np.float32)
    traffic_oh[_TRAFFIC_IDX[traffic]] = 1.0

    arrow_oh = np.zeros(4, dtype=np.float32)
    arrow_oh[_ARROW_IDX[arrow]] = 1.0

    return {"image": img, "lidar": lidar, "traffic": traffic_oh, "arrow": arrow_oh}


# ── Reward ────────────────────────────────────────────────────────────────────

def compute_reward(action: RLAction,
                   obs_report: ObstacleReport,
                   traffic: TrafficState,
                   arrow: ArrowDir,
                   new_checkpoint: bool,
                   cfg: dict) -> tuple[float, bool]:
    """
    Returns (reward, done).
    done=True señala fin de episodio virtual (colisión o timeout).
    El robot sigue corriendo físicamente; no hay teleport.

    Filosofía de recompensa:
    - FORWARD a máxima velocidad recibe el mayor bono → el agente aprende a ir rápido.
    - Penalización de proximidad cuadrática: el agente aprende a anticipar, no
      a reaccionar solo cuando ya está tocando la pared.
    - Checkpoint QR: objetivo competitivo principal.
    """
    r   = cfg["rl"]["reward"]
    obs = cfg["obstacle"]
    reward = 0.0
    done   = False

    # ── Avance y velocidad ────────────────────────────────────────────────────
    if action == RLAction.FORWARD:
        # Bono máximo por ir a máxima velocidad
        reward += r["forward_step"] + r.get("speed_bonus", 0.0)
    elif action in (RLAction.GENTLE_LEFT, RLAction.GENTLE_RIGHT):
        # Progreso parcial mientras se curva
        reward += r["forward_step"] * 0.7
    else:
        # Giro en seco: pierde velocidad, penalizar
        reward -= r["spin_penalty"]

    # ── Proximidad al obstáculo (cuadrática) ─────────────────────────────────
    front_d   = obs_report.front_dist or _LIDAR_MAX_D
    safe_dist = obs.get("stopping_dist", obs.get("front_min_dist", 0.38))

    if front_d < safe_dist:
        ratio   = (safe_dist - front_d) / safe_dist   # 0 → 1 cuanto más cerca
        # Penalización cuadrática: mucho más agresiva cerca de la colisión
        reward -= r["proximity_penalty"] * (ratio ** 1.5)

    # ── Colisión: el robot está dentro de su propia huella → episodio termina ─
    collision_threshold = safe_dist * 0.50
    if obs_report.front_dist is not None and obs_report.front_dist < collision_threshold:
        reward -= r["collision_penalty"]
        done    = True

    # ── Seguimiento de flechas ────────────────────────────────────────────────
    if arrow != ArrowDir.UNKNOWN:
        correct = (
            (arrow == ArrowDir.LEFT    and action == RLAction.SHARP_LEFT)   or
            (arrow == ArrowDir.RIGHT   and action == RLAction.SHARP_RIGHT)  or
            (arrow == ArrowDir.STRAIGHT and action == RLAction.FORWARD)
        )
        reward += r["arrow_reward"] if correct else -r["arrow_penalty"]

    # ── Semáforo ──────────────────────────────────────────────────────────────
    if traffic == TrafficState.RED:
        if action in (RLAction.FORWARD, RLAction.GENTLE_LEFT, RLAction.GENTLE_RIGHT):
            reward -= r["red_light_penalty"]

    # ── Checkpoint QR ─────────────────────────────────────────────────────────
    if new_checkpoint:
        reward += r["checkpoint_reward"]

    return reward, done


# ── Conversión de formato de observación ─────────────────────────────────────
#
# SB3's DictRolloutBuffer guarda imágenes en formato channel-first con dim de batch:
#   predict()        espera  {"image": (H, W, C),    "lidar": (D,), ...}
#   rollout_buf.add() espera  {"image": (1, C, H, W), "lidar": (1, D), ...}

def to_buffer_obs(obs: dict) -> dict:
    """Convierte obs de formato predict a formato rollout buffer (SB3)."""
    result = {}
    for k, v in obs.items():
        if k == "image" and v.ndim == 3:          # (H, W, C) → (1, C, H, W)
            result[k] = v.transpose(2, 0, 1)[np.newaxis]
        else:                                      # (D,) → (1, D)
            result[k] = v[np.newaxis]
    return result


# ── Dummy env (solo para inicializar PPO en SB3) ─────────────────────────────

class _DummyEnv(gym.Env):
    """Nunca se ejecuta realmente — solo para que SB3 infiera los espacios."""
    def __init__(self, img_size: int):
        self.observation_space = obs_space(img_size)
        self.action_space      = act_space()

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}
