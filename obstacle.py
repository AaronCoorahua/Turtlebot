"""
Obstacle detection from LIDAR LaserScan data.

Coordinate convention (ROS REP-103):
  - angle 0    → frente del robot
  - angle +π/2 → izquierda
  - angle -π/2 → derecha

Detección de corredor (footprint-aware):
  Para cada lectura LIDAR a distancia r y ángulo θ se proyecta:
    fwd = r · cos(θ)   (componente hacia adelante)
    lat = r · sin(θ)   (componente lateral: + = izq, - = der)

  El robot ocupa un corredor de semi-ancho = robot_radius + lateral_margin.
  Una lectura es amenaza frontal si:
    |lat| < corridor_half  AND  0 < fwd < stopping_dist
  Una lectura es advertencia lateral si:
    |lat| < corridor_half * 1.5  AND  stopping_dist ≤ fwd < look_ahead_dist
"""
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ObstacleZone(Enum):
    CLEAR        = auto()
    FRONT        = auto()       # frena completamente → backup + giro
    FRONT_LEFT   = auto()       # más amenaza izq → backup + giro derecha
    FRONT_RIGHT  = auto()       # más amenaza der → backup + giro izquierda
    SIDE_LEFT    = auto()       # corrección suave a la derecha, sin frenar
    SIDE_RIGHT   = auto()       # corrección suave a la izquierda, sin frenar


@dataclass
class ScanData:
    angle_min: float
    angle_inc: float
    ranges: list[float]

    def angle_at(self, idx: int) -> float:
        return self.angle_min + idx * self.angle_inc

    def sector_min_reliable(self, center_deg: float, half_width_deg: float,
                            min_readings: int, min_range: float) -> Optional[float]:
        """
        Distancia mínima válida en el sector [center ± half_width].
        Filtra lecturas menores a min_range (reflecciones propias del robot).
        Exige al menos min_readings lecturas válidas antes de confiar en el resultado.
        """
        center = math.radians(center_deg)
        half   = math.radians(half_width_deg)
        lo = center - half
        hi = center + half

        valid = []
        for i, r in enumerate(self.ranges):
            if math.isfinite(r) and min_range < r:
                a = math.atan2(math.sin(self.angle_at(i)),
                               math.cos(self.angle_at(i)))
                if lo <= a <= hi:
                    valid.append(r)

        if len(valid) < min_readings:
            return None
        return min(valid)

    def sector_min_any(self, center_deg: float, half_width_deg: float,
                       min_range: float, max_d: float = 2.0) -> float:
        """
        Mínimo válido en sector. Devuelve max_d si no hay lecturas (nunca None).
        Usado para construir la observación panorámica de 8 sectores del agente RL.
        """
        center = math.radians(center_deg)
        half   = math.radians(half_width_deg)
        lo = center - half
        hi = center + half
        best = max_d
        for i, r in enumerate(self.ranges):
            if math.isfinite(r) and min_range < r < max_d:
                a = math.atan2(math.sin(self.angle_at(i)),
                               math.cos(self.angle_at(i)))
                if lo <= a <= hi:
                    best = min(best, r)
        return best


@dataclass
class ObstacleReport:
    zone: ObstacleZone         = ObstacleZone.CLEAR
    front_dist: Optional[float] = None
    left_dist:  Optional[float] = None
    right_dist: Optional[float] = None

    @property
    def is_front(self) -> bool:
        """Obstáculo al frente — requiere stop + backup + giro."""
        return self.zone in (ObstacleZone.FRONT,
                             ObstacleZone.FRONT_LEFT,
                             ObstacleZone.FRONT_RIGHT)

    @property
    def is_side_only(self) -> bool:
        """Solo obstáculo lateral — corrección suave sin frenar."""
        return self.zone in (ObstacleZone.SIDE_LEFT, ObstacleZone.SIDE_RIGHT)

    @property
    def is_clear(self) -> bool:
        return self.zone == ObstacleZone.CLEAR


class ObstacleDetector:
    """
    Detección de obstáculos con conciencia del footprint físico del robot.

    En lugar de medir sectores angulares fijos, proyecta cada lectura LIDAR
    al espacio (adelante, lateral) y comprueba si entra en el corredor que
    ocupa el robot durante su desplazamiento.
    """

    def __init__(self, cfg: dict):
        obs = cfg["obstacle"]
        robot_r  = obs["robot_radius"]
        lat_marg = obs["lateral_margin"]

        self._corridor_half   = robot_r + lat_marg           # semiancho del corredor estricto
        self._corridor_wide   = self._corridor_half * 1.5    # corredor amplio para advertencias
        self._stopping_dist   = obs["stopping_dist"]          # frena dentro del corredor estricto
        self._look_ahead_dist = obs["look_ahead_dist"]        # corrige dentro del corredor amplio
        self._side_min_dist   = obs["side_min_dist"]          # pared pura a ±90°
        self._min_range       = obs["min_range"]
        self._min_threats     = obs["min_threat_count"]       # lecturas mínimas para confirmar
        self._side_cone       = obs["side_cone_deg"]
        self._min_valid       = obs["min_valid_readings"]

    def analyze(self, scan: ScanData) -> ObstacleReport:
        """
        Escanea todo el rango frontal (±150°) y clasifica lecturas en:
          - Zona frontal de emergencia  → is_front = True  → backup + giro
          - Zona de advertencia lateral → is_side_only = True → corrección suave
          - Claro                       → is_clear = True
        """
        report = ObstacleReport()

        front_left_count  = 0
        front_right_count = 0
        look_left_count   = 0
        look_right_count  = 0
        min_front_fwd     = float("inf")

        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r <= self._min_range:
                continue

            a   = math.atan2(math.sin(scan.angle_at(i)),
                             math.cos(scan.angle_at(i)))
            fwd = r * math.cos(a)   # componente hacia adelante (+ = frente)
            lat = r * math.sin(a)   # componente lateral (+ = izq, - = der)

            if fwd <= 0:
                continue  # detrás del robot, ignorar

            in_strict = abs(lat) <= self._corridor_half
            in_wide   = abs(lat) <= self._corridor_wide

            # Solo considerar lecturas con componente frontal significativa.
            # Lecturas casi perpendiculares (≈90°) tienen fwd ≈ 0, lo que las
            # haría caer en la zona de emergencia incorrectamente; las manejamos
            # con sector_min_reliable (pure side check) más abajo.
            if fwd < self._min_range:
                continue

            if in_strict and fwd < self._stopping_dist:
                # Zona de emergencia: el robot CHOCARÁ si no para
                if lat >= 0:
                    front_left_count += 1
                else:
                    front_right_count += 1
                if fwd < min_front_fwd:
                    min_front_fwd = fwd

            elif in_wide and fwd < self._look_ahead_dist:
                # Zona de advertencia: hay margen para girar suavemente
                if lat >= 0:
                    look_left_count += 1
                else:
                    look_right_count += 1

        # Sectores puros laterales (paredes directas a ±90°)
        pure_left  = scan.sector_min_reliable(
            90,  self._side_cone, self._min_valid, self._min_range)
        pure_right = scan.sector_min_reliable(
            -90, self._side_cone, self._min_valid, self._min_range)

        report.front_dist = min_front_fwd if min_front_fwd < float("inf") else None
        report.left_dist  = pure_left
        report.right_dist = pure_right

        front_count   = front_left_count + front_right_count
        front_blocked = front_count >= self._min_threats

        look_count  = look_left_count + look_right_count
        look_warns  = look_count >= self._min_threats

        left_side  = pure_left  is not None and pure_left  < self._side_min_dist
        right_side = pure_right is not None and pure_right < self._side_min_dist

        if front_blocked:
            # Girar hacia el lado con menos amenazas en el corredor
            if front_left_count >= front_right_count:
                report.zone = ObstacleZone.FRONT_LEFT   # más amenaza izq → girar derecha
            else:
                report.zone = ObstacleZone.FRONT_RIGHT  # más amenaza der → girar izquierda

        elif look_warns:
            if look_left_count > look_right_count:
                report.zone = ObstacleZone.SIDE_LEFT
            elif look_right_count > look_left_count:
                report.zone = ObstacleZone.SIDE_RIGHT
            elif left_side:
                report.zone = ObstacleZone.SIDE_LEFT
            elif right_side:
                report.zone = ObstacleZone.SIDE_RIGHT

        elif left_side:
            report.zone = ObstacleZone.SIDE_LEFT
        elif right_side:
            report.zone = ObstacleZone.SIDE_RIGHT
        else:
            report.zone = ObstacleZone.CLEAR

        return report
