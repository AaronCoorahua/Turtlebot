"""
Autonomous controller for TurtleBot4.

Jerarquía de decisión (de mayor a menor prioridad):
  1. Obstáculo frontal crítico → backup + giro (regla dura, siempre activa)
  2. Semáforo rojo             → detener (regla dura)
  3. Agente RL                 → elige acción de movimiento
     - si rl.enabled = false → fallback: flechas + avance recto

El RL aprende online: cada paso de control alimenta la experiencia al replay
buffer y entrena periódicamente en background, persistiendo entre sesiones.
"""
import os
import base64
import logging
import socket
import struct
import threading
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import numpy as np
import yaml
import cv2

os.environ["QT_QPA_PLATFORM"] = "xcb"

from vision import VisionProcessor, TrafficState, ArrowDir
from vision_signs import SignDetector
from obstacle import ObstacleDetector, ObstacleZone, ScanData, ObstacleReport
from rl_agent import RLAgent
from rl_env import RLAction


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("tb4")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%H:%M:%S")
    fh = logging.FileHandler(f"logs/run_{ts}.txt")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


logger = setup_logger()


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── FSM ───────────────────────────────────────────────────────────────────────

class State(Enum):
    MOVING       = auto()   # RL controla el movimiento
    STOPPED      = auto()   # señal STOP o semáforo rojo
    OBSTACLE     = auto()   # recuperación de obstáculo frontal (regla dura)
    PENDING_DER  = auto()   # vio señal DER, va recto hasta que costado derecho se libere
    PENDING_IZQ  = auto()   # vio señal IZQ, va recto hasta que costado izquierdo se libere
    TURNING_DER  = auto()   # ejecutando giro 90° a la derecha
    TURNING_IZQ  = auto()   # ejecutando giro 90° a la izquierda


# ── Controller ────────────────────────────────────────────────────────────────

class AutonomousController:
    def __init__(self, cfg: dict):
        self._cfg     = cfg
        self._robot   = cfg["robot"]
        self._mv      = cfg["movement"]
        self._tl_cfg  = cfg["traffic_light"]
        self._obs_cfg = cfg["obstacle"]
        self._rl_cfg  = cfg["rl"]

        self._vision   = VisionProcessor(cfg)
        self._obstacle = ObstacleDetector(cfg)

        self._use_rl = self._rl_cfg["enabled"]
        if self._use_rl:
            self._rl = RLAgent(cfg)
            logger.info("[MAIN] RL agent activo — aprendizaje online habilitado")
        else:
            self._rl = None
            logger.info("[MAIN] RL desactivado — usando reglas de flechas")

        self._rl_step_interval = self._rl_cfg.get("step_interval", 5)
        self._rl_counter       = 0
        self._last_rl_action   = RLAction.FORWARD
        self._last_rl_v        = 0.0
        self._last_rl_w        = 0.0

        # FSM
        self._state: State = State.MOVING
        self._red_until: float = 0.0
        self._obs_backup_until: float = 0.0
        self._obs_turn_until:   float = 0.0
        self._obs_turn_w: float = 0.0

        # Sign detection
        self._sign_detector = SignDetector(cfg.get("signs", {}))
        self._sign_cfg      = cfg.get("signs", {})
        self._turn_until:   float = 0.0

        # Sensor data
        self._frame: np.ndarray | None = None
        self._scan:  ScanData   | None = None
        self._lock = threading.Lock()

        self._running = True
        self._ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ctrl_addr = (self._robot["ip"], self._robot["control_port"])

    # ── Velocity ──────────────────────────────────────────────────────────────

    def _send_vel(self, v: float, w: float):
        self._ctrl_sock.sendto(struct.pack("ff", float(v), float(w)),
                               self._ctrl_addr)

    def _stop(self):
        self._send_vel(0.0, 0.0)

    # ── Handshake ──────────────────────────────────────────────────────────────

    @staticmethod
    def _do_handshake(sock, addr, domain_id, pairing_code, expected_name):
        sock.settimeout(1.0)
        logger.info(f"Handshake → {addr}")
        while True:
            sock.sendto(f"HELLO {domain_id} {pairing_code}".encode(), addr)
            try:
                data, _ = sock.recvfrom(4096)
                parts = data.decode().strip().split()
                if (len(parts) >= 3 and parts[0] == "ACK"
                        and int(parts[1]) == domain_id
                        and " ".join(parts[2:]) == expected_name):
                    logger.info(f"Paired with '{expected_name}'")
                    sock.settimeout(0.033)
                    return
                logger.warning("Handshake mismatch, retrying…")
            except socket.timeout:
                logger.warning("Handshake timeout, retrying…")
            except KeyboardInterrupt:
                raise

    # ── Receiver thread ────────────────────────────────────────────────────────

    def _receiver_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._do_handshake(sock,
                           (self._robot["ip"], self._robot["data_port"]),
                           self._robot["domain_id"],
                           self._robot["pairing_code"],
                           self._robot["name"])
        logger.info("Receiving telemetry…")
        while self._running:
            try:
                data, _ = sock.recvfrom(65535)
                text  = data.decode("utf-8", errors="ignore")
                parts = text.split()
                if not parts:
                    continue
                if parts[0] == "IMG":
                    self._handle_img(parts)
                elif parts[0] == "SCAN":
                    self._handle_scan(parts)
            except socket.timeout:
                pass
            except Exception as e:
                logger.error(f"Recv: {e}")
        sock.close()

    def _handle_img(self, parts):
        if len(parts) < 6:
            return
        try:
            arr = np.frombuffer(base64.b64decode(" ".join(parts[5:])),
                                dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self._lock:
                    self._frame = img
        except Exception as e:
            logger.error(f"IMG: {e}")

    def _handle_scan(self, parts):
        if len(parts) < 9:
            return
        try:
            angle_min = float(parts[5])
            angle_inc = float(parts[6])
            n         = int(parts[7])
            ranges    = [float(r) for r in parts[8: 8 + n]]
            with self._lock:
                self._scan = ScanData(angle_min, angle_inc, ranges)
        except Exception as e:
            logger.error(f"SCAN: {e}")

    # ── Obstacle recovery (frontal) ────────────────────────────────────────────

    def _start_front_recovery(self, report: ObstacleReport):
        ang = self._mv["angular_speed"]
        self._obs_turn_w = (+ang
                            if report.zone in (ObstacleZone.FRONT_RIGHT,
                                               ObstacleZone.SIDE_RIGHT)
                            else -ang)
        now = time.time()
        self._obs_backup_until = now + self._obs_cfg["backup_seconds"]
        self._obs_turn_until   = self._obs_backup_until + self._obs_cfg["turn_seconds"]
        self._state = State.OBSTACLE
        fd = f"{report.front_dist:.2f}m" if report.front_dist is not None else "--"
        logger.warning(
            f"OBSTACLE {report.zone.name} | "
            f"front={fd} "
            f"L={report.left_dist} R={report.right_dist}")

    def _tick_obstacle(self):
        now = time.time()
        if now < self._obs_backup_until:
            self._send_vel(self._obs_cfg["backup_speed"], 0.0)
            return
        if now < self._obs_turn_until:
            self._send_vel(0.0, self._obs_turn_w)
            return
        logger.info("Obstacle recovery done → MOVING")
        self._state = State.MOVING

    # ── Sign turn states ───────────────────────────────────────────────────────

    def _tick_sign_turn(self, scan: ScanData | None):
        lin     = self._mv["linear_speed"]
        ang_spd = self._sign_cfg.get("turn_angular_speed", self._mv["angular_speed"])
        turn_dur = self._sign_cfg.get("turn_duration_90", 0.87)
        clear_thr = self._sign_cfg.get("clear_threshold", 0.50)

        if self._state == State.PENDING_DER:
            if scan is not None:
                d = scan.sector_min_reliable(
                    -90,
                    self._obs_cfg["side_cone_deg"],
                    self._obs_cfg["min_valid_readings"],
                    self._obs_cfg["min_range"])
                if d is None or d > clear_thr:
                    self._turn_until = time.time() + turn_dur
                    self._state = State.TURNING_DER
                    logger.info(f"Right clear ({d}) → TURNING_DER")
                    return
            self._send_vel(lin, 0.0)

        elif self._state == State.PENDING_IZQ:
            if scan is not None:
                d = scan.sector_min_reliable(
                    90,
                    self._obs_cfg["side_cone_deg"],
                    self._obs_cfg["min_valid_readings"],
                    self._obs_cfg["min_range"])
                if d is None or d > clear_thr:
                    self._turn_until = time.time() + turn_dur
                    self._state = State.TURNING_IZQ
                    logger.info(f"Left clear ({d}) → TURNING_IZQ")
                    return
            self._send_vel(lin, 0.0)

        elif self._state == State.TURNING_DER:
            if time.time() < self._turn_until:
                self._send_vel(0.05, -ang_spd)
            else:
                self._state = State.MOVING
                logger.info("TURNING_DER done → MOVING")

        elif self._state == State.TURNING_IZQ:
            if time.time() < self._turn_until:
                self._send_vel(0.05, +ang_spd)
            else:
                self._state = State.MOVING
                logger.info("TURNING_IZQ done → MOVING")

    # ── Decision logic ─────────────────────────────────────────────────────────

    def _decide(self, frame: np.ndarray, scan: ScanData | None):
        result     = self._vision.process(frame, debug=True)
        obs_report = self._obstacle.analyze(scan) if scan else ObstacleReport()
        sign       = self._sign_detector.detect(frame)

        if result.debug_frame is not None:
            cv2.imshow("Vision", result.debug_frame)
        self._draw_obstacle_hud(obs_report)

        new_checkpoint = result.qr_data is not None
        if new_checkpoint:
            logger.info(
                f"QR: '{result.qr_data}' "
                f"({self._vision.checkpoint_count}/{self._vision._max_checkpoints})")

        now = time.time()

        # ── 1. Obstáculo frontal — regla dura ────────────────────────────────
        if self._state == State.OBSTACLE:
            self._tick_obstacle()
            return

        if obs_report.is_front:
            self._stop()
            self._start_front_recovery(obs_report)
            return

        # ── 2. Timer STOPPED — señal STOP o semáforo rojo ────────────────────
        if self._state == State.STOPPED:
            if now >= self._red_until and result.traffic != TrafficState.RED:
                logger.info("STOPPED timer expired → MOVING")
                self._state = State.MOVING
            else:
                if result.traffic == TrafficState.RED:
                    self._red_until = now + self._tl_cfg["red_stop_seconds"]
                self._stop()
            return

        # ── 3. Estados de giro por señal ──────────────────────────────────────
        if self._state in (State.PENDING_DER, State.PENDING_IZQ,
                           State.TURNING_DER,  State.TURNING_IZQ):
            self._tick_sign_turn(scan)
            return

        # ── 4. Semáforo rojo detectado ────────────────────────────────────────
        if result.traffic == TrafficState.RED:
            logger.info("RED → STOPPED")
            self._stop()
            self._red_until = now + self._tl_cfg["red_stop_seconds"]
            self._state = State.STOPPED
            return

        # ── 5. Señal STOP detectada — detener 5 segundos ─────────────────────
        if sign == "STOP" and self._state == State.MOVING:
            logger.info("STOP sign → STOPPED 5s")
            self._stop()
            self._red_until = now + self._sign_cfg.get("stop_duration", 5.0)
            self._state = State.STOPPED
            return

        # ── 6. Señales de giro ────────────────────────────────────────────────
        if self._state == State.MOVING:
            if sign == "DER":
                logger.info("DER sign → PENDING_DER")
                self._state = State.PENDING_DER
            elif sign == "IZQ":
                logger.info("IZQ sign → PENDING_IZQ")
                self._state = State.PENDING_IZQ

        # ── 7. Corrección lateral suave (sin frenar) ──────────────────────────
        side_w = 0.0
        if obs_report.is_side_only:
            w_corr = self._obs_cfg["side_correct_w"]
            if obs_report.zone == ObstacleZone.SIDE_LEFT:
                side_w = -w_corr
            else:
                side_w = +w_corr

        # ── 8. Movimiento: RL o fallback de flechas ───────────────────────────
        if self._use_rl:
            self._rl_step(frame, obs_report, result, new_checkpoint, side_w, scan)
        else:
            self._arrow_fallback(result, side_w)

    def _rl_step(self, frame, obs_report, result, new_checkpoint, side_w, scan):
        """Llama al agente RL cada step_interval pasos de control."""
        self._rl_counter += 1

        # Siempre ejecutar observe() para alimentar el buffer de experiencias
        self._rl.observe(frame, obs_report, result.traffic, result.arrow,
                         new_checkpoint, scan=scan)

        # Cada step_interval pasos pedir una nueva acción
        if self._rl_counter % self._rl_step_interval == 0:
            action = self._rl.decide(frame, obs_report,
                                     result.traffic, result.arrow, scan=scan)
            self._last_rl_action = action
            v, w = self._rl.get_velocity(action)
            self._last_rl_v = v
            self._last_rl_w = w
            logger.debug(
                f"RL → {action.label} "
                f"v={v:.2f} "
                f"w={w:.2f} "
                f"steps={self._rl.steps}")
            # logger.debug(
            #     f"RL → {action.label} v={v:.2f} w={w:.2f} "
            #     f"ε={self._rl.epsilon:.3f} steps={self._rl.steps}")

        # Entre decisiones RL, mantener la última velocidad + corrección lateral
        self._send_vel(self._last_rl_v, self._last_rl_w + side_w)

    def _arrow_fallback(self, result, side_w):
        """Fallback sin RL: flechas + avance recto."""
        lin = self._mv["linear_speed"]
        ang = self._mv["angular_speed"]
        if result.arrow == ArrowDir.LEFT:
            self._send_vel(0.0, +ang)
        elif result.arrow == ArrowDir.RIGHT:
            self._send_vel(0.0, -ang)
        else:
            self._send_vel(lin, side_w)

    # ── HUD de obstáculos ──────────────────────────────────────────────────────

    def _draw_obstacle_hud(self, report: ObstacleReport):
        h, w = 190, 230
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        danger = (0, 0, 255)
        safe   = (0, 200, 0)

        def bar(d, max_d=2.0):
            return 0 if d is None else int((1 - min(d, max_d) / max_d) * 70)

        stop_d = self._obs_cfg.get("stopping_dist", self._obs_cfg.get("front_min_dist", 0.38))
        fc = danger if (report.front_dist or 9) < stop_d                          else safe
        lc = danger if (report.left_dist  or 9) < self._obs_cfg["side_min_dist"]  else safe
        rc = danger if (report.right_dist or 9) < self._obs_cfg["side_min_dist"]  else safe

        cv2.rectangle(canvas, (100, 5),  (130, 5 + bar(report.front_dist)), fc, -1)
        cv2.rectangle(canvas, (5,   60), (5 + bar(report.left_dist),  80),  lc, -1)
        cv2.rectangle(canvas, (225 - bar(report.right_dist), 60), (225, 80), rc, -1)

        def lbl(d): return f"{d:.2f}m" if d else "--"
        cv2.putText(canvas, f"F:{lbl(report.front_dist)}", (75,  95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, fc, 1)
        cv2.putText(canvas, f"L:{lbl(report.left_dist)}",  (5,  115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lc, 1)
        cv2.putText(canvas, f"R:{lbl(report.right_dist)}", (155, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, rc, 1)

        zc = danger if not report.is_clear else safe
        cv2.putText(canvas, report.zone.name,      (5, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, zc, 1)
        cv2.putText(canvas, f"FSM:{self._state.name}", (5, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        if self._use_rl:
            eps_color = (0, 200, 255)
            cv2.putText(canvas,
                        f"RL:{self._last_rl_action.label} "
                        f"s={self._rl.steps}",
                        (5, 183), cv2.FONT_HERSHEY_SIMPLEX, 0.33, eps_color, 1)

        cv2.imshow("Obstacle/RL", canvas)

    # ── Lifecycle (usado por SessionManager) ─────────────────────────────────

    def start_threads(self):
        """Inicia el hilo receptor. Llamar antes de step()."""
        self._recv_thread = threading.Thread(
            target=self._receiver_loop, daemon=True)
        self._recv_thread.start()
        logger.info("Controller threads started.")

    def step(self) -> bool:
        """
        Un paso del loop de control. Retorna True si una tecla 'q' fue pulsada.
        Llamar en un loop externo a ~send_hz.
        """
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            scan  = self._scan

        if frame is not None:
            self._decide(frame, scan)

        return cv2.waitKey(1) & 0xFF == ord("q")

    def stop(self):
        """Para el robot, cierra sockets y guarda el modelo."""
        self._running = False
        self._stop()
        if self._use_rl:
            self._rl.save()
        self._ctrl_sock.close()
        cv2.destroyAllWindows()
        logger.info(f"Controller stopped. Checkpoints: {self._vision.checkpoints}")

    def do_recovery(self, backup_s: float | None = None,
                    turn_s: float | None = None,
                    turn_w: float | None = None):
        """
        Maniobra de recuperación entre episodios: retrocede y gira.
        Bloquea hasta completar la maniobra.
        """
        backup_s = backup_s or self._obs_cfg["backup_seconds"] * 2
        turn_s   = turn_s   or self._obs_cfg["turn_seconds"]
        turn_w   = turn_w   or self._mv["angular_speed"]

        logger.info(f"[RECOVERY] backup={backup_s:.1f}s  turn={turn_s:.1f}s")
        self._stop()
        time.sleep(0.1)

        # Marcha atrás
        t0 = time.time()
        while time.time() - t0 < backup_s:
            self._send_vel(self._obs_cfg["backup_speed"], 0.0)
            time.sleep(0.02)

        # Giro
        t0 = time.time()
        while time.time() - t0 < turn_s:
            self._send_vel(0.0, turn_w)
            time.sleep(0.02)

        self._stop()
        self._state = State.MOVING
        logger.info("[RECOVERY] done")

    # ── Main loop (entrada directa sin SessionManager) ────────────────────────

    def run(self):
        self.start_threads()
        period = 1.0 / self._mv["send_hz"]
        logger.info("Controller running. Q para salir.")
        try:
            while True:
                if self.step():
                    break
                time.sleep(period)
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            self.stop()


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    AutonomousController(cfg).run()
