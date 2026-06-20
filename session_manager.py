"""
Gestor de sesiones de aprendizaje autónomo para TurtleBot4.

Uso:
    python session_manager.py          # correr indefinidamente
    python session_manager.py --stats  # solo mostrar estadísticas acumuladas

El programa:
  1. Carga el modelo RL y las estadísticas previas (si existen).
  2. Corre episodios en bucle sin intervención humana.
  3. Al terminar un episodio (colisión o timeout) ejecuta una maniobra de
     recuperación automática y comienza el siguiente.
  4. Al cerrar con Ctrl+C (o Q en la ventana) guarda todo y termina limpiamente.
  5. Al volver a ejecutar, retoma desde donde quedó.

Estadísticas persistentes: models/session_stats.json
"""
import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from main import AutonomousController, load_config, setup_logger

STATS_PATH = Path("models/session_stats.json")

logger = logging.getLogger("tb4")


# ── Estadísticas persistentes ────────────────────────────────────────────────

def _empty_stats() -> dict:
    return {
        "total_episodes":    0,
        "total_rl_steps":    0,
        "total_checkpoints": 0,
        "best_ep_reward":    float("-inf"),
        "sessions":          [],    # historial resumido de sesiones
        "episode_log":       [],    # últimos MAX_EP_LOG episodios
    }


MAX_EP_LOG = 500   # cuántos episodios guardar en el JSON


def load_stats() -> dict:
    if STATS_PATH.exists():
        try:
            with open(STATS_PATH) as f:
                data = json.load(f)
            logger.info(
                f"[SESSION] Stats cargadas — "
                f"{data['total_episodes']} eps | "
                f"{data['total_rl_steps']} pasos RL")
            return data
        except Exception as e:
            logger.warning(f"[SESSION] No se pudo leer stats: {e}")
    return _empty_stats()


def save_stats(stats: dict):
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Limitar tamaño del log de episodios
    stats["episode_log"] = stats["episode_log"][-MAX_EP_LOG:]
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def print_stats(stats: dict):
    sep = "─" * 50
    print(sep)
    print(f"  Total episodios  : {stats['total_episodes']}")
    print(f"  Total pasos RL   : {stats['total_rl_steps']}")
    print(f"  Total checkpoints: {stats['total_checkpoints']}")
    best = stats["best_ep_reward"]
    print(f"  Mejor reward ep  : {best:.1f}" if best != float("-inf") else
          "  Mejor reward ep  : --")
    print(f"  Sesiones previas : {len(stats['sessions'])}")
    if stats["sessions"]:
        last = stats["sessions"][-1]
        print(f"  Última sesión    : {last['id']}  "
              f"({last['episodes']} eps, {last['duration_min']:.1f} min)")
    if stats["episode_log"]:
        last_eps = stats["episode_log"][-5:]
        print("  Últimos 5 episodios:")
        for ep in last_eps:
            print(f"    ep={ep['ep']:4d}  reward={ep['reward']:7.1f}  "
                  f"steps={ep['steps']:4d}  reason={ep['reason']}")
    print(sep)


# ── Session Manager ───────────────────────────────────────────────────────────

class SessionManager:
    def __init__(self, cfg: dict, stats: dict):
        self._cfg   = cfg
        self._stats = stats
        self._ctrl  = AutonomousController(cfg)

        self._session_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_episodes = 0
        self._session_steps    = 0
        self._session_cps      = 0
        self._session_start    = time.time()

        self._period = 1.0 / cfg["movement"]["send_hz"]

    # ── Bucle principal ───────────────────────────────────────────────────────

    def run(self):
        self._ctrl.start_threads()
        logger.info(
            f"[SESSION] Sesión {self._session_id} iniciada. "
            f"Total histórico: {self._stats['total_episodes']} eps. "
            "Ctrl+C o Q para terminar.")

        try:
            while True:
                quit_requested = self._run_episode()
                if quit_requested:
                    break
        except KeyboardInterrupt:
            logger.info("\n[SESSION] Ctrl+C — cerrando limpiamente...")
        finally:
            self._on_session_end()

    # ── Episodio ──────────────────────────────────────────────────────────────

    def _run_episode(self) -> bool:
        """
        Corre hasta que el agente RL señale fin de episodio.
        Retorna True si el usuario pidió salir (tecla Q).
        """
        rl = self._ctrl._rl if self._ctrl._use_rl else None

        ep_num   = self._stats["total_episodes"] + self._session_episodes + 1
        ep_start = time.time()
        ep_cps_before = self._ctrl._vision.checkpoint_count

        logger.info(f"[SESSION] ── Episodio {ep_num} iniciando ──")

        while True:
            quit_req = self._ctrl.step()
            if quit_req:
                return True

            # ¿El RL terminó el episodio?
            if rl is not None and rl.episode_ended:
                break

            time.sleep(self._period)

        # ── Fin del episodio ──────────────────────────────────────────────────
        ep_info  = rl.last_episode_info if rl else {}
        ep_dur   = time.time() - ep_start
        new_cps  = self._ctrl._vision.checkpoint_count - ep_cps_before

        self._session_episodes += 1
        self._session_steps    += ep_info.get("steps", 0)
        self._session_cps      += new_cps
        self._stats["total_episodes"]    += 1
        self._stats["total_rl_steps"]    += ep_info.get("steps", 0)
        self._stats["total_checkpoints"] += new_cps

        reward = ep_info.get("reward", 0.0)
        if reward > self._stats["best_ep_reward"]:
            self._stats["best_ep_reward"] = reward

        ep_record = {
            "ep":      self._stats["total_episodes"],
            "session": self._session_id,
            "reward":  round(reward, 2),
            "steps":   ep_info.get("steps", 0),
            "reason":  ep_info.get("reason", "?"),
            "dur_s":   round(ep_dur, 1),
            "cps":     new_cps,
        }
        self._stats["episode_log"].append(ep_record)

        logger.info(
            f"[SESSION] Ep {ep_num} | {ep_info.get('reason','?')} | "
            f"reward={reward:.1f}  steps={ep_info.get('steps',0)}  "
            f"dur={ep_dur:.1f}s  cps_totales={self._stats['total_checkpoints']}")

        # Guardar stats y modelo después de cada episodio
        if rl:
            rl.save()
        save_stats(self._stats)

        # Reconocer fin de episodio y ejecutar recuperación
        if rl:
            rl.acknowledge_episode_end()

        self._do_inter_episode_recovery(ep_info.get("reason", "timeout"))

        return False

    # ── Recuperación entre episodios ──────────────────────────────────────────

    def _do_inter_episode_recovery(self, reason: str):
        """
        Maniobra automática entre episodios para librar al robot de un choque.
        - reason='collision': recuperación agresiva (más backup, giro aleatorio)
        - reason='timeout':   recuperación suave (solo un pequeño backup)
        """
        obs_cfg = self._cfg["obstacle"]
        mv_cfg  = self._cfg["movement"]

        if reason == "collision":
            backup_s = obs_cfg["backup_seconds"] * 3.0
            turn_s   = obs_cfg["turn_seconds"] * 1.5
            # Giro aleatorio para explorar distinto camino
            turn_w   = mv_cfg["angular_speed"] * random.choice([-1, 1])
        else:
            backup_s = obs_cfg["backup_seconds"]
            turn_s   = obs_cfg["turn_seconds"] * 0.5
            turn_w   = mv_cfg["angular_speed"] * random.choice([-1, 1])

        # Breve pausa para que los sensores se estabilicen
        time.sleep(0.3)
        self._ctrl.do_recovery(backup_s=backup_s, turn_s=turn_s, turn_w=turn_w)
        time.sleep(0.5)   # pausa antes del siguiente episodio

    # ── Cierre de sesión ──────────────────────────────────────────────────────

    def _on_session_end(self):
        dur_min = (time.time() - self._session_start) / 60.0
        session_record = {
            "id":           self._session_id,
            "episodes":     self._session_episodes,
            "rl_steps":     self._session_steps,
            "checkpoints":  self._session_cps,
            "duration_min": round(dur_min, 1),
        }
        self._stats["sessions"].append(session_record)
        save_stats(self._stats)

        self._ctrl.stop()

        logger.info(
            f"\n[SESSION] ════ Sesión terminada ════\n"
            f"  Episodios esta sesión : {self._session_episodes}\n"
            f"  Pasos RL esta sesión  : {self._session_steps}\n"
            f"  Checkpoints           : {self._session_cps}\n"
            f"  Duración              : {dur_min:.1f} min\n"
            f"  Total histórico       : {self._stats['total_episodes']} eps | "
            f"{self._stats['total_rl_steps']} pasos RL\n"
            f"  Stats guardadas en    : {STATS_PATH}")

        print_stats(self._stats)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurtleBot4 Session Manager")
    parser.add_argument("--stats", action="store_true",
                        help="Solo mostrar estadísticas acumuladas y salir")
    parser.add_argument("--config", default="config.yaml",
                        help="Ruta al archivo de configuración")
    args = parser.parse_args()

    setup_logger()
    stats = load_stats()

    if args.stats:
        print_stats(stats)
        sys.exit(0)

    cfg = load_config(args.config)
    SessionManager(cfg, stats).run()


if __name__ == "__main__":
    main()
