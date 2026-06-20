"""
Online PPO agent para TurtleBot4.

PPO ventajas vs DQN para este caso:
- Exploración natural via política estocástica (no requiere epsilon-greedy)
- ent_coef mantiene diversidad de acciones sin parámetros manuales
- Más estable en entornos continuos con ruido de sensores

Flujo:
  decide()  → forward del actor → acción muestreada de la distribución
  observe() → añade (obs, action, reward, done, value, log_prob) al rollout buffer
  Cada n_steps → computa ventajas (GAE) → entrena PPO → reset buffer
  Entrenamiento en hilo background para no bloquear el loop de control.
"""
import logging
import threading
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import configure_logger

from rl_env import (
    RLAction, _DummyEnv, build_obs, to_buffer_obs, compute_reward, action_to_vel,
)
from vision import TrafficState, ArrowDir
from obstacle import ObstacleReport, ScanData

logger = logging.getLogger("tb4")


class RLAgent:
    def __init__(self, cfg: dict):
        self._cfg    = cfg
        self._rl     = cfg["rl"]
        self._img_sz = self._rl["image_size"]

        model_path = Path(self._rl["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_path = model_path

        env = _DummyEnv(self._img_sz)

        if model_path.exists():
            try:
                logger.info(f"[RL] Cargando modelo PPO: {model_path}")
                self._model = PPO.load(str(model_path), env=env)
            except Exception as e:
                logger.warning(
                    f"[RL] No se pudo cargar modelo existente ({e}). "
                    "Puede ser incompatibilidad de espacio de observación. Creando nuevo.")
                self._model = None

        if not model_path.exists() or getattr(self, "_model", None) is None:
            logger.info("[RL] Iniciando modelo PPO nuevo")
            self._model = PPO(
                "MultiInputPolicy",
                env,
                n_steps             = self._rl["n_steps"],
                batch_size          = self._rl["batch_size"],
                n_epochs            = self._rl["n_epochs"],
                gamma               = self._rl["gamma"],
                gae_lambda          = self._rl["gae_lambda"],
                learning_rate       = self._rl["learning_rate"],
                clip_range          = self._rl["clip_range"],
                ent_coef            = self._rl["ent_coef"],
                verbose             = 0,
                device              = "auto",
            )

        # Logger interno de SB3 (normalmente lo inicializa learn())
        self._model.set_logger(configure_logger(verbose=0, tensorboard_log=None))
        self._device = self._model.device

        # ── Contadores y estado ────────────────────────────────────────────────
        self._global_step  = 0
        self._rollout_step = 0
        self._n_steps      = self._rl["n_steps"]
        self._ep_step      = 0
        self._ep_reward    = 0.0
        self._episode      = 0

        self._last_ep_reward = 0.0
        self._last_ep_steps  = 0
        self._last_ep_reason = "none"

        # ── Estado del paso anterior ───────────────────────────────────────────
        self._last_obs_buf:    dict | None     = None
        self._last_action_np:  np.ndarray | None = None
        self._last_value:      th.Tensor | None  = None
        self._last_log_prob:   th.Tensor | None  = None
        self._last_ep_start:   np.ndarray        = np.array([True])

        # Para el cálculo de GAE al final del rollout
        self._gae_obs_buf: dict | None = None
        self._gae_done:    bool        = False

        # ── Sincronización ─────────────────────────────────────────────────────
        self._model_lock   = threading.Lock()
        self._train_event  = threading.Event()
        self._episode_done_event = threading.Event()

        self._train_thread = threading.Thread(
            target=self._train_loop, daemon=True, name="ppo-train")
        self._train_thread.start()

    # ── API pública ───────────────────────────────────────────────────────────

    def decide(self,
               frame: np.ndarray,
               obs_report: ObstacleReport,
               traffic: TrafficState,
               arrow: ArrowDir,
               scan: ScanData | None = None) -> RLAction:
        """
        Forward del actor: muestrea una acción de la distribución de política.
        La exploración es implícita (política estocástica) — sin epsilon-greedy.
        scan: datos LIDAR crudos para construir observación panorámica de 8 sectores.
        """
        obs     = build_obs(frame, obs_report, traffic, arrow, self._img_sz, scan=scan)
        obs_buf = to_buffer_obs(obs)
        obs_t   = self._buf_to_tensor(obs_buf)

        with self._model_lock:
            with th.no_grad():
                actions, values, log_probs = self._model.policy.forward(obs_t)

        self._last_obs_buf   = obs_buf
        self._last_action_np = actions.cpu().numpy()
        self._last_value     = values
        self._last_log_prob  = log_probs

        return RLAction(int(actions.cpu().numpy()[0]))

    def observe(self,
                frame: np.ndarray,
                obs_report: ObstacleReport,
                traffic: TrafficState,
                arrow: ArrowDir,
                new_checkpoint: bool,
                scan: ScanData | None = None):
        """
        Registra el resultado del paso anterior en el rollout buffer.
        Llama después de ejecutar la acción dada por decide().
        scan: datos LIDAR crudos para construir la siguiente observación panorámica.
        """
        if self._last_obs_buf is None or self._last_action_np is None:
            return

        reward, done = compute_reward(
            RLAction(int(self._last_action_np[0])),
            obs_report, traffic, arrow, new_checkpoint, self._cfg)

        self._ep_step   += 1
        self._ep_reward += reward

        timeout = self._ep_step >= self._rl["max_episode_steps"]
        if timeout:
            done = True

        # Guardar siguiente obs para GAE
        next_obs_buf     = to_buffer_obs(
            build_obs(frame, obs_report, traffic, arrow, self._img_sz, scan=scan))
        self._gae_obs_buf = next_obs_buf
        self._gae_done    = done

        # Añadir al rollout buffer (thread-safe: solo este hilo escribe)
        try:
            self._model.rollout_buffer.add(
                self._last_obs_buf,
                self._last_action_np,
                np.array([reward]),
                self._last_ep_start,
                self._last_value,
                self._last_log_prob,
            )
        except Exception as e:
            logger.warning(f"[RL] rollout_buffer.add error: {e}")

        self._last_ep_start = np.array([float(done)])
        self._global_step  += 1
        self._rollout_step += 1

        # Episodio terminado
        if done:
            reason = "timeout" if timeout else "collision"
            self._episode        += 1
            self._last_ep_reward  = self._ep_reward
            self._last_ep_steps   = self._ep_step
            self._last_ep_reason  = reason
            logger.info(
                f"[RL] Ep {self._episode} | {reason} | "
                f"pasos={self._ep_step} reward={self._ep_reward:.1f}")
            self._ep_step         = 0
            self._ep_reward       = 0.0
            self._last_ep_start   = np.array([True])
            self._episode_done_event.set()

        # Disparar entrenamiento al completar n_steps
        if self._rollout_step >= self._n_steps:
            self._rollout_step = 0
            self._train_event.set()

    def get_velocity(self, action: RLAction) -> tuple[float, float]:
        return action_to_vel(action, self._cfg)

    def save(self):
        with self._model_lock:
            self._model.save(str(self._model_path))
        logger.info(f"[RL] Modelo guardado: {self._model_path}")

    # ── Propiedades para SessionManager ───────────────────────────────────────

    @property
    def episode_ended(self) -> bool:
        return self._episode_done_event.is_set()

    def acknowledge_episode_end(self):
        """El SessionManager llama esto tras gestionar el fin de episodio."""
        self._episode_done_event.clear()
        self._last_obs_buf   = None
        self._last_action_np = None
        self._last_value     = None
        self._last_log_prob  = None

    @property
    def last_episode_info(self) -> dict:
        return {
            "episode": self._episode,
            "reward":  self._last_ep_reward,
            "steps":   self._last_ep_steps,
            "reason":  self._last_ep_reason,
            "buf_pct": self._rollout_step / self._n_steps,
        }

    @property
    def steps(self) -> int:
        return self._global_step

    @property
    def episodes(self) -> int:
        return self._episode

    # ── Hilo de entrenamiento PPO ─────────────────────────────────────────────

    def _train_loop(self):
        save_every = self._rl["save_every_steps"]
        train_num  = 0
        while True:
            self._train_event.wait()
            self._train_event.clear()

            if self._gae_obs_buf is None:
                continue

            with self._model_lock:
                try:
                    # Valor del último estado para calcular ventajas (GAE)
                    last_obs_t = self._buf_to_tensor(self._gae_obs_buf)
                    with th.no_grad():
                        last_values = self._model.policy.predict_values(last_obs_t)

                    self._model.rollout_buffer.compute_returns_and_advantage(
                        last_values=last_values,
                        dones=np.array([self._gae_done]))

                    self._model.train()
                    self._model.rollout_buffer.reset()
                    train_num += 1

                    logger.info(
                        f"[RL] PPO update #{train_num} | "
                        f"ep={self._episode} global_step={self._global_step}")
                except Exception as e:
                    logger.warning(f"[RL] train error: {e}")
                    try:
                        self._model.rollout_buffer.reset()
                    except Exception:
                        pass

            if self._global_step % save_every < self._n_steps:
                self.save()

    # ── Utilidad interna ──────────────────────────────────────────────────────

    def _buf_to_tensor(self, obs_buf: dict) -> dict:
        """Convierte obs en formato buffer a dict de tensores float32."""
        return {
            k: th.tensor(v, dtype=th.float32, device=self._device)
            for k, v in obs_buf.items()
        }
