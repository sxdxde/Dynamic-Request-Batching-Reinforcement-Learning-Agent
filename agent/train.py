"""
agent/train.py

PPO training script for the Dynamic Request Batching agent.

Uses Stable-Baselines3 with:
  - 4 parallel envs (Monitor-wrapped)
  - MlpPolicy with [64, 64] hidden layers
  - EvalCallback  → saves best model to models/best/
  - CheckpointCallback → saves every 50 000 steps
  - TensorBoard logging → tensorboard_logs/

Usage:
    python agent/train.py

Monitor outputs will be in monitor_logs/.
TensorBoard:
    tensorboard --logdir tensorboard_logs/
"""

import os
import sys
import time

# ── ensure project root is on path ──────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from env.batching_env import BatchingEnv
from config import CONFIG

# ── Directories ──────────────────────────────────────────────────────────────
TENSORBOARD_DIR  = os.path.join(ROOT, "tensorboard_logs")
MODELS_DIR       = os.path.join(ROOT, "models")
BEST_MODEL_DIR   = os.path.join(MODELS_DIR, "best")
CHECKPOINT_DIR   = os.path.join(MODELS_DIR, "checkpoints")
MONITOR_LOG_DIR  = os.path.join(ROOT, "monitor_logs")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_batching_final")

for d in [TENSORBOARD_DIR, BEST_MODEL_DIR, CHECKPOINT_DIR, MONITOR_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── PPO hyperparameters ──────────────────────────────────────────────────────
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    ),
    tensorboard_log=TENSORBOARD_DIR,
    verbose=1,
)

TOTAL_TIMESTEPS  = 500_000
N_ENVS           = 4
EVAL_FREQ        = 10_000   # steps between evaluations (per env → divided by n_envs)
CHECKPOINT_FREQ  = 50_000   # steps between checkpoints


def make_env(seed: int = 0):
    """Factory that returns a Monitor-wrapped BatchingEnv."""
    def _init():
        env = BatchingEnv(seed=seed)
        env = Monitor(env, filename=os.path.join(MONITOR_LOG_DIR, f"env_{seed}"))
        return env
    return _init


def build_callbacks(eval_env):
    """Build EvalCallback + CheckpointCallback."""
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=os.path.join(MODELS_DIR, "eval_logs"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_batching",
        verbose=1,
    )

    return CallbackList([eval_callback, checkpoint_callback])


def train():
    print("=" * 60)
    print("  Dynamic Request Batching — PPO Training")
    print("=" * 60)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  TensorBoard     : {TENSORBOARD_DIR}")
    print(f"  Best model      : {BEST_MODEL_DIR}")
    print(f"  Checkpoints     : {CHECKPOINT_DIR}")
    print("=" * 60)

    # ── Training envs (vectorised + Monitor) ────────────────────────────────
    train_env = make_vec_env(
        env_id=BatchingEnv,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=Monitor,
        wrapper_kwargs={"filename": None},  # in-memory monitor; logs later
        env_kwargs={},
    )

    # ── Eval env (single, deterministic) ────────────────────────────────────
    eval_env = Monitor(BatchingEnv(seed=9999))

    # ── PPO model ────────────────────────────────────────────────────────────
    model = PPO("MlpPolicy", train_env, **PPO_KWARGS)

    print(f"\nPolicy network architecture:\n{model.policy}\n")

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = build_callbacks(eval_env)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,
        tb_log_name="PPO_batching",
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(FINAL_MODEL_PATH)
    print(f"\n✓  Final model saved → {FINAL_MODEL_PATH}.zip")

    # ── Training summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Training time         : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Total timesteps       : {TOTAL_TIMESTEPS:,}")
    print(f"  Throughput            : {TOTAL_TIMESTEPS / elapsed:,.0f} steps/s")
    print(f"  Final model           : {FINAL_MODEL_PATH}.zip")
    print(f"  Best model (eval)     : {BEST_MODEL_DIR}/best_model.zip")
    print(f"  TensorBoard logs      : {TENSORBOARD_DIR}")
    print(f"  Run:  tensorboard --logdir {TENSORBOARD_DIR}")
    print("=" * 60)

    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
