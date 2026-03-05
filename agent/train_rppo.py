"""
agent/train_rppo.py

Recurrent PPO (LSTM-PPO) training script for the Dynamic Request Batching agent.

Uses sb3-contrib RecurrentPPO with MlpLstmPolicy:
  - 4 parallel envs (Monitor-wrapped)
  - LSTM hidden state of size 64 — gives the agent sequential memory
  - Actor and Critic both have LSTM cells
  - n_steps=128 per env (sequences for LSTM training)
  - EvalCallback → saves best model to models/rppo_best/
  - CheckpointCallback → saves every 50k steps
  - TensorBoard logging → tensorboard_logs/RPPO_batching/

Key difference from train.py (standard PPO):
  This agent maintains a hidden state (h_t, c_t) between consecutive steps
  within an episode. This allows it to 'remember' past observations — for
  example, the fact that traffic was rising over the last 2 seconds — and
  make more informed batching decisions than a memoryless policy.

  Theoretical motivation: Batching revenue arrivals form a non-stationary
  Poisson process. An agent that can detect trend changes in arrival rate
  can make proactive serving decisions, rather than only reacting to the
  current queue size and oldest wait time.

Usage:
    python agent/train_rppo.py

TensorBoard:
    tensorboard --logdir tensorboard_logs/
    # Compare 'PPO_batching' vs 'DQN_batching' vs 'RPPO_batching'
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from env.batching_env import BatchingEnv
from config import CONFIG, RPPO_CONFIG

# ── Directories ───────────────────────────────────────────────────────────────
TENSORBOARD_DIR        = os.path.join(ROOT, "tensorboard_logs")
MODELS_DIR             = os.path.join(ROOT, "models")
RPPO_BEST_MODEL_DIR    = os.path.join(MODELS_DIR, "rppo_best")
RPPO_CHECKPOINT_DIR    = os.path.join(MODELS_DIR, "rppo_checkpoints")
MONITOR_LOG_DIR        = os.path.join(ROOT, "monitor_logs")
RPPO_FINAL_MODEL_PATH  = os.path.join(MODELS_DIR, "rppo_batching_final")

for d in [RPPO_BEST_MODEL_DIR, RPPO_CHECKPOINT_DIR, MONITOR_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ       = 10_000
CHECKPOINT_FREQ = 50_000
N_ENVS          = 4


def train():
    print("=" * 65)
    print("  Dynamic Request Batching — Recurrent PPO (LSTM-PPO) Training")
    print("=" * 65)
    print(f"  Total timesteps     : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs       : {N_ENVS}")
    print(f"  LSTM hidden size    : {RPPO_CONFIG['lstm_hidden_size']}")
    print(f"  LSTM layers         : {RPPO_CONFIG['n_lstm_layers']}")
    print(f"  Critic LSTM         : {RPPO_CONFIG['enable_critic_lstm']}")
    print(f"  n_steps per env     : {RPPO_CONFIG['n_steps']}")
    print(f"  TensorBoard         : {TENSORBOARD_DIR}")
    print("=" * 65)

    # ── Training envs (vectorised + Monitor) ──────────────────────────────────
    # RecurrentPPO works with make_vec_env just like standard PPO.
    # The key difference is INTERNAL: SB3 tracks LSTM states per-env in a
    # VecEnv buffer and resets them automatically when an episode ends.
    train_env = make_vec_env(
        env_id=BatchingEnv,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=Monitor,
        wrapper_kwargs={"filename": None},
        env_kwargs={},
    )

    # ── Eval env (single, deterministic) ──────────────────────────────────────
    eval_env = Monitor(BatchingEnv(seed=9999))

    # ── Recurrent PPO Model ───────────────────────────────────────────────────
    # MlpLstmPolicy — the new policy type. Under the hood:
    #   1. Observation passes through a small MLP feature extractor
    #   2. Features fed into LSTM cell → produces h_t
    #   3. h_t passed to separate Actor head (policy) and Critic head (value)
    # When enable_critic_lstm=True, the critic gets its own LSTM cell
    # with separate weights — it learns to estimate long-term value
    # using its own temporal context.
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=RPPO_CONFIG["learning_rate"],
        n_steps=RPPO_CONFIG["n_steps"],
        batch_size=RPPO_CONFIG["batch_size"],
        n_epochs=RPPO_CONFIG["n_epochs"],
        gamma=RPPO_CONFIG["gamma"],
        gae_lambda=RPPO_CONFIG["gae_lambda"],
        clip_range=RPPO_CONFIG["clip_range"],
        ent_coef=RPPO_CONFIG["ent_coef"],
        vf_coef=RPPO_CONFIG["vf_coef"],
        max_grad_norm=RPPO_CONFIG["max_grad_norm"],
        policy_kwargs=dict(
            lstm_hidden_size=RPPO_CONFIG["lstm_hidden_size"],
            n_lstm_layers=RPPO_CONFIG["n_lstm_layers"],
            enable_critic_lstm=RPPO_CONFIG["enable_critic_lstm"],
            net_arch=RPPO_CONFIG["net_arch"],
        ),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=1,
        seed=0,
    )

    print(f"\nRecurrent Policy architecture:\n{model.policy}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=RPPO_BEST_MODEL_DIR,
        log_path=os.path.join(MODELS_DIR, "rppo_eval_logs"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=RPPO_CHECKPOINT_DIR,
        name_prefix="rppo_batching",
        verbose=1,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,
        tb_log_name="RPPO_batching",
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # ── Save final model ───────────────────────────────────────────────────────
    model.save(RPPO_FINAL_MODEL_PATH)
    print(f"\n✓  RPPO final model saved → {RPPO_FINAL_MODEL_PATH}.zip")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Recurrent PPO Training Summary")
    print("=" * 65)
    print(f"  Training time     : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Total timesteps   : {TOTAL_TIMESTEPS:,}")
    print(f"  Throughput        : {TOTAL_TIMESTEPS / elapsed:,.0f} steps/s")
    print(f"  Final model       : {RPPO_FINAL_MODEL_PATH}.zip")
    print(f"  Best model        : {RPPO_BEST_MODEL_DIR}/best_model.zip")
    print(f"  TensorBoard log   : {TENSORBOARD_DIR}/RPPO_batching_*/")
    print("=" * 65)

    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()