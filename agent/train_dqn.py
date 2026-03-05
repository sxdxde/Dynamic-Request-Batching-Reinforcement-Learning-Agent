import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback, 
    CallbackList,
)

from stable_baselines3.common.monitor import Monitor

from env.batching_env import BatchingEnv
from config import CONFIG, DQN_CONFIG

TENSORBOARD_DIR       = os.path.join(ROOT, "tensorboard_logs")
MODELS_DIR            = os.path.join(ROOT, "models")
DQN_BEST_MODEL_DIR    = os.path.join(MODELS_DIR, "dqn_best")
DQN_CHECKPOINT_DIR    = os.path.join(MODELS_DIR, "dqn_checkpoints")
MONITOR_LOG_DIR       = os.path.join(ROOT, "monitor_logs")
DQN_FINAL_MODEL_PATH  = os.path.join(MODELS_DIR, "dqn_batching_final")

for d in [DQN_BEST_MODEL_DIR, DQN_CHECKPOINT_DIR, MONITOR_LOG_DIR]:
    os.makedirs(d, exist_ok = True)

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
CHECKPOINT_FREQ = 50_000

def train():
    print("=" * 60)
    print("  Dynamic Request Batching — DQN Training")
    print("=" * 60)
    print(f"  Total timesteps     : {TOTAL_TIMESTEPS:,}")
    print(f"  Replay buffer size  : {DQN_CONFIG['buffer_size']:,}")
    print(f"  Learning starts at  : {DQN_CONFIG['learning_starts']:,} steps")
    print(f"  Target update every : {DQN_CONFIG['target_update_interval']} steps")
    print(f"  Exploration: ε 1.0 → {DQN_CONFIG['exploration_final_eps']}")
    print(f"  TensorBoard         : {TENSORBOARD_DIR}")
    print("=" * 60)

    train_env = Monitor(
        BatchingEnv(seed=0),
        filename = os.path.join(MONITOR_LOG_DIR, "dqn_train")
    )

    eval_env = Monitor(BatchingEnv(seed=9999))

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=DQN_CONFIG["learning_rate"],
        buffer_size=DQN_CONFIG["buffer_size"],
        learning_starts=DQN_CONFIG["learning_starts"],
        batch_size=DQN_CONFIG["batch_size"],
        gamma=DQN_CONFIG["gamma"],
        target_update_interval=DQN_CONFIG["target_update_interval"],
        tau=DQN_CONFIG["tau"],
        exploration_fraction=DQN_CONFIG["exploration_fraction"],
        exploration_initial_eps=DQN_CONFIG["exploration_initial_eps"],
        exploration_final_eps=DQN_CONFIG["exploration_final_eps"],
        train_freq=DQN_CONFIG["train_freq"],
        gradient_steps=DQN_CONFIG["gradient_steps"],
        policy_kwargs=dict(net_arch=DQN_CONFIG["net_arch"]),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=DQN_CONFIG["verbose"],
        seed=0,
    )

    print(f"\nQ-Network architecture:\n{model.policy}\n")


    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=DQN_BEST_MODEL_DIR,
        log_path=os.path.join(MODELS_DIR, "dqn_eval_logs"),
        eval_freq=EVAL_FREQ,   # DQN: single env, so no n_envs division needed
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=DQN_CHECKPOINT_DIR,
        name_prefix="dqn_batching",
        verbose=1,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    t0 = time.time()
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS, 
        callback = callbacks, 
        reset_num_timesteps = True,
        tb_log_name = "DQN_batching",
        progress_bar = True,
    )
    elapsed = time.time() - t0

    model.save(DQN_FINAL_MODEL_PATH)
    print(f"\n✓  DQN final model saved → {DQN_FINAL_MODEL_PATH}.zip")

    # ── Training summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DQN Training Summary")
    print("=" * 60)
    print(f"  Training time     : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Total timesteps   : {TOTAL_TIMESTEPS:,}")
    print(f"  Throughput        : {TOTAL_TIMESTEPS / elapsed:,.0f} steps/s")
    print(f"  Final model       : {DQN_FINAL_MODEL_PATH}.zip")
    print(f"  Best model        : {DQN_BEST_MODEL_DIR}/best_model.zip")
    print(f"  TensorBoard       : {TENSORBOARD_DIR}")
    print("=" * 60)

    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
