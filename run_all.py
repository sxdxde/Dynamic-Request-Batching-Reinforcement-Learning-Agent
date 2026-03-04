"""
run_all.py

Orchestrator script: runs the full RL batching pipeline end-to-end.

Steps:
  1. Validate environment (gymnasium check_env)
  2. Evaluate baselines (Random, Greedy, Cloudflare) — 10 episodes each
  3. Train PPO agent (500k timesteps)
  4. Generate comparison plots (PPO vs all baselines, 30 episodes each)
  5. Launch live demo (real-time animation)

Each step is wrapped in a try/except so a failure doesn't stop the pipeline.
Run from project root:
    python3 run_all.py
"""

import os
import sys
import traceback
import time

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 64

def _header(step: int, msg: str):
    print(f"\n{SEP}")
    print(f"  Step {step}: {msg}")
    print(SEP)


def _ok(msg: str = ""):
    tag = f"  ✓  {msg}" if msg else "  ✓  Done."
    print(tag)


def _err(exc: Exception):
    print(f"\n  ✗  ERROR: {exc}")
    print("  Traceback:")
    for line in traceback.format_exc().splitlines():
        print(f"    {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Validate environment
# ─────────────────────────────────────────────────────────────────────────────

def step1_validate():
    _header(1, "Validating environment...")
    try:
        from gymnasium.utils.env_checker import check_env
        from env.batching_env import BatchingEnv

        env = BatchingEnv()
        check_env(env, skip_render_check=True)
        env.close()
        _ok("BatchingEnv passed gymnasium check_env.")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Evaluate baselines
# ─────────────────────────────────────────────────────────────────────────────

def step2_baselines():
    _header(2, "Evaluating baselines...")
    try:
        import numpy as np
        from env.batching_env import BatchingEnv
        from baselines.cloudflare_formula import CloudflareBaseline, evaluate_baseline
        from baselines.random_agent import RandomAgent
        from baselines.greedy_agent import GreedyAgent
        from config import CONFIG

        N = 10
        env = BatchingEnv()

        agents = {
            "Cloudflare": CloudflareBaseline(
                max_latency_ms=CONFIG["max_latency_ms"], seed=42),
            "Random":     RandomAgent(seed=42),
            "Greedy":     GreedyAgent(),
        }

        print(f"\n  {'Agent':<14}  {'Mean Reward':>12}  {'Std':>8}  {'Mean Latency (ms)':>18}")
        print(f"  {'-'*14}  {'-'*12}  {'-'*8}  {'-'*18}")

        for name, agent in agents.items():
            mean_r, std_r, mean_lat = evaluate_baseline(agent, env, n_episodes=N)
            print(f"  {name:<14}  {mean_r:>+12.2f}  {std_r:>8.2f}  {mean_lat:>18.2f}")

        env.close()
        _ok("Baseline evaluation complete.")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Train PPO agent
# ─────────────────────────────────────────────────────────────────────────────

def step3_train():
    _header(3, "Training PPO agent...")
    try:
        from agent.train import train
        model = train()
        _ok("Training complete. Model saved.")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Generate comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def step4_evaluate():
    _header(4, "Generating comparison plots...")
    try:
        # Try best model first, then final
        import os
        from agent.evaluate import run_all_agents, generate_figure, print_summary_table, PLOT_PATH

        best_path  = os.path.join(ROOT, "models", "best", "best_model")
        final_path = os.path.join(ROOT, "models", "ppo_batching_final")

        if os.path.exists(best_path + ".zip") or os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(final_path + ".zip") or os.path.exists(final_path):
            model_path = final_path
        else:
            raise FileNotFoundError(
                "No trained model found. Run Step 3 (training) first."
            )

        print(f"  Using model: {model_path}\n")
        data = run_all_agents(model_path)
        print_summary_table(data)
        path = generate_figure(data)
        _ok(f"Comparison plot saved → {path}")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Launch live demo
# ─────────────────────────────────────────────────────────────────────────────

def step5_demo():
    _header(5, "Launching live demo...")
    try:
        import os
        import matplotlib.pyplot as plt
        from demo.live_demo import build_demo

        best_path  = os.path.join(ROOT, "models", "best", "best_model")
        final_path = os.path.join(ROOT, "models", "ppo_batching_final")

        if os.path.exists(best_path + ".zip") or os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(final_path + ".zip") or os.path.exists(final_path):
            model_path = final_path
        else:
            raise FileNotFoundError(
                "No trained model found. Run Step 3 (training) first."
            )

        print(f"  Using model: {model_path}")
        print("  Close the animation window to exit.\n")
        fig, anim = build_demo(model_path, interval_ms=50)
        plt.show()
        _ok("Live demo closed.")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  Dynamic Request Batching — Full Pipeline")
    print(SEP)

    t_start = time.time()

    step1_validate()
    step2_baselines()
    step3_train()
    step4_evaluate()
    step5_demo()

    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
