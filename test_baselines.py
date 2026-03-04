"""
test_baselines.py

End-to-end integration test for the Dynamic Request Batching RL project.

Runs:
  1. gymnasium.utils.env_checker.check_env — validates the env conforms to
     the Gymnasium API spec.
  2. All 3 baselines evaluated for 10 episodes each, printing results
     side-by-side.

Usage:
    python test_baselines.py
"""

import sys
import os

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Imports ─────────────────────────────────────────────────────────────────
from gymnasium.utils.env_checker import check_env

from env.batching_env import BatchingEnv
from baselines.cloudflare_formula import CloudflareBaseline, evaluate_baseline
from baselines.random_agent import RandomAgent
from baselines.greedy_agent import GreedyAgent
from config import CONFIG

N_EPISODES = 10
SEED_OFFSET = 42


def main():
    # ── 1. Gymnasium environment check ──────────────────────────────────────
    print("=" * 60)
    print("Step 1: gymnasium check_env(BatchingEnv())")
    print("=" * 60)
    env_check = BatchingEnv()
    check_env(env_check, warn=True, skip_render_check=True)
    env_check.close()
    print("✓  check_env passed — environment is Gymnasium-compliant.\n")

    # ── 2. Evaluate all baselines ────────────────────────────────────────────
    print("=" * 60)
    print(f"Step 2: Baseline evaluation ({N_EPISODES} episodes each)")
    print("=" * 60)

    env = BatchingEnv()

    baselines = {
        "Cloudflare": CloudflareBaseline(
            max_latency_ms=CONFIG["max_latency_ms"], seed=SEED_OFFSET
        ),
        "Random":     RandomAgent(seed=SEED_OFFSET),
        "Greedy":     GreedyAgent(),
    }

    results: dict[str, tuple] = {}
    for name, agent in baselines.items():
        print(f"  Evaluating {name:12s}...", end=" ", flush=True)
        mean_r, std_r, mean_lat = evaluate_baseline(
            agent, env, n_episodes=N_EPISODES, seed_offset=SEED_OFFSET
        )
        results[name] = (mean_r, std_r, mean_lat)
        print(f"done  (mean_reward={mean_r:+.2f})")

    env.close()

    # ── 3. Side-by-side summary ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Baseline Comparison Summary")
    print("=" * 60)
    header = f"{'Agent':<14}  {'Mean Reward':>12}  {'Std Reward':>10}  {'Mean Latency (ms)':>18}"
    print(header)
    print("-" * len(header))
    for name, (mean_r, std_r, mean_lat) in results.items():
        print(
            f"{name:<14}  {mean_r:>+12.2f}  {std_r:>10.2f}  {mean_lat:>18.2f}"
        )
    print("=" * 60)

    # ── 4. Quick sanity assertions ───────────────────────────────────────────
    for name, (mean_r, std_r, mean_lat) in results.items():
        assert isinstance(mean_r, float), f"{name}: mean_reward not float"
        assert std_r >= 0,               f"{name}: std_reward negative"
        assert mean_lat >= 0,            f"{name}: mean_latency negative"
    print("\n✓  All sanity checks passed.")


if __name__ == "__main__":
    main()
