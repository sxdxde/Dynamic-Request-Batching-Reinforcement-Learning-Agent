"""
demo/live_demo.py

Real-time matplotlib animation of the trained PPO agent running on BatchingEnv.

2×2 subplot layout (updates every 50ms via FuncAnimation):
  ┌──────────────────────┬──────────────────────┐
  │ Pending queue size   │ Step reward           │
  │ (line + serve dots)  │ (line + y=0 baseline) │
  ├──────────────────────┼──────────────────────┤
  │ Wait vs Serve counts │ State readout         │
  │ (bar chart, last 50) │ (monospace text box)  │
  └──────────────────────┴──────────────────────┘

Usage:
    python demo/live_demo.py
    python demo/live_demo.py --model models/ppo_batching_final
    python demo/live_demo.py --model models/best/best_model --speed 2.0
"""

import os
import sys
import argparse
import collections
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from stable_baselines3 import PPO

from env.batching_env import BatchingEnv
from config import CONFIG

# ── Theme constants ────────────────────────────────────────────────────────────
BG_FIG    = "#0a0e1a"
BG_AX     = "#141c30"
COL_GRID  = "#1e2a45"
COL_SPINE = "#2a3450"
COL_TEXT  = "#e8eaf6"

COL_CYAN   = "#00e5ff"
COL_AMBER  = "#ffab00"
COL_PURPLE = "#bb86fc"
COL_GREEN  = "#69f0ae"
COL_RED    = "#ef5350"
COL_WAIT   = "#546e7a"
COL_SERVE  = COL_GREEN

# ── History lengths ────────────────────────────────────────────────────────────
HISTORY    = 200   # steps kept for line charts
BAR_WINDOW = 50    # last N steps for wait/serve bar

DEFAULT_MODEL = os.path.join(ROOT, "models", "best", "best_model")
FALLBACK_MODEL = os.path.join(ROOT, "models", "ppo_batching_final")


# ─────────────────────────────────────────────────────────────────────────────
# Live state object (shared between animation frames)
# ─────────────────────────────────────────────────────────────────────────────

class LiveState:
    """Holds all mutable state shared across animation frames."""

    def __init__(self, model, env):
        self.model = model
        self.env   = env

        # Rolling histories
        self.q_hist      = collections.deque(maxlen=HISTORY)  # queue size
        self.r_hist      = collections.deque(maxlen=HISTORY)  # step reward
        self.a_hist      = collections.deque(maxlen=BAR_WINDOW)  # 0/1 actions
        self.serve_steps = collections.deque(maxlen=HISTORY)  # steps where served

        # Current env state
        self.obs         = None
        self.last_action = 0
        self.last_reward = 0.0
        self.step_count  = 0
        self.ep_count    = 0
        self.ep_reward   = 0.0

        # obs component names (for readout panel)
        self.obs_names = [
            "pending_req",
            "oldest_wait_ms",
            "request_rate",
            "since_serve_ms",
            "fill_ratio",
            "time_of_day",
        ]

        self._reset()

    def _reset(self):
        self.obs, _ = self.env.reset()
        self.ep_reward  = 0.0
        self.ep_count  += 1

    def step(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        obs, reward, terminated, truncated, info = self.env.step(int(action))

        self.obs         = obs
        self.last_action = int(action)
        self.last_reward = float(reward)
        self.ep_reward  += self.last_reward
        self.step_count += 1

        # History
        self.q_hist.append(float(obs[0]))      # pending_requests
        self.r_hist.append(self.last_reward)
        self.a_hist.append(self.last_action)
        if self.last_action == 1:
            self.serve_steps.append(len(self.q_hist) - 1)

        if terminated or truncated:
            self._reset()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(COL_SPINE)
    ax.tick_params(colors=COL_TEXT, labelsize=8)
    ax.xaxis.label.set_color(COL_TEXT)
    ax.yaxis.label.set_color(COL_TEXT)
    if title:
        ax.set_title(title, color=COL_TEXT, fontsize=9.5, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    ax.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Main animation
# ─────────────────────────────────────────────────────────────────────────────

def build_demo(model_path: str, interval_ms: int = 50):
    # ── Load model + env ──────────────────────────────────────────────────────
    env   = BatchingEnv()
    model = PPO.load(model_path, env=env)
    state = LiveState(model, env)

    # ── Figure skeleton ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "text.color":     COL_TEXT,
        "figure.facecolor": BG_FIG,
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 7),
        facecolor=BG_FIG,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.08,
                        hspace=0.38, wspace=0.30)

    ax_queue, ax_reward, ax_bar, ax_text = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    # ── Super-title ────────────────────────────────────────────────────────────
    fig_title = fig.suptitle(
        "RL Batching Agent  —  Live Demo",
        color=COL_TEXT, fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Panel 0: Queue size line ───────────────────────────────────────────────
    _style_ax(ax_queue, "Pending Queue Size", xlabel="Step (last 200)", ylabel="Requests")
    (line_queue,) = ax_queue.plot([], [], color=COL_CYAN, linewidth=1.5, zorder=2)
    scatter_serve = ax_queue.scatter([], [], color=COL_GREEN, s=28, zorder=5,
                                     label="Serve", edgecolors="none")
    ax_queue.legend(facecolor=BG_AX, edgecolor=COL_SPINE,
                    labelcolor=COL_TEXT, fontsize=7.5, loc="upper left")
    ax_queue.set_xlim(0, HISTORY)
    ax_queue.set_ylim(0, CONFIG["max_batch_size"] * 1.08)

    # ── Panel 1: Step reward line ──────────────────────────────────────────────
    _style_ax(ax_reward, "Step Reward", xlabel="Step (last 200)", ylabel="Reward")
    (line_reward,) = ax_reward.plot([], [], color=COL_AMBER, linewidth=1.5, zorder=2)
    ax_reward.axhline(0, color=COL_RED, linewidth=1.0, linestyle="--",
                      alpha=0.75, zorder=1)
    ax_reward.set_xlim(0, HISTORY)

    # ── Panel 2: Wait / Serve bar chart ───────────────────────────────────────
    _style_ax(ax_bar, f"Decisions (last {BAR_WINDOW} steps)")
    bar_rects = ax_bar.bar(
        ["WAIT", "SERVE"], [0, 0],
        color=[COL_WAIT, COL_SERVE],
        width=0.5, alpha=0.9, zorder=3,
    )
    ax_bar.set_ylim(0, BAR_WINDOW)
    ax_bar.tick_params(axis="x", labelsize=9)
    count_labels = [
        ax_bar.text(rect.get_x() + rect.get_width() / 2, 0, "0",
                    ha="center", va="bottom", color=COL_TEXT,
                    fontsize=9, fontweight="bold")
        for rect in bar_rects
    ]

    # ── Panel 3: State readout (text box) ─────────────────────────────────────
    _style_ax(ax_text, "Current State")
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)
    ax_text.grid(False)
    # Rounded box
    fancy_box = mpatches.FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=COL_SPINE,
        facecolor="#0d1525",
        zorder=1,
    )
    ax_text.add_patch(fancy_box)
    readout_text = ax_text.text(
        0.5, 0.5, "",
        ha="center", va="center",
        color=COL_TEXT,
        fontsize=8.5,
        fontfamily="monospace",
        zorder=2,
        linespacing=1.8,
    )

    # Subtitle showing episode + step
    ep_label = fig.text(
        0.5, 0.935,
        "Episode 1  |  Step 0  |  Ep Reward: 0.00",
        ha="center", va="top",
        color=COL_PURPLE, fontsize=9,
    )

    # ── FuncAnimation update ───────────────────────────────────────────────────
    def update(_frame):
        # Advance environment
        state.step()

        xs = np.arange(len(state.q_hist))

        # -- Panel 0: queue --
        line_queue.set_data(xs, list(state.q_hist))

        serve_xs, serve_ys = [], []
        for si in state.serve_steps:
            if 0 <= si < len(state.q_hist):
                serve_xs.append(si)
                serve_ys.append(list(state.q_hist)[si])
        scatter_serve.set_offsets(
            np.c_[serve_xs, serve_ys] if serve_xs else np.empty((0, 2))
        )

        # -- Panel 1: reward --
        r_arr = list(state.r_hist)
        line_reward.set_data(np.arange(len(r_arr)), r_arr)
        if r_arr:
            lo, hi = min(r_arr), max(r_arr)
            margin  = max(abs(hi - lo) * 0.15, 1.0)
            ax_reward.set_ylim(lo - margin, hi + margin)

        # -- Panel 2: bar chart --
        a_arr  = list(state.a_hist)
        n_wait  = a_arr.count(0)
        n_serve = a_arr.count(1)
        for rect, height, label_obj, val in zip(
            bar_rects, [n_wait, n_serve], count_labels, [n_wait, n_serve]
        ):
            rect.set_height(height)
            label_obj.set_position((
                rect.get_x() + rect.get_width() / 2,
                height + 0.5,
            ))
            label_obj.set_text(str(val))

        # -- Panel 3: state readout --
        obs = state.obs
        action_label = "SERVE ▶" if state.last_action == 1 else "WAIT  ◼"
        action_color = COL_GREEN if state.last_action == 1 else COL_AMBER
        lines = [
            f"{'── STATE ──':^30}",
            "",
        ]
        for name, val in zip(state.obs_names, obs):
            lines.append(f"  {name:<18s}  {val:>7.2f}")
        lines += [
            "",
            f"{'── LAST ACTION ──':^30}",
            "",
            f"  {'Action:':18s}  {action_label}",
            f"  {'Step Reward:':18s}  {state.last_reward:>+7.3f}",
            f"  {'Ep Reward:':18s}  {state.ep_reward:>+7.2f}",
        ]
        readout_text.set_text("\n".join(lines))
        readout_text.set_color(COL_TEXT)  # base color; action annotation separate
        # Recolour action line dynamically via annotate isn't practical — keep static

        # -- Episode label --
        ep_label.set_text(
            f"Episode {state.ep_count}  │  "
            f"Step {state.step_count}  │  "
            f"Ep Reward: {state.ep_reward:+.1f}"
        )

        return (line_queue, scatter_serve, line_reward,
                *bar_rects, *count_labels, readout_text, ep_label)

    anim = animation.FuncAnimation(
        fig,
        update,
        interval=interval_ms,
        blit=False,   # blit=True conflicts with text updates on some backends
        cache_frame_data=False,
    )

    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="Live PPO Batching Demo")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained SB3 model (without .zip). "
             "Defaults to models/best/best_model, then models/ppo_batching_final.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (1.0 = real-time at 50ms/frame).",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model:
        model_path = args.model
    elif os.path.exists(DEFAULT_MODEL + ".zip") or os.path.exists(DEFAULT_MODEL):
        model_path = DEFAULT_MODEL
    elif os.path.exists(FALLBACK_MODEL + ".zip") or os.path.exists(FALLBACK_MODEL):
        model_path = FALLBACK_MODEL
    else:
        print(
            "[ERROR] No trained model found.\n"
            "  Run: python3 agent/train.py\n"
            "  Then retry."
        )
        sys.exit(1)

    print(f"[demo] Loading model: {model_path}")
    interval_ms = max(10, int(50 / args.speed))
    print(f"[demo] Frame interval: {interval_ms} ms  (speed ×{args.speed})")
    print("[demo] Close the window to exit.\n")

    fig, anim = build_demo(model_path, interval_ms=interval_ms)
    plt.show()


if __name__ == "__main__":
    main()
