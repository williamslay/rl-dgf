#!/usr/bin/env python3
"""Generate Phase 1B training and evaluation figures from raw JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return arr
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_learning_curve(episodes: list[dict], metrics: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    # 1. Episode Return
    ax = axes[0, 0]
    if episodes:
        x = [e["end_timestep"] for e in episodes]
        y = [e["ep_return"] for e in episodes]
        ax.plot(x, y, alpha=0.35, color="#1f77b4", label="Episode Return")
        ax.plot(x, smooth(y, window=15), color="#1f77b4", linewidth=2.0, label="Moving Avg (15)")
    ax.set_title("Episode Return (Target-Conditioned Coverage Delta)")
    ax.set_xlabel("RL Timesteps")
    ax.set_ylabel("Episode Return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Policy & Value Loss
    ax = axes[0, 1]
    loss_rows = [m for m in metrics if "train/value_loss" in m and "train/policy_gradient_loss" in m]
    if loss_rows:
        lx = [m["time/total_timesteps"] for m in loss_rows]
        vloss = [m["train/value_loss"] for m in loss_rows]
        ploss = [m["train/policy_gradient_loss"] for m in loss_rows]
        ax.plot(lx, vloss, color="#d62728", label="Value Loss")
        ax2 = ax.twinx()
        ax2.plot(lx, ploss, color="#2ca02c", linestyle="--", label="Policy Gradient Loss")
        ax2.set_ylabel("Policy Loss", color="#2ca02c")
    ax.set_title("PPO Training Losses")
    ax.set_xlabel("RL Timesteps")
    ax.set_ylabel("Value Loss", color="#d62728")
    ax.grid(alpha=0.3)

    # 3. Entropy (stored as negative entropy loss in SB3: entropy = -entropy_loss)
    ax = axes[1, 0]
    ent_rows = [m for m in metrics if "train/entropy_loss" in m]
    if ent_rows:
        ex = [m["time/total_timesteps"] for m in ent_rows]
        ent = [-m["train/entropy_loss"] for m in ent_rows]
        ax.plot(ex, ent, color="#9467bd", linewidth=2.0, label="Policy Entropy")
        ax.axhline(np.log(12.0), color="gray", linestyle=":", label=f"Max Entropy ln(12)={np.log(12):.2f}")
    ax.set_title("Policy Entropy Over Training")
    ax.set_xlabel("RL Timesteps")
    ax.set_ylabel("Entropy (nats)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 4. Energy Fraction over Training Episodes
    ax = axes[1, 1]
    if episodes:
        x = [e["end_timestep"] for e in episodes]
        for lvl, col in [("32", "#1f77b4"), ("128", "#ff7f0e"), ("512", "#2ca02c")]:
            fracs = [e["energy_fracs"].get(lvl, 0.0) for e in episodes]
            ax.plot(x, smooth(fracs, window=15), color=col, linewidth=2.0, label=f"Energy={lvl} (MA15)")
    ax.set_title("Energy Selection Trend During Training")
    ax.set_xlabel("RL Timesteps")
    ax.set_ylabel("Action Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def interpolate_cov_curve(steps: list[dict], grid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    by_ep: dict[int, list[dict]] = {}
    for s in steps:
        by_ep.setdefault(int(s["episode"]), []).append(s)

    curves = []
    for ep_steps in by_ep.values():
        ep_steps.sort(key=lambda r: int(r["step_id"]))
        xs = np.array([0] + [int(r["budget_used"]) for r in ep_steps], dtype=float)
        ys = np.array([0] + [int(r["target_cov_total"]) for r in ep_steps], dtype=float)
        interp = np.interp(grid_x, xs, ys)
        curves.append(interp)

    mat = np.vstack(curves)
    return mat.mean(axis=0), mat.std(axis=0)


def plot_coverage_vs_execs(eval_dir: Path, episode_budget: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    grid_x = np.linspace(0, episode_budget, 200)

    series = [
        ("ppo_det", "eval_ppo_det_steps.jsonl", "PPO (Deterministic)", "#1f77b4", "-"),
        ("ppo_stoch", "eval_ppo_stoch_steps.jsonl", "PPO (Stochastic)", "#2ca02c", "--"),
        ("random", "eval_random_steps.jsonl", "Random Baseline", "#ff7f0e", "-."),
    ]
    for _, fname, label, color, ls in series:
        rows = load_jsonl(eval_dir / fname)
        if not rows:
            continue
        mean_y, std_y = interpolate_cov_curve(rows, grid_x)
        ax.plot(grid_x, mean_y, color=color, linestyle=ls, linewidth=2.2, label=label)
        ax.fill_between(grid_x, np.maximum(0, mean_y - std_y), mean_y + std_y, color=color, alpha=0.15)

    ax.set_title("Target-Conditioned Coverage vs. Fuzzing Executions")
    ax.set_xlabel("Cumulative Target Executions")
    ax.set_ylabel("Target-Conditioned Coverage (AFL Bitmap Edge Slots)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_eval_comparison(eval_summary: dict, out_path: Path) -> None:
    summaries = eval_summary["summaries"]
    policies = list(summaries.keys())
    labels = [p.replace("_", "\n") for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)

    # 1. Mean Return / Target Coverage (+/- std)
    means = [summaries[p]["mean_return"] for p in policies]
    stds = [summaries[p]["std_return"] for p in policies]
    axes[0].bar(labels, means, yerr=stds, capsize=5, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(policies)])
    axes[0].set_title("Mean Episode Return (± std)")
    axes[0].set_ylabel("Target Coverage Delta")
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Mean Targets Reached (out of num_targets) & Corpus Size
    t_reached = [summaries[p]["mean_targets_reached"] for p in policies]
    axes[1].bar(labels, t_reached, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(policies)])
    axes[1].set_title("Mean Unique Targets Reached")
    axes[1].set_ylabel("Targets Reached")
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Mean Queued Seeds Discovered
    q_items = [summaries[p]["mean_queued_items"] for p in policies]
    axes[2].bar(labels, q_items, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(policies)])
    axes[2].set_title("Mean Final Corpus Size")
    axes[2].set_ylabel("Queue Entries")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_energy_distribution(episodes: list[dict], eval_summary: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)

    # Left: Training early vs late energy share
    ax = axes[0]
    lvls = ["32", "128", "512"]
    if episodes:
        mid = max(1, len(episodes) // 2)
        early = episodes[:mid]
        late = episodes[mid:]
        early_fracs = [np.mean([e["energy_fracs"].get(l, 0.0) for e in early]) for l in lvls]
        late_fracs = [np.mean([e["energy_fracs"].get(l, 0.0) for e in late]) for l in lvls]
        x = np.arange(len(lvls))
        w = 0.35
        ax.bar(x - w / 2, early_fracs, width=w, label="Training First Half", color="#6baed6")
        ax.bar(x + w / 2, late_fracs, width=w, label="Training Second Half", color="#08519c")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Energy {l}" for l in lvls])
        ax.set_ylim(0, 1.05)
        ax.legend()
    ax.set_title("Training Energy Action Share")
    ax.set_ylabel("Fraction of Steps")
    ax.grid(axis="y", alpha=0.3)

    # Right: Evaluation Energy distribution by policy
    ax = axes[1]
    summaries = eval_summary["summaries"]
    policies = list(summaries.keys())
    x = np.arange(len(lvls))
    w = 0.8 / max(1, len(policies))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for idx, p in enumerate(policies):
        fracs = [summaries[p]["energy_fracs"].get(l, 0.0) for l in lvls]
        offset = (idx - (len(policies) - 1) / 2.0) * w
        ax.bar(x + offset, fracs, width=w, label=p, color=colors[idx % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Energy {l}" for l in lvls])
    ax.set_ylim(0, 1.05)
    ax.set_title("Evaluation Energy Action Share")
    ax.set_ylabel("Fraction of Steps")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot Phase 1B training & evaluation figures.")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_jsonl(args.train_dir / "train_episodes.jsonl")
    metrics = load_jsonl(args.train_dir / "train_metrics.jsonl")
    eval_summary = json.loads((args.eval_dir / "eval_summary.json").read_text(encoding="utf-8"))

    plot_learning_curve(episodes, metrics, args.out_dir / "fig1_learning_curve.png")
    plot_coverage_vs_execs(
        args.eval_dir,
        episode_budget=int(eval_summary["episode_budget"]),
        out_path=args.out_dir / "fig2_target_cov_vs_execs.png",
    )
    plot_eval_comparison(eval_summary, args.out_dir / "fig3_ppo_vs_random.png")
    plot_energy_distribution(episodes, eval_summary, args.out_dir / "fig4_energy_distribution.png")
    print(f"Saved 4 figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
