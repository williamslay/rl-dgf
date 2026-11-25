#!/usr/bin/env python3
"""Evaluate PPO checkpoint(s) vs Random Scheduling Baseline under identical budgets."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rl_dgf import DirectedFuzzEnv


PRESETS = {
    "smoke": {
        "target_bin": ROOT_DIR / "benchmarks" / "smoke-target" / "target",
        "aflrun_dir": ROOT_DIR / "benchmarks" / "smoke-target" / "aflrun_tmp",
        "seeds_dir": ROOT_DIR / "benchmarks" / "smoke-target" / "seeds",
        "target_args": ["@@"],
    },
    "pilot": {
        "target_bin": ROOT_DIR / "benchmarks" / "pilot-target" / "target",
        "aflrun_dir": ROOT_DIR / "benchmarks" / "pilot-target" / "aflrun_tmp",
        "seeds_dir": ROOT_DIR / "benchmarks" / "pilot-target" / "seeds",
        "target_args": ["@@"],
    },
}


def run_policy_episodes(
    policy_name: str,
    env: DirectedFuzzEnv,
    n_episodes: int,
    seed_base: int,
    model: PPO | None = None,
    deterministic: bool = True,
    steps_out: Path | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed_base + 99991)
    episode_results: list[dict[str, Any]] = []
    fh = steps_out.open("w", encoding="utf-8") if steps_out is not None else None

    try:
        for ep in range(n_episodes):
            ep_seed = seed_base + ep
            obs, reset_info = env.reset(seed=ep_seed)
            done = False
            ep_return = 0.0
            ep_steps = 0
            energy_counts: Counter[int] = Counter()
            slot_counts: Counter[int] = Counter()
            last_info: dict[str, Any] = dict(reset_info)

            while not done:
                if model is not None:
                    action_arr, _ = model.predict(obs, deterministic=deterministic)
                    action = int(action_arr)
                else:
                    action = int(rng.integers(0, env.action_space.n))

                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_return += float(reward)
                ep_steps += 1
                energy_counts[int(info["action_energy_level"])] += 1
                slot_counts[int(info["slot_idx"])] += 1
                last_info = info

                if fh is not None:
                    fh.write(
                        json.dumps(
                            {
                                "policy": policy_name,
                                "episode": ep,
                                "seed": ep_seed,
                                "step_id": int(info["step_id"]),
                                "action": action,
                                "slot_idx": int(info["slot_idx"]),
                                "energy_level": int(info["action_energy_level"]),
                                "selected_seed_id": int(info["selected_seed_id"]),
                                "actual_exec_count": int(info["actual_exec_count"]),
                                "budget_used": int(info["budget_used"]),
                                "reward": float(reward),
                                "target_cov_delta": int(info["target_cov_delta"]),
                                "target_cov_total": int(info["target_cov_total"]),
                                "global_cov_total": int(info["global_cov_total"]),
                                "aflrun_reached_targets": int(info["aflrun_reached_targets"]),
                                "queued_items": int(info["queued_items"]),
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                            }
                        )
                        + "\n"
                    )

            total_actions = max(1, ep_steps)
            episode_results.append(
                {
                    "policy": policy_name,
                    "episode": ep,
                    "seed": ep_seed,
                    "ep_steps": ep_steps,
                    "ep_return": float(ep_return),
                    "budget_used": int(last_info["budget_used"]),
                    "final_target_cov": int(last_info["target_cov_total"]),
                    "final_global_cov": int(last_info["global_cov_total"]),
                    "final_queued_items": int(last_info["queued_items"]),
                    "aflrun_reached_targets": int(last_info["aflrun_reached_targets"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "energy_counts": {str(lvl): energy_counts.get(lvl, 0) for lvl in env.energy_levels},
                    "energy_fracs": {
                        str(lvl): round(energy_counts.get(lvl, 0) / total_actions, 4)
                        for lvl in env.energy_levels
                    },
                    "slot_counts": {str(s): slot_counts.get(s, 0) for s in range(env.k_candidates)},
                }
            )
    finally:
        if fh is not None:
            fh.close()

    return episode_results


def summarize_episodes(episodes: list[dict[str, Any]], energy_levels: tuple[int, ...]) -> dict[str, Any]:
    returns = [e["ep_return"] for e in episodes]
    tcovs = [e["final_target_cov"] for e in episodes]
    gcovs = [e["final_global_cov"] for e in episodes]
    targets_hit = [e["aflrun_reached_targets"] for e in episodes]
    queued = [e["final_queued_items"] for e in episodes]
    execs = [e["budget_used"] for e in episodes]
    total_e = Counter()
    for e in episodes:
        for lvl in energy_levels:
            total_e[str(lvl)] += e["energy_counts"][str(lvl)]
    grand_actions = max(1, sum(total_e.values()))
    return {
        "n_episodes": len(episodes),
        "mean_return": round(statistics.mean(returns), 3),
        "std_return": round(statistics.pstdev(returns), 3),
        "min_return": float(min(returns)),
        "max_return": float(max(returns)),
        "mean_target_cov": round(statistics.mean(tcovs), 3),
        "mean_global_cov": round(statistics.mean(gcovs), 3),
        "mean_targets_reached": round(statistics.mean(targets_hit), 3),
        "mean_queued_items": round(statistics.mean(queued), 3),
        "mean_execs": round(statistics.mean(execs), 1),
        "energy_counts": dict(total_e),
        "energy_fracs": {k: round(v / grand_actions, 4) for k, v in total_e.items()},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO vs Random baseline on DirectedFuzzEnv.")
    p.add_argument("--target", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--episode-budget", type=int, default=10_000)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--also-stochastic", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.target]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(args.model_path), device="cpu")
    env = DirectedFuzzEnv(
        target_bin=preset["target_bin"],
        aflrun_dir=preset["aflrun_dir"],
        seeds_dir=preset["seeds_dir"],
        work_root=out_dir / "eval_runs",
        target_args=preset["target_args"],
        k_candidates=4,
        energy_levels=(32, 128, 512),
        episode_budget=args.episode_budget,
        keep_artifacts=False,
    )

    all_summaries: dict[str, Any] = {}
    all_episodes: dict[str, list[dict[str, Any]]] = {}

    try:
        ppo_det_eps = run_policy_episodes(
            "ppo_deterministic",
            env,
            n_episodes=args.n_episodes,
            seed_base=args.seed_base,
            model=model,
            deterministic=True,
            steps_out=out_dir / "eval_ppo_det_steps.jsonl",
        )
        all_episodes["ppo_deterministic"] = ppo_det_eps
        all_summaries["ppo_deterministic"] = summarize_episodes(ppo_det_eps, env.energy_levels)

        if args.also_stochastic:
            ppo_stoch_eps = run_policy_episodes(
                "ppo_stochastic",
                env,
                n_episodes=args.n_episodes,
                seed_base=args.seed_base,
                model=model,
                deterministic=False,
                steps_out=out_dir / "eval_ppo_stoch_steps.jsonl",
            )
            all_episodes["ppo_stochastic"] = ppo_stoch_eps
            all_summaries["ppo_stochastic"] = summarize_episodes(ppo_stoch_eps, env.energy_levels)

        rand_eps = run_policy_episodes(
            "random",
            env,
            n_episodes=args.n_episodes,
            seed_base=args.seed_base,
            model=None,
            steps_out=out_dir / "eval_random_steps.jsonl",
        )
        all_episodes["random"] = rand_eps
        all_summaries["random"] = summarize_episodes(rand_eps, env.energy_levels)
    finally:
        env.close()

    output = {
        "target": args.target,
        "model_path": str(args.model_path),
        "n_episodes": args.n_episodes,
        "episode_budget": args.episode_budget,
        "seed_base": args.seed_base,
        "summaries": all_summaries,
        "episodes": all_episodes,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(all_summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
