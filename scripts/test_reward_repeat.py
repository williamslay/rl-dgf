#!/usr/bin/env python3
"""Standalone reward verification test for RL-DGF Phase 1A.

Verifies:
  1. Target-conditioned coverage only increases when an execution hits a target.
  2. Repeatedly hitting an already-covered target path gives reward == 0.0.
  3. Reward is always non-negative (target_cov_total is monotonic).
  4. Cumulative reward in an episode equals (final target_cov_total - init target_cov_total).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rl_dgf import DirectedFuzzEnv


AFLRUN_DIR = ROOT_DIR / "benchmarks" / "smoke-target" / "aflrun_tmp"
TARGET_BIN = ROOT_DIR / "benchmarks" / "smoke-target" / "target"
DEFAULT_SEEDS = ROOT_DIR / "benchmarks" / "smoke-target" / "seeds"
OUT_ROOT = ROOT_DIR / "results" / "phase1a"


def test_repeated_target_path_zero_reward() -> dict:
    """Seed directly at Target 1 ('BB2!') so path is already covered in dry run."""
    seeds_dir = OUT_ROOT / "reward_test_seeds_hit"
    if seeds_dir.exists():
        shutil.rmtree(seeds_dir)
    seeds_dir.mkdir(parents=True, exist_ok=True)
    (seeds_dir / "seed_hit_b").write_bytes(b"BB2!")

    env = DirectedFuzzEnv(
        target_bin=TARGET_BIN,
        aflrun_dir=AFLRUN_DIR,
        seeds_dir=seeds_dir,
        work_root=OUT_ROOT / "reward_work_hit",
        target_args=["@@"],
        episode_budget=2000,
    )
    try:
        _, reset_info = env.reset(seed=101)
        init_tcov = reset_info["target_cov_total"]
        assert init_tcov > 0, f"Expected initial target_cov_total > 0, got {init_tcov}"

        prev_tcov = init_tcov
        cum_reward = 0.0
        zero_reward_target_hit_steps = 0
        steps_log = []

        # Always pick slot 0 with energy=32 on seed 0 ('BB2!') initially;
        # even if mutations stay on already-covered target path, reward must be 0.
        for _ in range(20):
            _, reward, terminated, truncated, info = env.step(0)  # slot=0, budget=32
            assert reward >= 0.0, f"Negative reward: {reward}"
            assert info["target_cov_total"] >= prev_tcov, "target_cov_total decreased!"
            assert reward == float(info["target_cov_total"] - prev_tcov), (
                f"Reward mismatch: {reward} vs {info['target_cov_total'] - prev_tcov}"
            )
            if info["target_hit"] and info["target_cov_delta"] == 0:
                assert reward == 0.0
                zero_reward_target_hit_steps += 1
            prev_tcov = info["target_cov_total"]
            cum_reward += reward
            steps_log.append(
                {
                    "step_id": info["step_id"],
                    "seed_id": info["selected_seed_id"],
                    "target_hit": info["target_hit"],
                    "step_target_hits": info["step_target_hits"],
                    "target_cov_delta": info["target_cov_delta"],
                    "target_cov_total": info["target_cov_total"],
                    "reward": reward,
                }
            )
            if terminated or truncated:
                break

        assert zero_reward_target_hit_steps > 0, (
            "Expected repeated target-hitting steps with target_cov_delta == 0 and reward == 0"
        )
        assert cum_reward == float(prev_tcov - init_tcov), (
            f"Cumulative reward {cum_reward} != total delta {prev_tcov - init_tcov}"
        )
        return {
            "init_target_cov": init_tcov,
            "final_target_cov": prev_tcov,
            "cumulative_reward": cum_reward,
            "zero_reward_target_hit_steps": zero_reward_target_hit_steps,
            "steps": steps_log,
        }
    finally:
        env.close()
        shutil.rmtree(seeds_dir, ignore_errors=True)
        shutil.rmtree(OUT_ROOT / "reward_work_hit", ignore_errors=True)


def test_nontarget_never_increases_target_cov() -> dict:
    """Start from non-target seed ('AAAA'); verify target_cov_delta == 0 if !target_hit."""
    env = DirectedFuzzEnv(
        target_bin=TARGET_BIN,
        aflrun_dir=AFLRUN_DIR,
        seeds_dir=DEFAULT_SEEDS,
        work_root=OUT_ROOT / "reward_work_miss",
        target_args=["@@"],
        episode_budget=5000,
    )
    try:
        _, reset_info = env.reset(seed=7)
        init_tcov = reset_info["target_cov_total"]
        prev_tcov = init_tcov
        cum_reward = 0.0
        nontarget_with_global_gain = 0

        for _ in range(30):
            _, reward, terminated, truncated, info = env.step(0)  # slot=0, budget=32
            assert reward >= 0.0
            assert info["target_cov_total"] >= prev_tcov
            if not info["target_hit"]:
                assert info["target_cov_delta"] == 0, (
                    f"target_cov_delta={info['target_cov_delta']} without target_hit!"
                )
                assert reward == 0.0
                if info["global_cov_delta"] > 0:
                    nontarget_with_global_gain += 1
            prev_tcov = info["target_cov_total"]
            cum_reward += reward
            if terminated or truncated:
                break

        assert cum_reward == float(prev_tcov - init_tcov)
        return {
            "init_target_cov": init_tcov,
            "final_target_cov": prev_tcov,
            "cumulative_reward": cum_reward,
            "nontarget_with_global_gain": nontarget_with_global_gain,
        }
    finally:
        env.close()
        shutil.rmtree(OUT_ROOT / "reward_work_miss", ignore_errors=True)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    res_hit = test_repeated_target_path_zero_reward()
    res_miss = test_nontarget_never_increases_target_cov()
    summary = {"status": "PASS", "repeated_target_path": res_hit, "nontarget_guard": res_miss}
    out_file = OUT_ROOT / "reward_repeat_test.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
