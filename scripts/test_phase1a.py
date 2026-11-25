#!/usr/bin/env python3
"""Phase 1A full integration & acceptance test suite for DirectedFuzzEnv."""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rl_dgf import DirectedFuzzEnv

AFL_FUZZ_BIN = ROOT_DIR / "backend" / "AFLRun" / "afl-fuzz"
TARGET_BIN = ROOT_DIR / "benchmarks" / "smoke-target" / "target"
AFLRUN_DIR = ROOT_DIR / "benchmarks" / "smoke-target" / "aflrun_tmp"
SEEDS_DIR = ROOT_DIR / "benchmarks" / "smoke-target" / "seeds"
RESULTS_DIR = ROOT_DIR / "results" / "phase1a"


def make_env(
    work_subdir: str,
    episode_budget: int = 10_000,
    keep_artifacts: bool = False,
) -> DirectedFuzzEnv:
    return DirectedFuzzEnv(
        target_bin=TARGET_BIN,
        aflrun_dir=AFLRUN_DIR,
        seeds_dir=SEEDS_DIR,
        afl_fuzz_bin=AFL_FUZZ_BIN,
        work_root=RESULTS_DIR / work_subdir,
        target_args=["@@"],
        k_candidates=4,
        energy_levels=(32, 128, 512),
        episode_budget=episode_budget,
        keep_artifacts=keep_artifacts,
    )


def count_child_afl_processes() -> int:
    out = subprocess.check_output(["ps", "-eo", "pid,ppid,comm"], text=True)
    my_pid = os.getpid()
    cnt = 0
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and int(parts[1]) == my_pid and "afl-fuzz" in parts[2]:
            cnt += 1
    return cnt


def count_user_shm() -> int:
    user = os.environ.get("USER", "")
    out = subprocess.check_output(["ipcs", "-m"], text=True)
    cnt = 0
    for line in out.splitlines():
        cols = line.split()
        if len(cols) >= 3 and cols[2] == user:
            cnt += 1
    return cnt


def test_gymnasium_check_env() -> dict:
    env = make_env("check_env_work", episode_budget=2_000)
    try:
        # Gymnasium's check_env verifies spaces, reset(seed=...), step(), determinism API.
        check_env(env, skip_render_check=True)
        return {"passed": True}
    finally:
        env.close()
        shutil.rmtree(RESULTS_DIR / "check_env_work", ignore_errors=True)


def test_default_budget_episode() -> dict:
    """Run an episode with Phase 1A default 10,000 budget until truncation."""
    env = make_env("ep_10k_work", episode_budget=10_000)
    rng = np.random.default_rng(12345)
    t_start = time.perf_counter()
    try:
        _, reset_info = env.reset(seed=42)
        cum_reward = 0.0
        init_tcov = reset_info["target_cov_total"]
        prev_tcov = init_tcov
        steps = []
        truncated = False
        terminated = False

        while not (terminated or truncated):
            action = int(rng.integers(0, env.action_space.n))
            _, reward, terminated, truncated, info = env.step(action)
            assert reward >= 0.0
            assert info["target_cov_total"] >= prev_tcov
            assert reward == float(info["target_cov_total"] - prev_tcov)
            assert (
                info["actual_exec_count"]
                == info["total_execs_after"] - info["total_execs_before"]
            )
            assert (
                info["actual_exec_count"]
                == info["fuzz_execs"] + info["cal_execs"] + info["trim_execs"]
            )
            prev_tcov = info["target_cov_total"]
            cum_reward += reward
            steps.append({**info, **env.last_step_timing})

        wall_s = time.perf_counter() - t_start
        assert terminated and not truncated, (
            f"Expected terminated=True, truncated=False on budget exhaustion; "
            f"got {terminated=}, {truncated=}"
        )
        assert steps[-1]["budget_used"] >= 10_000
        assert cum_reward == float(prev_tcov - init_tcov)
        total_execs_rl = steps[-1]["budget_used"]
        mean_ipc_us = statistics.mean(s["ipc_roundtrip_us"] for s in steps)
        p95_ipc_us = float(np.percentile([s["ipc_roundtrip_us"] for s in steps], 95))
        mean_fuzz_us = statistics.mean(s["elapsed_us"] for s in steps)

        return {
            "passed": True,
            "steps": len(steps),
            "budget_used": total_execs_rl,
            "init_dry_run_execs": reset_info["init_dry_run_execs"],
            "wall_s": round(wall_s, 3),
            "throughput_execs_per_sec": round(total_execs_rl / wall_s, 2),
            "final_queued_items": steps[-1]["queued_items"],
            "init_target_cov": init_tcov,
            "final_target_cov": prev_tcov,
            "cumulative_reward": cum_reward,
            "final_global_cov": steps[-1]["global_cov_total"],
            "final_aflrun_reached_targets": steps[-1]["aflrun_reached_targets"],
            "mean_ipc_roundtrip_us": round(mean_ipc_us, 1),
            "p95_ipc_roundtrip_us": round(p95_ipc_us, 1),
            "mean_step_fuzz_us": round(mean_fuzz_us, 1),
        }
    finally:
        env.close()
        shutil.rmtree(RESULTS_DIR / "ep_10k_work", ignore_errors=True)


def test_100plus_steps_and_controls() -> dict:
    """Run >=100 consecutive RL steps, check energy levels, seed switching & corpus growth."""
    env = make_env("ep_100steps_work", episode_budget=40_000)
    rng = np.random.default_rng(2026)
    jsonl_path = RESULTS_DIR / "steps_100plus.jsonl"
    try:
        _, reset_info = env.reset(seed=42)
        prev_tcov = reset_info["target_cov_total"]
        init_tcov = prev_tcov
        cum_reward = 0.0
        steps = []

        # Verify <4 candidate wraparound initially when corpus has 1 seed.
        assert reset_info["valid_seed_count"] == 1
        assert reset_info["slot_wrapped"] == [False, True, True, True]
        assert all(sid == 0 for sid in reset_info["candidate_seed_ids"])

        # Step 1: pick slot 3 (wraps safely to valid seed 0) with budget 512 to bootstrap corpus.
        _, r0, term, trunc, info0 = env.step(3 * 3 + 2)  # slot=3, energy=512
        assert info0["slot_wrapped"] is True
        assert info0["selected_seed_id"] == 0
        cum_reward += r0
        prev_tcov = info0["target_cov_total"]
        steps.append({**info0, **env.last_step_timing})

        # Now run >= 119 more random steps (total 120 consecutive steps).
        distinct_selected_seeds = {info0["selected_seed_id"]}
        energy_actuals: dict[int, list[int]] = {32: [], 128: [], 512: []}
        energy_exact_no_cal: dict[int, list[int]] = {32: [], 128: [], 512: []}
        cal_overrun_steps = []

        for _ in range(119):
            action = int(rng.integers(0, env.action_space.n))
            _, reward, term, trunc, info = env.step(action)
            assert not term and not trunc
            assert reward >= 0.0
            assert info["target_cov_total"] >= prev_tcov
            assert reward == float(info["target_cov_total"] - prev_tcov)
            assert (
                info["actual_exec_count"]
                == info["total_execs_after"] - info["total_execs_before"]
            )
            assert (
                info["actual_exec_count"]
                == info["fuzz_execs"] + info["cal_execs"] + info["trim_execs"]
            )
            distinct_selected_seeds.add(info["selected_seed_id"])

            req_b = info["requested_exec_budget"]
            act_b = info["actual_exec_count"]
            energy_actuals[req_b].append(act_b)
            if info["cal_execs"] == 0 and info["trim_execs"] == 0 and not info["skipped"]:
                assert act_b == req_b, f"Expected exact budget {req_b}, got {act_b}"
                energy_exact_no_cal[req_b].append(act_b)
            else:
                cal_overrun_steps.append(
                    {
                        "step_id": info["step_id"],
                        "requested": req_b,
                        "actual": act_b,
                        "fuzz_execs": info["fuzz_execs"],
                        "cal_execs": info["cal_execs"],
                    }
                )

            prev_tcov = info["target_cov_total"]
            cum_reward += reward
            steps.append({**info, **env.last_step_timing})

        # Explicitly test all 4 candidate slots execute different queue entries if >= 4 valid seeds.
        slot_test_ids = []
        assert steps[-1]["valid_seed_count"] >= 4, "Expected corpus growth to >= 4 valid seeds"
        for slot in range(4):
            _, r_s, _, _, info_s = env.step(slot * 3 + 0)  # each slot with budget 32
            slot_test_ids.append(info_s["selected_seed_id"])
            cum_reward += r_s
            prev_tcov = info_s["target_cov_total"]
            steps.append({**info_s, **env.last_step_timing})
        assert len(set(slot_test_ids)) == 4, f"Expected 4 distinct seed IDs, got {slot_test_ids}"

        with jsonl_path.open("w", encoding="utf-8") as fh:
            for item in steps:
                fh.write(json.dumps(item) + "\n")

        assert len(steps) >= 100
        assert steps[-1]["queued_items"] > 1
        assert cum_reward == float(prev_tcov - init_tcov)
        assert statistics.mean(energy_actuals[32]) < statistics.mean(energy_actuals[128])
        assert statistics.mean(energy_actuals[128]) < statistics.mean(energy_actuals[512])

        return {
            "passed": True,
            "total_steps": len(steps),
            "final_queued_items": steps[-1]["queued_items"],
            "distinct_seeds_executed": sorted(distinct_selected_seeds),
            "slot_test_seed_ids": slot_test_ids,
            "energy_stats": {
                str(lvl): {
                    "count": len(vals),
                    "exact_no_cal_count": len(energy_exact_no_cal[lvl]),
                    "min_actual": min(vals),
                    "max_actual": max(vals),
                    "mean_actual": round(statistics.mean(vals), 2),
                }
                for lvl, vals in energy_actuals.items()
            },
            "cal_overrun_count": len(cal_overrun_steps),
            "init_target_cov": init_tcov,
            "final_target_cov": prev_tcov,
            "cumulative_reward": cum_reward,
            "jsonl_log": str(jsonl_path),
        }
    finally:
        env.close()
        shutil.rmtree(RESULTS_DIR / "ep_100steps_work", ignore_errors=True)


def test_consecutive_resets_and_leaks() -> dict:
    """Verify >= 10 consecutive resets leave no leaked processes, FDs, files, or SysV SHM."""
    shm_before = count_user_shm()
    fd_before = len(os.listdir("/proc/self/fd"))
    work_dir = RESULTS_DIR / "reset_leak_work"
    shutil.rmtree(work_dir, ignore_errors=True)

    env = make_env("reset_leak_work", episode_budget=1_000, keep_artifacts=False)
    try:
        for i in range(12):
            obs, info = env.reset(seed=100 + i)
            assert obs.shape == env.observation_space.shape
            assert info["queued_items"] >= 1
            _, _, _, _, s_info = env.step(i % env.action_space.n)
            assert s_info["actual_exec_count"] > 0
    finally:
        env.close()

    time.sleep(0.2)
    shm_after = count_user_shm()
    fd_after = len(os.listdir("/proc/self/fd"))
    child_afl = count_child_afl_processes()
    leftover_entries = list(work_dir.glob("**/*")) if work_dir.exists() else []
    shutil.rmtree(work_dir, ignore_errors=True)

    assert child_afl == 0, f"Leaked child afl-fuzz processes: {child_afl}"
    assert shm_after <= shm_before, f"SHM leak: before={shm_before}, after={shm_after}"
    assert abs(fd_after - fd_before) <= 2, f"FD leak: before={fd_before}, after={fd_after}"
    assert len(leftover_entries) <= 1, f"Leftover temp dirs: {leftover_entries}"

    return {
        "passed": True,
        "resets_completed": 12,
        "shm_before": shm_before,
        "shm_after": shm_after,
        "fd_before": fd_before,
        "fd_after": fd_after,
        "leaked_child_afl_processes": child_afl,
    }


def test_fixed_seed_reproducibility() -> dict:
    """Run two 20-step trajectories with reset(seed=42) and compare traces."""
    actions = [0, 1, 0, 2, 1, 0, 3, 4, 6, 9, 1, 2, 5, 7, 10, 0, 4, 8, 11, 2]
    trajectories = []
    for run_idx in range(2):
        env = make_env(f"repro_work_{run_idx}", episode_budget=10_000)
        try:
            obs0, rinfo = env.reset(seed=42)
            trace = [
                {
                    "queued_items": rinfo["queued_items"],
                    "global_cov_total": rinfo["global_cov_total"],
                    "target_cov_total": rinfo["target_cov_total"],
                    "candidate_seed_ids": rinfo["candidate_seed_ids"],
                }
            ]
            for act in actions:
                obs, reward, term, trunc, info = env.step(act)
                trace.append(
                    {
                        "selected_seed_id": info["selected_seed_id"],
                        "actual_exec_count": info["actual_exec_count"],
                        "fuzz_execs": info["fuzz_execs"],
                        "cal_execs": info["cal_execs"],
                        "new_seeds": info["new_seeds"],
                        "queued_items": info["queued_items"],
                        "global_cov_total": info["global_cov_total"],
                        "target_cov_total": info["target_cov_total"],
                        "reward": reward,
                        "candidate_seed_ids": info["candidate_seed_ids"],
                    }
                )
                if term or trunc:
                    break
            trajectories.append(trace)
        finally:
            env.close()
            shutil.rmtree(RESULTS_DIR / f"repro_work_{run_idx}", ignore_errors=True)

    exact_match = trajectories[0] == trajectories[1]
    mismatches = []
    if not exact_match:
        for i, (a, b) in enumerate(zip(trajectories[0], trajectories[1])):
            if a != b:
                mismatches.append({"step": i, "run0": a, "run1": b})
    return {
        "passed": exact_match,
        "exact_match": exact_match,
        "steps_compared": len(trajectories[0]),
        "mismatch_count": len(mismatches),
        "mismatches_sample": mismatches[:3],
    }


def test_native_aflrun_regression() -> dict:
    """Verify native AFLRun mode still runs unchanged on smoke-target (5s check)."""
    out_dir = RESULTS_DIR / "native_regression_out"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AFL_NO_UI"] = "1"
    env["AFL_SKIP_CPUFREQ"] = "1"
    env["AFL_PATH"] = str(ROOT_DIR / "backend" / "AFLRun")
    env["AFLRUN_TMP"] = str(AFLRUN_DIR)

    cmd = [
        str(AFL_FUZZ_BIN),
        "-d",
        "-V",
        "5",
        "-i",
        str(SEEDS_DIR),
        "-o",
        str(out_dir),
        "--config",
        "init_cov_quant=30:log_at_begin=1:log_check_interval=2000",
        "--",
        str(TARGET_BIN),
        "@@",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    log_text = proc.stdout + "\n" + proc.stderr
    (RESULTS_DIR / "native_regression.log").write_text(log_text, encoding="utf-8")

    stats_path = out_dir / "default" / "fuzzer_stats"
    assert proc.returncode == 0, f"Native AFLRun failed (rc={proc.returncode}): {log_text[-500:]}"
    assert stats_path.exists(), "Missing fuzzer_stats from native AFLRun"
    stats_text = stats_path.read_text(encoding="utf-8")

    def extract_num(key: str) -> float:
        m = re.search(rf"^{re.escape(key)}\s*:\s*([0-9.]+)", stats_text, re.MULTILINE)
        return float(m.group(1)) if m else 0.0

    execs_done = int(extract_num("execs_done"))
    execs_per_sec = extract_num("execs_per_sec")
    corpus_count = int(extract_num("corpus_count"))
    shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "passed": execs_done > 0 and corpus_count >= 1,
        "returncode": proc.returncode,
        "execs_done": execs_done,
        "execs_per_sec": execs_per_sec,
        "corpus_count": corpus_count,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    print("[1/6] Running gymnasium.utils.env_checker.check_env...", flush=True)
    report["check_env"] = test_gymnasium_check_env()

    print("[2/6] Running 10,000-exec default episode & latency test...", flush=True)
    report["default_10k_episode"] = test_default_budget_episode()

    print("[3/6] Running >=100-step random trajectory & energy/seed tests...", flush=True)
    report["steps_100plus_and_controls"] = test_100plus_steps_and_controls()

    print("[4/6] Running 12x consecutive reset() & leak checks...", flush=True)
    report["consecutive_resets"] = test_consecutive_resets_and_leaks()

    print("[5/6] Running fixed-seed reproducibility test (seed=42)...", flush=True)
    report["fixed_seed_reproducibility"] = test_fixed_seed_reproducibility()

    print("[6/6] Running native AFLRun regression & throughput check (5s)...", flush=True)
    report["native_aflrun_regression"] = test_native_aflrun_regression()

    summary_path = RESULTS_DIR / "phase1a_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
