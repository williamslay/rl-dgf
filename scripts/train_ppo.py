#!/usr/bin/env python3
"""Minimal SB3 PPO training script for Phase 1B (Discrete(12), 30-dim obs)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import KVWriter, configure
from stable_baselines3.common.monitor import Monitor

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


class JsonlKVWriter(KVWriter):
    """Captures SB3 logger dumps into a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        record: dict[str, Any] = {"step": int(step)}
        for k, v in key_values.items():
            if isinstance(v, (np.floating, float)):
                record[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                record[k] = int(v)
            elif isinstance(v, str):
                record[k] = v
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class FuzzTrainingCallback(BaseCallback):
    """Logs per-step and per-episode fuzzing/RL telemetry during PPO training."""

    def __init__(self, out_dir: Path, energy_levels: tuple[int, ...]) -> None:
        super().__init__(verbose=0)
        self.out_dir = out_dir
        self.energy_levels = energy_levels
        self.steps_path = out_dir / "train_steps.jsonl"
        self.episodes_path = out_dir / "train_episodes.jsonl"
        self._steps_fh = None
        self._ep_fh = None
        self._episode_num = 0
        self._reset_ep_accumulators()

    def _reset_ep_accumulators(self) -> None:
        self._ep_steps = 0
        self._ep_return = 0.0
        self._ep_energy_counts: Counter[int] = Counter()
        self._ep_slot_counts: Counter[int] = Counter()
        self._ep_execs = 0

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._steps_fh = self.steps_path.open("w", encoding="utf-8")
        self._ep_fh = self.episodes_path.open("w", encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if not infos:
            return True

        info = infos[0]
        action = int(actions[0]) if len(actions) > 0 else -1
        reward = float(rewards[0]) if len(rewards) > 0 else 0.0
        done = bool(dones[0]) if len(dones) > 0 else False

        # SB3 VecEnv moves terminal info into `terminal_observation` / preserves fields,
        # and sets TimeLimit.truncated if truncated=True.
        energy_level = int(info.get("action_energy_level", 0))
        slot_idx = int(info.get("slot_idx", -1))

        self._ep_steps += 1
        self._ep_return += reward
        self._ep_execs = int(info.get("budget_used", self._ep_execs))
        if energy_level:
            self._ep_energy_counts[energy_level] += 1
        if slot_idx >= 0:
            self._ep_slot_counts[slot_idx] += 1

        step_rec = {
            "global_step": int(self.num_timesteps),
            "episode_idx": self._episode_num + 1,
            "ep_step": int(info.get("step_id", self._ep_steps)),
            "action": action,
            "slot_idx": slot_idx,
            "energy_level": energy_level,
            "selected_seed_id": int(info.get("selected_seed_id", -1)),
            "requested_exec_budget": int(info.get("requested_exec_budget", 0)),
            "actual_exec_count": int(info.get("actual_exec_count", 0)),
            "budget_used": int(info.get("budget_used", 0)),
            "reward": reward,
            "target_cov_delta": int(info.get("target_cov_delta", 0)),
            "target_cov_total": int(info.get("target_cov_total", 0)),
            "global_cov_total": int(info.get("global_cov_total", 0)),
            "aflrun_reached_targets": int(info.get("aflrun_reached_targets", 0)),
            "new_seeds": int(info.get("new_seeds", 0)),
            "queued_items": int(info.get("queued_items", 0)),
            "done": done,
            "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
        }
        assert self._steps_fh is not None
        self._steps_fh.write(json.dumps(step_rec) + "\n")

        if done:
            self._episode_num += 1
            total_actions = max(1, self._ep_steps)
            energy_dist = {
                str(lvl): self._ep_energy_counts.get(lvl, 0) for lvl in self.energy_levels
            }
            energy_frac = {
                str(lvl): round(self._ep_energy_counts.get(lvl, 0) / total_actions, 4)
                for lvl in self.energy_levels
            }
            slot_dist = {str(s): self._ep_slot_counts.get(s, 0) for s in range(4)}

            ep_rec = {
                "episode_idx": self._episode_num,
                "end_timestep": int(self.num_timesteps),
                "ep_steps": self._ep_steps,
                "ep_return": float(self._ep_return),
                "actual_execs": self._ep_execs,
                "final_target_cov": int(info.get("target_cov_total", 0)),
                "final_global_cov": int(info.get("global_cov_total", 0)),
                "final_queued_items": int(info.get("queued_items", 0)),
                "aflrun_reached_targets": int(info.get("aflrun_reached_targets", 0)),
                "energy_counts": energy_dist,
                "energy_fracs": energy_frac,
                "slot_counts": slot_dist,
                "budget_exhausted": bool(info.get("budget_exhausted", False)),
                "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
            }
            assert self._ep_fh is not None
            self._ep_fh.write(json.dumps(ep_rec) + "\n")
            self._ep_fh.flush()
            self._steps_fh.flush()

            for lvl in self.energy_levels:
                self.logger.record(
                    f"fuzz/energy_{lvl}_frac",
                    self._ep_energy_counts.get(lvl, 0) / total_actions,
                )
            self.logger.record("fuzz/ep_target_cov", int(info.get("target_cov_total", 0)))
            self.logger.record("fuzz/ep_global_cov", int(info.get("global_cov_total", 0)))
            self.logger.record("fuzz/ep_queued_items", int(info.get("queued_items", 0)))
            self.logger.record("fuzz/ep_execs", self._ep_execs)
            self._reset_ep_accumulators()

        return True

    def _on_training_end(self) -> None:
        if self._steps_fh is not None:
            self._steps_fh.flush()
            self._steps_fh.close()
        if self._ep_fh is not None:
            self._ep_fh.flush()
            self._ep_fh.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal SB3 PPO trainer for DirectedFuzzEnv.")
    p.add_argument("--target", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--target-bin", type=Path, default=None)
    p.add_argument("--aflrun-dir", type=Path, default=None)
    p.add_argument("--seeds-dir", type=Path, default=None)
    p.add_argument("--total-timesteps", type=int, default=2048)
    p.add_argument("--episode-budget", type=int, default=10_000)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--checkpoint-freq", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.target]
    target_bin = args.target_bin or preset["target_bin"]
    aflrun_dir = args.aflrun_dir or preset["aflrun_dir"]
    seeds_dir = args.seeds_dir or preset["seeds_dir"]

    out_dir: Path = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    work_root = out_dir / "rl_runs"
    tb_dir = out_dir / "tb"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_env = DirectedFuzzEnv(
        target_bin=target_bin,
        aflrun_dir=aflrun_dir,
        seeds_dir=seeds_dir,
        work_root=work_root,
        target_args=preset["target_args"],
        k_candidates=4,
        energy_levels=(32, 128, 512),
        episode_budget=args.episode_budget,
        keep_artifacts=False,
    )
    monitored_env = Monitor(raw_env, filename=str(out_dir / "monitor.csv"))
    monitored_env.reset(seed=args.seed)

    logger = configure(str(tb_dir), ["stdout", "csv", "tensorboard"])
    jsonl_writer = JsonlKVWriter(out_dir / "train_metrics.jsonl")
    logger.output_formats.append(jsonl_writer)

    model = PPO(
        policy="MlpPolicy",
        env=monitored_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    model.set_logger(logger)

    fuzz_cb = FuzzTrainingCallback(out_dir=out_dir, energy_levels=raw_env.energy_levels)
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(ckpt_dir),
        name_prefix="ppo_ckpt",
        verbose=0,
    )
    callbacks = CallbackList([fuzz_cb, ckpt_cb])

    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=False)
        final_model_path = ckpt_dir / "ppo_final"
        model.save(str(final_model_path))
        wall_s = time.perf_counter() - t0
    finally:
        jsonl_writer.close()
        monitored_env.close()

    # Verify saved model loads cleanly via PPO.load
    loaded_model = PPO.load(str(ckpt_dir / "ppo_final.zip"), device="cpu")
    dummy_obs = np.zeros(raw_env.observation_space.shape, dtype=np.float32)
    loaded_action, _ = loaded_model.predict(dummy_obs, deterministic=True)

    episodes = [
        json.loads(line)
        for line in (out_dir / "train_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "target": args.target,
        "target_bin": str(target_bin),
        "total_timesteps_requested": args.total_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "episode_budget": args.episode_budget,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "seed": args.seed,
        "wall_seconds": round(wall_s, 3),
        "episodes_completed": len(episodes),
        "final_model_path": str(ckpt_dir / "ppo_final.zip"),
        "load_verify_action": int(loaded_action),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
