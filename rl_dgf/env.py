"""Gymnasium environment wrapping AFLRun for Phase 1A directed greybox fuzzing."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_dgf.candidates import CandidateSelectionResult, select_candidates
from rl_dgf.ipc import AFLRunIPCClient


DEFAULT_ENERGY_LEVELS = (32, 128, 512)
GLOBAL_OBS_DIM = 6
CAND_OBS_DIM = 6


@dataclass
class EnvConfig:
    """Configuration for DirectedFuzzEnv."""

    target_bin: str | Path
    seeds_dir: str | Path
    aflrun_dir: str | Path | None = None
    afl_fuzz_bin: str | Path = Path("/home/sylei/RL_miniproject/backend/AFLRun/afl-fuzz")
    work_root: str | Path = Path("/home/sylei/RL_miniproject/results/rl_runs")
    target_args: list[str] = field(default_factory=list)
    k_candidates: int = 4
    energy_levels: tuple[int, ...] = DEFAULT_ENERGY_LEVELS
    episode_budget: int = 10_000
    exec_timeout_ms: int | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    keep_artifacts: bool = False
    cap_final_step_budget: bool = True


class DirectedFuzzEnv(gym.Env[np.ndarray, int]):
    """Minimal Gymnasium control environment for joint seed + energy selection."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        target_bin: str | Path | None = None,
        aflrun_dir: str | Path | None = None,
        seeds_dir: str | Path | None = None,
        afl_fuzz_bin: str | Path = Path("/home/sylei/RL_miniproject/backend/AFLRun/afl-fuzz"),
        work_root: str | Path = Path("/home/sylei/RL_miniproject/results/rl_runs"),
        target_args: list[str] | None = None,
        k_candidates: int = 4,
        energy_levels: tuple[int, ...] = DEFAULT_ENERGY_LEVELS,
        episode_budget: int = 10_000,
        exec_timeout_ms: int | None = None,
        extra_env: dict[str, str] | None = None,
        keep_artifacts: bool = False,
        cap_final_step_budget: bool = True,
    ) -> None:
        super().__init__()
        if config is None:
            if target_bin is None or seeds_dir is None:
                raise ValueError(
                    "Provide either `config` or (`target_bin`, `seeds_dir`)."
                )
            config = EnvConfig(
                target_bin=target_bin,
                aflrun_dir=aflrun_dir,
                seeds_dir=seeds_dir,
                afl_fuzz_bin=afl_fuzz_bin,
                work_root=work_root,
                target_args=list(target_args or []),
                k_candidates=k_candidates,
                energy_levels=tuple(energy_levels),
                episode_budget=episode_budget,
                exec_timeout_ms=exec_timeout_ms,
                extra_env=dict(extra_env or {}),
                keep_artifacts=keep_artifacts,
                cap_final_step_budget=cap_final_step_budget,
            )

        if config.k_candidates <= 0:
            raise ValueError("k_candidates must be >= 1")
        if not config.energy_levels or any(e <= 0 for e in config.energy_levels):
            raise ValueError("energy_levels must contain positive integers")
        if config.episode_budget <= 0:
            raise ValueError("episode_budget must be positive")

        self.config = config
        self.k_candidates = config.k_candidates
        self.energy_levels = tuple(int(e) for e in config.energy_levels)
        self.n_energy = len(self.energy_levels)
        self.episode_budget = int(config.episode_budget)

        self.action_space = spaces.Discrete(self.k_candidates * self.n_energy)
        obs_dim = GLOBAL_OBS_DIM + self.k_candidates * CAND_OBS_DIM
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._client: AFLRunIPCClient | None = None
        self._episode_counter = 0
        self._step_id = 0
        self._budget_used = 0
        self._steps_since_target_cov = 0
        self._init_dry_run_execs = 0
        self._last_state: dict[str, Any] | None = None
        self._candidates: CandidateSelectionResult | None = None
        self._episode_active = False
        self.last_step_timing: dict[str, int] = {"elapsed_us": 0, "ipc_roundtrip_us": 0}

    def decode_action(self, action: int) -> tuple[int, int]:
        """Map discrete action index -> (candidate_slot_idx, energy_level)."""
        act = int(action)
        if not self.action_space.contains(act):
            raise ValueError(f"Invalid action {act} for {self.action_space}")
        slot_idx = act // self.n_energy
        energy_idx = act % self.n_energy
        return slot_idx, self.energy_levels[energy_idx]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._cleanup_backend()

        self._episode_counter += 1
        self._step_id = 0
        self._budget_used = 0
        self._steps_since_target_cov = 0
        self.last_step_timing = {"elapsed_us": 0, "ipc_roundtrip_us": 0}

        # Derive a deterministic positive 31-bit AFLRun seed whenever Gymnasium was seeded
        # (via reset(seed=...) or prior seeding of self.np_random).
        rl_seed = int(self.np_random.integers(1, 2**31 - 1))

        # Keep socket path well under Unix 107-byte sun_path limit.
        run_dir = (
            Path(self.config.work_root)
            / f"ep_{os.getpid()}_{self._episode_counter}"
        )

        client = AFLRunIPCClient(
            afl_fuzz_bin=self.config.afl_fuzz_bin,
            target_bin=self.config.target_bin,
            aflrun_dir=self.config.aflrun_dir,
            seeds_dir=self.config.seeds_dir,
            work_dir=run_dir,
            target_args=self.config.target_args,
            rl_seed=rl_seed,
            exec_timeout_ms=self.config.exec_timeout_ms,
            extra_env=self.config.extra_env,
            keep_artifacts=self.config.keep_artifacts,
        )
        ready = client.start()

        self._client = client
        self._last_state = ready
        self._init_dry_run_execs = int(ready["total_execs"])
        self._real_map_size = max(1.0, float(ready["real_map_size"]))
        self._num_targets = max(1.0, float(ready["num_targets"]))
        self._candidates = select_candidates(ready["seeds"], k=self.k_candidates)
        self._episode_active = True

        obs = self._build_observation()
        info = {
            "episode_id": self._episode_counter,
            "rl_seed": rl_seed,
            "init_dry_run_execs": self._init_dry_run_execs,
            "budget_used": 0,
            "remaining_budget": self.episode_budget,
            "queued_items": int(ready["queued_items"]),
            "target_cov_total": int(ready["target_cov_total"]),
            "global_cov_total": int(ready["global_cov_total"]),
            "num_targets": int(ready["num_targets"]),
            "num_reachables": int(ready["num_reachables"]),
            "real_map_size": int(ready["real_map_size"]),
            "aflrun_reached_targets": int(ready["aflrun_reached_targets"]),
            "candidate_seed_ids": list(self._candidates.slot_seed_ids),
            "slot_wrapped": list(self._candidates.slot_wrapped),
            "valid_seed_count": self._candidates.valid_seed_count,
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._episode_active or self._client is None or self._candidates is None:
            raise RuntimeError("Call reset() before step().")

        slot_idx, energy_level = self.decode_action(action)
        selected_seed_id = self._candidates.slot_seed_ids[slot_idx]
        slot_wrapped = self._candidates.slot_wrapped[slot_idx]

        remaining_before = max(0, self.episode_budget - self._budget_used)
        if self.config.cap_final_step_budget and remaining_before > 0:
            requested_budget = min(energy_level, remaining_before)
        else:
            requested_budget = energy_level

        t0 = time.perf_counter_ns()
        res = self._client.step(seed_id=selected_seed_id, exec_budget=requested_budget)
        ipc_roundtrip_us = (time.perf_counter_ns() - t0) // 1000

        self._step_id += 1
        actual_exec_count = int(res["actual_exec_count"])
        self._budget_used += actual_exec_count
        remaining_after = max(0, self.episode_budget - self._budget_used)

        target_cov_delta = int(res["target_cov_delta"])
        if target_cov_delta > 0:
            self._steps_since_target_cov = 0
        else:
            self._steps_since_target_cov += 1

        reward = float(target_cov_delta)
        self._last_state = res
        self._candidates = select_candidates(res["seeds"], k=self.k_candidates)

        budget_exhausted = self._budget_used >= self.episode_budget
        backend_stopped = bool(int(res.get("stop_soon", 0)))
        # Fixed execution budget is a finite-horizon task boundary, so budget exhaustion
        # is a true MDP termination (V(s_T) = 0), not an external time truncation.
        terminated = budget_exhausted
        truncated = backend_stopped and not budget_exhausted
        if terminated or truncated:
            self._episode_active = False

        obs = self._build_observation()
        self.last_step_timing = {
            "elapsed_us": int(res["elapsed_us"]),
            "ipc_roundtrip_us": int(ipc_roundtrip_us),
        }
        info = {
            "step_id": self._step_id,
            "slot_idx": slot_idx,
            "slot_wrapped": slot_wrapped,
            "selected_seed_id": int(res["selected_seed_id"]),
            "action_energy_level": energy_level,
            "requested_exec_budget": int(res["requested_exec_budget"]),
            "actual_exec_count": actual_exec_count,
            "fuzz_execs": int(res["fuzz_execs"]),
            "cal_execs": int(res["cal_execs"]),
            "trim_execs": int(res["trim_execs"]),
            "total_execs_before": int(res["total_execs_before"]),
            "total_execs_after": int(res["total_execs_after"]),
            "new_seeds": int(res["new_seeds"]),
            "new_seed_count": int(res["new_seeds"]),
            "queued_items": int(res["queued_items"]),
            "target_hit": bool(int(res["target_hit"])),
            "step_target_hits": int(res["step_target_hits"]),
            "target_cov_delta": target_cov_delta,
            "target_cov_total": int(res["target_cov_total"]),
            "global_cov_total": int(res["global_cov_total"]),
            "global_cov_delta": int(res["global_cov_delta"]),
            "saved_crashes": int(res["saved_crashes"]),
            "saved_hangs": int(res["saved_hangs"]),
            "aflrun_reached": int(res["aflrun_reached"]),
            "aflrun_reached_targets": int(res["aflrun_reached_targets"]),
            "skipped": bool(int(res["skipped"])),
            "budget_exhausted": bool(budget_exhausted),
            "budget_used": self._budget_used,
            "remaining_budget": remaining_after,
            "candidate_seed_ids": list(self._candidates.slot_seed_ids),
            "valid_seed_count": self._candidates.valid_seed_count,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._cleanup_backend()

    def _cleanup_backend(self) -> None:
        self._episode_active = False
        if self._client is not None:
            self._client.close()
            self._client = None

    def _build_observation(self) -> np.ndarray:
        assert self._last_state is not None and self._candidates is not None
        st = self._last_state
        real_map = self._real_map_size
        num_targets = self._num_targets
        remaining_ratio = max(0.0, (self.episode_budget - self._budget_used) / float(self.episode_budget))

        global_feats = [
            min(1.0, float(st["global_cov_total"]) / real_map),
            min(1.0, float(st["target_cov_total"]) / real_map),
            min(1.0, float(st["aflrun_reached_targets"]) / num_targets),
            min(1.0, remaining_ratio),
            min(1.0, float(self._candidates.valid_seed_count) / 256.0),
            min(1.0, float(self._steps_since_target_cov) / 50.0),
        ]

        cand_feats: list[float] = []
        for seed in self._candidates.slot_seeds:
            if seed is None:
                cand_feats.extend([0.0] * CAND_OBS_DIM)
                continue
            has_dist = bool(seed.get("has_target_dist", 0))
            min_dist = float(seed.get("min_target_dist", -1.0))
            dist_prox = (1.0 / (1.0 + max(0.0, min_dist))) if (has_dist and min_dist >= 0.0) else 0.0
            exec_bucket = float(int(seed.get("exec_us", 0)) // 500)
            cand_feats.extend(
                [
                    1.0,
                    min(1.0, max(0.0, dist_prox)),
                    min(1.0, float(seed.get("fuzzed_times", 0)) / 5000.0),
                    min(1.0, exec_bucket / 20.0),
                    min(1.0, float(seed.get("fringe_cnt", 0)) / 32.0),
                    1.0 if seed.get("favored", 0) else 0.0,
                ]
            )

        obs = np.array(global_feats + cand_feats, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
