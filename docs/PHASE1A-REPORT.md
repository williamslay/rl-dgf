# RL-DGF Phase 1A Verification Report: Minimal Gymnasium Control Loop

**Date**: 2026-11-25  
**Workspace**: `/home/sylei/RL_miniproject`  
**Verdict**: **PASS**

---

## 1. Executive Summary

Phase 1A implemented and verified the minimal end-to-end RL control loop connecting `gymnasium.Env` (`rl_dgf.DirectedFuzzEnv`) to AFLRun over a Unix Domain Socket (`src/afl-fuzz-rl.c`). No PPO training or distance-based reward shaping was added. All integration, execution budget, monotonic reward, leak, reproducibility, and native regression checks passed after the final build.

Key verified results on `benchmarks/smoke-target/target`:
- **`gymnasium.utils.env_checker.check_env`**: **PASS** (unwrapped `DirectedFuzzEnv`, `skip_render_check=True`).
- **10,000-Execution Episode (`Discrete(12)`)**: **33** RL steps, exact `budget_used = 10,000`, `wall_s = 0.959s`, throughput **`10,426.33 execs/s`** vs native AFLRun **`10,374.15 execs/s`**.
- **IPC Latency Overhead**: mean round-trip `27,261.2 us` vs backend step fuzz `27,150.8 us` (**~110.4 us/step** IPC + JSON + Python overhead, **~0.4%** of step time).
- **124-Step Control Run (`budget=40,000`)**: Corpus grew `1 -> 18` seeds; all 4 candidate slots (`[16, 15, 13, 11]`) mapped deterministically to distinct queue entries; energy levels `32 / 128 / 512` cleanly controlled `actual_exec_count` (exact `32/32`, `128/128`, `512/512` mean actual execs).
- **Repeated-Path Reward Guard**: Target-hitting seed `"BB2!"` produced `init_target_cov = 4 -> final_target_cov = 4` (`cumulative_reward = 0.0`, `4` target-hit steps yielding `reward = 0.0`); non-target steps never incremented `target_cov_total` despite `6` global coverage gains.
- **12 Consecutive `reset()` Leak Check**: Zero leaked child `afl-fuzz` processes (`0`), zero FD leak (`9 -> 9`), zero SysV SHM leak (`108 -> 108`, after fixing an upstream AFLRun `aflrun_shm_deinit` omission in `stop_fuzzing`).
- **Fixed-Seed Reproducibility (`seed=42`)**: **21/21** checkpoints (initial + 20 steps) matched identically (`mismatch_count = 0`).

---

## 2. Architecture & AFLRun Backend Changes

AFLRun modifications were kept minimal and strictly gated behind `--rl-sock <path>` (`afl->rl_sock_fd >= 0`):

| File | Summary of Change |
| --- | --- |
| `backend/AFLRun/include/afl-fuzz.h` | Added `min_target_dist`, `avg_target_dist`, `has_target_dist` to `struct queue_entry`; added RL socket/budget/coverage accounting fields to `afl_state_t`; extended `rand_below` guard with `!afl->rl_seed_mode`; declared `afl_update_target_coverage` and `afl_rl_fuzz_loop`. |
| `backend/AFLRun/include/aflrun.h` & `src/aflrun.cpp` | Added `aflrun_compute_seed_dist` (using loaded `bb_to_dists` and `bb_to_avg_dists`) and `aflrun_get_seed_fringe_cnt` (reading `path_fringes`, `path_pro_fringes`, `reached_targets`). |
| `backend/AFLRun/src/afl-fuzz-state.c` | Initialized `afl->rl_sock_fd = -1` in `afl_state_init` and freed `target_edge_covered` in `afl_state_deinit`. |
| `backend/AFLRun/src/afl-fuzz-bitmap.c` | Implemented monotonic `afl_update_target_coverage(afl, count_hit)` using AFL bitmap edge slots (`trace_bits` when any target BB in `trace_reachables[0..num_targets-1]` fires); computed seed distances after `add_to_queue`; tracked `rl_step_cal_execs` around inline `calibrate_case`. |
| `backend/AFLRun/src/afl-fuzz-run.c` | Populated seed distances in `aflrun_new_bits`; updated $C_0^T$ in `calibrate_case` (`from_queue && !stage_cur`); invoked `afl_update_target_coverage(afl, 1)` in `common_fuzz_stuff` before `save_if_interesting` overwrites `trace_bits`; incremented `rl_step_fuzz_execs`. |
| `backend/AFLRun/src/afl-fuzz-one.c` | Bypassed native skip probabilities and `HAVOC_MIN` in RL mode; set `num_execs = rl_exec_budget`; accounted for calibration & trim execs; enforced `(total_execs - rl_step_start_execs) >= rl_exec_budget` inside havoc and splicing; prevented `perf_score *= 2` and premature `aflrun_end_cycle()` aborts in RL mode; added bounded partner fallback in splicing (`aflrun_extra` infinite-loop guard). |
| `backend/AFLRun/src/afl-fuzz.c` | Added `--rl-sock` and `--rl-seed` CLI options; allocated `target_edge_covered`; launched `afl_rl_fuzz_loop(afl)` when `--rl-sock` is active (`disable_trim = 1`); added missing `aflrun_shm_deinit(&afl->shm_run)` in `stop_fuzzing`. |
| `backend/AFLRun/src/afl-fuzz-rl.c` | New module implementing the 4-byte length-prefixed JSON client over `AF_UNIX`, seed ID validation (`seed_id < queued_items && !disabled`), `fuzz_one` dispatch, `cull_queue`, and execution/coverage telemetry. |

---

## 3. Execution Budget Audit

Every RL step tracks real executions via `afl->fsrv.total_execs` (`actual_exec_count = total_execs_after - total_execs_before`) and breaks them down into:

```text
actual_exec_count == fuzz_execs + cal_execs + trim_execs
```

Checked execution sources:
1. **Initial dry run (`perform_dry_run`)**: Runs once during `reset()` before the `ready` handshake (`init_dry_run_execs = 7` on `smoke-target`). Reported separately in `reset()` `info` and not deducted from the `10,000` RL action budget.
2. **Trimming (`trim_case`)**: Disabled in RL mode (`afl->disable_trim = 1` in `src/afl-fuzz.c`) so trimming introduces `0` unexpected executions (`trim_execs = 0` across all observed steps).
3. **Calibration (`calibrate_case`)**:
   - Inline calibration of newly discovered interesting seeds executes inside `save_if_interesting` and is counted in `cal_execs` (e.g., Step 1 of `steps_100plus.jsonl`: `new_seeds=8`, `fuzz_execs=456`, `cal_execs=56`, `actual_exec_count=512`; Step 2: `new_seeds=1`, `fuzz_execs=121`, `cal_execs=7`, `actual_exec_count=128`).
   - Because havoc/splice checks `(fsrv.total_execs - rl_step_start_execs) >= rl_exec_budget` on every loop iteration, `cal_execs` is absorbed inside `rl_exec_budget` and counted toward `budget_used`.
   - **Boundary caveat**: If a step's *final* mutation (`used == rl_exec_budget - 1`) discovers a new seed, inline `calibrate_case` runs `CAL_CYCLES` (`~3–7` execs) before control returns to the loop check, so `actual_exec_count` can exceed `requested_exec_budget` by at most one calibration burst (`<= 7` execs) and is still faithfully recorded and charged to `budget_used`.
4. **Havoc & Splicing**: Both loops share the remaining step budget via AFLRun's `AFLRUN_HAVOC_TIMES` / `AFLRUN_SPLICE_TIMES` split and terminate immediately at the loop check once `actual_exec_count >= rl_exec_budget`. Notice `HAVOC_MIN` (`16`) is suppressed in RL mode so small budgets like `32` with 15 splice cycles are never inflated.
5. **CmpLog**: Not enabled (`-c` omitted).
6. **Random Skip Logic**: Native coverage-mode skip probabilities (`SKIP_TO_NEW_PROB`, `SKIP_NFAV_*`) and AFLRun probabilistic skip (`min_num_exec`) are disabled when `rl_sock_fd >= 0`.

Observed budget accuracy in the 124-step run (`steps_100plus.jsonl`):
- Budget `32`: 35/35 steps, `min=32, max=32, mean=32.0` (`35` without calibration).
- Budget `128`: 41/41 steps, `min=128, max=128, mean=128.0` (`37` without calibration, `4` with inline calibration).
- Budget `512`: 43/43 steps, `min=512, max=512, mean=512.0` (`38` without calibration, `5` with inline calibration).

---

## 4. Reward & Target Coverage Verification

Executed via `scripts/test_reward_repeat.py` (raw output saved to `results/phase1a/reward_repeat_test.json`):

1. **Repeated Target Path Test (`seed = "BB2!"`)**:
   - Dry-run target coverage: `init_target_cov = 4` AFL Bitmap Edge Slots.
   - Over 20 steps (`budget=32`), 4 steps hit Target 1 (`target_hit=True, step_target_hits=1`) without discovering new target edges: all 4 yielded `target_cov_delta = 0` and `reward = 0.0`.
   - Final `target_cov_total = 4`, `cumulative_reward = 0.0 == final - init`.
2. **Non-Target Guard (`seed = "AAAA"`)**:
   - Over 30 steps (`budget=32`), `6` steps increased `global_cov_total` (`global_cov_delta > 0`) on non-target branches without hitting a target (`target_hit = False`), and every one yielded `target_cov_delta = 0`, `reward = 0.0`, keeping `target_cov_total = 0`.
3. **124-Step Monotonicity Check**:
   - Target reached at Step 36 (`seed_id = 13`, `target_hit = True`, `target_cov_delta = 4`, `reward = 4.0`).
   - Subsequent target-hitting steps (Steps 40, 41, 54, 55, 63, 70, 88, 91, 97 on `seed_id = 16`, up to 3 target hits/step) yielded `reward = 0.0`; `cumulative_reward = 4.0 == 4 - 0`.

---

## 5. Integration Test Results Matrix

Raw outputs archived in:
- `results/phase1a/reward_repeat_test.json`
- `results/phase1a/phase1a_summary.json`
- `results/phase1a/steps_100plus.jsonl`
- `results/phase1a/native_regression.log`

| # | Check | Observed Result | Status |
| --- | --- | --- | --- |
| 1 | `gymnasium.utils.env_checker.check_env` | Passed on `DirectedFuzzEnv` (`skip_render_check=True`) | **PASS** |
| 2 | `>=100` consecutive RL steps | Ran `124` consecutive steps cleanly (`budget=40,000`) | **PASS** |
| 3 | Distinct seed execution per slot | `slot_test_seed_ids = [16, 15, 13, 11]`; `11` distinct seeds fuzzed overall; `<4` wraparound `[False, True, True, True]` on initial 1 seed | **PASS** |
| 4 | Energy control (`32 / 128 / 512`) | Mean actual execs: `32.0`, `128.0`, `512.0`; `actual == fuzz + cal + trim` on 100% of steps | **PASS** |
| 5 | Reward monotonicity & repeat-path 0 | `reward >= 0`; `sum(r) == delta(C^T)`; repeat target path gives `reward == 0.0` | **PASS** |
| 6 | Corpus growth across RL steps | `1 -> 13` (10k episode), `1 -> 18` (124-step run) | **PASS** |
| 7 | `>=10` consecutive `reset()` & leak audit | `12` resets; `shm: 108 -> 108`, `fd: 9 -> 9`, child `afl-fuzz: 0` | **PASS** |
| 8 | Budget limit truncation | `10,000`-exec episode stopped cleanly at `budget_used=10000`, `truncated=True, terminated=False` (`33` steps) | **PASS** |
| 9 | Fixed-seed reproducibility (`seed=42`) | Exact step-by-step equality across `20` steps (`0` mismatches) | **PASS** |
| 10 | Native AFLRun regression | 5s native smoke run `rc=0`, `51,933` execs, `10,374.15 execs/s`, `15` seeds | **PASS** |
| 11 | IPC latency & throughput | RL `10,426.33 execs/s` vs Native `10,374.15 execs/s`; IPC overhead `~110.4 us/step` (`~0.4%`) | **PASS** |

---

## 6. Issues Found & Resolved During Phase 1A, and Remaining Non-Determinism

### Issues Diagnosed & Fixed
1. **SysV SHM Leak in Upstream `afl-fuzz.c`**:
   - During initial 12-reset leak testing, SysV SHM segment count grew by `+24` (`84 -> 108`). Root cause: `afl-fuzz.c` called `aflrun_shm_init(&afl->shm_run, ...)` at startup without calling `aflrun_shm_deinit(&afl->shm_run)` in `stop_fuzzing`. Added guarded `aflrun_shm_deinit(&afl->shm_run)` cleanup at `stop_fuzzing`; re-verified `108 -> 108` across 12 consecutive resets.
2. **`--rl-seed` vs `-s` Execution Timing Side-Effect**:
   - Using AFL++'s native `-s` (`afl->fixed_seed = 1`) causes `calibrate_case` to overwrite `q->exec_us` with `exec_tmout - 1`. Added `--rl-seed` (`afl->rl_seed_mode = 1`) which fixes `rand_below` seeding cleanly while keeping genuine target `exec_us` measurements.
3. **Gymnasium `check_step_determinism` vs Wall-Clock Metadata**:
   - `check_env` verifies exact equality of `obs` and `info` across seeded resets. Raw `exec_us` in observations and wall-clock `elapsed_us` / `ipc_roundtrip_us` in `info` fluctuated by microsecond scheduling jitter. Resolved by bucketing `exec_cost_norm` at `500 us` granularity in observations, removing `exec_us` from candidate tie-breaking, and storing wall-clock timings in `env.last_step_timing`.

### Remaining Sources of Non-Determinism (Documented)
- **OS Execution Timing (`exec_us` / `elapsed_us`)**: Microsecond fluctuations remain in `exec_us` (used in `500 us` observation buckets if a target straddles a `500 us` boundary) and in `last_step_timing`.
- **Target Non-Determinism**: If future real-world benchmarks exhibit ASLR, uninitialized memory, thread scheduling, or timeout jitter near `exec_tmout`, identical `--rl-seed` sequences can diverge across runs.

---

## 7. Readiness for Next Phase

Phase 1A (`DirectedFuzzEnv` + AFLRun RL Adapter) is verified **PASS** and ready for baseline comparison experiments and Phase 1B PPO training integration in future phases.
