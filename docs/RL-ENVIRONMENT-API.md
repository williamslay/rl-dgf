# RL-DGF Environment API Specification (Phase 1A)

This document specifies the Python Gymnasium environment (`rl_dgf.DirectedFuzzEnv`), the deterministic candidate selector (`rl_dgf.select_candidates`), and the AFLRun Unix Domain Socket wire protocol implemented for Phase 1A.

---

## 1. Python Package Layout

```text
rl_dgf/
├── __init__.py      # Exports DirectedFuzzEnv, EnvConfig, AFLRunIPCClient, SeedCandidateSelector, select_candidates
├── env.py           # DirectedFuzzEnv(gymnasium.Env)
├── candidates.py    # Deterministic K-slot candidate selection & wraparound handling
└── ipc.py           # Length-prefixed AF_UNIX JSON client + afl-fuzz subprocess lifecycle
```

Python runtime environment: `/home/sylei/RL_miniproject/.venv` (`gymnasium==1.4.0`, `numpy==2.5.5`).

---

## 2. `DirectedFuzzEnv` Interface

```python
from rl_dgf import DirectedFuzzEnv

env = DirectedFuzzEnv(
    target_bin="/home/sylei/RL_miniproject/benchmarks/smoke-target/target",
    aflrun_dir="/home/sylei/RL_miniproject/benchmarks/smoke-target/aflrun_tmp",
    seeds_dir="/home/sylei/RL_miniproject/benchmarks/smoke-target/seeds",
    target_args=["@@"],
    k_candidates=4,
    energy_levels=(32, 128, 512),
    episode_budget=10_000,
)

obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### 2.1 Configuration (`EnvConfig` / Keyword Arguments)

| Parameter | Default | Description |
| --- | --- | --- |
| `target_bin` | required | Path to AFLRun-instrumented target binary |
| `seeds_dir` | required | Initial seed corpus directory (`-i`) |
| `aflrun_dir` | `None` | Path to AFLRun metadata dir (`BBreachable.txt`, `distance.cfg/`) or parent `AFLRUN_TMP` dir containing one target subdirectory; `None` lets `afl-fuzz` auto-discover from binary |
| `afl_fuzz_bin` | `/home/sylei/RL_miniproject/backend/AFLRun/afl-fuzz` | Path to compiled `afl-fuzz` binary |
| `work_root` | `/home/sylei/RL_miniproject/results/rl_runs` | Parent directory for ephemeral episode run dirs and `rl.sock` |
| `target_args` | `[]` | Arguments passed after `-- <target_bin>` (e.g., `["@@"]`) |
| `k_candidates` | `4` | Fixed candidate slot count $K$ |
| `energy_levels` | `(32, 128, 512)` | Discrete per-step execution budget levels |
| `episode_budget` | `10_000` | Total execution budget per episode before `terminated=True` (finite-horizon task boundary) |
| `exec_timeout_ms` | `None` | Optional `-t` timeout passed to `afl-fuzz` |
| `extra_env` | `{}` | Optional environment overrides for `afl-fuzz` |
| `keep_artifacts` | `False` | Retain ephemeral episode output directory & logs after `close()`/`reset()` |
| `cap_final_step_budget` | `True` | Cap final step budget to `min(energy_level, remaining_budget)` |

---

## 3. Action Space & Deterministic Candidate Mapping

- **`action_space`**: `gymnasium.spaces.Discrete(K * len(energy_levels))` (`Discrete(12)` under defaults `K=4`, `energy_levels=(32, 128, 512)`).
- **Decoding**:
  - `slot_idx = action // len(energy_levels)` (`0..3`)
  - `energy_idx = action % len(energy_levels)` (`0..2`)
  - `energy_level = energy_levels[energy_idx]` (`32`, `128`, or `512`)

### Deterministic Slot Mapping (`rl_dgf.candidates.select_candidates`)

1. Filter out any queue entry with `disabled != 0`.
2. Sort all valid seeds deterministically by key:
   ```text
   (missing_dist_flag, min_target_dist, not_favored_flag, fuzzed_times, len, id)
   ```
   - `min_target_dist` comes directly from AFLRun's static CFG/CG distance tables (`bb_to_dists` in `src/aflrun.cpp`), computed over basic blocks reached by the seed (`aflrun_compute_seed_dist`).
   - Using `len` and queue `id` instead of wall-clock microsecond `exec_us` ensures ranking cannot flip due to microsecond OS jitter.
3. Take the top `K` entries.
4. **Handling `< K` valid seeds** (`0 < N < K`):
   - Slots `0 .. N-1` map 1-to-1 to the `N` valid seeds (`slot_wrapped[slot] = False`).
   - Slots `N .. K-1` explicitly wrap via `slot % N` onto real valid queue entries (`slot_wrapped[slot] = True`), while observation features for slots `>= N` are zeroed (`slot_valid = 0.0`).
   - Invalid/disabled seeds are never silently executed (both Python and C `afl-fuzz-rl.c` enforce validity).

---

## 4. Observation Space

- **`observation_space`**: `gymnasium.spaces.Box(low=0.0, high=1.0, shape=(30,), dtype=np.float32)` (`6 + 4 * 6 = 30`).

All features are derived directly from live AFLRun runtime state (no synthetic/fake distances):

| Index | Scope | Feature | Formula / AFLRun Source |
| --- | --- | --- | --- |
| 0 | Global | `global_cov_norm` | `min(1.0, count_non_255_bytes(virgin_bits) / real_map_size)` |
| 1 | Global | `target_cov_norm` | `min(1.0, target_cov_total / real_map_size)` |
| 2 | Global | `aflrun_target_ratio` | `min(1.0, aflrun_reached_targets / num_targets)` |
| 3 | Global | `remaining_budget_ratio` | `max(0.0, remaining_budget / episode_budget)` |
| 4 | Global | `corpus_size_norm` | `min(1.0, valid_seed_count / 256.0)` |
| 5 | Global | `stagnation_norm` | `min(1.0, steps_since_last_target_cov / 50.0)` |
| `6 + 6*i + 0` | Slot `i` | `slot_valid` | `1.0` if `i < min(K, valid_seed_count)` else `0.0` |
| `6 + 6*i + 1` | Slot `i` | `dist_prox` | `1.0 / (1.0 + min_target_dist)` if `has_target_dist` else `0.0` |
| `6 + 6*i + 2` | Slot `i` | `fuzz_count_norm` | `min(1.0, fuzzed_times / 5000.0)` |
| `6 + 6*i + 3` | Slot `i` | `exec_cost_norm` | `min(1.0, (exec_us // 500) / 20.0)` (500 us buckets) |
| `6 + 6*i + 4` | Slot `i` | `fringe_norm` | `min(1.0, aflrun_get_seed_fringe_cnt(id) / 32.0)` |
| `6 + 6*i + 5` | Slot `i` | `favored` | `1.0` if `q->favored` else `0.0` |

Note on `exec_cost_norm`: bucketing `exec_us` in `500 us` steps isolates fast seeds (`< 500 us`) from microsecond OS scheduling jitter in observations while preserving exact `exec_us` in backend metadata.

---

## 5. Reward Definition

Target-conditioned Coverage is tracked monotonically inside `afl_state_t`:

$$
C_t^T = \bigcup_{x : \mathrm{HitTarget}(x)} E(x)
$$

$$
r_t = |C_{t+1}^T| - |C_t^T| \ge 0
$$

- **Coverage Unit**: Covered **AFL Bitmap Edge Slots** (`afl->target_edge_covered[idx] == 1` where `afl->fsrv.trace_bits[idx] != 0` across `0 <= idx < afl->fsrv.map_size`), recorded in `afl_update_target_coverage()` right before `save_if_interesting()`. Note that these are AFL shared-memory bitmap slots (edge-hash slots with `map_size` resolution), **not** exact static CFG edge counts.
- **Target Hit Condition**: At least one target basic block bit `0 <= i < afl->fsrv.num_targets` is set in `afl->fsrv.trace_reachables`.
- **Dry-run Initialization**: Target-hitting initial seeds update `target_edge_covered` during `calibrate_case(..., from_queue=1)` (`C_0^T`) without double-counting hits.
- **Original AFLRun Diversity Preserved**: `aflrun_new_bits`, `path_fringes`, `path_pro_fringes`, `reached_targets`, and `clusters` remain untouched.

---

## 6. Returned `info` & Timing Fields

`step()` returns deterministic transition/coverage/budget metadata in `info`:

- `step_id`, `slot_idx`, `slot_wrapped`, `selected_seed_id`
- `action_energy_level`, `requested_exec_budget`, `actual_exec_count`
- `fuzz_execs`, `cal_execs`, `trim_execs`
- `total_execs_before`, `total_execs_after`
- `new_seeds`, `new_seed_count`, `queued_items`
- `target_hit`, `step_target_hits`, `target_cov_delta`, `target_cov_total`
- `global_cov_total`, `global_cov_delta`
- `saved_crashes`, `saved_hangs`, `aflrun_reached`, `aflrun_reached_targets`
- `skipped`, `budget_used`, `remaining_budget`, `candidate_seed_ids`, `valid_seed_count`

Wall-clock per-step timing is exposed in `env.last_step_timing`:
- `elapsed_us`: backend `fuzz_one` + `cull_queue` duration measured via `CLOCK_MONOTONIC` in `src/afl-fuzz-rl.c`
- `ipc_roundtrip_us`: Python client `send(step)` -> `recv(step_result)` round-trip duration

Separating fluctuating wall-clock microseconds from `info` ensures `gymnasium.utils.env_checker.check_env` verifies exact `obs`, `reward`, `terminated`, `truncated`, and deterministic `info` invariants under fixed seeding.

---

## 7. Backend Wire Protocol (`--rl-sock` / `--rl-seed`)

When `afl-fuzz` is invoked with `--rl-sock <unix_socket_path>` (and optional `--rl-seed <u64>`):

- **Framing**: 4-byte big-endian `uint32` length header followed by UTF-8 JSON payload.
- **Handshake (`afl-fuzz` -> Python)**: `{"type":"ready","status":"ok","num_targets":...,"num_reachables":...,"map_size":...,"real_map_size":...,"total_execs":...,"queued_items":...,"target_cov_total":...,"global_cov_total":...,"aflrun_reached":...,"aflrun_reached_targets":...,"seeds":[...]}`
- **Step Request (Python -> `afl-fuzz`)**: `{"op":"step","seed_id":<u32>,"exec_budget":<u32>}`
- **Step Response (`afl-fuzz` -> Python)**: `{"type":"step_result","status":"ok","selected_seed_id":...,"requested_exec_budget":...,"actual_exec_count":...,"fuzz_execs":...,"cal_execs":...,"trim_execs":...,"total_execs_before":...,"total_execs_after":...,"elapsed_us":...,"new_seeds":...,"target_hit":...,"target_cov_delta":...,"target_cov_total":...,"global_cov_total":...,"seeds":[...]}`
- **Shutdown Request (Python -> `afl-fuzz`)**: `{"op":"stop"}` -> `{"type":"stopped","status":"ok"}`, triggering `stop_fuzzing` cleanup (`aflrun_shm_deinit`, `afl_shm_deinit`, forkserver termination).
