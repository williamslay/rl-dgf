# IMPLEMENTATION-PLAN — Minimal RL-DGF Architecture & 2–3 Week Roadmap

> **Decision**: **CONDITIONAL GO** (based on [ENVIRONMENT-REPORT.md](file:///home/sylei/RL_miniproject/docs/ENVIRONMENT-REPORT.md) and [RL-INTERFACE-AUDIT.md](file:///home/sylei/RL_miniproject/docs/RL-INTERFACE-AUDIT.md))  
> **Scope**: Single developer, 2–3 weeks, course Mini Project. Reuse AFLRun's mutation engine, coverage/target instrumentation, forkserver, and corpus management; reuse Stable-Baselines3 PPO.

---

## 1. Selected Architecture (Single Preferred Design)

```text
+-------------------------------------------------------------------+
| Python Process (RL Training & Evaluation)                         |
|  - Stable-Baselines3 PPO (or MaskablePPO / PPO with fixed K slots)|
|  - Gymnasium DirectedFuzzEnv (reset / step / observation / reward)|
|  - CandidateWindowManager (maps K candidate slots -> real seed_id)|
+-------------------------------------------------------------------+
            ^                                       |
            |  Unix Domain Socket (AF_UNIX, SOCK_STREAM)
            |  Length-prefixed JSON or flat C struct messages
            |  (StepRequest: seed_id, exec_budget | StepResult: state, deltas)
            v
+-------------------------------------------------------------------+
| C/C++ AFLRun Subprocess (afl-fuzz --rl-sock /path/to/sock)        |
|  - RL Scheduling Adapter (src/aflrun_rl.c/.cpp, ~150-250 LOC)     |
|    * Receives (seed_id, exec_budget) at fuzz_one() boundary       |
|    * Sets queue_cur = queue_buf[seed_id], disables probabilistic  |
|      skipping, bounds havoc/splice loop by actual execs delta     |
|    * Maintains monotonic virgin_target_bits & per-seed distance   |
|  - Reused Unmodified Core:                                        |
|    * Mutation engine (fuzz_one_original havoc + splice)           |
|    * Coverage & target maps (virgin_bits, virgin_reachables,      |
|      virgin_ctx, Clusters, FringeBlocks)                          |
|    * Forkserver & shared memory (afl-forkserver.c, afl-sharedmem) |
|    * Corpus storage (add_to_queue, queue_buf, out_dir/queue)      |
+-------------------------------------------------------------------+
```

### Why Unix Domain Socket (`AF_UNIX`)?
- **Clean episode reset**: Python creates the listening socket server once (`socket_path`), spawns `afl-fuzz --rl-sock <socket_path>` in `env.reset()`, and accepts a fresh connection per episode. When an episode finishes (`terminated=True` on budget exhaustion) or crashes, closing the socket / killing the child cleanly tears down the episode without colliding with `stdin`/`stdout` (which AFLRun uses for startup logs and `afl-forkserver` uses for target I/O).
- **Low latency**: Local `AF_UNIX` round-trip is ~5–15 microseconds, negligible compared to a single RL step of 32–512 target executions (~3–50 ms at ~11k execs/s).
- **Simple synchronous lockstep**: `Python send(StepCmd)` -> `AFLRun runs fuzz_one` -> `AFLRun send(StepObs)` -> `Python computes obs & reward`. Zero lock/race-condition headaches compared to shared-memory ring buffers.

---

## 2. Module Breakdown & Exact Modified Files

### 2.1 C/C++ Side (Minimal Patch to `backend/AFLRun`)
| File | Planned Change | Est. LOC |
|---|---|---|
| `include/afl-fuzz.h` | Add `rl_sock_path`, `rl_sock_fd`, `rl_exec_budget`, `rl_step_start_execs`, `u8 *virgin_target_bits`, `u32 target_edges_found` to `afl_state_t`; add `double min_target_dist, avg_target_dist` to `struct queue_entry` | ~15 |
| `include/aflrun.h` & `src/aflrun.cpp` | Add helper `aflrun_compute_seed_dist(const u8* reached, double* min_d, double* avg_d)` using existing `bb_to_dists` / `bb_to_avg_dists`; add getter `aflrun_get_rl_metrics(...)` returning `num_reached`, `num_reached_targets`, `virgin_ctx` set-bit count, active fringe counts, and per-seed fringe count | ~70 |
| `src/afl-fuzz-bitmap.c` | In `save_if_interesting()`, when the run hits any target BB (`b < g->num_targets` in `trace_targets` or `trace_reachables`), update monotonic `afl->virgin_target_bits` (`has_new_bits` against `fsrv.trace_bits`) and increment `afl->target_edges_found` | ~20 |
| `src/afl-fuzz-run.c` | In `aflrun_new_bits()` / `calibrate_case()`, populate `q->min_target_dist` and `q->avg_target_dist` on initial seed calibration | ~10 |
| `src/afl-fuzz-one.c` | In `fuzz_one_original()`: when `afl->rl_sock_fd >= 0`, (1) bypass `!afl->is_aflrun` skip checks (`478-518`), (2) set `num_execs = afl->rl_exec_budget` directly instead of `quant_score` division, (3) disable `aflrun_end_cycle()` early abort (`3116`), and (4) break out of the havoc/splice loop if `(afl->fsrv.total_execs - afl->rl_step_start_execs) >= afl->rl_exec_budget` | ~30 |
| `src/afl-fuzz.c` | Add `--rl-sock` CLI option; after `perform_dry_run` + `cull_queue`, enter `rl_fuzz_loop(afl)` that exchanges `StepResult` / `StepCmd` over the Unix socket and invokes `fuzz_one(afl)` on the chosen `seed_id` | ~80 |
| `src/aflrun_rl.c` (new, optional split) | Socket connect, fixed-size binary/JSON serialize/deserialize for global state + top-K candidate seed features | ~100 |

**Total C/C++ footprint**: ~225–325 LOC. **Zero** edits to mutators, forkserver, LLVM passes, or `afl-sharedmem.c`.

### 2.2 Python Side (New Modules under `/home/sylei/RL_miniproject/rl_dgf/`)
| Module | Responsibility |
|---|---|
| `rl_dgf/ipc_client.py` | `AF_UNIX` server managing `afl-fuzz` subprocess lifecycle, timeouts, and message framing |
| `rl_dgf/candidates.py` | Maps fixed `K` candidate slots (e.g., `K = 16` or `32`) from the dynamic corpus `queue_buf` (e.g., top-K prioritized by target distance, recency, unexplored status, favored status, plus random/diverse slots so newly discovered seeds enter immediately) |
| `rl_dgf/env.py` | `DirectedFuzzEnv(gymnasium.Env)` implementing `reset()`, `step(action)`, fixed-size observation vector `o_t`, discrete action `(slot_idx, energy_idx)`, reward computation, and budget termination |
| `rl_dgf/wrappers.py` | Observation normalization (`VecNormalize` or bounded feature scaling) and optional action flattening (`MultiDiscrete([K, M])` -> `Discrete(K * M)` if needed) |
| `rl_dgf/train_ppo.py` | SB3 PPO training entrypoint + TensorBoard / CSV logging |
| `rl_dgf/eval.py` | Head-to-head evaluation comparing Trained PPO vs Vanilla AFLRun vs Random/Uniform under identical execution budgets & multiple seeds |

---

## 3. Concrete RL Formulation Adjustments Needed

Based on the source audit, adjust the Phase 0 draft RL design in three specific ways:

1. **Candidate Window for Dynamic Corpus (`i_t`)**:
   - Because `queue_buf` grows dynamically (`1` -> hundreds/thousands of seeds) while SB3 PPO requires a fixed-size `Discrete` / `MultiDiscrete` action space, Python selects a **fixed candidate window of `K` active (`!disabled`) seeds** at each step (e.g., `K = 16`, constructed deterministically from top closest-to-target seeds + least-fuzzed seeds + newly discovered seeds + favored seeds).
   - Action `a_t = (i_t, e_t)` selects `i_t in {0..K-1}` (mapped to real `seed_id`) and `e_t in {0..M-1}` mapped to discrete execution budgets, e.g., `E = [16, 32, 64, 128, 256, 512]`.
   - If fewer than `K` valid seeds exist early in an episode, modulo/clamp `i_t % num_valid` (or use invalid-action fallback to best valid seed) so PPO never crashes on an invalid `seed_id`.

2. **Target-Conditioned Reward Definition (`r_t`)**:
   - Do **not** compute reward directly from `Clusters::size()` or `FringeBlocks::fringes.size()`, because both shrink when clusters merge or fringes are crossed (`try_del_fringe`).
   - Use strictly **monotonic, non-negative deltas** per step:
     - `delta_target_edges = delta(target_edges_found)` (new edges observed on runs hitting target BBs via `virgin_target_bits`),
     - `delta_ctx_reach = delta(virgin_ctx_bits)` (new context-sensitive reachable BBs covered),
     - `delta_targets = delta(num_reached_targets)` (bonus for reaching new target BBs),
     - `delta_reach_bbs = delta(num_reached)` (intermediate progress toward targets).
   - Example Phase 1 baseline reward:
     `r_t = w_1 * delta_target_edges + w_2 * delta_targets + w_3 * delta_ctx_reach + w_4 * delta_reach_bbs` (normalized by budget scale).

3. **Disable Trimming & Early-Cycle-Abort in RL Mode**:
   - Pass `--config trim_thr=inf` (or `afl->disable_trim = 1`) during RL episodes so `trim_case()` does not consume hundreds of unbudgeted executions on the first step a large seed is picked.
   - Always decrement remaining budget `B_t` by `actual_execs = total_execs_after - total_execs_before` (which includes `calibrate_case` on new finds), and cap `max_steps` / terminate cleanly when `B_t <= 0`.

---

## 4. Development Tasks, Dependencies & 2–3 Week Schedule

| Week / Phase | Task | Depends On | Deliverable / Milestone |
|---|---|---|---|
| **Week 1, Days 1–2** | Setup Python `.venv` (`gymnasium`, `stable-baselines3`, `torch`, `tensorboard`); write `aflrun_rl` C/C++ hook (`--rl-sock`, `virgin_target_bits`, per-seed distance, exact `rl_exec_budget` cap) | Phase 0 complete | `afl-fuzz --rl-sock` compiles & passes lockstep echo test |
| **Week 1, Days 3–4** | Implement `ipc_client.py`, `candidates.py`, `DirectedFuzzEnv` (`env.py`), and Gymnasium `check_env()` validation | C/C++ RL hook | `check_env(env)` passes; 5-episode random policy smoke run on `smoke-target` completes cleanly with deterministic reset |
| **Week 1, Day 5** | Select 1–2 realistic small/medium C benchmarks (e.g., `libxml2`/`mjs`/`binutils` single small binary or multi-branch synthetic + real OSS-Fuzz small target) and verify target reachability | `DirectedFuzzEnv` working | Benchmarks compiled with `afl-clang-lto` and verified with baseline AFLRun |
| **Week 2, Days 1–3** | Implement `train_ppo.py`; run initial PPO training on fast target (e.g., `B = 50,000`–`100,000` execs/episode, using `/dev/shm` for `out_dir`); tune observation scaling, candidate selection `K`, energy levels `E`, and reward weights | Realistic benchmarks + `env.py` | Training curves showing reward & target-conditioned coverage improving over random policy |
| **Week 2, Days 4–5** | Run multi-seed baseline comparison (`eval.py`): Vanilla AFLRun vs Random Policy vs Trained PPO (same fixed exec budget `-E B`) | Trained PPO checkpoint | Reproducible CSV + plots comparing target coverage, target-conditioned edge coverage, and time-to-targets |
| **Week 3 (Buffer & Polish)** | Ablations (e.g., `seed-only` vs `joint seed+energy`, or distance feature ablation); finalize final project report & presentation | Eval results | Final report, clean scripts, reproducible results |

---

## 5. Risks & Minimal Viable Fallbacks

1. **Risk: High variance & slow episodes on real targets**
   - *Mitigation*: Run `out_dir` inside `/dev/shm` (167 GB free tmpfs), keep initial corpus small (1–5 seeds), and cap per-episode budget `B` so each episode takes ~2–10s (at 10k execs/s, `B = 50,000` takes ~5s -> ~360 episodes in 30 min of training).
2. **Risk: Candidate window `K` hides a good seed in large corpora**
   - *Mitigation*: Construct the `K=16` candidate list as a stratified mix (e.g., top-6 closest to targets, top-4 most recently discovered, top-4 least fuzzed / favored, 2 uniform random active seeds).
3. **Risk: Multi-target sparse reward**
   - *Mitigation*: Already mitigated by including `delta(num_reached)` and `delta(virgin_ctx_bits)` as dense intermediate progress signals alongside `delta(target_edges_found)`.
4. **Minimal Fallback (if full PPO on real programs underperforms in 2 weeks)**:
   - Keep the exact same Gymnasium environment and evaluate (a) contextual bandit / short-horizon PPO (`gamma=0.95`) on synthetic + 1 small real target (`mjs` / `cjson` / `sqlite` parser subset), and (b) compare against Vanilla AFLRun and Uniform Random Scheduling.
