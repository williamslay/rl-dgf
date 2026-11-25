# RL-INTERFACE-AUDIT — AFLRun Scheduling, Feedback & RL Feasibility Audit

> **Audited Repo**: `/home/sylei/RL_miniproject/backend/AFLRun` (`ea58c0e9f49615a02753f0c32dcc5dec8a46a87f`)  
> **Primary Files Audited**:
> - `include/aflrun.h`, `include/afl-fuzz.h`, `include/config.h`, `include/sharedmem.h`, `include/trace.h`
> - `src/aflrun.cpp`, `src/afl-fuzz.c`, `src/afl-fuzz-one.c`, `src/afl-fuzz-queue.c`, `src/afl-fuzz-bitmap.c`, `src/afl-fuzz-run.c`, `src/afl-fuzz-init.c`, `src/afl-fuzz-stats.c`, `src/afl-sharedmem.c`
> - `instrumentation/aflrun-pass.cc`, `instrumentation/afl-compiler-rt.o.c`

---

## 1. Executive Summary & Feasibility Verdict

- **Overall Verdict**: **CONDITIONAL GO**
- **Summary**: AFLRun's mutation engine (`fuzz_one_original`), forkserver (`afl-forkserver.c`), corpus storage (`queue_buf`, `add_to_queue`), and target/diversity/coverage tracking (`virgin_bits`, `virgin_reachables`, `virgin_ctx`, `Clusters`, `FringeBlocks`) are modular, clean, and directly reusable. However, AFLRun's native scheduling operates **at queue-cycle granularity** (`select_aflrun_seeds` + `aflrun_assign_energy` once per cycle in `afl-fuzz.c:2485-2524`), whereas our RL formulation steps **per `(seed_index, energy_level)` decision**. Bridging this cleanly requires a **small, localized C/C++ RL Scheduling Adapter** (~150–250 LOC) rather than any rewrite of mutation, coverage, or forkserver components.

---

## 2. Task 4 — Seed Selection, Energy Assignment & Step Boundary Audit

### A. Seed Selection Call Path `[Source-verified only]`

1. **Native AFLRun Selection Flow**:
   - Main loop in [`src/afl-fuzz.c:2385-2524`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz.c#L2385-L2524):
     - Queries `afl->is_aflrun = aflrun_get_mode()` (`0 = kCoverage`, `1 = kFringe`, `2 = kProFringe`, `3 = kTarget`, `4 = kUnite`, defined in [`src/aflrun.cpp:662-665`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L662-L665)).
     - Note: AFLRun starts in `kCoverage` (`init_cov = true`) until `cov_quant >= config.init_cov_quant` (default `6000` = 600s equivalent quantum, [`src/aflrun.cpp:100,710-726`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L710-L726)).
     - At cycle start when `afl->is_aflrun` is non-zero (`afl-fuzz.c:2485-2524`):
       1. `u32 n = select_aflrun_seeds(afl);` ([`src/afl-fuzz-queue.c:1489-1526`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-queue.c#L1489-L1526)) collects all `!q->disabled` entries from `afl->queue_buf[0 .. queued_items-1]`, recalculates `q->perf_score = calculate_score(afl, q)`, resets `q->quant_score = 0`, calls `cull_queue_div(...)` ([`afl-fuzz-queue.c:868-911`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-queue.c#L868-L911)), and calls `aflrun_cull_queue(afl->aflrun_seeds, idx)` ([`src/aflrun.cpp:3846-3861`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L3846-L3861)).
       2. `aflrun_cull_queue` / `cull_queue_unite` ([`src/aflrun.cpp:1190-1253`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L1190-L1253)) keeps favored/div-favored seeds deterministically and **randomly drops unfavored seeds** (`SKIP_NFAV_OLD_PROB=95%`, `SKIP_NFAV_NEW_PROB=75%`).
       3. `aflrun_assign_energy(n, seeds, afl->perf_scores)` ([`src/aflrun.cpp:3118-3126`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L3118-L3126)) assigns `q->quant_score` for all `n` selected seeds.
       4. Seeds with `q->quant_score > afl->queue_quant_thr` are placed into `afl->aflrun_queue[]`, sorted descending by `cmp_quant_score`, and iterated sequentially via `afl->queue_cur = afl->aflrun_queue[++afl->aflrun_idx]` (`afl-fuzz.c:2766`).
   - When `afl->is_aflrun == 0` (`kCoverage` mode, `afl-fuzz.c:2680-2699`): uses AFL++'s weighted alias table (`create_alias_table` + `select_next_queue_entry`).

2. **Can an external RL Agent specify a Seed ID?**
   - **Yes (`[Source-verified only]`)**. In [`src/afl-fuzz.c:2697`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz.c#L2697), `fuzz_one(afl)` simply operates on `afl->queue_cur` and `afl->current_entry`. Setting:
     ```c
     afl->current_entry = seed_id;
     afl->queue_cur = afl->queue_buf[seed_id];
     ```
     directly selects any seed before calling `fuzz_one(afl)`.

3. **What if the specified Seed ID is not in the active/valid queue?**
   - Out-of-bounds (`seed_id >= afl->queued_items`): `afl->queue_buf[seed_id]` causes an out-of-bounds read / segfault (`[Source-verified only]`).
   - Disabled seed (`afl->queue_buf[seed_id]->disabled == 1`):
     - Seeds can be marked `disabled = 1` in three places: timeout during `perform_dry_run` ([`src/afl-fuzz-init.c:920,1062`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-init.c#L920)), duplicate trace/path during `perform_dry_run` ([`src/afl-fuzz-init.c:1181,1195`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-init.c#L1181)), and `disable_aflrun_extra` when an `+div`/`+path`-only extra seed loses all active fringes ([`src/afl-fuzz-queue.c:1543-1569`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-queue.c#L1543-L1569)).
     - `fuzz_one_original` does **not** check `q->disabled` at entry; if forced, it would fuzz a disabled/stale seed. Therefore, the RL adapter **must** filter candidate seeds to `!q->disabled` (and handle invalid action indices via candidate masking or clamping).

4. **Seed ID Stability under Corpus Growth**:
   - **Stable (`[Source-verified only]`)**: [`add_to_queue`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-queue.c#L523-L581) strictly appends new entries to `afl->queue_buf[afl->queued_items - 1]` and sets `q->id = afl->queued_items - 1`. Entries are **never removed or re-indexed** from `queue_buf` (even when `disabled = 1`). Thus `q->id` is an immutable monotonic integer ID for the entire campaign.

5. **Conflict with Queue Culling & Favored Mechanisms**:
   - **Direct conflict if left enabled, easy to decouple (`[Source-verified only]`)**:
     - In `fuzz_one_original` (`afl-fuzz-one.c:478-518`), skipping logic (`aflrun_extra`, `SKIP_TO_NEW_PROB=99%`, `SKIP_NFAV_OLD_PROB=95%`, `SKIP_NFAV_NEW_PROB=75%`) only runs when `!afl->is_aflrun`.
     - In `aflrun` mode, probabilistic skipping happens inside `aflrun_cull_queue` / `cull_queue_unite` (`src/aflrun.cpp:1190-1253`).
     - Additionally, `cull_queue` (`afl-fuzz-queue.c:791`) and `cull_queue_div` (`afl-fuzz-queue.c:868`) compute `q->favored` and `q->div_favored` flags, which **are useful read-only features** for RL, while `aflrun_cull_queue` should be **bypassed** (or used only to populate candidate pools) when RL makes the selection, and `fuzz_one_original`'s `if (!afl->is_aflrun)` skip block (`afl-fuzz-one.c:478-518`) must be skipped in RL mode so RL's chosen seed is never silently rejected (`return 1`).

---

### B. Energy Assignment Call Path & Actual Execution Budget `[Source-verified only]`

1. **Actual Unit of AFLRun Energy**:
   - `quant_score` (`struct queue_entry::quant_score`, `double`, [`include/afl-fuzz.h:199`](file:///home/sylei/RL_miniproject/backend/AFLRun/include/afl-fuzz.h#L199)) is in units of **time quanta**, where `1 quantum = QUANTUM_TIME = 100,000 us = 0.1 s` ([`include/config.h:546`](file:///home/sylei/RL_miniproject/backend/AFLRun/include/config.h#L546)).
   - Default cycle energy budget in AFLRun is `cycle_energy = 60 * 10 = 600` quanta (= 60 seconds per cycle, [`src/aflrun.cpp:95`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L95)).

2. **Conversion from `quant_score` to Execution Count (`num_execs`)**:
   - In [`src/afl-fuzz-one.c:609-626`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-one.c#L609-L626) (and `mopt_common_fuzzing` at `3430-3447`):
     ```c
     orig_perf = perf_score = afl->queue_cur->quant_score;
     num_execs = QUANTUM_TIME * perf_score / afl->queue_cur->exec_us;
     if (num_execs < afl->min_num_exec) {
       if (rand_below(afl, afl->min_num_exec) < num_execs)
         num_execs = afl->min_num_exec;
       else
         goto abandon_entry;
     }
     ```
   - Then `num_execs` is split between havoc and splicing ([`src/afl-fuzz-one.c:2178-2201`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-one.c#L2178-L2201)):
     - Havoc stage (`splice_cycle == 0`): `AFLRUN_HAVOC_TIMES(num_execs, SPLICE_CYCLES)` if splicing is active, else `num_execs`.
     - Each splice cycle (`splice_cycle = 1..SPLICE_CYCLES`, `SPLICE_CYCLES=15`): `AFLRUN_SPLICE_TIMES(num_execs, SPLICE_CYCLES) = (num_execs * 32) / (256 + 32 * 15)` (`include/config.h:538-541`).
     - Notice: `AFLRUN_HAVOC_TIMES + SPLICE_CYCLES * AFLRUN_SPLICE_TIMES == num_execs` when all `SPLICE_CYCLES` run.

3. **Can RL Override Energy for a Single Seed?**
   - **Yes (`[Source-verified only]`)**. Even cleaner than overriding `quant_score` (which undergoes floating-point division by `exec_us` and probabilistic `min_num_exec` skipping), the RL adapter can directly set an **explicit execution count `rl_exec_budget`** (e.g., `[16, 32, 64, 128, 256, 512]`) and bypass the `quant_score -> num_execs` formula in `afl-fuzz-one.c:609-637` when RL mode is active.

4. **Sources of Budget Imprecision (Extra / Early-Exit Executions)**:
   Auditing `afl-fuzz-one.c`, `afl-fuzz-run.c`, and `afl-fuzz-bitmap.c` reveals **5 exact causes** why `num_execs` differs from actual `fsrv.total_execs` delta:
   1. **Re-calibration (`cal_failed`)**: `afl-fuzz-one.c:538-564` runs `calibrate_case()` (`3` or `7` or `12` execs).
   2. **Trimming**: `afl-fuzz-one.c:570-602` calls `trim_case()`, which executes `fuzz_run_target()` many times **without** incrementing `afl->fuzzed_times` (`afl-fuzz-run.c:1044`). Fixable via `-D`/disabling trim in RL mode (`afl->disable_trim = 1`) OR deducting trim execs from the step budget.
   3. **New-find inline calibration**: Whenever `common_fuzz_stuff()` discovers an interesting input, `save_if_interesting()` calls `calibrate_case()` ([`afl-fuzz-bitmap.c:676`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-bitmap.c#L676)), consuming `CAL_CYCLES_FAST=3` or `CAL_CYCLES=7` extra `fuzz_run_target()` calls (up to `CAL_CYCLES_LONG=12` if unstable).
   4. **Timeout verification**: `save_if_interesting()` re-runs timeouts with `hang_tmout` (`afl-fuzz-bitmap.c:774`).
   5. **Early `aflrun_end_cycle()` abort**: [`afl-fuzz-one.c:3116-3121`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-one.c#L3116-L3121) cuts `stage_max = stage_cur` whenever `state.is_reset() || state.is_end_cov()` becomes true after discovering a new seed!
   - **Recommendation**: In RL mode, (i) disable `aflrun_end_cycle()` early abort inside `fuzz_one_original`, (ii) optionally set `afl->disable_trim = 1` (or pass `-a`/config `trim_thr=inf`) so trimming does not cause massive unpredictable budget spikes on large inputs, and (iii) enforce a hard step-level cap `(afl->fsrv.total_execs - step_start_execs) >= rl_exec_budget` inside `fuzz_one_original` and report the exact `actual_execs = fsrv.total_execs - step_start_execs` back to Python to decrement `B_t`.

---

### C. Step Boundary Design Comparison `[Source-verified only]`

| Dimension | **Scheme A: 1 RL Step = 1 `(seed, budget)` execution** | **Scheme B: 1 RL Step = Energy vector for K seeds** | **Scheme C: 1 RL Step = 1 AFLRun Queue Cycle** |
|---|---|---|---|
| **Native Alignment** | Requires bypassing cycle loop (`afl-fuzz.c:2480-2524`) and jumping directly to `(seed_id, exec_budget)` -> `fuzz_one` | Matches `aflrun_assign_energy` signature, still needs batch execution loop | Matches `aflrun_cycle_end` naturally |
| **Action Space** | **Discrete `(i_t, e_t)`** (`MultiDiscrete([K, M])` or flattened `Discrete(K*M)`), directly fits SB3 PPO | Continuous/Multi-dim vector `R^K` | High-dim vector `R^N` (variable `N`) |
| **Feedback Granularity** | **Immediate per-seed credit assignment**: reward `r_t` directly attributed to `(i_t, e_t)` | Delayed/blurred across K seeds | Very coarse (~hundreds of seconds / tens of thousands of execs per step; too few steps per episode!) |
| **Episode Length (fixed budget B)** | E.g., `B = 100,000` execs, `e_t in {32..512}` -> **~200–1,000 steps/episode** (ideal for PPO) | ~20–100 steps/episode | ~2–10 steps/episode (unsuitable for PPO learning) |
| **C/C++ Modification Size** | **Small (~150–200 LOC)**: hook right before `fuzz_one` in `afl-fuzz.c:2700`, override `queue_cur` & `num_execs` (with hard exec cap), pause at `fuzz_one` return | Medium (~200–300 LOC) | Smallest (~100 LOC), but poor RL sample efficiency |

**Recommended Scheme**: **Scheme A (1 RL Step = 1 `(seed_slot, exec_budget)` execution)**.
- **Why**: AFLRun's `afl-fuzz.c` already has a clean per-seed boundary around `fuzz_one(afl)` (`afl-fuzz.c:2702-2745`), and `aflrun_has_new_path` / `save_if_interesting` already update all coverage/fringe/diversity maps *synchronously* during `fuzz_one(afl)`. Placing the IPC pause/resume hook immediately around `fuzz_one(afl)` (while skipping `aflrun_cycle_end` / `select_aflrun_seeds` / `aflrun_assign_energy` scheduling control, yet still calling `aflrun_update_fuzzed_quant` and `update_bitmap_score` to keep metadata fresh) gives exact credit assignment, ~200–1,000 PPO steps per episode, and minimal C/C++ diff.

---

## 3. Task 5 — Observation & Reward Feasibility Audit `[Source-verified only]`

### 3.1 Distinction of the Three Coverage / Diversity Metrics (CRITICAL)
Do **not** conflate these three distinct metrics inside AFLRun:

1. **Global Edge Coverage (`C_t`)**:
   - **Definition**: Number of discovered `(prev_loc ^ cur_loc)` edges / hit-count buckets in AFL++'s primary bitmap `afl->virgin_bits` (initially all `0xFF`, cleared on discovery).
   - **Source**: `count_non_255_bytes(afl, afl->virgin_bits)` ([`src/afl-fuzz-bitmap.c:118-141`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-bitmap.c#L118-L141)), also reported as `edges_found` in `fuzzer_stats`.
   - **Scope**: Whole program, including non-target paths like `path_a`.

2. **Target-Conditioned Coverage (`C_t^T`, our RL proxy objective)**:
   - **Definition(s) directly extractable from AFLRun**:
     - **(i) Target-Reachable Basic Block / Context Coverage**:
       - `g->num_reached` (out of `g->num_reachables`) and `g->num_reached_targets` (out of `g->num_targets`) from `afl->virgin_reachables` ([`src/aflrun.cpp:3601-3615`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L3601-L3615)).
       - Function-level: `g->num_freached`, `g->num_freached_targets` from `afl->virgin_freachables`.
       - Context-sensitive reachable BBs: bits cleared in `afl->virgin_ctx` (`MAP_TR_SIZE(num_reachables)` bytes, [`src/aflrun.cpp:3670-3757`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L3670-L3757), [`instrumentation/afl-compiler-rt.o.c:2598-2603`](file:///home/sylei/RL_miniproject/backend/AFLRun/instrumentation/afl-compiler-rt.o.c#L2598-L2603)).
     - **(ii) Target-Cluster Conditioned Edge Coverage (from `Clusters::cluster_maps`)**:
       - AFLRun maintains a separate edge virgin map `cluster_maps[c]` (`size = afl->fsrv.map_size`) for each active target/diversity block cluster `c >= 1` ([`src/aflrun.cpp:2004,2141-2143`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L2004)), updated via `has_new_bits_mul(..., modify=1)` ([`src/afl-fuzz-bitmap.c:580-593`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-bitmap.c#L580-L593)) ONLY when an execution reaches the target/diversity block(s) in cluster `c`!
       - **Important caveat**: `Clusters` dynamically merges clusters (`try_merge`, `aflrun.cpp:2043-2107`) and invalidates/removes non-target diversity blocks (`div_seed_thr`, `div_switch`), whereas target blocks (`b < g->num_targets`) are never switched off (`aflrun.cpp:3630,2413`). Thus, **tracking newly cleared bits across target clusters (or fixing `no_diversity=0` and counting unique `(target_block, edge)` or target-hitting `virgin_ctx` + target `cluster_maps`)** requires care if clusters merge mid-episode!

3. **AFLRun's Original Target Path-Diversity Metric**:
   - **Definition**: In the AFLRun paper & [`src/aflrun.cpp`](file:///home/sylei/RL_miniproject/backend/AFLRun/src/aflrun.cpp#L3448-L3793), path diversity towards targets is operationalized via:
     - **Context-sensitive Fringe / Pro-Fringe / Reached-Target sets**: `path_fringes`, `path_pro_fringes`, `reached_targets` (`FringeBlocks` storing `(block, call_ctx)` pairs) + `div_blocks` (`DiversityBlocks`).
     - **Target Clustering**: `Clusters<reach_t>` (or `Clusters<Fringe>` if `AFLRUN_CTX_DIV`) clustering targets/fringes whose conditional edge-discovery bitmaps correlate above confidence `conf_thr=0.9` and support `supp_cnt_thr=100`, plus `div_seed_thr=100` invalidation.
     - **Note**: In `afl-compiler-rt.o.c:2588-2603`, `aflrun_inst` maintains `virgin_ctx` for *every* `(reachable_block, call_ctx)` pair, and `aflrun_has_new_path` returns `cf >= 2 || ct` (new context-sensitive fringe or target).

> **Reward Feasibility Conclusion**:
> Yes, a reliable per-step delta `r_t` **can** be obtained synchronously before and after `fuzz_one(afl)` (`[Source-verified only]`), **but** `|C_{t+1}^T| - |C_t^T|` is NOT identical to AFLRun's internal clustering/fringe heuristic. For clean, non-decreasing RL rewards without cluster-merge discontinuities:
> - **Recommended Primary Reward Signal**: Weighted combination (or exact tracked counter) of **Target-Conditioned Edge Coverage** (edges discovered specifically during executions that hit target `b < g->num_targets`, or non-255 bytes in a dedicated `virgin_target_edges` bitmap updated whenever `trace_targets->num > 0` for `b < num_targets`, ~10 lines in `save_if_interesting`) + **Context-Sensitive Target / Reachable Discovery** (`delta(num_reached_targets)`, `delta(virgin_ctx_set_bits)`, `delta(num_reached)`). Adding a tiny dedicated `virgin_target_bits[map_size]` bitmap (or counting `virgin_ctx` bits + `num_reached_targets`) guarantees **monotonic** `r_t >= 0` unaffected by `Clusters::try_merge()`!

---

### 3.2 Feature Availability Table `[Source-verified only]`

| Feature | Source File & Symbol | Data Type | Update Timing | Exportability | Fuzzer Modification Needed? |
|---|---|---|---|---|---|
| **Seed ID** | `include/afl-fuzz.h:157` (`q->id`), `src/afl-fuzz-queue.c:563` | `u32` (monotonic index into `queue_buf[]`) | Once at `add_to_queue` | Trivial | **No** (just expose in state message) |
| **Target Distance** | `src/aflrun.cpp:630,1910,2779-2834` (`bb_to_dists`, `bb_to_avg_dists`); `afl->fsrv.trace_reachables` | `double` (static CFG Dijkstra distance per BB, **not** stored per seed!) | Loaded once at `aflrun_load_dists`; trace cleared per exec | Easy (~20 LOC) | **Yes (minimal)**: compute min/harmonic/avg distance across `trace_reachables` during calibration/`save_if_interesting` and store `q->min_target_dist` / `q->avg_target_dist` |
| **Seed Execution Time** | `include/afl-fuzz.h:185` (`q->exec_us`) | `u64` (microseconds) | In `calibrate_case` (`afl-fuzz-run.c:682`) | Trivial | **No** |
| **Seed Fuzz Count** | `include/afl-fuzz.h:173,189` (`q->fuzz_level`, `q->fuzzed_times`); `src/aflrun.cpp:3056` (`seed_quant[id]`) | `u32` / `u64` / `double` | `fuzz_level` at end of `fuzz_one` (`afl-fuzz-one.c:3251`); `fuzzed_times` per `common_fuzz_stuff` (`afl-fuzz-run.c:1164`); `seed_quant` after `fuzz_one` (`afl-fuzz.c:2742`) | Trivial | **Minor**: note `fuzz_one_original` increments `fuzz_level` at line 3251 (`abandon_entry`), whereas `mopt` didn't; in RL mode update both consistently |
| **Seed Energy** | `include/afl-fuzz.h:199` (`q->quant_score`, `q->perf_score`); `src/aflrun.cpp:3118` | `double` | Cycle start (`afl-fuzz.c:2504`) | Trivial | **N/A** (RL sets energy; can export `perf_score` + `seed_quant[id]`) |
| **Global Coverage** | `afl->virgin_bits` (`afl-fuzz-bitmap.c:118` `count_non_255_bytes`) | `u32` (edges) / `double` (`bitmap_cvg`) | Every `has_new_bits_mul` (`afl-fuzz-bitmap.c:592`) | Trivial | **No** (expose delta & total) |
| **Target-related Coverage** | `src/aflrun.cpp:412-413,3061` (`num_reached`, `num_reached_targets`, `num_freached`, `num_freached_targets`); `afl->virgin_ctx` | `reach_t` (`u32`) & bitcount of `virgin_ctx` | Every `aflrun_has_new_path` (`aflrun.cpp:3576`) | Easy | **Minimal**: expose `aflrun_get_reached()` + `virgin_ctx` bitcount; add monotonic `virgin_target_bits` if edge-level target-conditioned cov is used |
| **Target Diversity** | `src/aflrun.cpp:543,2004,2372` (`path_fringes`, `path_pro_fringes`, `reached_targets`, `clusters`, `div_blocks`) | `size_t` counts (`fringes.size()`, `clusters.size()`, `num_invalid`, `num_fringes`) | Every `aflrun_has_new_path` & `commit_bit_seqs` | Easy | **Minor**: add getter functions in `aflrun.cpp` for active fringe counts & per-seed fringe counts (`seed_fringes[id].size()`) |
| **Newly Discovered Seeds** | `afl->queued_items`, `afl->queued_discovered`, `afl->queued_extra`, `q->mother` (`afl-fuzz.h:207`) | `u32` delta (`after - before` around `fuzz_one`) | Synchronously inside `save_if_interesting` (`afl-fuzz-bitmap.c:614`) | Trivial | **No** (`fuzz_one` is single-threaded; delta before/after `fuzz_one` attributes 100% of finds to `(i_t, e_t)`) |
| **Remaining Budget** | `afl->fsrv.total_execs` (`include/forkserver.h:112`, incremented in `afl-forkserver.c:1446,1606`) | `u64` | Every target execution (including calibration & trimming) | Trivial | **Minimal**: enforce hard budget in `fuzz_one` + report `total_execs` delta |

---

## 4. Task 6 — Episode Reset Feasibility Audit `[Source-verified only]`

### 4.1 All Mutable State Inside AFLRun
Resetting AFLRun **in-process** would require resetting dozens of coupled global/static structures across C and C++:
1. `afl_state_t` (`virgin_bits`, `virgin_tmout`, `virgin_crash`, `var_bytes`, `top_rated`, `queue_buf`, `q_testcase_cache`, `n_fuzz`, `alias_table`, `fsrv.total_execs`, `last_find_time`, `rand_seed`, `expand_havoc`, etc.).
2. `aflrun.cpp` process-wide globals (`g`, `state`, `path_fringes`, `path_pro_fringes`, `reached_targets`, `clusters`, `div_blocks`, `seed_quant`, `update_time`, `TargetGrouper::all_targets`, `num_active_seeds`, plus `std::mt19937 gen` at `aflrun.cpp:1144`).
3. `aflrun_shm_t` SysV shared memory (`map_reachables`, `map_freachables`, `map_ctx`, `map_virgin_ctx`, `map_new_blocks`, `map_targets`, `div_switch`).
4. Target forkserver child process & `out_dir/queue/` files on disk.

### 4.2 Recommended Reset Strategy: Clean Process Restart
- **Recommendation**: **Restart the `afl-fuzz` subprocess inside `env.reset()`**, using a fresh (or cleaned) temporary `out_dir` (ideally on `/dev/shm` tmpfs if desired — 167 GiB free on `/dev/shm`!), re-reading the immutable `in_dir` initial corpus (`read_testcases` uses `scandir(..., alphasort)` at `afl-fuzz-init.c:692`, deterministic!).
- **Why**:
  - Guarantees **100% bit-exact isolation** across episodes: zero stale bitmap bytes, zero leaked `virgin_ctx` / `Clusters` state, zero queue cache or `top_rated` contamination, clean SysV SHM teardown (`afl_shm_deinit` / `aflrun_shm_deinit`), and clean forkserver restart.
  - **Startup cost**: On our smoke target, `afl-fuzz` dry-run + forkserver spin-up took **< 0.2 seconds** (`121 us` * `CAL_CYCLES_FAST` per initial seed). Even with 20 initial seeds, restart overhead is < 0.5–1.0s compared to a 10–30s episode!
- **RNG & Reproducibility Control**:
  - `afl-fuzz` already supports `-s <seed>` (`src/afl-fuzz.c:615-622`), which calls `rand_set_seed(afl, seed)` and sets `afl->fixed_seed = 1`.
  - Note two minor caveats in vanilla AFLRun when `-s` is used:
    1. `afl->fixed_seed = 1` forces `q->exec_us` to a fake constant `(exec_tmout - 1)` in `calibrate_case` (`src/afl-fuzz-run.c:657-660`) and disables speed-based scoring (`afl-fuzz-queue.c:674,939`)! If we want reproducible RNG *without* faking `exec_us`, pass a separate `--rl-seed <seed>` (or decouple `rand_set_seed` from `fixed_seed`'s timing override).
    2. `src/aflrun.cpp:1143-1144` uses `random_device rd; mt19937 gen(rd());` for `cull_queue`, though under RL Scheme A `cull_queue` random skipping is bypassed anyway.

---

## 5. Summary Answer Matrix (Completion Criteria 1–8)

| # | Question | Answer | Evidence Status |
|---|---|---|---|
| 1 | AFLRun compiled & running on server? | **Yes.** Built cleanly with user-space LLVM/Clang/LLD 15.0.6 (`EXIT=0`, LTO mode & `test_build` passed). | **Verified** |
| 2 | Target-related Coverage verified? | **Yes.** Verified `BBreachable.txt` (`2,21`), `Freachable.txt` (`2,3`), `distance.cfg/`, `100%` target hit (`2/2`), `aflrun_targets.txt`, `seeds.txt`, and `+cov2,+div2,+path` tags in 30s smoke test (`356,985` execs, `11,898.31 execs/s`). | **Verified** |
| 3 | Can RL control Seed Selection? | **Yes**, by setting `afl->current_entry` & `afl->queue_cur = afl->queue_buf[id]` before `fuzz_one()`, filtering `!q->disabled`, and bypassing `aflrun_cull_queue` / skip probs. | **Source-verified only** |
| 4 | Can RL control Energy Assignment? | **Yes**, by overriding `num_execs` directly in `fuzz_one_original()` and enforcing a step-level `total_execs` cap (accounting for calibration/trim). | **Source-verified only** |
| 5 | Where should the RL Step boundary be placed? | **Scheme A**: immediately around `fuzz_one(afl)` in `src/afl-fuzz.c` (1 step = 1 `(seed_slot, exec_budget)`). | **Source-verified only** |
| 6 | Can Reward be computed from existing target coverage data? | **Yes**, with clarification: distinguish global `virgin_bits`, target-conditioned edges/blocks (`virgin_reachables`, `virgin_ctx`, `Clusters` or a monotonic `virgin_target_bits`), and AFLRun's `FringeBlocks` diversity. | **Source-verified only** |
| 7 | Is full Episode Reset feasible? | **Yes**, cleanly via subprocess restart of `afl-fuzz` in `env.reset()` with fresh `out_dir` and deterministic `in_dir` scan. | **Source-verified only** |
| 8 | Any engineering blocker for a 2–3 week solo mini project? | **No.** All components are localized; required C/C++ hooks are ~150–250 LOC. | **Source-verified only** |

**Final Phase 0 Decision**: **CONDITIONAL GO** — Proceed to Phase 1 after setting up the Python `.venv` (`gymnasium`, `stable-baselines3`) and implementing the lightweight per-step C/C++ IPC adapter + monotonic target-conditioned coverage tracker.
