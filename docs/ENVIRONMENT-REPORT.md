# ENVIRONMENT-REPORT — Phase 0 Environment Audit, AFLRun Build & Smoke Test

> **Project Root**: `/home/sylei/RL_miniproject`  
> **Audit Date**: 2026-11-25  
> **Evidence Tags**: `[Verified]` = directly executed and observed; `[Source-verified only]` = confirmed from source/config files without runtime test; `[Unverified]` = not tested in Phase 0; `[Blocked]` = unavailable without extra setup.

---

## 1. Server Environment Inventory `[Verified]`

### 1.1 Hardware, OS & User Permissions
| Item | Observed Value | Status |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (`noble`) | `[Verified]` |
| Kernel | `Linux fineserver 6.8.0-87-generic #88-Ubuntu SMP PREEMPT_DYNAMIC x86_64` | `[Verified]` |
| CPU | `AMD EPYC 9455 48-Core Processor` (192 logical CPUs, 2 NUMA nodes, 2 threads/core) | `[Verified]` |
| Memory | 377 GiB total, ~256 GiB available (`free -h`) | `[Verified]` |
| GPU | 4x NVIDIA GeForce RTX 4090 (24,564 MiB each; GPU 3 has 20,781 MiB free) | `[Verified]` |
| User & Groups | `uid=1006(sylei) gid=1006(sylei) groups=1006(sylei),27(sudo),108(kvm),147(libvirt)` | `[Verified]` |
| Disk (`/home`) | `/dev/nvme6n1p4`: 1.6T total, 1.4T used, 121G available (92% used) | `[Verified]` |
| Disk (`/`) | `/dev/nvme6n1p3`: 196G total, 116G used, 71G available (63% used) | `[Verified]` |
| `core_pattern` | `/proc/sys/kernel/core_pattern` = `core` (no crash-handler pipe blocking AFL++) | `[Verified]` |
| Host Load | Active multi-user server (high-CPU jobs present on other users, plenty of free cores/RAM) | `[Verified]` |

### 1.2 Compilers, Build Tools & System LLVM
| Tool | Version & Path | Status |
|---|---|---|
| `git` | `2.43.0` (`/usr/bin/git`) | `[Verified]` |
| `cmake` | `3.28.3` (`/usr/bin/cmake`) | `[Verified]` |
| `make` | `GNU Make 4.3` (`/usr/bin/make`) | `[Verified]` |
| `ninja` | `1.13.0.git.kitware.jobserver-pipe-1` (`/home/sylei/.local/bin/ninja`) | `[Verified]` |
| `gcc` / `g++` | `13.3.0` (`Ubuntu 13.3.0-6ubuntu2~24.04.1`, `/usr/bin/gcc`, `/usr/bin/g++`) | `[Verified]` |
| System `clang` / `clang++` | **Not installed** (`NOT FOUND`) | `[Verified]` |
| System `lld` | **Not installed** (`NOT FOUND`) | `[Verified]` |
| System `llvm-config` / `opt` / `llc` | `18.1.3` (`/usr/bin/llvm-config`, `/usr/lib/llvm-18/`) | `[Verified]` |
| Boost headers | `libboost-all-dev` `1.74.0` installed (`/usr/include/boost/`) | `[Verified]` |
| `python3-dev` | `3.12.3-0ubuntu2.1` installed | `[Verified]` |
| `binutils` / `plugin-api.h` | `binutils 2.42`, `/usr/include/plugin-api.h` **not present** | `[Verified]` |

### 1.3 Python & RL Dependencies
| Component | Status | Detail |
|---|---|---|
| `python3` | `[Verified]` | `3.12.3` (`/usr/bin/python3`) |
| `pip` / `pip3` | `[Verified]` | `24.0` (`/usr/bin/pip3`) |
| `venv` | `[Verified]` | `import venv` succeeds |
| `gymnasium` | `[Verified]` (Missing) | `ModuleNotFoundError: No module named 'gymnasium'` in global Python |
| `stable_baselines3` | `[Verified]` (Missing) | `ModuleNotFoundError: No module named 'stable_baselines3'` in global Python |
| `torch` | `[Verified]` (Missing) | `ModuleNotFoundError: No module named 'torch'` in global Python |

> **Action for Phase 1**: Create a user-level virtual environment `/home/sylei/RL_miniproject/.venv` and install `gymnasium`, `stable-baselines3`, `torch`, `numpy` via `pip` without modifying system Python.

---

## 2. LLVM Toolchain Compatibility Analysis `[Verified]`

### 2.1 Why System LLVM 18.1.3 Could Not Build AFLRun
1. System lacks `clang`, `clang++`, and `ld.lld` completely (`GNUmakefile.llvm:233` disables LTO mode).
2. `instrumentation/afl-llvm-common.h:24` includes `"llvm/Transforms/IPO/PassManagerBuilder.h"`, which was removed in LLVM 17/18 (`fatal error: llvm/Transforms/IPO/PassManagerBuilder.h: No such file or directory` in [aflrun-build-attempt1.log](file:///home/sylei/RL_miniproject/results/aflrun-build-attempt1.log)).
3. `instrumentation/aflrun-pass.cc:111` calls `StringRef::startswith(...)` (renamed to `starts_with` in LLVM 18), and legacy pass registration APIs were removed.

### 2.2 Important Finding on AFLRun README's Commit vs Clang 16.0.3
- AFLRun's [README.md](file:///home/sylei/RL_miniproject/backend/AFLRun/README.md#L5-L12) states *"tested with clang 16.0.3"* while pinning LLVM commit `4a2c05b05ed07f1f620e94f6524a8b4b2760a0b1`.
- Inspecting `llvm/CMakeLists.txt` at commit `4a2c05b05ed07f1f620e94f6524a8b4b2760a0b1` directly revealed it is actually **LLVM 15.0.3** (`LLVM_VERSION_MAJOR=15, MINOR=0, PATCH=3`).
- Attempting to compile AFLRun with official prebuilt **LLVM 16.0.3** failed (`results/aflrun-build-llvm16-*.log`, `results/aflrun-lto-check.log`) due to LLVM 16 breaking API changes:
  - `llvm::None` removed in favor of `std::nullopt` (`afl-llvm-pass.so.cc`, `SanitizerCoverageLTO.so.cc`, `aflrun-pass.cc`)
  - `RegisterStandardPasses` / `PassManagerBuilder::EP_*` removed (`SanitizerCoverageLTO.so.cc:1832-1840`)
  - `BasicBlock::getInstList()` became private (`split-compares-pass.so.cc:268-947`)
  - `ValueName::Create` renamed to `create` (`aflrun-pass.cc:608`)
  - `-fexperimental-new-pass-manager` removed in Clang 16 (`src/afl-cc.c:720`)
  - C++17 required by LLVM 16 headers, conflicting with `GNUmakefile.llvm:80` (`LLVM_STDCXX = c++14`).

### 2.3 Chosen User-Space Alternative (No Full LLVM Source Build, No `sudo`)
- Instead of building LLVM from source (~30–60 min, ~30–50 GB build tree) or modifying system packages via `apt`, we installed the official prebuilt **Clang+LLVM 15.0.6** (`clang+llvm-15.0.6-x86_64-linux-gnu-ubuntu-18.04.tar.xz`, 817 MB archive) entirely inside `/home/sylei/RL_miniproject/toolchain/llvm-15.0.6`.
- Verified `ldd` on `toolchain/llvm-15.0.6/bin/clang`: zero missing shared libraries on Ubuntu 24.04.
- Removed the temporary LLVM 16.0.3 directory to avoid wasting disk space.

---

## 3. AFLRun Repository & Compilation `[Verified]`

### 3.1 Repository Metadata
- **Upstream**: `https://github.com/Mem2019/AFLRun`
- **Local Path**: `/home/sylei/RL_miniproject/backend/AFLRun`
- **Branch**: `main`
- **Commit SHA**: `ea58c0e9f49615a02753f0c32dcc5dec8a46a87f`
- **Submodule**: `9145f963d80d6a02f0f96a47758050a89184a3ed robin-hood-hashing (3.11.5)` (checked out cleanly)
- **License**: Apache License, Version 2.0 ([LICENSE](file:///home/sylei/RL_miniproject/backend/AFLRun/LICENSE))
- **Core Dependencies**: Clang/LLVM 15 + LLD 15 (`ld.lld`), `libboost-all-dev` (BGL Dijkstra in `aflrun-pass.cc`), `robin-hood-hashing` header-only submodule.

### 3.2 Minimal Compatibility Fixes Applied
Only two tiny compatibility fixes (4 lines total in `git diff`, zero architectural or scheduling changes) were needed:
1. [instrumentation/afl-llvm-common.cc](file:///home/sylei/RL_miniproject/backend/AFLRun/instrumentation/afl-llvm-common.cc#L12-L16): added `#include <cmath>` for `pow`/`round` under GCC 13 `libstdc++`.
2. [src/afl-cc.c](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-cc.c#L719-L724): guarded `-fexperimental-new-pass-manager` with `#if LLVM_MAJOR < 16` (matching line 806).

And two build flags were passed to `make` ([build_aflrun.sh](file:///home/sylei/RL_miniproject/scripts/build_aflrun.sh)):
- `NO_PYTHON=1`: bypasses an upstream AFLRun bug in [src/afl-fuzz-python.c](file:///home/sylei/RL_miniproject/backend/AFLRun/src/afl-fuzz-python.c#L441) where `fuzz_count_py` takes 3 parameters while `afl_custom_fuzz_count` in [include/afl-fuzz.h](file:///home/sylei/RL_miniproject/backend/AFLRun/include/afl-fuzz.h#L890) was changed to 4 parameters (`-Wincompatible-function-pointer-types` under Clang).
- `AFL_CLANG_FLTO=-flto=full`: bypasses `GNUmakefile.llvm:199` testing `-flto=full` with system `ld` (which fails without `LLVMgold.so`), allowing `GNUmakefile.llvm:214` to detect and bind `toolchain/llvm-15.0.6/bin/ld.lld`.

### 3.3 Build Outcome (`EXIT=0`)
Full build log: [aflrun-build-llvm15.log](file:///home/sylei/RL_miniproject/results/aflrun-build-llvm15.log)
- Built `afl-fuzz`, `afl-showmap`, `afl-tmin`, `afl-analyze`, `afl-gotcpu`, `afl-as`, `afl-cc`, `aflrun-cc`, `afl-ld-lto`, `afl-clang-lto` / `afl-clang-lto++`.
- Built all LLVM passes including `SanitizerCoverageLTO.so`, `lto-marker.so`, `afl-llvm-pass.so`, `SanitizerCoveragePCGUARD.so`, `afl-llvm-rt-lto.o`, `afl-compiler-rt.o`.
- Built-in `test_build` (`./afl-clang-lto ./test-instr.c` + `afl-showmap` differential check) **passed**.
- Optional `gcc_plugin` (`afl-gcc-pass.so`) was not built because `gcc-13-plugin-dev` is not installed (not required by AFLRun, which uses LLVM LTO mode).

---

## 4. Minimal Smoke Test (`benchmarks/smoke-target/`) `[Verified]`

### 4.1 Target Design & Instrumentation
- **Source**: [target.c](file:///home/sylei/RL_miniproject/benchmarks/smoke-target/target.c)
- **Targets** ([BBtargets.txt](file:///home/sylei/RL_miniproject/benchmarks/smoke-target/BBtargets.txt)):
  - `target.c:30` inside `path_b` (reached only by inputs starting with `"BB2!"`)
  - `target.c:49` inside `path_c` (reached only by inputs starting with `"CCXY"`)
- **Non-target path**: `path_a` (input starting with `"A"`, initial seed `seeds/seed0` = `"AAAA"`).
- **Compilation command**:
  ```bash
  AFLRUN_BB_TARGETS=BBtargets.txt AFLRUN_TARGETS="target" AFLRUN_TMP=aflrun_tmp \
    afl-clang-lto -g -O1 target.c -o target
  ```
- **Static CFG & Distance Output** (`aflrun_tmp/target.mEpkJU7B1To7/`):
  - `BBreachable.txt`: `2,21` (2 target BBs, 21 reachable BBs)
  - `Freachable.txt`: `2,3` (2 target functions: `path_b`, `path_c`; 3 reachable functions: `path_b`, `path_c`, `main` — `path_a` correctly excluded)
  - `distance.cfg/0.txt` and `distance.cfg/1.txt`: per-BB shortest path distances to target 0 (`22:target.c:30`, 14 reachable BBs) and target 1 (`24:target.c:49`, 13 reachable BBs).

### 4.2 Fuzzing Execution & Observed Metrics
- **Script**: [run_smoke.sh](file:///home/sylei/RL_miniproject/scripts/run_smoke.sh)
- **Command**:
  ```bash
  AFL_NO_UI=1 AFL_SKIP_CPUFREQ=1 afl-fuzz -d -V 30 -i seeds -o out \
    --config "init_cov_quant=30:log_at_begin=1:log_check_interval=2000" \
    -- ./target @@
  ```
  *(Note: `init_cov_quant=30` sets initial coverage warm-up to 3.0s instead of the default 600s so AFLRun transitions into directed `kUnite` (`U`) scheduling within the 30s smoke window.)*
- **Full Logs**: [aflrun-smoke.log](file:///home/sylei/RL_miniproject/results/smoke/aflrun-smoke.log) and [fuzzer_stats](file:///home/sylei/RL_miniproject/results/smoke/out/default/fuzzer_stats)

| Metric | Observed Result | Source / Evidence |
|---|---|---|
| Exit Code | `0` (clean exit on `-V 30` *"Time limit was reached"*) | `aflrun-smoke.log` |
| Runtime | `30s` | `fuzzer_stats` (`run_time: 30`) |
| Total Executions | `356,985` | `fuzzer_stats` (`execs_done: 356985`) |
| Throughput | `11,898.31 execs/sec` | `fuzzer_stats` (`execs_per_sec: 11898.31`) |
| Corpus Size Growth | `1` initial (`AAAA`) -> `19` total (`18` newly discovered) | `queue/id:000000` .. `id:000018` |
| Edge Coverage | `25 / 33` edges (`75.76%` `bitmap_cvg`), `100.00%` stability | `fuzzer_stats` |
| Reachable BB Coverage | `6/21 (28%)` -> `18/21 (85%)` -> `21/21 (100%)` | `aflrun-smoke.log`, `aflrun_log.txt` |
| Target Hit Progression | `0/2 (0%)` -> `1/2 (50%)` (at 3.36s, `id:000016` `CCXY`) -> `2/2 (100%)` (at 28.72s, `id:000018` `BB2!`) | `aflrun-smoke.log`, `queue/` |
| Target-Related Feedback Files | `aflrun_log.txt`, `aflrun_targets.txt` (clusters & targets), `seeds.txt` (seed->fringe/target map), `queue/*,+cov2,+div2,+path` | `results/smoke/out/default/` |

### 4.3 Blocking / Environment Caveats
1. **Python RL packages missing in global Python**: `gymnasium`, `stable_baselines3`, and `torch` are not installed yet; must be installed into a project-local `.venv` in Phase 1. `[Verified]`
2. **Debug symbols & optimization caveat**: Target binaries MUST be compiled with `-g` and avoid aggressive dead-code/branch-merging optimizations (e.g., `select` folding of target statements) so `BBtargets.txt` matches surviving basic blocks in `aflrun-pass.cc`. `[Verified]`
3. **Python custom mutator disabled**: Built with `NO_PYTHON=1`; this does not affect our plan because Phase 1 will use C/C++ IPC (Unix domain socket / pipe) rather than AFL++'s `USE_PYTHON` custom mutator API. `[Verified]`
