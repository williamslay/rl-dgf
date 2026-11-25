# RL-DGF Mini Project

Reinforcement Learning for Joint Seed Selection and Energy Assignment in Directed Greybox Fuzzing (AFLRun + Gymnasium + Stable-Baselines3 PPO).

## Current Phase

**Phase 1B — PPO Minimal Training & Baseline Evaluation (Completed)**

Implements SB3 `PPO("MlpPolicy")` training and evaluation against a Random Scheduling baseline over `rl_dgf.DirectedFuzzEnv`, with updated finite-horizon `terminated=True` budget exhaustion semantics and a multi-branch controllable `pilot-target`.

## Directory Structure

```text
RL_miniproject/
├── .venv/                   # User-local Python 3.12 venv (gymnasium, numpy, torch, sb3, matplotlib)
├── backend/
│   └── AFLRun/              # Cloned Mem2019/AFLRun (commit ea58c0e) + RL Unix socket adapter
├── benchmarks/
│   ├── smoke-target/        # Minimal Phase 0/1A C target + BBtargets.txt + initial seed
│   └── pilot-target/        # Multi-subsystem C target (5 BBs, 62 reachables) for Phase 1B PPO pilot
├── docs/
│   ├── ENVIRONMENT-REPORT.md
│   ├── RL-INTERFACE-AUDIT.md
│   ├── IMPLEMENTATION-PLAN.md
│   ├── RL-ENVIRONMENT-API.md
│   ├── PHASE1A-REPORT.md
│   └── PHASE1B-REPORT.md
├── rl_dgf/
│   ├── __init__.py
│   ├── env.py               # DirectedFuzzEnv(gymnasium.Env), Discrete(12), Box(30,)
│   ├── candidates.py        # Deterministic K=4 candidate selector & <K wraparound
│   └── ipc.py               # AF_UNIX length-prefixed JSON IPC + afl-fuzz subprocess manager
├── results/
│   ├── aflrun-build-*.log   # Compilation logs
│   ├── smoke/               # Phase 0 native smoke test logs & outputs
│   ├── phase1a/             # Phase 1A verification logs (steps_100plus.jsonl, summaries)
│   └── phase1b/             # Phase 1B PPO checkpoints, JSONL logs, eval outputs, and figures
├── scripts/
│   ├── build_aflrun.sh      # Reproducible AFLRun build script using local LLVM 15.0.6
│   ├── build_pilot_target.sh# Reproducible build script for pilot-target
│   ├── run_smoke.sh         # Reproducible native AFLRun smoke test script
│   ├── test_reward_repeat.py# Standalone target-conditioned reward verification
│   ├── test_phase1a.py      # Full Phase 1A integration test suite
│   ├── train_ppo.py         # Minimal SB3 PPO trainer with full JSONL/TB telemetry
│   ├── evaluate.py          # PPO vs Random baseline evaluator
│   └── plot_phase1b.py      # Phase 1B figure generator
└── toolchain/
    └── llvm-15.0.6/         # Prebuilt user-space Clang/LLVM/LLD 15.0.6 toolchain
```

## Reports

- [ENVIRONMENT-REPORT.md](file:///home/sylei/RL_miniproject/docs/ENVIRONMENT-REPORT.md)
- [RL-INTERFACE-AUDIT.md](file:///home/sylei/RL_miniproject/docs/RL-INTERFACE-AUDIT.md)
- [IMPLEMENTATION-PLAN.md](file:///home/sylei/RL_miniproject/docs/IMPLEMENTATION-PLAN.md)
- [RL-ENVIRONMENT-API.md](file:///home/sylei/RL_miniproject/docs/RL-ENVIRONMENT-API.md)
- [PHASE1A-REPORT.md](file:///home/sylei/RL_miniproject/docs/PHASE1A-REPORT.md)
- [PHASE1B-REPORT.md](file:///home/sylei/RL_miniproject/docs/PHASE1B-REPORT.md)
