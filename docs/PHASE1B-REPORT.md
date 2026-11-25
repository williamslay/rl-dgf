# RL-DGF Phase 1B Report: PPO Minimal Training & Baseline Evaluation

**Date**: 2026-11-25  
**Workspace**: `/home/sylei/RL_miniproject`  
**Pipeline Status**: **PASS (Runnable End-to-End PPO + Baseline + Diagnostics Complete)**

---

## 1. Executive Summary

Phase 1B built and verified a runnable SB3 `PPO("MlpPolicy")` training and evaluation pipeline over `rl_dgf.DirectedFuzzEnv` (`Discrete(12)`, 30-dim continuous observations in `[0.0, 1.0]`), fixed the finite-horizon MDP termination semantics, installed `torch==2.15.0+cpu` and `stable-baselines3==2.9.2` inside `.venv`, passed a 2,048-step smoke test on `smoke-target`, and ran a 15,360-step (272-episode) Pilot Training and 10-episode comparative evaluation against a Random Scheduling Baseline on a controllable multi-branch target (`benchmarks/pilot-target`).

**Headline Scientific Findings (Unedited / Honest Baseline Behavior)**:
- **Pipeline Verification**: PPO cleanly collects rollouts, updates policy and value networks, writes TensorBoard/JSONL telemetry, saves periodic and final checkpoints (`ppo_final.zip`), reloads via `PPO.load`, and runs multi-episode evaluations deterministically against Random under identical budgets (`15,000` executions/episode) and initial seeds (`"AAAA"`).
- **Critic Learned, Policy Did Not Beat Random**: On `pilot-target`, `train/explained_variance` improved from **`0.0092`** (iteration 2) to **`0.5752`** (iteration 30), confirming the value network predicts expected remaining target-coverage return. However, episode returns during training stayed near **`17.8–19.2`** with high variance (`std ~ 12.1–12.7`), and in 10-episode held-out evaluation (`seeds 2000..2009`), **Random Baseline achieved `21.4 ± 10.26`** vs **`17.6 ± 10.38` for PPO (deterministic)** and **`13.8 ± 13.88` for PPO (stochastic)**.
- **Root Causes Diagnosed**: (1) **Severe reward sparsity** (`91.01%` of training steps and `23.9%` of training episodes return `0.0`); (2) **High-energy bias / logit collapse** (`Energy = 512` grew from `39.4%` to `52.3%` during training and reached **`100.0%`** under deterministic evaluation because per-step coverage reward is unnormalized by execution cost); (3) **Top-4 deterministic candidate starvation** (`PPO deterministic` executed **Slot 0 on `95.7%`** of steps, starving alternative subsystems and discovering fewer seeds: `28.5` vs `39.5` for Random).

---

## 2. Environment & Termination Semantics Update

Before starting PPO training, `rl_dgf/env.py` was audited and updated:
- **Finite-Horizon Termination (`terminated=True`)**: Exhausting `episode_budget` now sets `terminated = True, truncated = False` (and `info["budget_exhausted"] = True`), because the fixed execution budget is a true task-boundary constraint where zero remaining executions yield $V(s_T) = 0$. Premature unexpected backend termination before budget exhaustion sets `truncated = True, terminated = False`.
- **Phase 1A Regression Re-verified**: Updated the corresponding assertion in `scripts/test_phase1a.py` and re-ran the full Phase 1A verification suite (`check_env`, 10,000-exec episode, 124-step run, 12 consecutive resets, fixed-seed reproducibility, and 5s native AFLRun regression) — all checks passed (`exit 0`).
- **Backend & Adapter Preserved**: No modifications were made to `backend/AFLRun/`.

---

## 3. Dependencies & Training Setup

### Local Environment (`.venv`)
Installed inside `/home/sylei/RL_miniproject/.venv`:
- `torch==2.15.0+cpu`
- `stable-baselines3==2.9.2`
- `gymnasium==1.4.0`, `numpy==2.5.5`
- `matplotlib==3.11.2`, `tensorboard==2.21.0`, `pandas==3.1.1`

### Training Scripts & Architecture
- [scripts/train_ppo.py](file:///home/sylei/RL_miniproject/scripts/train_ppo.py): Minimal SB3 `PPO("MlpPolicy")` trainer wrapped with `Monitor`, `CheckpointCallback`, `JsonlKVWriter`, and `FuzzTrainingCallback` logging every step (`train_steps.jsonl`), every episode (`train_episodes.jsonl`), and every rollout update (`train_metrics.jsonl`).
- [scripts/evaluate.py](file:///home/sylei/RL_miniproject/scripts/evaluate.py): Evaluates loaded PPO checkpoints (`deterministic=True` and `deterministic=False`) against a seeded `Random` baseline under identical seeds, initial corpus, execution budget, and reward definition.
- [scripts/plot_phase1b.py](file:///home/sylei/RL_miniproject/scripts/plot_phase1b.py): Generates all 4 analysis figures from raw JSONL logs.

---

## 4. Pipeline Smoke Test (`smoke-target`, 2,048 RL Steps)

- **Command / Parameters**: `target=smoke`, `total_timesteps=2048`, `episode_budget=10000`, `n_steps=256`, `batch_size=64`, `lr=3e-4`, `ent_coef=0.01`, `gamma=0.99`, `seed=42`.
- **Execution**: Completed **46 consecutive episodes** in `46.20s` (~44 fps), saved checkpoints (`ppo_ckpt_1024_steps.zip`, `ppo_ckpt_2048_steps.zip`, `ppo_final.zip`), and verified reload (`load_verify_action=1`).
- **Observed Saturation on `smoke-target`**: `ep_rew_mean` remained `0.0` through the first 6 iterations (~1,536 steps) and reached `0.348` by step 2,048; hits yield an all-or-nothing reward of `4.0` (`mean_return=1.6 ± 1.96` across 5 eval episodes vs `0.8 ± 1.6` for Random), while `ppo_deterministic` collapsed 100% onto `Energy=128` (`395/395` steps).

Because `smoke-target` saturates in a single jump (`0` or `4` slots), a richer controllable C target (`benchmarks/pilot-target/`) was compiled for Pilot Training.

---

## 5. Controllable Pilot Target (`benchmarks/pilot-target/`)

Built via [scripts/build_pilot_target.sh](file:///home/sylei/RL_miniproject/scripts/build_pilot_target.sh) from [target.c](file:///home/sylei/RL_miniproject/benchmarks/pilot-target/target.c) and [BBtargets.txt](file:///home/sylei/RL_miniproject/benchmarks/pilot-target/BBtargets.txt):
- **Static Instrumentation (`BBreachable.txt`)**: `5` target basic blocks, `62` reachable basic blocks (vs `2` / `21` on `smoke-target`).
- **Structure**: Three independent multi-stage input subsystems (`PK` packet TLV/stateful records, `CM` ASCII commands `AUTH`/`EXEC`, and `VM` accumulator/stack bytecode interpreter) plus non-target decoy paths (`N*`/`D*`). Initial seed is `"AAAA"`.
- **Dynamics at `episode_budget=15,000`**: Episodes span `~50–70` RL steps with progressive target-coverage returns (`0–41` bitmap edge slots) and corpus growth (`1 -> 14..72` seeds), avoiding instant single-step saturation.

---

## 6. Pilot Training Results (`pilot-target`, 15,360 RL Steps)

### Configuration
- `total_timesteps = 15,360` (`30` PPO updates × `512` rollout steps), `batch_size = 64`, `n_epochs = 10`, `learning_rate = 3e-4`, `gamma = 0.99`, `ent_coef = 0.01`, `episode_budget = 15,000`, `seed = 42`, `device = cpu`.
- Wall time: `393.31s` (`272` completed episodes).
- Artifacts: [pilot_ppo/checkpoints/ppo_final.zip](file:///home/sylei/RL_miniproject/results/phase1b/pilot_ppo/checkpoints/ppo_final.zip), [train_episodes.jsonl](file:///home/sylei/RL_miniproject/results/phase1b/pilot_ppo/train_episodes.jsonl), [train_metrics.jsonl](file:///home/sylei/RL_miniproject/results/phase1b/pilot_ppo/train_metrics.jsonl), [train_steps.jsonl](file:///home/sylei/RL_miniproject/results/phase1b/pilot_ppo/train_steps.jsonl).

### Training Progression by Quartile (`68` episodes per quartile)

| Quartile | Mean Return ± Std | Zero-Return Episodes | Mean Episode Steps | Energy Share (`32` / `128` / `512`) |
| --- | --- | --- | --- | --- |
| Q1 (ep 1–68) | `17.84 ± 12.09` | `18 / 68` (26.5%) | `61.7` | `0.304` / `0.302` / `0.394` |
| Q2 (ep 69–136) | `19.24 ± 12.70` | `14 / 68` (20.6%) | `57.1` | `0.292` / `0.263` / `0.444` |
| Q3 (ep 137–204) | `18.24 ± 12.27` | `17 / 68` (25.0%) | `55.8` | `0.316` / `0.219` / `0.465` |
| Q4 (ep 205–272) | `18.40 ± 12.12` | `16 / 68` (23.5%) | `50.8` | `0.257` / `0.220` / `0.523` |

### PPO Optimization Diagnostics

| Iteration | Timesteps | `explained_variance` | `value_loss` | `policy_gradient_loss` | Entropy (`-entropy_loss`) |
| --- | --- | --- | --- | --- | --- |
| 2 | 1,024 | `0.0092` | `22.57` | `-0.0150` | `2.479` |
| 5 | 2,560 | `0.1337` | `16.09` | `-0.0061` | `2.472` |
| 10 | 5,120 | `0.6300` | `17.27` | `-0.0161` | `2.379` |
| 15 | 7,680 | `0.3973` | `23.19` | `-0.0080` | `2.336` |
| 20 | 10,240 | `0.4611` | `15.54` | `-0.0062` | `2.294` |
| 25 | 12,800 | `0.2989` | `19.91` | `-0.0105` | `2.381` |
| 30 | 15,360 | `0.5752` | `15.97` | `-0.0086` | `2.261` |

![Episode Return Learning Curve, Losses, Entropy, and Energy Trend](/home/sylei/RL_miniproject/results/phase1b/figures/fig1_learning_curve.png)

---

## 7. PPO vs. Random Baseline Evaluation (`pilot-target`, 10 Episodes)

Evaluated over seeds `2000..2009` with identical initial corpus (`"AAAA"`), `episode_budget=15,000`, and target-conditioned bitmap edge-slot reward ([eval_summary.json](file:///home/sylei/RL_miniproject/results/phase1b/pilot_eval/eval_summary.json)):

| Policy | Mean Return ± Std | Min / Max | Targets Reached ( /5) | Mean Global Cov | Mean Corpus Size | Energy Share (`32`/`128`/`512`) | Slot Share (`0`/`1`/`2`/`3`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PPO (Deterministic)** | `17.60 ± 10.38` | `0.0 / 32.0` | `3.1` | `42.9` | `28.5` | `0.0% / 0.0% / 100.0%` | `95.7% / 0.0% / 0.0% / 4.3%` |
| **PPO (Stochastic)** | `13.80 ± 13.88` | `0.0 / 35.0` | `1.9` | `39.3` | `31.8` | `27.4% / 19.3% / 53.3%` | `34.8% / 15.5% / 15.9% / 33.8%` |
| **Random Baseline** | **`21.40 ± 10.26`** | `0.0 / 35.0` | **`3.4`** | **`48.5`** | **`39.5`** | `35.2% / 33.1% / 31.7%` | `22.9% / 26.4% / 26.0% / 24.7%` |

![Target Coverage vs Executions](/home/sylei/RL_miniproject/results/phase1b/figures/fig2_target_cov_vs_execs.png)
![PPO vs Random Evaluation Comparison](/home/sylei/RL_miniproject/results/phase1b/figures/fig3_ppo_vs_random.png)
![Training and Evaluation Energy Action Distribution](/home/sylei/RL_miniproject/results/phase1b/figures/fig4_energy_distribution.png)

---

## 8. Diagnostic Answers & Key Takeaways

1. **Does PPO show a learning trend?**
   - **Value learning is clear; policy improvement is not yet achieved.** Critic `explained_variance` climbed from `0.0092` to `0.5752`, showing the MLP accurately estimates expected future target coverage from remaining budget and corpus state. However, policy return stayed flat (`~18`), and PPO did not outperform Random in held-out evaluation (`17.6` vs `21.4`).
2. **Did PPO suffer from reward sparsity?**
   - **Yes.** `91.01%` (`13,979 / 15,360`) of training transitions and `23.9%` (`65 / 272`) of full training episodes received `0.0` reward. Discovering intermediate branch seeds (`PK`, `CM`, `VM` headers and intermediate checks) yields `reward = 0.0` until a deep `BBtarget` is reached, providing weak credit assignment for seed selection.
3. **Did PPO bias toward a single Energy level?**
   - **Yes — strongly toward `Energy = 512`.** In training, `Energy = 512` grew monotonically from `39.4%` (Q1) to `52.3%` (Q4), while deterministic evaluation selected `Energy = 512` on **`100%` (`300 / 300`)** of steps (`smoke-target` similarly collapsed `100%` onto `128`). Because $r_t = |C_{t+1}^T| - |C_t^T|$ gives `512` executions 16× more chances per RL step to hit a target without normalizing reward per execution or modeling exploration granularity, PPO learns to maximize single-step hit probability via max energy.
4. **Main limitations of the current Candidate Seed Strategy (`K = 4`)**:
   - **Single-Slot Starvation (`95.7%` Slot 0)**: Deterministic evaluation concentrated `287 / 300` steps on Slot 0. Once a seed achieves smaller `min_target_dist` in one subsystem (e.g., `PK`), it pins Slot 0 while new frontier seeds from other subsystems (`CM`, `VM`) wait in or drop below the top-4 queue, reducing discovered seeds (`28.5` vs `39.5` for Random) and hurting multi-target coverage.
   - **Static Greedy Sorting without Diversity**: Sorting purely by `(min_target_dist, not_favored, fuzzed_times, len, id)` packs Slots 0..3 with structurally similar children of the first subsystem reached rather than maintaining diverse frontier candidates across targets/clusters.
   - **Low-Corpus Slot Aliasing (`N < 4`)**: Modulo wraparound aliases slots `1..3` onto Seed 0 early in each episode, skewing initial action gradients across distinct discrete logits (`0..11`) that temporarily execute identical actions.

---

## 9. Readiness for Next Phase

Phase 1B delivers a working, reproducible PPO training and evaluation pipeline with full logging, checkpoints, and diagnostic evidence explaining why naive `Discrete(12)` + sparse bitmap-slot reward lags Uniform Random. Next phases (outside Phase 1B scope) can focus directly on fixing those diagnosed bottlenecks: diversity-aware candidate pool construction (e.g., target-cluster / frontier representation), execution-cost-aware reward or non-sparse distance/fringe progress shaping, and larger real-world benchmark evaluation.
