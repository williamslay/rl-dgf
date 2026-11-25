#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM15="$ROOT_DIR/toolchain/llvm-15.0.6"
AFLRUN_DIR="$ROOT_DIR/backend/AFLRun"
BENCH_DIR="$ROOT_DIR/benchmarks/smoke-target"
OUT_DIR="$ROOT_DIR/results/smoke/out"
DURATION="${1:-30}"

export PATH="$LLVM15/bin:$PATH"
export AFL_PATH="$AFLRUN_DIR"
export AFLRUN_BB_TARGETS="$BENCH_DIR/BBtargets.txt"
export AFLRUN_TARGETS="target"
export AFLRUN_TMP="$BENCH_DIR/aflrun_tmp"
export AFL_NO_UI=1
export AFL_SKIP_CPUFREQ=1

rm -rf "$AFLRUN_TMP" "$OUT_DIR"
mkdir -p "$AFLRUN_TMP" "$OUT_DIR"

cd "$BENCH_DIR"
"$AFLRUN_DIR/afl-clang-lto" -g -O1 target.c -o target 2>&1 | tee "$ROOT_DIR/results/smoke/target-compile.log"

"$AFLRUN_DIR/afl-fuzz" \
  -d -V "$DURATION" \
  -i "$BENCH_DIR/seeds" \
  -o "$OUT_DIR" \
  --config "init_cov_quant=30:log_at_begin=1:log_check_interval=2000" \
  -- "$BENCH_DIR/target" @@ 2>&1 | tee "$ROOT_DIR/results/smoke/aflrun-smoke.log"
