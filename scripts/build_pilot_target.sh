#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM15="$ROOT_DIR/toolchain/llvm-15.0.6"
AFLRUN_DIR="$ROOT_DIR/backend/AFLRun"
BENCH_DIR="$ROOT_DIR/benchmarks/pilot-target"

export PATH="$LLVM15/bin:$PATH"
export AFL_PATH="$AFLRUN_DIR"
export AFLRUN_BB_TARGETS="$BENCH_DIR/BBtargets.txt"
export AFLRUN_TARGETS="target"
export AFLRUN_TMP="$BENCH_DIR/aflrun_tmp"

rm -rf "$AFLRUN_TMP"
mkdir -p "$AFLRUN_TMP" "$ROOT_DIR/results/phase1b"

cd "$BENCH_DIR"
"$AFLRUN_DIR/afl-clang-lto" -g -O1 target.c -o target 2>&1 | tee "$ROOT_DIR/results/phase1b/pilot-target-compile.log"
ls -la "$AFLRUN_TMP"
