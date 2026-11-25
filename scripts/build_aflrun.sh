#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM15="$ROOT_DIR/toolchain/llvm-15.0.6"
AFLRUN_DIR="$ROOT_DIR/backend/AFLRun"
LOG_FILE="$ROOT_DIR/results/aflrun-build-llvm15.log"

export PATH="$LLVM15/bin:$PATH"
export CC="$LLVM15/bin/clang"
export CXX="$LLVM15/bin/clang++"
export LLVM_CONFIG="$LLVM15/bin/llvm-config"

cd "$AFLRUN_DIR"
make clean all NO_PYTHON=1 AFL_CLANG_FLTO=-flto=full 2>&1 | tee "$LOG_FILE"
