#!/bin/bash
# Clona la rama indicada (por defecto ec2) en ~/ec2inf/src y compila delivery_sim en Release con CUDA.
# Uso (WSL): bash analysis/scripts/compilar_wsl.sh [rama]
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
BRANCH="${1:-ec2}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # raiz del repositorio (este script vive en analysis/scripts)
BASE="$HOME/ec2inf"
rm -rf "$BASE/src"
mkdir -p "$BASE"
timeout 300 git clone --quiet --branch "$BRANCH" --single-branch "$REPO" "$BASE/src"
cd "$BASE/src"
echo "commit: $(git rev-parse --short HEAD)"
mkdir -p build && cd build
timeout 900 cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release > cmake.out 2>&1 || { tail -30 cmake.out; exit 1; }
timeout 1800 make -j6 delivery_sim > make.out 2>&1 || { tail -40 make.out; exit 1; }
echo "warnings: $(grep -ci 'warning' make.out || true)"
