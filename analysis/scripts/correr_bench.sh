#!/bin/bash
# Corre el modo bench en primer plano y registra el estado P de la GPU cada 500 ms.
# Uso (WSL): bash analysis/scripts/correr_bench.sh <carpeta_salida> [argumentos extra de --bench, p. ej. --apps x]
# Deja en la carpeta: results.csv, results.csv.env.json, bench_stdout.txt, bench_stderr.txt, gpu_pstate.csv,
# perf_antes.txt, perf_despues.txt (nvidia-smi -q -d PERFORMANCE) y carga_antes.txt (uptime y procesos).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
OUT="$1"; shift
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$HOME/ec2inf/src/build"
timeout 20 nvidia-smi -q -d PERFORMANCE > "$OUT/perf_antes.txt" 2>&1
{ uptime; ps -eo pcpu,comm --sort=-pcpu | head -8; } > "$OUT/carga_antes.txt"
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -lms 500 > "$OUT/gpu_pstate.csv" 2>&1 &
MONITOR_PID=$!
start=$(date +%s)
timeout 3000 ./delivery_sim ../config/ciudad.json --headless --bench "$OUT/results.csv" "$@" \
    > "$OUT/bench_stdout.txt" 2> "$OUT/bench_stderr.txt"
rc=$?
end=$(date +%s)
kill "$MONITOR_PID" 2>/dev/null; wait "$MONITOR_PID" 2>/dev/null
timeout 20 nvidia-smi -q -d PERFORMANCE > "$OUT/perf_despues.txt" 2>&1
echo "rc=$rc segundos=$((end - start))"
tail -2 "$OUT/bench_stderr.txt"
exit $rc
