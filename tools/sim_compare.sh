#!/usr/bin/env bash
# Efecto de la GPU sobre la simulacion completa (PDF 2.2 "The simulation itself" y 2.6.5): corre la misma
# ciudad N veces por backend, ALTERNANDO --backend auto y --backend cpu (auto, cpu, auto, cpu, ...), con una
# duracion REAL fija y un timeout en cada corrida; valida cada log con tools/validate_log.py y agrega las
# metricas de tools/sim_metrics.py en un CSV (una fila por corrida).
#
# Uso: tools/sim_compare.sh [-b binario] [-c config.json] [-n corridas_por_backend] [-d segundos]
#                           [-m no-window|headless|window] [-o salida.csv]
#   -b  binario (por defecto build/delivery_sim)           -c  config (por defecto config/ciudad.json)
#   -n  corridas por backend (por defecto 3)               -d  segundos REALES por corrida (por defecto 30)
#   -m  no-window (por defecto: dibuja cada cuadro sin pantalla, mide FPS), headless (sin cuadros) o window
#       (con ventana: necesita pantalla/WSLg)
#   -o  CSV de salida (por defecto sim_compare.csv)
# La corrida termina con SIGINT a los -d segundos (el programa cierra limpio y escribe el resumen), asi que
# todas duran lo mismo en tiempo real. Sale con codigo != 0 si algun log no valida o alguna corrida falla.
BIN=build/delivery_sim
CONFIG=config/ciudad.json
RUNS=3
SECONDS_PER_RUN=30
MODE=no-window
OUT=sim_compare.csv
while getopts "b:c:n:d:m:o:" opt; do
    case $opt in
        b) BIN=$OPTARG ;; c) CONFIG=$OPTARG ;; n) RUNS=$OPTARG ;; d) SECONDS_PER_RUN=$OPTARG ;;
        m) MODE=$OPTARG ;; o) OUT=$OPTARG ;;
        *) sed -n '2,16p' "$0"; exit 2 ;;
    esac
done
[[ $RUNS =~ ^[1-9][0-9]*$ && $SECONDS_PER_RUN =~ ^[1-9][0-9]*$ ]] || { echo "-n y -d deben ser enteros >= 1"; exit 2; }
case $MODE in
    no-window) MODE_FLAG=--no-window ;;
    headless) MODE_FLAG=--headless ;;
    window) MODE_FLAG= ;;
    *) echo "-m: no-window, headless o window"; exit 2 ;;
esac
BIN=$(realpath "$BIN") || exit 1
CONFIG=$(realpath "$CONFIG") || exit 1
OUT=$(realpath -m "$OUT")
HERE=$(dirname "$(realpath "$0")")
cd "$HERE/.." || exit 1
[[ -x $BIN ]] || { echo "No existe o no es ejecutable: $BIN"; exit 1; }
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT

fails=0
header_written=0
for ((run = 1; run <= RUNS; ++run)); do
    for backend in auto cpu; do
        log=$tmp/$backend.$run.log; stdout=$tmp/$backend.$run.out; stderr=$tmp/$backend.$run.err
        echo "== corrida $run/$RUNS backend=$backend modo=$MODE (${SECONDS_PER_RUN}s)"
        # timeout con SIGINT: cierre limpio; 124 = llego al tiempo fijo (lo esperado). Margen de 20 s para cerrar.
        timeout -s INT -k 20 "$SECONDS_PER_RUN" "$BIN" "$CONFIG" $MODE_FLAG --backend "$backend" --log "$log" \
            >"$stdout" 2>"$stderr"
        rc=$?
        if [[ $rc -ne 0 && $rc -ne 124 ]]; then
            echo "FALLA: el programa salio con codigo $rc"; fails=$((fails + 1)); continue
        fi
        validator_ok=yes
        if ! python3 tools/validate_log.py "$log" "$stdout" "$CONFIG" "$stderr" >"$tmp/validator.txt" 2>&1; then
            validator_ok=no; fails=$((fails + 1)); echo "FALLA: validate_log.py"; tail -5 "$tmp/validator.txt" | cut -c1-200
        fi
        if [[ $header_written -eq 0 ]]; then
            echo "backend,run,mode,validator_ok,$(python3 tools/sim_metrics.py "$log" --format csv --header | head -1)" >"$OUT"
            header_written=1
        fi
        echo "$backend,$run,$MODE,$validator_ok,$(python3 tools/sim_metrics.py "$log" --stderr "$stderr" --format csv)" >>"$OUT"
    done
done
echo
[[ -f $OUT ]] && { echo "CSV: $OUT"; cat "$OUT"; }
if [[ $fails -eq 0 ]]; then echo "OK: todas las corridas validaron"; else echo "FALLAS: $fails"; fi
exit $((fails > 0))
