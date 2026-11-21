#!/usr/bin/env bash
# Prueba de equivalencia GPU = CPU: para cada aplicacion registrada en el bench corre
# --bench-once en 3 tamanos (minimo, uno que NO es multiplo del bloque y uno mediano) y exige
# exit 0, cabecera correcta, filas GPU (salvo las apps solo CPU) y TODAS las filas con correct=yes.
# La columna correct la calcula el propio bench contra cpu1: exacta para enteros y con la
# tolerancia documentada de cada app para float.
# Se niega a correr (exit 2) si el binario no tiene CUDA o si no hay una GPU usable.
# Uso: tools/check_gpu_equivalence.sh [binario]      (por defecto build/delivery_sim)
BIN=${1:-build/delivery_sim}
BIN=$(realpath "$BIN") || exit 1          # antes del cd: la ruta es relativa al cwd del usuario
cd "$(dirname "$0")/.." || exit 1
[[ -x $BIN ]] || { echo "No existe o no es ejecutable: $BIN"; exit 1; }

# Tipo de build: lo dice el CMakeCache de la carpeta del binario.
cache="$(dirname "$BIN")/CMakeCache.txt"
useCuda=$(sed -n 's/^USE_CUDA:BOOL=//p' "$cache" 2>/dev/null)
if [[ $useCuda != ON ]]; then
    echo "check_gpu_equivalence.sh se niega a correr: el build tiene USE_CUDA='${useCuda:-desconocido}', no ON."
    echo "Compila con -DUSE_CUDA=ON: sin CUDA no hay version GPU que comparar con la CPU."
    exit 2
fi

fails=0
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
ok()   { echo "OK    $1"; }
fail() { echo "FALLA $1"; fails=$((fails + 1)); }

CFG=config/ciudad.json
HEADER="application,version,size,rep,ms,correct,threads,seed,h2d_ms,kernel_ms,d2h_ms"
TIMEOUT=120

# Aplicaciones solo CPU: no tienen version GPU, se aceptan sin filas gpu_kernel/gpu_e2e.
CPU_ONLY=" apsp_dijkstra "

# app:tamanos. Tamanos: minimo (1), no multiplo del bloque (tiles de 16/32, bloques de 256) y mediano.
APPS=(
    "apsp_floyd:1,100,1000"
    "apsp_dijkstra:1,100,1000"
    "eta_batch_soa:1,777,2000"
    "eta_batch_aos:1,777,2000"
    "demand_heatmap:1,100,1024"
    "demand_heatmap_tiles:1,100,1024"
    "coverage_field:1,100,1024"
    "delivery_histogram:1,100001,1000000"
    "histogram_naive:1,100001,1000000"
    "best_courier:1,777,2000"
    "best_courier_divergent:1,777,2000"
    "orders_in_progress:1,100001,1000000"
    "candidate_filter:1,10001,50000"
    "candidate_filter_naive:1,10001,50000"
    "delivery_timeline:1,100001,1000000"
    "delivery_percentiles:1,100001,1000000"
)

# Control: la lista cubre todas las apps registradas en src/bench (menos las de calibracion).
registered=$(grep -rhoE 'BENCH_REGISTER_APP\("[a-z_0-9]+"' src/bench | sed 's/.*("//; s/"//' | grep -v '^calib_' | sort)
listed=$(printf '%s\n' "${APPS[@]%%:*}" | sort)
if [[ $registered == "$listed" ]]; then
    ok "la lista cubre las ${#APPS[@]} apps registradas"
else
    fail "la lista no coincide con BENCH_REGISTER_APP (sobran/faltan: $(diff <(echo "$listed") <(echo "$registered") | grep '^[<>]' | tr '\n' ' '))"
fi

# GPU usable: una corrida de una app con GPU debe producir filas gpu_kernel/gpu_e2e.
timeout -s KILL $TIMEOUT "$BIN" $CFG --headless --bench-once --app delivery_histogram --size 1000 >"$tmp/gpu.out" 2>"$tmp/gpu.err"
code=$?
if [[ $code -ne 0 ]] || ! awk -F, '$2 ~ /^gpu_/ { found = 1 } END { exit !found }' "$tmp/gpu.out"; then
    echo "check_gpu_equivalence.sh se niega a correr: no hay una GPU usable (exit $code)."
    echo "Salida: $(tail -n 3 "$tmp/gpu.err" | tr '\n' ' ')"
    exit 2
fi

start=$SECONDS
for entry in "${APPS[@]}"; do
    app=${entry%%:*}
    IFS=, read -ra sizes <<<"${entry#*:}"
    for size in "${sizes[@]}"; do
        label="$app size=$size"
        timeout -s KILL $TIMEOUT "$BIN" $CFG --headless --bench-once --app "$app" --size "$size" >"$tmp/o.csv" 2>"$tmp/o.err"
        code=$?
        if [[ $code -ne 0 ]]; then fail "$label: exit $code ($(tail -n 2 "$tmp/o.err" | tr '\n' ' '))"; continue; fi
        if [[ $(head -n 1 "$tmp/o.csv") != "$HEADER" ]]; then fail "$label: cabecera distinta"; continue; fi
        rows=$(( $(wc -l <"$tmp/o.csv") - 1 ))
        gpuRows=$(awk -F, 'NR > 1 && $2 ~ /^gpu_/ { n++ } END { print n + 0 }' "$tmp/o.csv")
        kernelRows=$(awk -F, 'NR > 1 && $2 == "gpu_kernel" { n++ } END { print n + 0 }' "$tmp/o.csv")
        endToEndRows=$(awk -F, 'NR > 1 && $2 == "gpu_e2e" { n++ } END { print n + 0 }' "$tmp/o.csv")
        notYes=$(awk -F, 'NR > 1 && $6 != "yes" { n++ } END { print n + 0 }' "$tmp/o.csv")
        if [[ $rows -lt 1 ]]; then fail "$label: sin filas"; continue; fi
        if [[ $CPU_ONLY != *" $app "* && ( $kernelRows -eq 0 || $endToEndRows -eq 0 ) ]]; then
            fail "$label: faltan filas GPU (gpu_kernel=$kernelRows, gpu_e2e=$endToEndRows)"; continue
        fi
        if [[ $notYes -ne 0 ]]; then fail "$label: $notYes de $rows filas con correct distinto de yes"; continue; fi
        ok "$label: $rows filas ($gpuRows GPU), todas correct=yes"
    done
done

echo "----"
echo "tiempo total: $((SECONDS - start)) s"
if [[ $fails -eq 0 ]]; then echo "Equivalencia GPU = CPU OK."; else echo "$fails casos fallaron"; fi
# 0 = todo correcto, 1 = algun caso fallo (2 queda reservado para "se niega a correr").
exit $(( fails > 0 ? 1 : 0 ))
