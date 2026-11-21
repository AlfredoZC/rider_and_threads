#!/usr/bin/env bash
# Comprueba el modo bench en un build Release: corre --bench con --reps 2 y pocas aplicaciones y
# tamanos, y valida el CSV (columnas, filas esperadas, orden rotado, sin correct=no) y el .env.json.
# Se niega a correr si el build no es Release (el binario tambien se niega a medir).
# Uso: tools/check_bench.sh [binario]      (por defecto build/delivery_sim)
BIN=${1:-build/delivery_sim}
BIN=$(realpath "$BIN") || exit 1          # antes del cd: la ruta es relativa al cwd del usuario
cd "$(dirname "$0")/.." || exit 1
[[ -x $BIN ]] || { echo "No existe o no es ejecutable: $BIN"; exit 1; }

# Tipo de build: lo dice el CMakeCache de la carpeta del binario.
cache="$(dirname "$BIN")/CMakeCache.txt"
btype=$(sed -n 's/^CMAKE_BUILD_TYPE:[A-Z]*=//p' "$cache" 2>/dev/null)
if [[ $btype != Release ]]; then
    echo "check_bench.sh se niega a correr: el build es '${btype:-desconocido}', no Release."
    echo "Never time a Debug or sanitizer build: compila con -DCMAKE_BUILD_TYPE=Release."
    exit 2
fi

fails=0
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
ok()   { echo "OK    $1"; }
fail() { echo "FALLA $1"; fails=$((fails + 1)); }

CFG=config/tests/headless_short.json
HEADER="application,version,size,rep,ms,correct,threads,seed,h2d_ms,kernel_ms,d2h_ms"

# 1) Corrida completa pequena.
timeout -s KILL 300 "$BIN" $CFG --headless --bench "$tmp/b.csv" --apps calib_copy,calib_launch \
    --sizes 100,4096 --reps 2 --threads 1,2 --seed 7 >"$tmp/b.out" 2>"$tmp/b.err"
code=$?
if [[ $code -eq 0 ]]; then ok "--bench: exit 0"; else fail "--bench: exit $code ($(tail -n 3 "$tmp/b.err" | tr '\n' ' '))"; fi

[[ -f $tmp/b.csv ]] || { fail "no se creo el CSV"; echo "$fails casos fallaron"; exit 1; }

# 2) Encabezado y numero de columnas.
if [[ $(head -n 1 "$tmp/b.csv") == "$HEADER" ]]; then ok "encabezado exacto"; else fail "encabezado distinto: $(head -n 1 "$tmp/b.csv")"; fi
bad=$(awk -F, 'NR > 1 && NF != 11 { n++ } END { print n + 0 }' "$tmp/b.csv")
if [[ $bad -eq 0 ]]; then ok "todas las filas tienen 11 columnas"; else fail "$bad filas sin 11 columnas"; fi

# 3) Nada incorrecto, y solo yes/no.
no=$(awk -F, 'NR > 1 && $6 == "no" { n++ } END { print n + 0 }' "$tmp/b.csv")
other=$(awk -F, 'NR > 1 && $6 != "yes" && $6 != "no" { n++ } END { print n + 0 }' "$tmp/b.csv")
if [[ $no -eq 0 && $other -eq 0 ]]; then ok "ninguna fila con correct=no"; else fail "correct=no en $no filas ($other valores raros)"; fi

# 4) Filas esperadas: 2 apps x 2 tamanos x 2 reps x versiones (cpu1 + 2 cpuN [+ 2 GPU si hay GPU]).
gpu=0; grep -q '"available": true' "$tmp/b.csv.env.json" 2>/dev/null && gpu=1
variants=$((3 + 2 * gpu))
expected=$((2 * 2 * 2 * variants))
rows=$(( $(wc -l <"$tmp/b.csv") - 1 ))
if [[ $rows -eq $expected ]]; then ok "$rows filas (esperadas $expected, GPU=$gpu)"; else fail "$rows filas, se esperaban $expected (GPU=$gpu)"; fi

# 5) Misma semilla en todas las filas; ms numerico y positivo.
seeds=$(awk -F, 'NR > 1 && $8 != 7 { n++ } END { print n + 0 }' "$tmp/b.csv")
if [[ $seeds -eq 0 ]]; then ok "semilla fija en todas las filas"; else fail "$seeds filas con otra semilla"; fi
badms=$(awk -F, 'NR > 1 && ($5 !~ /^[0-9]+[.][0-9]+$/ || $5 + 0 < 0) { n++ } END { print n + 0 }' "$tmp/b.csv")
if [[ $badms -eq 0 ]]; then ok "ms numerico en todas las filas"; else fail "$badms filas con ms invalido"; fi

# 6) Orden rotado: la primera version de la repeticion 0 y la de la 1 (misma app y tamano) difieren.
first0=$(awk -F, '$1 == "calib_launch" && $3 == 4096 && $4 == 0 { print $2 "/" $7; exit }' "$tmp/b.csv")
first1=$(awk -F, '$1 == "calib_launch" && $3 == 4096 && $4 == 1 { print $2 "/" $7; exit }' "$tmp/b.csv")
if [[ -n $first0 && -n $first1 && $first0 != "$first1" ]]; then ok "orden rotado ($first0 en la rep 0, $first1 en la rep 1)"; else fail "el orden no rota ($first0 / $first1)"; fi

# 7) .env.json valido con las claves de hardware y build.
if python3 -I - "$tmp/b.csv.env.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
assert j["cpu"]["model"] and j["cpu"]["hardware_concurrency"] >= 1
assert j["build"]["type"] == "Release" and j["build"]["sanitizer"] == "none"
assert j["build"]["compiler"] and j["build"]["cxx_flags"]
assert j["run"]["seed"] == 7 and j["run"]["reps"] == 2
g = j["gpu"]
if g["available"]:
    assert g["name"] and g["sm_count"] > 0 and g["peak_bandwidth_gbs"] > 0
    assert j["cuda"]["runtime"] and j["cuda"]["driver"]
PY
then ok ".env.json valido"; else fail ".env.json invalido o incompleto"; fi

# 8) No sobrescribe sin --force; con --force si; flags desconocidos rechazados.
cp "$tmp/b.csv" "$tmp/keep.csv"
timeout -s KILL 60 "$BIN" $CFG --headless --bench "$tmp/b.csv" --apps calib_launch --sizes 10 --reps 1 >/dev/null 2>"$tmp/o.err"
code=$?
if [[ $code -ne 0 ]] && cmp -s "$tmp/b.csv" "$tmp/keep.csv"; then ok "no sobrescribe un CSV existente (exit $code)"; else fail "sobrescribio o no fallo (exit $code)"; fi
timeout -s KILL 60 "$BIN" $CFG --headless --bench "$tmp/b.csv" --apps calib_launch --sizes 10 --reps 1 --force >/dev/null 2>&1
code=$?
if [[ $code -eq 0 ]] && ! cmp -s "$tmp/b.csv" "$tmp/keep.csv"; then ok "--force sobrescribe"; else fail "--force no funciono (exit $code)"; fi
timeout -s KILL 30 "$BIN" $CFG --headless --bench "$tmp/x.csv" --frobnicate >/dev/null 2>&1
code=$?
if [[ $code -ne 0 && ! -e $tmp/x.csv ]]; then ok "flag desconocido rechazado (exit $code)"; else fail "flag desconocido aceptado (exit $code)"; fi
timeout -s KILL 30 "$BIN" $CFG --headless --bench "$tmp/y.csv" --reps 0 >/dev/null 2>&1
code=$?
if [[ $code -ne 0 && ! -e $tmp/y.csv ]]; then ok "--reps 0 rechazado (exit $code)"; else fail "--reps 0 aceptado (exit $code)"; fi

# 9) --bench-once: una corrida de cada version, todas correctas.
timeout -s KILL 60 "$BIN" $CFG --headless --bench-once --app calib_copy --size 4096 >"$tmp/once.out" 2>"$tmp/once.err"
code=$?
n=$(awk -F, 'NR > 1 && $6 == "yes" { n++ } END { print n + 0 }' "$tmp/once.out")
if [[ $code -eq 0 && $n -ge 2 && $(head -n 1 "$tmp/once.out") == "$HEADER" ]]; then ok "--bench-once: $n corridas correctas"; else fail "--bench-once: exit $code, $n filas yes"; fi

echo "----"
if [[ $fails -eq 0 ]]; then echo "--bench OK."; else echo "$fails casos fallaron"; fi
exit $fails
