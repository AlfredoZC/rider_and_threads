#!/usr/bin/env bash
# Comprueba --headless: sin imagen, termina por duracion (exit 0, resumen JSON valido,
# log valido) y por SIGINT/SIGTERM en < 5 s, sin reportes de sanitizers.
# Uso: tools/check_headless.sh [binario]
BIN=${1:-build/delivery_sim}
BIN=$(realpath "$BIN") || exit 1          # antes del cd: la ruta es relativa al cwd del usuario
cd "$(dirname "$0")/.." || exit 1
[[ -x $BIN ]] || { echo "No existe o no es ejecutable: $BIN"; exit 1; }
fails=0
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT

ok()   { echo "OK    $1"; }
fail() { echo "FALLA $1"; fails=$((fails + 1)); }
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }

# Falla si el .err trae un reporte de TSan/UBSan/ASan.
no_sanitizer() { # archivo.err descripcion
    if grep -Eq 'WARNING: ThreadSanitizer|runtime error|ERROR: AddressSanitizer' "$1"; then
        fail "$2: reporte de sanitizer en stderr: $(grep -Em1 'WARNING: ThreadSanitizer|runtime error|ERROR: AddressSanitizer' "$1")"
    else
        ok "$2: sin reportes de sanitizers"
    fi
}

# Espera hasta 10 s a que el proceso termine (un zombi cuenta como terminado). 0 = termino.
wait_exit() { # pid
    local i state
    for ((i = 0; i < 100; i++)); do
        state=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)
        [[ -z $state || $state == Z ]] && return 0
        sleep 0.1
    done
    return 1
}

# 1) Imagen inexistente: falla sin --headless y con --no-window, funciona con --headless.
for flag in "" "--no-window"; do
    timeout -s KILL 30 "$BIN" config/tests/headless_short.json --log "$tmp/n.log" $flag >/dev/null 2>"$tmp/n.err"
    code=$?
    if [[ $code -eq 124 || $code -eq 137 ]]; then
        fail "imagen inexistente (${flag:-ventana}): se colgo (timeout 30 s, exit $code)"
    elif [[ $code -ne 0 ]] && grep -q "no se pudo leer la imagen" "$tmp/n.err"; then
        ok "imagen inexistente se rechaza sin --headless (${flag:-ventana}) (exit $code)"
    else
        fail "imagen inexistente deberia rechazarse sin --headless (${flag:-ventana}) (exit $code)"
    fi
done

# 2) Termina por duracion simulada, sin senal.
start=$(now_ms)
timeout -s KILL 30 "$BIN" config/tests/headless_short.json --log "$tmp/d.log" --headless >"$tmp/d.out" 2>"$tmp/d.err"
code=$?
el=$(( $(now_ms) - start ))
if [[ $code -eq 0 ]]; then ok "duracion: exit 0 (${el} ms)"
elif [[ $code -eq 124 || $code -eq 137 ]]; then fail "duracion: se colgo (timeout 30 s, exit $code)"
else fail "duracion: exit $code"; fi
if python3 -I tools/validate_log.py "$tmp/d.log" "$tmp/d.out" config/tests/headless_short.json "$tmp/d.err" >"$tmp/v.txt" 2>&1; then
    ok "duracion: log y resumen validos"
else
    fail "duracion: validate_log: $(tail -n 3 "$tmp/v.txt" | tr '\n' ' ')"
fi
no_sanitizer "$tmp/d.err" "duracion"

# 3) Cierre limpio por senal en < 5 s (si no responde en 10 s, se mata y falla).
for combo in "INT headless_long" "TERM headless_long" "INT headless_huge"; do
    sig=${combo% *}; cfgname=${combo#* }; tag="SIG$sig $cfgname"
    "$BIN" config/tests/$cfgname.json --log "$tmp/s.log" --headless >"$tmp/s.out" 2>"$tmp/s.err" &
    pid=$!
    sleep 1.5
    if ! kill -0 $pid 2>/dev/null; then
        fail "$tag: el proceso termino antes de la senal (duracion mal calculada?)"
    fi
    t0=$(now_ms)
    kill -$sig $pid 2>/dev/null
    if wait_exit $pid; then
        wait $pid; code=$?
    else
        kill -KILL $pid 2>/dev/null
        wait $pid 2>/dev/null
        fail "$tag: no respondio a la senal en 10 s (se mato con SIGKILL)"
        code=137
    fi
    el=$(( $(now_ms) - t0 ))
    if [[ $code -eq 0 && $el -lt 5000 ]]; then
        ok "$tag: exit 0 en ${el} ms"
    else
        fail "$tag: exit $code en ${el} ms"
    fi
    if python3 -I tools/validate_log.py "$tmp/s.log" "$tmp/s.out" config/tests/$cfgname.json "$tmp/s.err" >"$tmp/v.txt" 2>&1; then
        ok "$tag: log y resumen validos"
    else
        fail "$tag: validate_log: $(tail -n 3 "$tmp/v.txt" | tr '\n' ' ')"
    fi
    no_sanitizer "$tmp/s.err" "$tag"
done

echo "----"
if [[ $fails -eq 0 ]]; then echo "--headless OK."; else echo "$fails casos fallaron"; fi
exit $fails
