#!/usr/bin/env bash
# Verifica que cada configuración inválida termina con código != 0, con un mensaje
# en stderr y SIN imprimir la línea de inicio (es decir, sin simular).
# Uso (desde la raíz del proyecto): tools/check_invalid_configs.sh build/delivery_sim
BIN=${1:-build/delivery_sim}
fails=0

check() {   # check <descripción> <argumentos...>
    local desc=$1; shift
    local out err code
    out=$("$BIN" "$@" 2>/tmp/check_err.txt); code=$?
    err=$(head -c 160 /tmp/check_err.txt | tr '\n' ' ')
    if [[ $code -ne 0 && -n $err && $out != *"Simulation has started"* ]]; then
        echo "OK    ($code) $desc :: $err"
    else
        echo "FALLA ($code) $desc :: stdout='$out' stderr='$err'"
        fails=$((fails + 1))
    fi
}

check "sin argumentos"
check "archivo inexistente" config/tests/no_existe.json
check "flag desconocido" config/tests/small.json --verbose
check "--log sin ruta" config/tests/small.json --log
for f in config/tests/invalid_*.json; do
    check "$(basename "$f")" "$f"
done

echo "----"
if [[ $fails -eq 0 ]]; then echo "Todas las configuraciones inválidas se rechazan."; else echo "$fails casos fallaron"; fi
exit $fails
