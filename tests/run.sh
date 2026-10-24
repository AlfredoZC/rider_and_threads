#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.." || exit 1

FLAGS="-std=c++17 -Wall -Wextra -g -pthread -Isrc -Ithird_party"
SRC=$(ls src/*.cpp | grep -v '^src/main\.cpp$')
mkdir -p tests/bin

STRAY=$(grep -l '^[[:space:]]*int[[:space:]]\+main[[:space:]]*(' $SRC 2>/dev/null)
if [ -n "$STRAY" ]; then
    echo "ERROR: estos archivos de src/ definen main() y rompen el build oficial:"
    echo "$STRAY" | sed 's/^/  /'
    echo "Moveles el contenido a tests/ y borralos de src/."
    exit 2
fi

if [ $# -gt 0 ]; then
    FILES="tests/test_$1.cpp"
    if [ ! -f "$FILES" ]; then
        echo "No existe $FILES. Disponibles:"
        ls tests/test_*.cpp 2>/dev/null | sed 's|tests/test_|  |; s|\.cpp$||'
        exit 1
    fi
else
    FILES=$(ls tests/test_*.cpp)
fi

fail=0
for f in $FILES; do
    name=$(basename "$f" .cpp)
    printf '=== %s\n' "$name"
    if ! g++ $FLAGS -o "tests/bin/$name" "$f" $SRC; then
        echo "    COMPILACION FALLIDA"
        fail=1
        continue
    fi
    if ! "./tests/bin/$name"; then
        echo "    FALLO EN EJECUCION (exit $?)"
        fail=1
    fi
done

echo
if [ $fail -eq 0 ]; then
    echo "TODAS LAS PRUEBAS PASARON"
else
    echo "HUBO PRUEBAS FALLIDAS" >&2
    echo "HUBO PRUEBAS FALLIDAS"
fi
exit $fail
