#!/usr/bin/env bash
# Compila y corre las pruebas de este proyecto.
#
#   bash tests/run.sh              -> corre todas
#   bash tests/run.sh city_graph   -> corre solo tests/test_city_graph.cpp
#
# Las pruebas quedan FUERA del CMakeLists a proposito: el brief exige que
# "cmake .. && make" produzca exactamente el target delivery_sim y sin warnings.
#
# Se enlazan todos los src/*.cpp menos main.cpp, asi ningun test necesita OpenCV.
# No se pasa -DNDEBUG: si se pasara, los assert desaparecerian y los tests no
# probarian nada.
set -u

cd "$(dirname "$0")/.." || exit 1          # siempre desde la raiz del proyecto,
                                           # para que "config/..." resuelva bien

FLAGS="-std=c++17 -Wall -Wextra -g -pthread -Isrc -Ithird_party"
SRC=$(ls src/*.cpp | grep -v '^src/main\.cpp$')
mkdir -p tests/bin

# Ningun archivo de src/ aparte de main.cpp puede definir main(): si lo hace, el build
# oficial (cmake usa file(GLOB src/*.cpp)) falla con "multiple definition of main".
# Los programas de prueba van en tests/, no en src/.
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
[ $fail -eq 0 ] && echo "TODAS LAS PRUEBAS PASARON" || echo "HUBO PRUEBAS FALLIDAS"
exit $fail
