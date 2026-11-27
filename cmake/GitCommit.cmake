# Escribe BuildGit.h con `git describe --always --dirty` (o "unknown" sin git ni repositorio).
# Se ejecuta en cada build (no solo al configurar) y solo toca el archivo si cambia el contenido,
# para no recompilar sin necesidad.
# Uso: cmake -DSRC_DIR=<repo> -DOUT=<BuildGit.h> -P cmake/GitCommit.cmake
set(commit "unknown")
find_program(GIT_EXECUTABLE git)
if(GIT_EXECUTABLE)
  execute_process(COMMAND "${GIT_EXECUTABLE}" describe --always --dirty
    WORKING_DIRECTORY "${SRC_DIR}"
    OUTPUT_VARIABLE described OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE rc ERROR_QUIET)
  if(rc EQUAL 0 AND NOT "${described}" STREQUAL "")
    string(REGEX REPLACE "[^A-Za-z0-9._-]" "" commit "${described}")
  endif()
endif()
file(WRITE "${OUT}.tmp" "// Generado en cada build por cmake/GitCommit.cmake. No se edita ni se versiona.\n#pragma once\n#define BENCH_GIT_COMMIT \"${commit}\"\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${OUT}.tmp" "${OUT}")
file(REMOVE "${OUT}.tmp")
