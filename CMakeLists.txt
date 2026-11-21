cmake_minimum_required(VERSION 3.11...3.18)
project(delivery_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Tipo de build" FORCE)
endif()

option(USE_CUDA "Compilar el backend GPU (CUDA). Sin esto se compila la version solo CPU" OFF)
option(ENABLE_TSAN "Compilar con ThreadSanitizer (detecta data races)" OFF)
option(ENABLE_ASAN "Compilar con AddressSanitizer + UBSan (errores de memoria)" OFF)

if(ENABLE_TSAN AND ENABLE_ASAN)
  message(FATAL_ERROR "ENABLE_TSAN y ENABLE_ASAN no se pueden activar a la vez")
endif()

if(USE_CUDA)
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "USE_CUDA requiere CMake >= 3.18 (version actual: ${CMAKE_VERSION})")
  endif()
  if(ENABLE_TSAN OR ENABLE_ASAN)
    message(FATAL_ERROR "USE_CUDA no se combina con ENABLE_TSAN/ENABLE_ASAN: los sanitizers se corren sobre el build CPU (el codigo host de los .cu no queda instrumentado y ASan rompe el runtime de CUDA sin protect_shadow_gap=0)")
  endif()
  # T4 de Colab = sm_75, RTX 4050 = sm_89. El usuario puede pisarlo con -DCMAKE_CUDA_ARCHITECTURES.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;89")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenCV 4.1 REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Threads REQUIRED)

file(GLOB SOURCES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/accel/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/bench/*.cpp")
if(USE_CUDA)
  file(GLOB ACCEL_CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/accel/cuda/*.cu")
  list(APPEND SOURCES ${ACCEL_CUDA_SOURCES})
else()
  # Versiones de respaldo (sin CUDA) de cada modulo de src/accel/.
  file(GLOB ACCEL_CPU_SOURCES "${PROJECT_SOURCE_DIR}/src/accel/cpu/*.cpp")
  list(APPEND SOURCES ${ACCEL_CPU_SOURCES})
endif()
add_executable(delivery_sim ${SOURCES})

# Datos del build para el modo bench (src/bench/BuildInfo.cpp): tipo, sanitizers, flags y commit.
# El bench se niega a medir si el tipo no es Release o si hay sanitizers.
set(BENCH_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
set(BENCH_SANITIZER "none")
if(ENABLE_TSAN)
  set(BENCH_SANITIZER "thread")
elseif(ENABLE_ASAN)
  set(BENCH_SANITIZER "address,undefined")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" BENCH_BUILD_TYPE_UPPER)
set(BENCH_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BENCH_BUILD_TYPE_UPPER}} -Wall -Wextra")
set(BENCH_CUDA_FLAGS "")
if(USE_CUDA)
  set(BENCH_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BENCH_BUILD_TYPE_UPPER}} -lineinfo archs=${CMAKE_CUDA_ARCHITECTURES}")
endif()
foreach(var BENCH_BUILD_TYPE BENCH_CXX_FLAGS BENCH_CUDA_FLAGS)
  string(REPLACE "\\" "\\\\" ${var} "${${var}}")
  string(REPLACE "\"" "\\\"" ${var} "${${var}}")
  string(STRIP "${${var}}" ${var})
endforeach()
configure_file("${PROJECT_SOURCE_DIR}/src/bench/BuildInfoGenerated.h.in"
               "${PROJECT_BINARY_DIR}/generated/BuildInfoGenerated.h" @ONLY)

# El commit (git describe --always --dirty) se regenera en CADA build, no solo al configurar.
set(BENCH_GIT_HEADER "${PROJECT_BINARY_DIR}/generated/BuildGit.h")
set(BENCH_GIT_COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${PROJECT_SOURCE_DIR}" "-DOUT=${BENCH_GIT_HEADER}"
    -P "${PROJECT_SOURCE_DIR}/cmake/GitCommit.cmake")
execute_process(COMMAND ${BENCH_GIT_COMMAND})
add_custom_target(bench_git_commit ALL COMMAND ${BENCH_GIT_COMMAND}
  BYPRODUCTS "${BENCH_GIT_HEADER}" COMMENT "Actualizando el commit del build")
add_dependencies(delivery_sim bench_git_commit)

target_include_directories(delivery_sim PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/generated)
target_include_directories(delivery_sim SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
  ${OpenCV_INCLUDE_DIRS}
)

# Las opciones de gcc solo van a C++: nvcc no las entiende.
target_compile_options(delivery_sim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)

target_link_libraries(delivery_sim PRIVATE ${OpenCV_LIBS} Threads::Threads)

if(USE_CUDA)
  target_compile_definitions(delivery_sim PRIVATE USE_CUDA=1)
  target_compile_options(delivery_sim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra>)
  target_link_libraries(delivery_sim PRIVATE CUDA::cudart)
endif()

if(ENABLE_TSAN)
  target_compile_options(delivery_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread;-g;-O1;-fno-omit-frame-pointer>)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=thread)
endif()

if(ENABLE_ASAN)
  target_compile_options(delivery_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-g;-O1;-fno-omit-frame-pointer>)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=address,undefined)
endif()
