# Equipetrol Delivery - simulación concurrente (C++17)
# Build exigido por el brief:  mkdir build && cd build && cmake .. && make

# --- 1. Versión mínima y proyecto ---------------------------------------------
# 3.11 es el mínimo del brief: no usar comandos posteriores (p. ej. target_link_options, CONFIGURE_DEPENDS)
cmake_minimum_required(VERSION 3.11)
project(delivery_sim LANGUAGES CXX)

# --- 2. C++17 estricto ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # -std=c++17, no -std=gnu++17

# --- 3. Tipo de build por defecto -------------------------------------------------
# Sin esto, "cmake .." compila sin optimizar y sin símbolos de depuración
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Tipo de build" FORCE)
endif()

# --- 4. Opciones de sanitizers --------------------------------------------------
option(ENABLE_TSAN "Compilar con ThreadSanitizer (detecta data races)" OFF)
option(ENABLE_ASAN "Compilar con AddressSanitizer + UBSan (errores de memoria)" OFF)

if(ENABLE_TSAN AND ENABLE_ASAN)
  message(FATAL_ERROR "ENABLE_TSAN y ENABLE_ASAN no se pueden activar a la vez")
endif()

# --- 5. Dependencias ------------------------------------------------------------
# OpenCV solo para la ventana: matrices, dibujo, carga del PNG y highgui
find_package(OpenCV 4.1 REQUIRED COMPONENTS core imgproc imgcodecs highgui)
# Librería de hilos del sistema que necesita std::thread (-pthread)
find_package(Threads REQUIRED)

# --- 6. Ejecutable --------------------------------------------------------------
# Al agregar un .cpp nuevo en src/ hay que volver a correr "cmake .."
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(delivery_sim ${SOURCES})

# --- 7. Headers -------------------------------------------------------------------
# SYSTEM: los warnings de json.hpp y OpenCV no cuentan como nuestros
target_include_directories(delivery_sim SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
  ${OpenCV_INCLUDE_DIRS}
)

# --- 8. Warnings ------------------------------------------------------------------
# Sin -Werror a propósito: un warning nuevo en el compilador del revisor no debe romper el build
target_compile_options(delivery_sim PRIVATE -Wall -Wextra)

# --- 9. Enlace --------------------------------------------------------------------
target_link_libraries(delivery_sim PRIVATE ${OpenCV_LIBS} Threads::Threads)

# --- 10. Sanitizers ---------------------------------------------------------------
# La bandera va al compilar (instrumenta el código) y al enlazar (agrega la librería del sanitizer)
if(ENABLE_TSAN)
  target_compile_options(delivery_sim PRIVATE -fsanitize=thread -g -O1 -fno-omit-frame-pointer)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=thread)
endif()

if(ENABLE_ASAN)
  target_compile_options(delivery_sim PRIVATE -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=address,undefined)
endif()
