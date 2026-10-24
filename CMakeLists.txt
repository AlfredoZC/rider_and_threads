cmake_minimum_required(VERSION 3.11)
project(delivery_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Tipo de build" FORCE)
endif()

option(ENABLE_TSAN "Compilar con ThreadSanitizer (detecta data races)" OFF)
option(ENABLE_ASAN "Compilar con AddressSanitizer + UBSan (errores de memoria)" OFF)

if(ENABLE_TSAN AND ENABLE_ASAN)
  message(FATAL_ERROR "ENABLE_TSAN y ENABLE_ASAN no se pueden activar a la vez")
endif()

find_package(OpenCV 4.1 REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Threads REQUIRED)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(delivery_sim ${SOURCES})

target_include_directories(delivery_sim SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(delivery_sim PRIVATE -Wall -Wextra)

target_link_libraries(delivery_sim PRIVATE ${OpenCV_LIBS} Threads::Threads)

if(ENABLE_TSAN)
  target_compile_options(delivery_sim PRIVATE -fsanitize=thread -g -O1 -fno-omit-frame-pointer)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=thread)
endif()

if(ENABLE_ASAN)
  target_compile_options(delivery_sim PRIVATE -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=address,undefined)
endif()
