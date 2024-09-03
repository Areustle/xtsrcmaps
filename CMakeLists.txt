# Fermi xtsrcmaps Author: Alexander Reustle (2022)
cmake_minimum_required(VERSION 3.21)

project(
  xtsrcmaps
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Fermi LAT experimental source map generator.")

cmake_policy(SET CMP0072 NEW)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake tool support packages. These provide additional funcitons to our Cmake
# files.
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(FetchContent)

# Add the 'cmake' directory to the CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# use ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache in ${CCACHE_PROGRAM}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

add_subdirectory(external)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Detect the architecture and add appropriate SIMD flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-mavx2)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(-xAVX2)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/arch:AVX2)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_NAME STREQUAL
                                                 "Darwin")
  # Apple Silicon (M1/M2) uses NEON SIMD
  add_compile_options(-march=armv8-a+simd)
endif()

# ##############################################################################

add_compile_options(
  # -Wall #
  # -Wextra #
  # -Werror # -mbmi2 #
  $<$<CONFIG:RELEASE>:-march=native> #
  -fexperimental-library #
  $<$<CONFIG:RELEASE>:-O3> #
  -Wno-unused-parameter #
  -Wno-array-bounds #
  -Wno-deprecated-enum-enum-conversion #
  -Wno-unused-but-set-variable #
  # -Wno-class-memaccess # $<$<CONFIG:RELEASE>:-Og> #
  $<$<CONFIG:DEBUG>:-pedantic> #
  # $<$<CONFIG:DEBUG>:-g3> #
  $<$<CONFIG:DEBUG>:-Og> #
  $<$<CONFIG:DEBUG>:-fsanitize=address> #
  $<$<CONFIG:DEBUG>:-fsanitize=leak> #
  $<$<CONFIG:DEBUG>:-fsanitize=undefined> #
  $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer> #
  $<$<CONFIG:DEBUG>:-fno-common> #
)
add_link_options(
  $<$<CONFIG:DEBUG>:-fsanitize=address> #
  $<$<CONFIG:DEBUG>:-fsanitize=leak> #
  $<$<CONFIG:DEBUG>:-fsanitize=undefined> #
  $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer> #
  $<$<CONFIG:DEBUG>:-fno-common> #
  $<$<CONFIG:DEBUG>:-Og> #
)

add_subdirectory(xtsrcmaps)


# Add OpenMP flags to your compiler
if(OpenMP_CXX_FOUND)
  # Check if we're on macOS and set the correct flags
  if(APPLE)
    # Force linking against the OpenMP library from Homebrew
    set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    set(OpenMP_CXX_FLAGS
        "${OpenMP_CXX_FLAGS} -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIBRARIES "${OpenMP_omp_LIBRARY}")
  endif()
  target_compile_options(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_FLAGS})
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

#
# ##############################################################################

FetchContent_Declare(
  GTest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.15.2)
FetchContent_MakeAvailable(GTest )

# target_include_directories(my_target PUBLIC ${DOCTEST_INCLUDE_DIR})
include(CTest)
enable_testing()
add_subdirectory(tests)
add_subdirectory(scripts)
