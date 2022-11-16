# Fermi xtsrcmaps Author: Alexander Reustle (2022)
cmake_minimum_required(VERSION 3.21)

project(
  xtsrcmaps
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "Fermi LAT experimental source map generator."
)

cmake_policy(SET CMP0072 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# CMake tool support packages. These provide additional funcitons to our Cmake
# files.
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(FetchContent)

# Externals: The dependent packages on which the tool relies. ##
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan
  GIT_TAG stable
)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG 3.4
)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 9.0.0
)
FetchContent_MakeAvailable(mdspan Eigen fmt)

# find_package(fmt REQUIRED)
find_package(PkgConfig REQUIRED)

# Pkg-Config Externals ##
pkg_check_modules(cfitsio REQUIRED IMPORTED_TARGET GLOBAL cfitsio)
pkg_check_modules(wcslib REQUIRED IMPORTED_TARGET GLOBAL wcslib)

# ##############################################################################

add_compile_options(
  -Wall
  -Wextra
  -Werror # -mbmi2
  -march=native
  -Wno-unused-parameter
  $<$<CONFIG:RELEASE>:-O3>
  # $<$<CONFIG:RELEASE>:-Og>
  $<$<CONFIG:DEBUG>:-pedantic>
  $<$<CONFIG:DEBUG>:-g3>
  $<$<CONFIG:DEBUG>:-Og>
  $<$<CONFIG:DEBUG>:-fsanitize=address>
  $<$<CONFIG:DEBUG>:-fsanitize=leak>
  $<$<CONFIG:DEBUG>:-fsanitize=undefined>
  $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>
  $<$<CONFIG:DEBUG>:-fno-common>
)
add_link_options(
  $<$<CONFIG:DEBUG>:-fsanitize=address> $<$<CONFIG:DEBUG>:-fsanitize=leak>
  $<$<CONFIG:DEBUG>:-fsanitize=undefined>
  $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer> $<$<CONFIG:DEBUG>:-fno-common>
)

add_subdirectory(xtsrcmaps)

# ##############################################################################

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest
  GIT_TAG master
)
FetchContent_MakeAvailable(doctest)

# target_include_directories(my_target PUBLIC ${DOCTEST_INCLUDE_DIR})
include(CTest)
enable_testing()
