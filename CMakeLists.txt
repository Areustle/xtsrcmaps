# Fermi xtsrcmaps Author: Alexander Reustle (2022)
cmake_minimum_required(VERSION 3.21)

project(
  xtsrcmaps
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "Fermi LAT experimental source map generator."
)

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
FetchContent_MakeAvailable(mdspan)

find_package(fmt REQUIRED)
find_package(PkgConfig REQUIRED)

# Pkg-Config Externals ##
pkg_check_modules(cfitsio REQUIRED IMPORTED_TARGET GLOBAL cfitsio)

# ##############################################################################

add_compile_options(
  -Wall -Wextra -pedantic -Werror # -mbmi2 -march=native
  # -fsanitize=address -fsanitize=leak -fsanitize=undefined
  # -fno-omit-frame-pointer -fno-common
)
add_link_options(
  # -fsanitize=address -fsanitize=leak -fsanitize=undefined
  # -fno-omit-frame-pointer -fno-common
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
