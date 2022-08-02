# Fermi xtsrcmaps Author: Alexander Reustle (2022)
cmake_minimum_required(VERSION 3.18)

project(
  xtsrcmaps
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "Fermi xtsrcmaps experimental source map generator."
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# CMake tool support packages. These provide additional funcitons to our Cmake
# files.
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CTest)
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
# foreach(pkg ${FERMI_COMPONENTS}) add_subdirectory(src/${pkg}
# ${CMAKE_BINARY_DIR}/${pkg}) endforeach()

add_compile_options(
  -Wall
  -Wextra
  -pedantic
  -Werror
  -march=native
  # -fsanitize=address
  # -fsanitize=leak
  # -fsanitize=undefined
  # -fno-omit-frame-pointer
  # -fno-common
)
# add_link_options(
#   -fsanitize=address -fsanitize=leak -fsanitize=undefined
#   -fno-omit-frame-pointer -fno-common
# )

add_subdirectory(xtsrcmaps)
