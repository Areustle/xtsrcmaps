# Fermi xtsrcmaps Author: Alexander Reustle (2022)
cmake_minimum_required(VERSION 3.21)

project(
  xtsrcmaps
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "Fermi LAT experimental source map generator.")

cmake_policy(SET CMP0072 NEW)
set(CMAKE_CXX_STANDARD 20)

# CMake tool support packages. These provide additional funcitons to our Cmake
# files.
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(FetchContent)

# use ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache in ${CCACHE_PROGRAM}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

add_subdirectory(external)

# ##############################################################################

add_compile_options(
  -Wall #
  -Wextra #
  -Werror # -mbmi2 #
  $<$<CONFIG:RELEASE>:-march=native> #
  $<$<CONFIG:RELEASE>:-O3> #
  -Wno-unused-parameter #
  -Wno-array-bounds #
  -Wno-deprecated-enum-enum-conversion #
  -Wno-unused-but-set-variable #
  # -Wno-class-memaccess #
  # $<$<CONFIG:RELEASE>:-Og> #
  $<$<CONFIG:DEBUG>:-pedantic> #
  # $<$<CONFIG:DEBUG>:-g3> #
  # $<$<CONFIG:DEBUG>:-Og> #
  # $<$<CONFIG:DEBUG>:-fsanitize=address> #
  # $<$<CONFIG:DEBUG>:-fsanitize=leak> #
  # $<$<CONFIG:DEBUG>:-fsanitize=undefined> #
  $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer> #
  $<$<CONFIG:DEBUG>:-fno-common> #
)
add_link_options(
  # $<$<CONFIG:DEBUG>:-fsanitize=address> #
  # $<$<CONFIG:DEBUG>:-fsanitize=leak> #
  # $<$<CONFIG:DEBUG>:-fsanitize=undefined> #
  $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer> #
  $<$<CONFIG:DEBUG>:-fno-common> #
)

add_subdirectory(xtsrcmaps)

# #
# ##############################################################################
#
# FetchContent_Declare( doctest GIT_REPOSITORY
# https://github.com/doctest/doctest GIT_TAG master )
# FetchContent_MakeAvailable(doctest)
#
# # target_include_directories(my_target PUBLIC ${DOCTEST_INCLUDE_DIR})
# include(CTest) enable_testing()
