# Add the 'cmake' directory to the CMake module path list(PREPEND
# CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Externals: The dependent packages on which the tool relies.
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 10.2.0)
# FetchContent_Declare( indicators GIT_REPOSITORY
# https://github.com/p-ranav/indicators GIT_TAG v2.3)
FetchContent_MakeAvailable(fmt)

# Pkg-Config Externals ##
find_package(PkgConfig REQUIRED)
pkg_check_modules(cfitsio REQUIRED IMPORTED_TARGET GLOBAL cfitsio)
pkg_check_modules(wcslib REQUIRED IMPORTED_TARGET GLOBAL wcslib)
pkg_check_modules(fftw3 REQUIRED IMPORTED_TARGET GLOBAL fftw3)

#
add_subdirectory(cxxopts)
