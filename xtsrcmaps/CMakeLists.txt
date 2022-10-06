#
# ... #
#
include_directories(${CMAKE_SOURCE_DIR})

add_library(
  xtsrcmaps_lib OBJECT
  tensor_ops.cxx
  bilerp.cxx
  exposure.cxx
  fitsfuncs.cxx
  healpix.cxx
  irf.cxx
  parse_src_mdl.cxx
  psf.cxx
  source_utils.cxx
)

add_library(Fermi::xtsrcmaps_lib ALIAS xtsrcmaps_lib)
target_link_libraries(
  xtsrcmaps_lib
  PUBLIC std::mdspan
  PRIVATE Eigen3::Eigen fmt::fmt std::mdspan PkgConfig::cfitsio
)

add_executable(xtsrcmaps xtsrcmaps.cxx)
target_link_libraries(xtsrcmaps PRIVATE fmt::fmt Fermi::xtsrcmaps_lib)

# Tests ####
add_subdirectory(tests)
