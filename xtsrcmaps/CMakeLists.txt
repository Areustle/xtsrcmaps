#
# ... #
#
include_directories(${CMAKE_SOURCE_DIR})

add_library(
  xtsrcmaps_lib OBJECT
  tensor_ops.cxx
  fitsfuncs.cxx
  parse_src_mdl.cxx
  source_utils.cxx
  sky_geom.cxx
  irf.cxx
  healpix.cxx
  bilerp.cxx
  exposure.cxx
  psf.cxx
  genz_malik.cxx
  model_map.cxx
)

add_library(Fermi::xtsrcmaps_lib ALIAS xtsrcmaps_lib)
target_link_libraries(
  xtsrcmaps_lib
  PUBLIC PkgConfig::wcslib Eigen3::Eigen
  PRIVATE fmt::fmt PkgConfig::cfitsio
)

add_executable(xtsrcmaps xtsrcmaps.cxx)
target_link_libraries(xtsrcmaps PRIVATE fmt::fmt Fermi::xtsrcmaps_lib)

# Tests ####
add_subdirectory(tests)
