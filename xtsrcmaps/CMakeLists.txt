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
  interp.cxx
  healpix.cxx
  bilerp.cxx
  exposure.cxx
  psf.cxx
  genz_malik.cxx
  model_map.cxx
  model_map/grid_operations.cxx
)

add_library(Fermi::xtsrcmaps_lib ALIAS xtsrcmaps_lib)
target_link_libraries(
  xtsrcmaps_lib
  PUBLIC PkgConfig::wcslib Eigen3::Eigen
  PRIVATE fmt::fmt PkgConfig::cfitsio  #  OpenMP::OpenMP_CXX
)

add_executable(xtsrcmaps xtsrcmaps.cxx)
target_link_libraries(xtsrcmaps PRIVATE fmt::fmt Fermi::xtsrcmaps_lib)

add_executable(gen_model_map scripts/gen_model_map.cxx)
target_link_libraries(gen_model_map PRIVATE fmt::fmt Fermi::xtsrcmaps_lib)

# Tests ####
add_subdirectory(tests)
