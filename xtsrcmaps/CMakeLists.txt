#
# ... #
#
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(utils)
add_subdirectory(psf)

add_library(xtsrcmaps_lib
    STATIC  tensor_ops.cxx
            fitsfuncs.cxx
            parse_src_mdl.cxx
            source_utils.cxx
            sky_geom.cxx
            irf.cxx
            interp.cxx
            healpix.cxx
            exposure.cxx
            genz_malik.cxx
            model_map.cxx
)

add_library(Fermi::xtsrcmaps_lib ALIAS xtsrcmaps_lib)
target_link_libraries(
  xtsrcmaps_lib
  PUBLIC  PkgConfig::wcslib
  PRIVATE fmt::fmt
         Eigen3::Eigen
          PkgConfig::cfitsio  #  OpenMP::OpenMP_CXX
          Fermi::utils
          Fermi::psf
)

add_executable(xtsrcmaps xtsrcmaps.cxx)
target_link_libraries(xtsrcmaps fmt::fmt Eigen3::Eigen Fermi::xtsrcmaps_lib)

# add_executable(gen_model_map scripts/gen_model_map.cxx)
# target_link_libraries(gen_model_map PRIVATE fmt::fmt Fermi::xtsrcmaps_lib)

# Tests ####
# add_subdirectory(tests)
