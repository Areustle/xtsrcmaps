#
# ... #
#
include_directories(${CMAKE_SOURCE_DIR})

add_library(bilerp OBJECT bilerp.cxx)
target_link_libraries(bilerp PRIVATE fmt::fmt)
#
add_library(exposure OBJECT exposure.cxx)
target_link_libraries(
  exposure
  PUBLIC irf
  PRIVATE fmt::fmt std::mdspan
)
#
add_library(fitsfuncs OBJECT fitsfuncs.cxx)
target_link_libraries(fitsfuncs PRIVATE fmt::fmt PkgConfig::cfitsio)
#
add_library(irf OBJECT irf.cxx)
target_link_libraries(
  irf
  PUBLIC std::mdspan
  PRIVATE fmt::fmt
)
#
add_library(source_utils OBJECT source_utils.cxx)
target_link_libraries(source_utils)
#
add_library(parse_src_mdl OBJECT parse_src_mdl.cxx)
#
add_library(psf_v2 OBJECT psf_v2.cxx psf_bilerp.cxx)
target_link_libraries(
  psf_v2
  PUBLIC irf
  PRIVATE fmt::fmt std::mdspan
)

#
add_executable(xtsrcmaps xtsrcmaps.cxx)
target_link_libraries(
  xtsrcmaps
  PRIVATE
    fmt::fmt
    std::mdspan
    PkgConfig::cfitsio
    bilerp
    exposure
    fitsfuncs
    irf
    parse_src_mdl
    psf_v2
    source_utils
)

# Tests ####
add_subdirectory(tests)
