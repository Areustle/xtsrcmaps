# Make test executable
#
add_executable(
  tests
  test_exposure.cxx
  test_fitsfuncs.cxx
  test_healpix.cxx
  test_irf.cxx
  test_king.cxx
  test_main.cxx #
  test_model_map_v2.cxx
  test_psf.cxx
  test_sky_geom.cxx
  test_src_utils.cxx
)
target_link_libraries(
  tests PRIVATE fmt::fmt PkgConfig::cfitsio doctest::doctest
                Fermi::xtsrcmaps_lib
)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME tests COMMAND tests)
