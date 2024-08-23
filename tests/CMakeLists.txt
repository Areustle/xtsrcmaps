# Make test executable
#
add_executable(
  tests #
  test_exposure.cxx #
  test_fitsfuncs.cxx #
  test_healpix.cxx #
  test_irf.cxx #
  test_king.cxx #
  test_main.cxx #
  tensor/test_read_tensor.cpp #
  tensor/test_write_tensor.cpp #
  tensor/test_tensor.constructors.cpp #
  tensor/test_tensor.subscript.cpp #
  tensor/test_tensor.iterators.cpp #
  tensor/test_tensor.utility.cpp #
  tensor/test_tensor.reshape.cpp #
  tensor/test_tensor.broadcast.cpp #
  tensor/test_tensor.arithmetic.cpp #
  tensor/test_broadcast_iterator.cpp #
  # model_map/test_pixel_convol_generic.cpp #
  # test_model_map_v2.cxx #
  test_psf.cxx #
  test_irf.psf.cpp #
  # test_sky_geom.cxx #
  # test_src_utils.cxx #
  # model_map/test_pixel_offsets.cxx #
  # model_map/test_riemann_slow.cxx #
)
target_link_libraries(tests PRIVATE fmt::fmt PkgConfig::cfitsio
                                    doctest::doctest xtsrcmaps)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME tests COMMAND tests)
