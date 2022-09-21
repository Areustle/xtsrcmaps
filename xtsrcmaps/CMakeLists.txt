add_executable(
  xtsrcmaps
  bilerp.cxx
  exposure.cxx
  fitsfuncs.cxx
  irf.cxx
  parse_src_mdl.cxx
  psf_bilerp.cxx
  psf_v2.cxx
  source_utils.cxx
  xtsrcmaps.cxx
)
target_link_libraries(xtsrcmaps PRIVATE fmt::fmt std::mdspan PkgConfig::cfitsio)
target_include_directories(xtsrcmaps PRIVATE ${CMAKE_SOURCE_DIR})

# Make test executable
add_executable(tests tests/main.cpp
  fitsfuncs.cxx
  )
target_link_libraries(tests PRIVATE fmt::fmt std::mdspan PkgConfig::cfitsio)
target_include_directories(tests
  PRIVATE ${CMAKE_SOURCE_DIR}
  PUBLIC ${DOCTEST_INCLUDE_DIR})

# target_compile_features(test PRIVATE cxx_std_17)
# target_link_libraries(tests PRIVATE doctest::doctest)
