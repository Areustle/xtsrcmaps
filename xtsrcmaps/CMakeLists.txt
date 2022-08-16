add_executable(
  xtsrcmaps
  bilerp.cxx
  fitsfuncs.cxx
  parse_src_mdl.cxx
  prepare_psf_data.cxx
  psf_v2.cxx
  source_utils.cxx
  xtsrcmaps.cxx
)
target_link_libraries(xtsrcmaps PRIVATE fmt::fmt std::mdspan PkgConfig::cfitsio)
target_include_directories(xtsrcmaps PRIVATE ${CMAKE_SOURCE_DIR})
