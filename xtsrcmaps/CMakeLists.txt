#
# ... #
#
include_directories(${CMAKE_SOURCE_DIR})

add_library(xtsrcmaps SHARED)

add_subdirectory(cli)
add_subdirectory(config)
add_subdirectory(exposure)
add_subdirectory(fits)
add_subdirectory(healpix)
add_subdirectory(irf)
add_subdirectory(math)
add_subdirectory(model_map)
add_subdirectory(observation)
add_subdirectory(psf)
add_subdirectory(sky_geom)
add_subdirectory(source)

target_link_libraries(xtsrcmaps PRIVATE fmt::fmt PkgConfig::cfitsio
                                        PkgConfig::wcslib cxxopts)

add_executable(xtsrcmaps_exe main.cxx)
set_target_properties(xtsrcmaps_exe PROPERTIES OUTPUT_NAME "xtsrcmaps")
target_link_libraries(xtsrcmaps_exe PRIVATE xtsrcmaps)

### add_executable(gen_model_map scripts/gen_model_map.cxx)
### target_link_libraries(gen_model_map PRIVATE fmt::fmt Fermi::xtsrcmaps_lib)

# Tests #### add_subdirectory(tests)
