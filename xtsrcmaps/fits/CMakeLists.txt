target_sources(
  xtsrcmaps
  PRIVATE fits.read_energies.cxx
          fits.read_image_meta.cxx
          fits.safe_open.cpp
          fits.read_allsky_cropped.cpp
          fits.read_expcube.cxx
          fits.read_irf_pars.cxx
          fits.write_src_model.cxx #
)
target_link_libraries(xtsrcmaps PRIVATE PkgConfig::cfitsio)
