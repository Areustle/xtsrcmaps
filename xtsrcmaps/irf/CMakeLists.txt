target_sources(
  xtsrcmaps
  PRIVATE irf.cxx
          irf.aeff_value.cxx
          irf.collect_irf_data.cpp
          irf.normalize_rpsf.cpp
          irf.prepare_aeff_data.cpp
          irf.prepare_effic.cpp
          irf.prepare_grid.cpp
          irf.prepare_psf_data.cpp
          irf.prepare_scale.cpp)
