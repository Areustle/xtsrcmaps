target_sources(
  xtsrcmaps
  PRIVATE model_map.cxx
          model_map.integral.cxx
          model_map.get_init_points.cxx
          model_map.integral.cxx
          model_map.map_correction_factor.cxx
          model_map.map_integral.cxx
          model_map.pixel_mean_psf_genz.cxx
          model_map.psf_boundary_radius.cxx
          model_map.psf_fast_lut.cxx
          # model_map.riemann.cxx
          model_map.scale_map_by_correction_factor.cxx
          model_map.scale_map_by_exposure.cxx
          model_map.scale_map_by_solid_angle.cxx)
