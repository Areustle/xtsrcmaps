target_sources(
  xtsrcmaps
  PRIVATE model_map.compute_src_maps.cxx
          mm_cubature/cubature.implementation.cxx
          mm_cubature/weights.arpist.4.cpp
          # mm_cubature/weights.asymsquare.12.cpp
          # mm_cubature/weights.square.11.cpp
          mm_cubature/weights.naive.4096.cpp
          model_map.point_src_model_map_wcs.cxx
          # model_map.integral.cxx
          # model_map.get_init_points.cxx
          # model_map.integral.cxx
          model_map.map_correction_factor.cxx
          model_map.map_integral.cxx
          # model_map.pixel_mean_psf_genz.cxx
          model_map.psf_boundary_radius.cxx
          # model_map.psf_fast_lut.cxx
          model_map.scale_map_by_correction_factor.cxx
          model_map.scale_map_by_exposure.cxx
          model_map.scale_map_by_solid_angle.cxx
)
