target_sources(
  xtsrcmaps PRIVATE bilerp.cxx # convolve/fft/fft.apple_convolve3d.cpp
  convolve/fft/fft.fftw_convolve_CHW.cpp
) # genz_malik.cxx)

target_link_libraries(xtsrcmaps PRIVATE PkgConfig::fftw3)
