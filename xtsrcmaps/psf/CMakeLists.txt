include_directories(${CMAKE_SOURCE_DIR})

add_library(psf STATIC
            psf.bilerp.cxx
            psf.corrected_exposure_psf.cxx
            psf.fast_separation_lower_index.cxx
            psf.king.cxx
            psf.mean_psf.cxx
            psf.normalize.cxx
            psf.partial_total_integral.cxx
            psf.peak_psf.cxx
            psf.psf_lookup_table_and_partial_integrals.cxx
)

add_library(Fermi::psf ALIAS psf)
target_link_libraries(
    psf 
    PUBLIC  Eigen3::Eigen
    PRIVATE Fermi::utils
)
