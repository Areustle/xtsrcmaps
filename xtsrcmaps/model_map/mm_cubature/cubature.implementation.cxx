
#include "xtsrcmaps/model_map/mm_cubature/cubature.hpp"
#include "xtsrcmaps/model_map/mm_cubature/convolve_pixel_psf_logsep.hpp"
/* #include "xtsrcmaps/model_map/mm_cubature/convolve_pixel_psf_searchsep.hpp"
 */
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <omp.h>

#include <array>
#include <cassert>

namespace Fermi {

/********************************************************************************
 * Convolve the PSF via lookup table, with each of the pixels for every source.
 * ----
 ********************************************************************************/

auto
cubature_convolve_map_psf(
    size_t const             Nh,
    size_t const             Nw,
    Tensor<double, 2> const& src_sph,
    Tensor<double, 3> const& psf_lut, // psf lookup table [Source, Seps, Energy]
    SkyGeom<double> const&   skygeom) -> Tensor<double, 4> {

    size_t const Ns = src_sph.extent(0);
    assert(src_sphcrds.extent(1) == 2);
    size_t const Ne = psf_lut.extent(2);

    Tensor<double, 4> model_map({ Ns, Nh, Nw, Ne });
    model_map.clear();

    auto points_weights = Fermi::square_ptswts(
        Nh, Nw, skygeom, Fermi::CubatureSets::square_deg7);

#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        auto source = skygeom.sph2dir({ src_sph[s, 0], src_sph[s, 1] });
        for (size_t w = 0; w < Nw; ++w) {
            for (size_t h = 0; h < Nh; ++h) {
                convolve_pixel_psf_logsep<double,
                                          Fermi::CubatureSets::N_SQUARE_D7>(
                    &(model_map[s, h, w, 0]),
                    Ne,
                    &(points_weights[h, w, 0]),
                    &(psf_lut[s, 0, 0]),
                    source);
            }
        }
    }
    return model_map;
}
} // namespace Fermi
