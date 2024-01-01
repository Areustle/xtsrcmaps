#include "xtsrcmaps/model_map/model_map.hxx"

auto
Fermi::ModelMap::map_correction_factor(
    Tensor2d const& inv_map_integ, /* [E, F] */
    Tensor1d const& psf_radius,
    Tensor1b const& is_in_fov,
    Tensor3d const& mean_psf,         /* [D,E,S] */
    Tensor3d const& partial_integrals /* [D,E,S] */
    ) -> Tensor2d {
    long const Nd = mean_psf.dimension(0);
    long const Ne = mean_psf.dimension(1);
    long const Ns = mean_psf.dimension(2);
    long const Nf = psf_radius.dimension(0);

    Tensor3d filtered_psf(Nd, Ne, Nf);
    Tensor3d filtered_parint(Nd, Ne, Nf);

    long f = 0;
    for (long s = 0; s < Ns; ++s) {
        if (!is_in_fov(s)) { continue; }
        filtered_psf.slice(Idx3 { 0, 0, f }, Idx3 { Nd, Ne, 1 })
            = mean_psf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 });
        filtered_parint.slice(Idx3 { 0, 0, f }, Idx3 { Nd, Ne, 1 })
            = partial_integrals.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 });
        ++f;
    }

    Tensor2d const rad_integ
        = Fermi::ModelMap::integral(psf_radius, filtered_psf, filtered_parint);
    Tensor2d cor_fac = rad_integ * inv_map_integ;

    Tensor2d correction_factor(Ne, Ns);
    correction_factor.setConstant(1.);

    f = 0;
    for (long s = 0; s < Ns; ++s) {
        if (!is_in_fov(s)) { continue; }
        correction_factor.slice(Idx2 { 0, s }, Idx2 { Ne, 1 })
            = cor_fac.slice(Idx2 { 0, f }, Idx2 { Ne, 1 });
        ++f;
    }

    return correction_factor;
}
