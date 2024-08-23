#include "xtsrcmaps/model_map/model_map.hxx"

auto
Fermi::ModelMap::map_correction_factor(
    Tensor<float, 2> const&  inv_map_integ, // [SE]
    Tensor<double, 1> const& psf_radius,
    std::vector<bool> const& is_in_fov,
    Tensor<float, 3> const&  mean_psf,         // [SDE]
    Tensor<double, 3> const& partial_integrals // [SDE]
    ) -> Tensor<float, 2> {
    size_t const Ns = mean_psf.extent(0);
    /* size_t const Nd = mean_psf.extent(1); */
    size_t const Ne = mean_psf.extent(2);

    /* assert(Nd == partial_integrals.extent(1)); */
    assert(Ne == partial_integrals.extent(2));

    Tensor<float, 2> correction_factor(Ns, Ne);

    PSF::SepArr const    seps = PSF::separations();
    Tensor<int, 1> const sep_idxs
        = Fermi::PSF::fast_separation_lower_index(psf_radius);

#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        if (!is_in_fov[s]) {
            for (size_t e = 0; e < Ne; ++e) { correction_factor[s, e] = 1.0f; }
            continue;
        }

        // Use Midpoint Rule to compute approximate sum of psf from each
        // separation entry over the lookup table.
        size_t const d      = sep_idxs[s] >= PSF::sep_arr_len - 2
                                  ? PSF::sep_arr_len - 2
                                  : sep_idxs[s];
        float const  theta1 = seps[d] * deg2rad;
        float const  theta2 = seps[d + 1] * deg2rad;

        for (size_t e = 0; e < Ne; ++e) {
            float const y1 = twopi * mean_psf[s, d, e] * std::sin(theta1);
            float const y2 = twopi * mean_psf[s, d + 1, e] * std::sin(theta2);
            float const m  = (y2 - y1) / (theta2 - theta1);
            float const b  = y1 - theta1 * m;
            float const v  = 0.5f * m * (theta2 * theta2 - theta1 * theta1)
                            + b * (theta2 - theta1);
            float integ             = v + partial_integrals[s, d, e];
            integ                   = integ > 1.0f ? 1.0f : integ;
            correction_factor[s, e] = integ * inv_map_integ[s, e];
        }
    }

    return correction_factor;
}
