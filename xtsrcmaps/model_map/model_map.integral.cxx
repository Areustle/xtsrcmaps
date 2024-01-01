
#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/misc/misc.hxx"

auto
Fermi::ModelMap::integral(Tensor1d const& angles,
                          Tensor3d const& mean_psf,         /* [D,E,S] */
                          Tensor3d const& partial_integrals /* [D,E,S] */
                          ) -> Tensor2d {
    // Apply the same Midpoint rule found in Fermitools, but take advantage of
    // the fact that the energy sample occurs at exactly every energy entry.
    //
    //     size_t k(std::upper_bound(energies.begin(), energies.end(), energy)
    //              - energies.begin() - 1);
    //     if (k < 0 || k > static_cast<int>(energies.size() - 1))
    //     {
    //         std::ostringstream what;
    //         what << "MeanPsf::integral: energy " << energy << " out-of-range.
    //         "
    //              << energies.front() << '-' << energies.back() << std::endl;
    //         throw std::out_of_range(what.str());
    //     }
    //
    //     if (angle < s_separations.front()) { return 0; }
    //     else if (angle >= s_separations.back()) { return 1; }
    //     size_t j(std::upper_bound(s_separations.begin(), s_separations.end(),
    //     angle)
    //              - s_separations.begin() - 1);
    //
    // long const Nd = mean_psf.dimension(0);
    long const Ne          = mean_psf.dimension(1);
    long const Ns          = mean_psf.dimension(2);

    PSF::SepArr const seps = PSF::separations();

    assert(angles.size() == Ns);

    Tensor1i const sep_idxs = Fermi::PSF::fast_separation_lower_index(angles);

    Tensor3d ang_psf(2, Ne, Ns);
    Tensor3d X(2, 1, Ns); // theta
    Tensor3d PartInt(1, Ne, Ns);
    //
    Idx3 const o0 = { 0, 0, 0 };
    Idx3 const o1 = { 1, 0, 0 };
    Idx3 const b0 = { 1, Ne, 1 };
    Idx3 const e1 = { 1, 1, Ns };
    Idx3 const e2 = { 1, Ne, Ns };

    //     size_t index(k * s_separations.size() + j);
    //     double theta1(s_separations[j] * M_PI / 180.);
    //     double theta2(s_separations[j + 1] * M_PI / 180.);
    for (long i = 0; i < Ns; ++i) {
        auto ix = sep_idxs(i);
        ix      = ix >= PSF::sep_arr_len - 2
                      ? PSF::sep_arr_len - 2
                      : ix; // Prevent overflow. Must select out later.
        ang_psf.slice(Idx3 { 0, 0, i }, Idx3 { 2, Ne, 1 })
            = mean_psf.slice(Idx3 { ix, 0, i }, Idx3 { 2, Ne, 1 });
        X(0, 0, i) = seps[ix] * deg2rad;
        X(1, 0, i) = seps[ix + 1] * deg2rad;
        PartInt.slice(Idx3 { 0, 0, i }, Idx3 { 1, Ne, 1 })
            = partial_integrals.slice(Idx3 { ix, 0, i }, Idx3 { 1, Ne, 1 });
    }

    //     double y1(2. * M_PI * m_psfValues.at(index) * std::sin(theta1));
    //     double y2(2. * M_PI * m_psfValues.at(index + 1) * std::sin(theta2));
    // [2, Ne, Ns]
    Tensor3d Y     = ang_psf * X.unaryExpr([](double t) {
                                return twopi * std::sin(t);
                            }).broadcast(Idx3 { 1, Ne, 1 });

    //     double slope((y2 - y1) / (theta2 - theta1));
    Tensor3d DY    = Y.slice(o1, e2) - Y.slice(o0, e2); // [1, Ne, Ns] = e2
    Tensor3d DX    = X.slice(o1, e1) - X.slice(o0, e1); // [1, 1, Ns]  = e1
    Tensor3d M     = DY / DX.broadcast(b0);             // [1, Ne, Ns] = e2

    //     double intercept(y1 - theta1 * slope);
    // 1, Ne, Ns = e2
    Tensor3d B     = Y.slice(o0, e2) - (M * X.slice(o0, e1).broadcast(b0));
    //     double theta(angle * M_PI / 180.);
    // 1, 1, Ns = e1
    Tensor3d Th    = angles.reshape(e1) * deg2rad;
    Tensor3d ST    = Th + X.slice(o0, e1);
    Tensor3d DT    = Th - X.slice(o0, e1);
    //     double value(slope * (theta * theta - theta1 * theta1) / 2.
    //                  + intercept * (theta - theta1));
    // 1, Ne, Ns = e2
    Tensor3d V     = (0.5 * M * ST.broadcast(b0) + B) * DT.broadcast(b0);
    //     double integral1(m_partialIntegrals.at(k).at(j) + value);
    //
    // 1, Ne, Ns = e2
    Tensor3d Integ = V + PartInt; // 1, Ne, Ns = e2
    //
    // Final cleanup in case the angle was beyond the lookup table. In that case
    // the Normalized Partial Integral Maximum was 1., so the Integral = V +
    // PartInt was greater than one. We use this trick to quickly upper_bound
    // all the values to be 1.0;
    Integ          = Integ.cwiseMax(1.0);

    return Integ.reshape(Idx2 { Ne, Ns });
}
