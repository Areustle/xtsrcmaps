#include "xtsrcmaps/irf/_irf_private.hpp"

#include "xtsrcmaps/fits/fits.hxx"

#include <cassert>
#include <mdspan>

namespace irf_private {

//************************************************************************************
// Convert the raw arrays read in from an IRF FITS file into a useable set
// of gridded data tensors for future sampling.
//
// The format of an IRF grid is a single FITS row with 5 or more columns.
// The first 4 columns are the low and high columns of the energy and
// costheta range vectors. Every subsequent column is a data entry for 1
// paramater entry in a linearized grid.
//
// column 0: energy low
// column 1: energy high
// column 2: costheta low
// column 3: costheta high
// column 4: Data parameter in linearized (costheta x energy) grid.
// column n: Data parameter in linearized (costheta x energy) grid.
//************************************************************************************
auto
prepare_grid(Fermi::fits::TablePars const& pars) -> Fermi::Irf::IrfData3 {

    assert(pars.extents.size() >= 5);
    assert(pars.extents[0] > 1);
    assert(pars.extents[0] == pars.extents[1]);
    assert(pars.extents[2] > 1);
    assert(pars.extents[2] == pars.extents[3]);
    assert(pars.extents[4] == pars.extents[0] * pars.extents[2]);
    assert(pars.rowdata.extent(0) == 1uz);

    auto const& extents               = pars.extents;
    auto const& offsets               = pars.offsets;

    size_t const             M_e_base = extents[0];
    size_t const             M_e      = extents[0] + 2;
    size_t const             M_t_base = extents[2];
    size_t const             M_t      = extents[2] + 2;
    size_t const             off_cos0 = offsets[2];
    size_t const             off_cos1 = offsets[3];
    Fermi::Tensor<double, 1> cosths(M_t);

    // Arithmetic mean of cosine bins
    auto const cos0view = std::span { &pars.rowdata[0, off_cos0], M_t_base };
    auto const cos1view = std::span { &pars.rowdata[0, off_cos1], M_t_base };
    std::transform(cos0view.begin(),
                   cos0view.end(),
                   cos1view.begin(),
                   cosths.begin_at(1uz),
                   [](auto const& c0, auto const& c1) {
                       return 0.5 * (c0 + c1);
                   });
    cosths[0]                        = -1.;
    cosths[M_t - 1]                  = 1.;

    // scale and pad the energy data
    size_t const             off_Es0 = offsets[0];
    size_t const             off_Es1 = offsets[1];
    Fermi::Tensor<double, 1> logEs(M_e);
    /* logEs.(); */
    auto const Es0view = std::span { &pars.rowdata[0, off_Es0], M_e_base };
    auto const Es1view = std::span { &pars.rowdata[0, off_Es1], M_e_base };
    std::transform(Es0view.begin(),
                   Es0view.end(),
                   Es1view.begin(),
                   logEs.begin_at(1uz),
                   [](auto const& c0, auto const& c1) {
                       return 0.5 * std::log10((c0 * c1));
                   });
    logEs[0]                        = 0.;
    logEs[M_e - 1]                  = 10.;

    size_t const             Ngrids = extents.size() - 4;
    Fermi::Tensor<double, 3> params(M_t, M_e, Ngrids);
    std::mdspan              pv {
        pars.rowdata.data() + offsets[4], Ngrids, M_t_base, M_e_base
    };

    // Let's assign the data values into the params block structure. Pad by
    // value duplication.
    for (size_t p = 0; p < Ngrids; ++p) {  // params
        for (size_t t = 0; t < M_t; ++t) { // costheta
            size_t const t_ = t == 0 ? 0 : t >= M_t_base ? M_t_base - 1 : t - 1;
            for (size_t e = 0; e < M_e; ++e) { // energy
                size_t const e_ = e == 0          ? 0
                                  : e >= M_e_base ? M_e_base - 1
                                                  : e - 1;
                params[t, e, p] = pv[p, t_, e_];
            }
        }
    }

    return { .cosths      = cosths,
             .logEs       = logEs,
             .params      = params,
             .minCosTheta = pars.rowdata[0, off_cos0] };
};
} // namespace irf_private
