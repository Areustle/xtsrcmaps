#include "xtsrcmaps/irf/_irf_private.hpp"

namespace irf_private {

auto
prepare_psf_data(Fermi::fits::TablePars const& front_rpsf,
                 Fermi::fits::TablePars const& front_scaling,
                 Fermi::fits::TablePars const& front_fisheye,
                 Fermi::fits::TablePars const& back_rpsf,
                 Fermi::fits::TablePars const& back_scaling,
                 Fermi::fits::TablePars const& back_fisheye)
    -> Fermi::irf::psf::Pass8FB {

    auto front = Fermi::irf::psf::Data { prepare_grid(front_rpsf),
                                         prepare_scale(front_scaling),
                                         prepare_grid(front_fisheye) };
    auto back  = Fermi::irf::psf::Data { prepare_grid(back_rpsf),
                                        prepare_scale(back_scaling),
                                        prepare_grid(back_fisheye) };

    normalize_rpsf(front);
    normalize_rpsf(back);

    return { front, back };
};
} // namespace irf_private
