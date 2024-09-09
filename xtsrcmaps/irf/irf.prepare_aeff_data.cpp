#include "xtsrcmaps/irf/_irf_private.hpp"

#include "xtsrcmaps/fits/fits.hxx"

namespace irf_private {

auto
prepare_aeff_data(Fermi::fits::TablePars const& front_eff_area,
                  Fermi::fits::TablePars const& front_phi_dep,
                  Fermi::fits::TablePars const& front_effici,
                  Fermi::fits::TablePars const& back_eff_area,
                  Fermi::fits::TablePars const& back_phi_dep,
                  Fermi::fits::TablePars const& back_effici)
    -> Fermi::Irf::aeff::Pass8FB {
    auto front = Fermi::Irf::aeff::Data { prepare_grid(front_eff_area),
                                          prepare_grid(front_phi_dep),
                                          prepare_effic(front_effici) };
    auto back  = Fermi::Irf::aeff::Data { prepare_grid(back_eff_area),
                                         prepare_grid(back_phi_dep),
                                         prepare_effic(back_effici) };


    return { front, back };
}

} // namespace irf_private
