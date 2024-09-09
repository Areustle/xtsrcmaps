#include "xtsrcmaps/irf/_irf_private.hpp"

#include <cassert>

namespace irf_private {

auto
prepare_effic(Fermi::fits::TablePars const& pars) -> Fermi::Irf::IrfEffic {

    assert(pars.extents.size() == 1);
    assert(pars.extents[0] == 6);
    assert(pars.rowdata.extent(0) == 2);
    assert(pars.rowdata.extent(1) == 6);

    auto p0 = std::array<float, 6> { 0.0 };
    auto p1 = std::array<float, 6> { 0.0 };
    std::copy(&pars.rowdata[0, 0], &pars.rowdata[0, 6], p0.begin());
    std::copy(&pars.rowdata[1, 0], &pars.rowdata[1, 6], p1.begin());

    return { p0, p1 };
};
} // namespace irf_private
