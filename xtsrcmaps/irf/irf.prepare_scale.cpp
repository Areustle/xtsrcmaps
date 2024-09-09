#include "xtsrcmaps/irf/_irf_private.hpp"

#include <cassert>

namespace irf_private {
auto
prepare_scale(Fermi::fits::TablePars const& pars) -> Fermi::Irf::IrfScale {
    assert(pars.rowdata.extent(0) == 1);
    assert(pars.rowdata.extent(1) == 3);

    return { pars.rowdata[0, 0], pars.rowdata[0, 1], pars.rowdata[0, 2] };
};
} // namespace irf_private
