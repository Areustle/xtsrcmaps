#include "xtsrcmaps/interp.hxx"
#include "xtsrcmaps/psf/psf.hxx"

#include <algorithm>

auto
Fermi::psf_lerp_slow(const Tensor2d& lut, long eidx, double offset) -> double
{
    PSF::SepArr seps = PSF::separations();
    long        oidx = std::distance(seps.begin(),
                              std::upper_bound(seps.begin(), seps.end(), offset) - 1);
    double      wgt  = offset - oidx;
    return (1. - wgt) * lut(eidx, oidx) + (wgt)*lut(eidx, oidx + 1);
}
