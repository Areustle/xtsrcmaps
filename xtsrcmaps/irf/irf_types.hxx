#pragma once

#include "xtsrcmaps/math/tensor_types.hxx"

#include <array>

namespace Fermi {

struct IrfData3 {
    Tensor1d cosths;
    Tensor1d logEs;
    Tensor3d params;
    double   minCosTheta;
};

struct IrfScale {
    float scale0;
    float scale1;
    float scale_index;
};

struct IrfEffic {
    std::array<float, 6> p0;
    std::array<float, 6> p1;
};

namespace irf::psf {

struct Data {
    IrfData3 rpsf;
    IrfScale psf_scaling_params;
    IrfData3 fisheye_correction;
};

struct Pass8FB {
    irf::psf::Data front;
    irf::psf::Data back;
};
} // namespace irf::psf

namespace irf::aeff {

struct Data {
    IrfData3 effective_area;
    IrfData3 phi_dependence;
    IrfEffic efficiency_params;
};

struct Pass8FB {
    irf::aeff::Data front;
    irf::aeff::Data back;
};
} // namespace irf::aeff

struct XtIrf {
    irf::aeff::Pass8FB                                  aeff_irf;
    irf::psf::Pass8FB                                   psf_irf;
    std::pair<std::vector<double>, std::vector<double>> front_LTF;
};


} // namespace Fermi
