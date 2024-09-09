#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"

#include <array>

namespace Fermi::Irf {

struct IrfData3 {
    Tensor<double, 1> cosths;
    Tensor<double, 1> logEs;
    Tensor<double, 3> params;
    double            minCosTheta;
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

namespace psf {

struct Data {
    IrfData3 rpsf;
    IrfScale psf_scaling_params;
    IrfData3 fisheye_correction;
};

struct Pass8FB {
    psf::Data front;
    psf::Data back;
};
} // namespace psf

namespace aeff {

struct Data {
    IrfData3 effective_area;
    IrfData3 phi_dependence;
    IrfEffic efficiency_params;
};

struct Pass8FB {
    aeff::Data front;
    aeff::Data back;
};
} // namespace aeff

struct XtIrf {
    aeff::Pass8FB     aeff_irf;
    psf::Pass8FB      psf_irf;
    Tensor<double, 2> front_LTF;
};


} // namespace Fermi::Irf
