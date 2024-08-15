#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/model_map/mm_cubature/convolve_pixel_psf.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

TEST_CASE("Test float Convolve psf with pixel.") {
    size_t const Nd = 401;
    size_t const Ne = 38;
    auto         mm = Fermi::Tensor<float, 4>({ 1, 1, 1, Ne });
    auto         pw = Fermi::Tensor<float, 3>({ 1, Nd, Ne });
    auto         lt = Fermi::Tensor<float, 3>({ 1, Nd, Ne });
    mm.clear();
    pw.clear();
    lt.clear();

    Fermi::convolve_pixel_psf<float>(
        &mm[0, 0, 0, 0], Ne, &pw[0, 0, 0], &lt[0, 0, 0], { 1.0f, 1.0f, 1.0f });

    CHECK(mm[0, 0, 0, 0] == 0.0f);
}
