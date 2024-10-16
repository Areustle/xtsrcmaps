#include "xtsrcmaps/tensor/tensor.hpp"
#include "fftw3.h"
#include <vector>

namespace Fermi::FFT {

Tensor<double, 3>
convolve_map_psf(const Tensor<double, 3>& highresgaldiff,
                 const Tensor<double, 3>& kernel) {
    // Extract dimensions from the input tensors
    auto [C, H, W]    = highresgaldiff.extents();
    auto [KC, KH, KW] = kernel.extents();

    // Ensure channel dimensions match
    if (C != KC) {
        throw std::invalid_argument(
            "Channel dimensions of image and kernel must match.");
    }

    // Extract data from the tensors
    const double* image_data  = highresgaldiff.data();
    const double* kernel_data = kernel.data();

    // Prepare the output data container
    std::vector<double> output_data(C * (H + KH - 1) * (W + KW - 1), 0.0);

    // Call the FFTW convolution function
    fftw_convolve_3d(image_data, C, H, W, kernel_data, KH, KW, output_data);

    // Determine the extents of the output tensor
    std::array<size_t, 3> output_extents = { C, H + KH - 1, W + KW - 1 };

    // Create the output tensor from the convolved data
    Tensor<double, 3> model_map(output_data, output_extents);

    return model_map;
}

} // namespace Fermi::FFT
