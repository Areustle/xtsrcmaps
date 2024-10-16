#include <complex>
#include <vector>

#include "fftw3.h"

// Function to perform FFT-based 3D convolution using FFTW
void
fftw_convolve_3d(const double*        image,
                 size_t               C,
                 size_t               H,
                 size_t               W,
                 const double*        kernel,
                 size_t               KH,
                 size_t               KW,
                 std::vector<double>& output) {
    // Determine padded dimensions for "same" padding
    size_t paddedH = H + KH - 1;
    size_t paddedW = W + KW - 1;

    // Size of FFT input/output
    size_t fftSize = C * paddedH * paddedW;

    // Allocate memory for FFT input, output, and kernel in frequency domain
    fftw_complex* image_fft
        = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSize);
    fftw_complex* kernel_fft
        = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSize);
    fftw_complex* result_fft
        = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSize);

    // Plan FFT and inverse FFT
    fftw_plan forward_image = fftw_plan_dft_r2c_3d(
        C, paddedH, paddedW, (double*)image_fft, image_fft, FFTW_ESTIMATE);
    fftw_plan forward_kernel = fftw_plan_dft_r2c_3d(
        C, paddedH, paddedW, (double*)kernel_fft, kernel_fft, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_dft_c2r_3d(
        C, paddedH, paddedW, result_fft, (double*)result_fft, FFTW_ESTIMATE);

    // Zero-pad the input image and kernel
    std::vector<double> padded_image(fftSize, 0.0);
    std::vector<double> padded_kernel(fftSize, 0.0);

    // Copy input image into padded_image
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                padded_image[c * paddedH * paddedW + h * paddedW + w]
                    = image[c * H * W + h * W + w];
            }
        }
    }

    // Copy kernel into padded_kernel
    for (size_t c = 0; c < C; ++c) {
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                padded_kernel[c * paddedH * paddedW + kh * paddedW + kw]
                    = kernel[c * KH * KW + kh * KW + kw];
            }
        }
    }

    // Execute FFT on both the image and the kernel
    fftw_execute_dft_r2c(forward_image, padded_image.data(), image_fft);
    fftw_execute_dft_r2c(forward_kernel, padded_kernel.data(), kernel_fft);

    // Perform element-wise multiplication in the frequency domain
    for (size_t i = 0; i < fftSize; ++i) {
        std::complex<double> img(image_fft[i][0], image_fft[i][1]);
        std::complex<double> ker(kernel_fft[i][0], kernel_fft[i][1]);
        std::complex<double> res = img * ker;
        result_fft[i][0]         = res.real();
        result_fft[i][1]         = res.imag();
    }

    // Execute the inverse FFT to get the spatial domain result
    fftw_execute_dft_c2r(backward, result_fft, output.data());

    // Normalize the output, Scaling by the total number of elements
    double norm_factor = 1.0 / paddedH * paddedW * C;
    for (size_t i = 0; i < fftSize; ++i) { output[i] *= norm_factor; }

    // Free FFTW resources
    fftw_destroy_plan(forward_image);
    fftw_destroy_plan(forward_kernel);
    fftw_destroy_plan(backward);
    fftw_free(image_fft);
    fftw_free(kernel_fft);
    fftw_free(result_fft);
}
