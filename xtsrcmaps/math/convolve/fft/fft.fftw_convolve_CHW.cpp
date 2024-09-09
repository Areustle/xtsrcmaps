#include <vector>

#include "fftw3.h"

// Function to perform FFT-based 3D convolution using FFTW
void
fftw_convolve_3d(const std::vector<double>& image,
                 int                        C,
                 int                        H,
                 int                        W,
                 const std::vector<double>& kernel,
                 int                        KH,
                 int                        KW,
                 std::vector<double>&       output) {
    // Determine padded dimensions for "same" padding
    int paddedH = H + KH - 1;
    int paddedW = W + KW - 1;

    // Size of FFT input/output
    int fftSize = C * paddedH * paddedW;

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
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                padded_image[c * paddedH * paddedW + h * paddedW + w]
                    = image[c * H * W + h * W + w];
            }
        }
    }

    // Copy kernel into padded_kernel
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                padded_kernel[c * paddedH * paddedW + kh * paddedW + kw]
                    = kernel[c * KH * KW + kh * KW + kw];
            }
        }
    }

    // Execute FFT on both the image and the kernel
    fftw_execute_dft_r2c(forward_image, padded_image.data(), image_fft);
    fftw_execute_dft_r2c(forward_kernel, padded_kernel.data(), kernel_fft);
}
