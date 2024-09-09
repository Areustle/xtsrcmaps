#include <iostream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
// Example function to perform a simple convolution using BNNS
void
convolve(const float* input,
         int          H,
         int          W,
         int          C,
         const float* kernel,
         int          KH,
         int          KW,
         int          KC,
         float*       output) {
    // Check that the channel dimensions match
    if (C != KC) {
        throw std::invalid_argument(
            "Input and kernel must have the same number of channels.");
    }

    // Define the descriptors for input, kernel, and output tensors using
    // BNNSDataLayout3DFirstMajor
    BNNSNDArrayDescriptor input_desc = {
        .layout    = BNNSDataLayout3DFirstMajor,
        .size      = {     static_cast<size_t>(H),
                      static_cast<size_t>(W),
                      static_cast<size_t>(C)                                },
        .stride    = { static_cast<size_t>(W * C), static_cast<size_t>(C), 1 },
        .data      = const_cast<float*>(input), // Cast input to void*
        .data_type = BNNSDataTypeFloat32,
    };

    BNNSNDArrayDescriptor kernel_desc = {
        .layout = BNNSDataLayout3DFirstMajor,
        .size   = {      static_cast<size_t>(KH),
                   static_cast<size_t>(KW),
                   static_cast<size_t>(KC)                                  },
        .stride = { static_cast<size_t>(KW * KC), static_cast<size_t>(KC), 1 },
        .data   = const_cast<float*>(kernel), // Cast kernel to void*
        .data_type = BNNSDataTypeFloat32,
    };

    BNNSNDArrayDescriptor output_desc = {
        .layout    = BNNSDataLayout3DFirstMajor,
        .size      = {     static_cast<size_t>(H),
                      static_cast<size_t>(W),
                      static_cast<size_t>(C)                                },
        .stride    = { static_cast<size_t>(W * C), static_cast<size_t>(C), 1 },
        .data      = output, // Output already fits void*
        .data_type = BNNSDataTypeFloat32,
    };

    // Convolution layer parameters
    BNNSLayerParametersConvolution conv_params = {
        .i_desc = input_desc,
        .w_desc = kernel_desc,
        .o_desc = output_desc,
        .bias   = { .data = nullptr }, // No bias
        .activation
        = { .function = BNNSActivationFunctionIdentity }, // No activation
        .x_stride          = 1,
        .y_stride          = 1,
        .x_dilation_stride = 1,
        .y_dilation_stride = 1,
        .x_padding         = 0,
        .y_padding         = 0,
        .groups            = 1 // Number of groups for grouped convolution
    };

    // Create the convolution filter
    BNNSFilter conv_filter
        = BNNSFilterCreateLayerConvolution(&conv_params, nullptr);
    if (!conv_filter) {
        std::cerr << "Failed to create BNNS convolution layer." << std::endl;
        return;
    }

    // Apply the convolution filter
    BNNSFilterApply(conv_filter, input, &output);

    // Clean up the filter
    BNNSFilterDestroy(conv_filter);
}
#else
#endif
