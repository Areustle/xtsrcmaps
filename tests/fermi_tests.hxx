#pragma once

#include <gtest/gtest.h>

#include <filesystem>

#include "xtsrcmaps/tensor/read_file_tensor.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

/* using Tensor1b = Fermi::Tensor<bool, 1>; */
/* using Tensor2b = Fermi::Tensor<bool, 2>; */
/* using Tensor3b = Fermi::Tensor<bool, 3>; */
using Tensor1d = Fermi::Tensor<double, 1>;
using Tensor2d = Fermi::Tensor<double, 2>;
using Tensor3d = Fermi::Tensor<double, 3>;
using Tensor4d = Fermi::Tensor<double, 4>;
using Tensor1f = Fermi::Tensor<float, 1>;
using Tensor2f = Fermi::Tensor<float, 2>;
using Tensor3f = Fermi::Tensor<float, 3>;
using Tensor4f = Fermi::Tensor<float, 4>;

// Custom matcher for relative tolerance comparison
template <typename T = double, typename U = double>
::testing::AssertionResult
NearRelative(T const      a,
             U const      b,
             double const rtol = 1e-5,
             double const atol = 1e-5) {
    double diff = std::fabs(a - b);
    double err  = atol + rtol * std::fabs(b);
    if (diff <= err) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure()
               << "Values " << a << " and " << b << " differ by " << diff
               << ", which exceeds the allowable error of " << err
               << " (rtol: " << rtol << ", atol: " << atol << ")\n";
    }
}

template <typename T1, typename T2>
auto
et2comp(Fermi::Tensor<T1, 2> const& computed,
        std::vector<T2> const&      expected,
        double const                rtol = 1e-5,
        double const                atol = 1e-5) -> void {

    ASSERT_TRUE(NearRelative(computed.size(), expected.size(), rtol, atol)
                << "Tensor and expected vector sizes do not match.");

    Fermi::Tensor<T2, 2> sp_b(expected, computed.extent(0), computed.extent(1));

    for (size_t i = 0; i < computed.extent(0); ++i) {
        for (size_t j = 0; j < computed.extent(1); ++j) {
            auto x = computed[i, j];
            auto y = sp_b[i, j];
            ASSERT_TRUE(NearRelative(y, x, rtol, atol)
                        << "Mismatch at index (" << i << ", " << j << ")");
        }
    }
}


template <typename T = double, typename U = double, size_t Rank>
auto
filecomp(Fermi::Tensor<T, Rank> const& computed,
         std::string const&            filebase,
         double const                  rtol = 1e-5,
         double const                  atol = 1e-5) -> void {

    std::string const filename = "./tests/expected/" + filebase + ".bin";

    ASSERT_TRUE(std::filesystem::exists(filename))
        << "File does not exist: " << filename;

    auto const expected
        = Fermi::read_file_tensor<U, Rank>(filename, computed.extents());

    auto comp_it  = computed.begin();
    auto expe_it  = expected.begin();

    auto comp_end = computed.end();
    auto expe_end = expected.end();

    ASSERT_EQ(std::distance(comp_it, comp_end),
              std::distance(expe_it, expe_end))
        << "The computed and expected tensors have different sizes.";

    for (; comp_it != comp_end && expe_it != expe_end; ++comp_it, ++expe_it) {
        ASSERT_TRUE(NearRelative(*comp_it, *expe_it, rtol, atol)
                    << "Mismatch at index: "
                    << std::distance(computed.begin(), comp_it));
    }
}
