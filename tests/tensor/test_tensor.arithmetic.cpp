#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/tensor.hpp"

TEST_CASE("Tensor element-wise addition") {
    using Tensor2I = Fermi::Tensor<int, 2, false>;
    using Tensor2F = Fermi::Tensor<float, 2, false>;

    std::array<std::size_t, 2> extents = {2, 3};

    Tensor2I tensor1(extents);
    Tensor2F tensor2(extents);

    // Fill tensors with values
    std::iota(tensor1.begin(), tensor1.end(), 1); // [1, 2, 3, 4, 5, 6]
    std::iota(tensor2.begin(), tensor2.end(), 0.5f); // [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    // Perform element-wise addition
    auto result = tensor1 + tensor2;

    // Expected result after addition: [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]
    std::vector<float> expected_result = {1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f};

    CHECK(result.extents() == extents);

    for (std::size_t i = 0; i < result.size(); ++i) {
        CHECK(result.data()[i] == doctest::Approx(expected_result[i]));
    }
}
