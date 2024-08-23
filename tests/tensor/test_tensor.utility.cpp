#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/tensor.hpp"

TEST_SUITE("Fermi::Tensor Utility Methods") {

    // Test the extent() method
    TEST_CASE("Tensor Utility: Extent Method") {
        using Tensor2D = Fermi::Tensor<int, 2>;
        
        Tensor2D tensor({ 3, 4 });

        CHECK(tensor.extent(0) == 3);
        CHECK(tensor.extent(1) == 4);

        // Test out-of-bounds access
        /* CHECK_THROWS_AS(tensor.extent(2), std::out_of_range); */
        /* CHECK_THROWS_AS(tensor.extent(10), std::out_of_range); */
    }

    // Test the total_size() method
    TEST_CASE("Tensor Utility: Total Size Method") {
        using Tensor2D = Fermi::Tensor<int, 2>;
        
        Tensor2D tensor({ 3, 4 });
        
        CHECK(tensor.size() == 12); // 3 * 4 = 12
    }

    // Test the data() method
    TEST_CASE("Tensor Utility: Data Method") {
        using Tensor2D = Fermi::Tensor<int, 2>;

        Tensor2D tensor({ 2, 2 });
        
        int* raw_data = tensor.data();

        CHECK(raw_data != nullptr);

        tensor[0, 0] = 42;
        CHECK(raw_data[0] == 42);
    }

    // Test the extents() method
    TEST_CASE("Tensor Utility: Extents Method") {
        using Tensor2D = Fermi::Tensor<int, 2>;

        Tensor2D tensor({ 3, 4 });
        auto extents = tensor.extents();

        CHECK(extents[0] == 3);
        CHECK(extents[1] == 4);
    }

    // Test the strides() method
    TEST_CASE("Tensor Utility: Strides Method") {
        using Tensor2D = Fermi::Tensor<int, 2>;

        Tensor2D tensor({ 3, 4 });
        auto strides = tensor.strides();

        CHECK(strides[0] == 4); // Row-major order: 4 elements per row
        CHECK(strides[1] == 1); // Single element per column
    }

    // Test the clear() method
    TEST_CASE("Tensor Utility: Clear Method") {
        using Tensor2D = Fermi::Tensor<int, 2>;

        Tensor2D tensor({ 2, 2 });

        tensor[0, 0] = 42;
        tensor[1, 1] = 84;

        tensor.clear();

        CHECK(tensor[0, 0] == 0);
        CHECK(tensor[0, 1] == 0);
        CHECK(tensor[1, 0] == 0);
        CHECK(tensor[1, 1] == 0);
    }
}
