#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/tensor.hpp"

TEST_SUITE("Tensor Reshape Tests") {

    TEST_CASE("Reshape to Same Number of Elements") {
        using Tensor2I = Fermi::Tensor<int, 2>;

        // Create a 3x4 tensor
        Tensor2I tensor({ 3, 4 });

        // Initialize the tensor with sequential values
        std::iota(tensor.begin(),
                  tensor.end(),
                  0); // Fill with values starting from 0

        // Reshape to 2x6x2
        auto reshaped = tensor.reshape<2>({ 2, 6 });

        // Check the extents
        CHECK(reshaped.extent(0) == 2);
        CHECK(reshaped.extent(1) == 6);

        // Check the contents
        int value = 0;
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 6; ++j) {
                CHECK(reshaped[i, j] == value++);
            }
        }
    }
    TEST_CASE("Reshape to Same Number of Elements") {
        using Tensor3I = Fermi::Tensor<int, 3>;

        // Create a 3x4x2 tensor
        Tensor3I tensor({ 3, 4, 2 });

        // Initialize the tensor with sequential values
        int value = 1;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                for (std::size_t k = 0; k < 2; ++k) {
                    tensor[i, j, k] = value++;
                }
            }
        }

        // Reshape to 2x6x2
        auto reshaped = tensor.reshape<3>({ 2, 6, 2 });

        // Check the extents
        CHECK(reshaped.extent(0) == 2);
        CHECK(reshaped.extent(1) == 6);
        CHECK(reshaped.extent(2) == 2);

        // Check the contents
        value = 1;
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 6; ++j) {
                for (std::size_t k = 0; k < 2; ++k) {
                    CHECK(reshaped[i, j, k] == value++);
                }
            }
        }
    }

    TEST_CASE("Invalid Reshape with Different Number of Elements") {
        using Tensor3D = Fermi::Tensor<int, 3>;

        // Create a 3x4x2 tensor
        Tensor3D tensor({ 3, 4, 2 });

        // Attempt to reshape to an invalid size
        /* CHECK_THROWS_AS(tensor.reshape<2>({ 4, 5 }), std::invalid_argument); */
    }

    /* TEST_CASE("Reshape with Broadcasting") { */
    /*     using Tensor2D = Fermi::Tensor<int, 2>; */
    /**/
    /*     // Create a 1x3 tensor */
    /*     Tensor2D tensor({ 1, 3 }); */
    /**/
    /*     // Initialize the tensor */
    /*     tensor[0, 0]  = 1; */
    /*     tensor[0, 1]  = 2; */
    /*     tensor[0, 2]  = 3; */
    /**/
    /*     // Reshape to 3x3 (expecting to broadcast the first dimension) */
    /*     auto reshaped = tensor.reshape<2>({ 3, 3 }).broadcast(3, 3); */
    /**/
    /*     // Check the extents */
    /*     CHECK(reshaped.extent(0) == 3); */
    /*     CHECK(reshaped.extent(1) == 3); */
    /**/
    /*     // Check the contents */
    /*     for (std::size_t i = 0; i < 3; ++i) { */
    /*         CHECK(reshaped[i, 0] == 1); */
    /*         CHECK(reshaped[i, 1] == 2); */
    /*         CHECK(reshaped[i, 2] == 3); */
    /*     } */
    /* } */

    TEST_CASE("Reshape with Reduction in Rank") {
        using Tensor3D = Fermi::Tensor<int, 3>;

        // Create a 3x4x2 tensor
        Tensor3D tensor({ 3, 4, 2 });

        // Initialize the tensor with sequential values
        int value = 1;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                for (std::size_t k = 0; k < 2; ++k) {
                    tensor[i, j, k] = value++;
                }
            }
        }

        // Reshape to 24x1
        auto reshaped = tensor.reshape<2>({ 24, 1 });

        // Check the extents
        CHECK(reshaped.extent(0) == 24);
        CHECK(reshaped.extent(1) == 1);

        // Check the contents
        value = 1;
        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(reshaped[i, 0] == value++);
        }
    }

    TEST_CASE("Reshape with Increase in Rank") {
        using Tensor2D = Fermi::Tensor<int, 2>;

        // Create a 4x3 tensor
        Tensor2D tensor({ 4, 3 });

        // Initialize the tensor with sequential values
        int value = 1;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 3; ++j) { tensor[i, j] = value++; }
        }

        // Reshape to 2x2x3
        auto reshaped = tensor.reshape<3>({ 2, 2, 3 });

        // Check the extents
        CHECK(reshaped.extent(0) == 2);
        CHECK(reshaped.extent(1) == 2);
        CHECK(reshaped.extent(2) == 3);

        // Check the contents
        value = 1;
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                for (std::size_t k = 0; k < 3; ++k) {
                    CHECK(reshaped[i, j, k] == value++);
                }
            }
        }
    }
}
