#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/tensor/tensor.hpp"

TEST_SUITE("Fermi::Tensor Subscript Operations") {

    // 2D Tensor Subscripting Tests
    TEST_CASE("Tensor Subscript: 2D Tensor") {
        using Tensor2D = Fermi::Tensor<int, 2>;

        Tensor2D tensor({ 3, 4 });

        // Fill the tensor with sequential values
        int value = 1;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) { tensor[i, j] = value++; }
        }

        // Check access via ExtentsType
        CHECK(tensor[{ 0, 0 }] == 1);
        CHECK(tensor[{ 1, 1 }] == 6);
        CHECK(tensor[{ 2, 3 }] == 12);

        // Check access via multiple indices
        CHECK(tensor[0, 0] == 1);
        CHECK(tensor[1, 1] == 6);
        CHECK(tensor[2, 3] == 12);

        // Check that data is modifiable
        tensor[0, 0] = 100;
        CHECK(tensor[0, 0] == 100);
    }

    // 3D Tensor Subscripting Tests
    TEST_CASE("Tensor Subscript: 3D Tensor") {
        using Tensor3D = Fermi::Tensor<int, 3>;

        Tensor3D tensor({ 2, 3, 4 });

        // Fill the tensor with sequential values
        int value = 1;
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t k = 0; k < 4; ++k) {
                    tensor[i, j, k] = value++;
                }
            }
        }

        // Check access via ExtentsType
        CHECK(tensor[{ 0, 0, 0 }] == 1);
        CHECK(tensor[{ 1, 2, 3 }] == 24);

        // Check access via multiple indices
        CHECK(tensor[0, 0, 0] == 1);
        CHECK(tensor[1, 2, 3] == 24);

        // Check that data is modifiable
        tensor[1, 2, 3] = 200;
        CHECK(tensor[1, 2, 3] == 200);
    }

    // 4D Tensor Subscripting Tests
    TEST_CASE("Tensor Subscript: 4D Tensor") {
        using Tensor4D = Fermi::Tensor<int, 4>;

        Tensor4D tensor({ 2, 3, 4, 5 });

        // Fill the tensor with sequential values
        int value = 1;
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t k = 0; k < 4; ++k) {
                    for (std::size_t l = 0; l < 5; ++l) {
                        tensor[i, j, k, l] = value++;
                    }
                }
            }
        }

        // Check access via ExtentsType
        CHECK(tensor[{ 0, 0, 0, 0 }] == 1);
        CHECK(tensor[{ 1, 2, 3, 4 }] == 120);

        // Check access via multiple indices
        CHECK(tensor[0, 0, 0, 0] == 1);
        CHECK(tensor[1, 2, 3, 4] == 120);

        // Check that data is modifiable
        tensor[1, 2, 3, 4] = 300;
        CHECK(tensor[1, 2, 3, 4] == 300);
    }

    // Test const correctness
    TEST_CASE("Tensor Subscript: Const Correctness") {
        using Tensor2D = Fermi::Tensor<int, 2>;

        const Tensor2D tensor({ 2, 2 });

        // Ensure that the const Tensor can still be accessed
        CHECK(tensor[0, 0] == 0);
        CHECK(tensor[1, 1] == 0);

        // Ensure that writing to const Tensor causes compilation error
        // tensor[0, 0] = 10; // Uncommenting this line should cause a
        // compilation error
    }

    /* // Test bounds checking */
    /* TEST_CASE("Tensor Subscript: Bounds Checking") { */
    /*     using Tensor2D = Fermi::Tensor<int, 2>; */
    /**/
    /*     Tensor2D tensor({ 3, 3 }); */
    /**/
    /*     CHECK_THROWS_AS(tensor[3, 0], std::out_of_range); */
    /*     CHECK_THROWS_AS(tensor[0, 3], std::out_of_range); */
    /*     CHECK_THROWS_AS(tensor[4, 4], std::out_of_range); */
    /* } */
}
