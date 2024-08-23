#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <array>
#include <numeric>
// Unit tests for broadcasting functionality

TEST_SUITE("Tensor Broadcasting") {

    TEST_CASE("Broadcast 2D Tensor") {
        using Tensor2I = Fermi::Tensor<int, 2>;

        Tensor2I tensor({ 1, 9 });
        std::iota(tensor.begin(), tensor.end(), 0);

        std::array<std::size_t, 2> broadcast_extents = { 5, 9 };
        auto broadcast_tensor = tensor.broadcast(broadcast_extents);

        CHECK(broadcast_tensor.extent(0) == 5);
        CHECK(broadcast_tensor.extent(1) == 9);

        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 9; ++j) {
                CHECK(broadcast_tensor[i, j] == tensor[0, j]);
            }
        }
    }

    TEST_CASE("Broadcast 1D Tensor") {
        using Tensor1I = Fermi::Tensor<int, 1>;

        // Test broadcasting from {3} to {3}
        Tensor1I tensor({ 3 });
        std::iota(tensor.begin(), tensor.end(), 1); // 1, 2, 3

        std::array<std::size_t, 1> broadcast_extents = { 3 };
        auto broadcast_tensor = tensor.broadcast(broadcast_extents);

        CHECK(broadcast_tensor.extent(0) == 3);

        for (std::size_t i = 0; i < 3; ++i) {
            CHECK(broadcast_tensor[i] == tensor[i]);
        }
    }

    TEST_CASE("Broadcast 3D Tensor") {
        using Tensor3I = Fermi::Tensor<int, 3>;

        // Test broadcasting from {1, 1, 5} to {2, 3, 5}
        Tensor3I tensor({ 1, 1, 5 });
        std::iota(tensor.begin(), tensor.end(), 0);

        std::array<std::size_t, 3> broadcast_extents = { 2, 3, 5 };
        auto broadcast_tensor = tensor.broadcast(broadcast_extents);

        CHECK(broadcast_tensor.extent(0) == 2);
        CHECK(broadcast_tensor.extent(1) == 3);
        CHECK(broadcast_tensor.extent(2) == 5);

        for (long i = 0; i < 2; ++i) {
            for (long j = 0; j < 3; ++j) {
                for (long k = 0; k < 5; ++k) {
                    CHECK(broadcast_tensor[{ i, j, k }] == tensor[{ 0, 0, k }]);
                }
            }
        }
    }

    TEST_CASE("Broadcast and Modify Broadcasted Tensor") {
        using Tensor2I = Fermi::Tensor<int, 2>;

        Tensor2I tensor({ 1, 5 });
        std::iota(tensor.begin(), tensor.end(), 1);

        auto broadcast_tensor = tensor.broadcast({ 3, 5 });

        CHECK(broadcast_tensor.extent(0) == 3);
        CHECK(broadcast_tensor.extent(1) == 5);

        for (long i = 0; i < 3; ++i) {
            for (long j = 0; j < 5; ++j) {
                CHECK(broadcast_tensor[{ i, j }] == tensor[{ 0, j }]);
            }
        }

        // Modify the broadcasted tensor and check the original tensor
        broadcast_tensor[{ 2, 4 }] = 99;
        CHECK(tensor[{ 0, 4 }] == 99);
    }
}
