#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/tensor.hpp"
#include <vector>

// Basic construction tests
TEST_CASE("Tensor: Basic Construction with extents") {
    using Tensor2D = Fermi::Tensor<int, 2>;

    SUBCASE("Construction with extents only") {
        Tensor2D tensor({2, 3});
        CHECK(tensor.extent(0) == 2);
        CHECK(tensor.extent(1) == 3);
        CHECK(tensor.size() == 6);
    }

    SUBCASE("Construction with variadic extents") {
        Tensor2D tensor(2, 3);
        CHECK(tensor.extent(0) == 2);
        CHECK(tensor.extent(1) == 3);
        CHECK(tensor.size() == 6);
    }
}

// Test construction from iterators
TEST_CASE("Tensor: Construction from iterators") {
    using Tensor2D = Fermi::Tensor<int, 2>;

    std::vector<int> data = {1, 2, 3, 4, 5, 6};

    SUBCASE("Construction from iterator range") {
        Tensor2D tensor(data.begin(), data.end(), {2, 3});
        CHECK(tensor.extent(0) == 2);
        CHECK(tensor.extent(1) == 3);
        CHECK(tensor.size() == 6);
        CHECK(tensor[0, 0] == 1);
        CHECK(tensor[1, 2] == 6);
    }
}

// Test construction from unaligned vector
TEST_CASE("Tensor: Construction from unaligned vector") {
    using Tensor2D = Fermi::Tensor<int, 2>;

    std::vector<int> data = {1, 2, 3, 4, 5, 6};

    SUBCASE("Construction from unaligned vector") {
        Tensor2D tensor(data, {2, 3});
        CHECK(tensor.extent(0) == 2);
        CHECK(tensor.extent(1) == 3);
        CHECK(tensor.size() == 6);
        CHECK(tensor[0, 0] == 1);
        CHECK(tensor[1, 2] == 6);
    }

    SUBCASE("Construction from unaligned vector with variadic extents") {
        Tensor2D tensor(data, 2, 3);
        CHECK(tensor.extent(0) == 2);
        CHECK(tensor.extent(1) == 3);
        CHECK(tensor.size() == 6);
        CHECK(tensor[0, 0] == 1);
        CHECK(tensor[1, 2] == 6);
    }
}

/* // Test invalid constructions */
/* TEST_CASE("Tensor: Invalid Construction Tests") { */
/*     using Tensor2D = Fermi::Tensor<int, 2>; */
/*     std::vector<int> data = {1, 2, 3, 4, 5}; */
/**/
/*     SUBCASE("Mismatched data size and extents") { */
/*         CHECK_THROWS_AS(Tensor2D(data.begin(), data.end(), {2, 3}), std::invalid_argument); */
/*     } */
/**/
/*     SUBCASE("Mismatched data size and extents with strides") { */
/*         CHECK_THROWS_AS(Tensor2D(data.begin(), data.end(), {2, 3}, {3, 1}), std::invalid_argument); */
/*     } */
/**/
/*     SUBCASE("Mismatched data size in unaligned vector construction") { */
/*         CHECK_THROWS_AS(Tensor2D(data, {2, 3}), std::invalid_argument); */
/*     } */
/* } */
