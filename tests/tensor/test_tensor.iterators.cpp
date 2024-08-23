#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/tensor.hpp"

TEST_CASE("Tensor Iterator Methods") {
    using Tensor2D = Fermi::Tensor<int, 2>;

    // Initialize a 2x3 tensor with values 1, 2, 3, 4, 5, 6
    // 1 2 3
    // 4 5 6
    const Tensor2D tensor(std::vector<int> { 1, 2, 3, 4, 5, 6 }, { 2, 3 });

    SUBCASE("Testing begin and end iterators") {
        auto it_begin = tensor.begin();
        auto it_end   = tensor.end();

        CHECK(it_begin != it_end);
        CHECK(*it_begin == 1);

        ++it_begin;
        CHECK(*it_begin == 2);

        --it_end;
        CHECK(*it_end == 6);
    }

    SUBCASE("Testing cbegin and cend iterators") {
        auto it_cbegin = tensor.cbegin();
        auto it_cend   = tensor.cend();

        CHECK(it_cbegin != it_cend);
        CHECK(*it_cbegin == 1);

        ++it_cbegin;
        CHECK(*it_cbegin == 2);

        --it_cend;
        CHECK(*it_cend == 6);
    }

    SUBCASE("Testing rbegin and rend iterators") {
        auto it_rbegin = tensor.rbegin();
        auto it_rend   = tensor.rend();

        CHECK(it_rbegin != it_rend);
        CHECK(*it_rbegin == 6);

        ++it_rbegin;
        CHECK(*it_rbegin == 5);

        --it_rend;
        CHECK(*it_rend == 1);
    }

    SUBCASE("Testing crbegin and crend iterators") {
        auto it_crbegin = tensor.crbegin();
        auto it_crend   = tensor.crend();

        CHECK(it_crbegin != it_crend);
        CHECK(*it_crbegin == 6);

        ++it_crbegin;
        CHECK(*it_crbegin == 5);

        --it_crend;
        CHECK(*it_crend == 1);
    }

    SUBCASE("Testing begin_at and end_at methods") {
        auto it = tensor.begin_at(1, 2);
        CHECK(*it == 6);

        auto it_end = tensor.end_at(0, 3);
        CHECK(std::distance(tensor.begin(), it_end) == 3);
        it_end = tensor.end_at(1, 3);
        CHECK(std::distance(tensor.begin(), it_end) == 6);
    }
}
