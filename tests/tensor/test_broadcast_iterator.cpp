#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <numeric>
#include <vector>

TEST_SUITE("BroadcastIterator Tests") {
    constexpr std::size_t R = 3;
    using ExtentsType       = std::array<std::size_t, R>;
    using IndicesType       = std::array<long, R>;

    TEST_CASE("Increment Index - Simple Case") {
        std::vector<int> data(6);
        std::iota(data.begin(), data.end(), 0);

        auto tensor = Fermi::Tensor(data, 1, 1, 6);
        auto bcastr = tensor.broadcast(1, 2, 6);
        auto it     = bcastr.begin();

        // 0 1 2 3 4 5
        //
        // 0 1 2 3 4 5
        // 0 1 2 3 4 5


        CHECK(it.extents() == ExtentsType { 1, 2, 6 });
        CHECK(it.strides() == ExtentsType { 0, 0, 1 });
        CHECK(it.indices() == IndicesType { 0, 0, 0 });

        for (long i = 0; i < 1; ++i) {
            for (long j = 0; j < 2; ++j) {
                for (long k = 0; k < 6; ++k) {
                    CHECK(it.indices() == IndicesType { i, j, k });
                    CHECK(bcastr[i, j, k] == tensor[i, j % 1, k % 6]);
                    CHECK(*it == tensor[i, j % 1, k % 6]);
                    ++it;
                }
            }
        }

        it = bcastr.begin();
        for (size_t i = 0; i < 1; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 6; ++k) {
                    CHECK(*it++ == tensor[i, j % 1, k % 6]);
                }
            }
        }
        it = bcastr.begin();
        for (size_t i = 0; i < 1; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 6; ++k) {
                    CHECK(*it == tensor[i, j % 1, k % 6]);
                    it += 1;
                }
            }
        }

        it = bcastr.begin();
        for (size_t i = 0; i < 1; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 6; ++k) {
                    CHECK(*it == tensor[i, j % 1, k % 6]);
                    it = it + 1;
                }
            }
        }
    }

    TEST_CASE("Increment Index - Strided Case") {
        std::vector<int> data(6);
        std::iota(data.begin(), data.end(), 0);

        auto tensor = Fermi::Tensor(data, 1, 6, 1);
        auto bcastr = tensor.broadcast(4, 6, 2);
        auto it     = bcastr.begin();

        // 0
        // 1
        // 2
        // 3
        // 4
        // 5
        //
        // 0 0    0 0    0 0    0 0
        // 1 1    1 1    1 1    1 1
        // 2 2    2 2    2 2    2 2
        // 3 3    3 3    3 3    3 3
        // 4 4    4 4    4 4    4 4
        // 5 5    5 5    5 5    5 5


        CHECK(it.extents() == ExtentsType { 4, 6, 2 });
        CHECK(it.strides() == ExtentsType { 0, 1, 0 });
        CHECK(it.indices() == IndicesType { 0, 0, 0 });

        for (long i = 0; i < 4; ++i) {
            for (long j = 0; j < 6; ++j) {
                for (long k = 0; k < 2; ++k) {
                    CHECK(it.indices() == IndicesType { i, j, k });
                    CHECK(bcastr[i, j, k] == tensor[i % 1, j % 6, k % 1]);
                    CHECK(*it++ == tensor[i % 1, j % 6, k % 1]);
                }
            }
        }
    }

    TEST_CASE("Decrement Index - Simple Case") {
        std::vector<int> data(6);
        std::iota(data.begin(), data.end(), 0);

        auto tensor = Fermi::Tensor(data, 1, 1, 6);
        auto bcastr = tensor.broadcast(1, 2, 6);
        auto it     = bcastr.end(); // Start at the end iterator
        CHECK(it.extents() == ExtentsType { 1, 2, 6 });
        CHECK(it.strides() == ExtentsType { 0, 0, 1 });
        CHECK(it.indices() == IndicesType { 1, 0, 0 });

        // 0 1 2 3 4 5
        //
        // 0 1 2 3 4 5
        // 0 1 2 3 4 5

        // Decrement the iterator to start at the last valid index
        --it;

        CHECK(it.indices() == IndicesType { 0, 1, 5 });
        CHECK(it.indices()[0] == 0);
        CHECK(it.indices()[1] == 1);
        CHECK(it.indices()[2] == 5);
        CHECK(it.parent_indices() == IndicesType { 0, 0, 5 });
        CHECK(it.parent_indices()[0] == 0);
        CHECK(it.parent_indices()[1] == 0);
        CHECK(it.parent_indices()[2] == 5);
        CHECK(*it == 5);
        --it;
        CHECK(it.indices() == IndicesType { 0, 1, 4 });
        CHECK(it.indices()[0] == 0);
        CHECK(it.indices()[1] == 1);
        CHECK(it.indices()[2] == 4);
        CHECK(*it == 4);

        it = bcastr.end();
        for (long i = 1; i > 0; --i) {
            for (long j = 2; j > 0; --j) {
                for (long k = 6; k > 0; --k) {
                    --it;
                    CHECK(it.indices() == IndicesType { i - 1, j - 1, k - 1 });
                    CHECK(it.parent_indices()
                          == IndicesType { i - 1, (j - 1) % 1, (k - 1) % 6 });
                    CHECK(bcastr[it.indices()] == tensor[it.parent_indices()]);
                    CHECK_MESSAGE((*it == tensor[it.parent_indices()]),
                                  (i - 1) << " " << (j - 1) << " " << (k - 1)
                                          << " ");
                    CHECK(*it == tensor[it.parent_indices()]);
                }
            }
        }
    }

    /* TEST_CASE("Decrement Index - Simple Case") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 1, 2, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     --it; */
    /**/
    /*     CHECK(it[0] == 1); */
    /*     CHECK(it[1] == 2); */
    /*     CHECK(it[2] == 2); */
    /* } */
    /**/
    /* TEST_CASE("Add to Index") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 0, 0, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     it += 4; */
    /**/
    /*     CHECK(it[0] == 1); */
    /*     CHECK(it[1] == 2); */
    /*     CHECK(it[2] == 3); */
    /* } */
    /**/
    /* TEST_CASE("Subtract from Index") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 1, 2, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     it -= 2; */
    /**/
    /*     CHECK(it[0] == 1); */
    /*     CHECK(it[1] == 1); */
    /*     CHECK(it[2] == 2); */
    /* } */
    /**/
    /* TEST_CASE("STL Algorithm - std::copy") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 0, 0, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     std::vector<int> copy(6); */
    /**/
    /*     std::copy(it, it + 6, copy.begin()); */
    /**/
    /*     CHECK(copy[0] == 1); */
    /*     CHECK(copy[1] == 2); */
    /*     CHECK(copy[2] == 3); */
    /*     CHECK(copy[3] == 4); */
    /*     CHECK(copy[4] == 5); */
    /*     CHECK(copy[5] == 6); */
    /* } */
    /**/
    /* TEST_CASE("STL Algorithm - std::reverse") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 0, 0, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     std::reverse(it, it + 6); */
    /**/
    /*     CHECK(it[0] == 6); */
    /*     CHECK(it[1] == 5); */
    /*     CHECK(it[2] == 4); */
    /*     CHECK(it[3] == 3); */
    /*     CHECK(it[4] == 2); */
    /*     CHECK(it[5] == 1); */
    /* } */
    /**/
    /* TEST_CASE("Edge Case - Maximum Index") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 1, 2, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     ++it; */
    /**/
    /*     CHECK(it[0] == 6); // Expect wrapping or the appropriate element */
    /* } */
    /**/
    /* TEST_CASE("Edge Case - Minimum Index") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 0, 0, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     --it; */
    /**/
    /*     CHECK(it[0] == 1); // Expect wrapping or the appropriate element */
    /* } */
    /**/
    /* TEST_CASE("Large Value Operations - Add N") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 0, 0, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     it += 100; */
    /**/
    /*     CHECK(it[0] == 1); // Depends on the wrapping behavior */
    /* } */
    /**/
    /* TEST_CASE("Large Value Operations - Subtract N") { */
    /*     std::vector<int> data(6); */
    /*     std::iota(data.begin(), data.end(), 1); */
    /**/
    /*     ExtentsType broadcast_extents = { 2, 3, 1 }; */
    /*     ExtentsType memory_strides    = { 3, 1, 1 }; */
    /*     ExtentsType broadcast_indices = { 0, 0, 0 }; */
    /**/
    /*     Fermi::BroadcastIterator<int, R> it( */
    /*         data.data(), broadcast_extents, memory_strides,
     * broadcast_indices); */
    /*     it -= 100; */
    /**/
    /*     CHECK(it[0] == 1); // Depends on the wrapping behavior */
    /* } */
}
