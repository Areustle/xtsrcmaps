#pragma once

#include "doctest/doctest.h"

#include <fstream>
#include <utility>
#include <vector>

#include "xtsrcmaps/tensor_types.hxx"

auto
veq(auto v1, auto v2) -> void
{
    REQUIRE(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); ++i) REQUIRE(v1[i] == doctest::Approx(v2[i]));
}

template <typename T>
auto
md2comp(mdarray2 const& computed, std::vector<T> const& expected) -> void
{
    REQUIRE(computed.size() == expected.size());
    auto sp_b = std::experimental::mdspan(
        expected.data(), computed.extent(0), computed.extent(1));
    for (size_t i = 0; i < computed.extent(0); ++i)
        for (size_t j = 0; j < computed.extent(1); ++j)
            REQUIRE_MESSAGE(computed(i, j) == doctest::Approx(sp_b(i, j)),
                            i << " " << j);
}

template <typename T = double>
auto
filecomp2(mdarray2 const& computed, std::string const& filebase) -> void
{
    const size_t sz_exp = computed.extent(0) * computed.extent(1);
    REQUIRE(computed.size() == sz_exp);
    auto expected = std::vector<T>(sz_exp);

    std::ifstream ifs("./xtsrcmaps/tests/expected/" + filebase + ".bin",
                      std::ios::in | std::ios::binary);
    ifs.read((char*)(&expected[0]), sizeof(T) * sz_exp);
    ifs.close();

    // md2comp(computed, expected);
    REQUIRE(computed.size() == expected.size());
    auto sp_b = std::experimental::mdspan(
        expected.data(), computed.extent(0), computed.extent(1));
    for (size_t i = 0; i < computed.extent(0); ++i)
        for (size_t j = 0; j < computed.extent(1); ++j)
            REQUIRE_MESSAGE(computed(i, j) == doctest::Approx(sp_b(i, j)),
                            i << " " << j << " " << filebase);
}


template <typename T = double>
auto
filecomp3(mdarray3 const& computed, std::string const& filebase) -> void
{
    const size_t sz_exp = computed.extent(0) * computed.extent(1) * computed.extent(2);
    REQUIRE(computed.size() == sz_exp);
    auto expected = std::vector<T>(sz_exp);

    std::ifstream ifs("./xtsrcmaps/tests/expected/" + filebase + ".bin",
                      std::ios::in | std::ios::binary);
    ifs.read((char*)(&expected[0]), sizeof(T) * sz_exp);
    ifs.close();

    // md2comp(computed, expected);
    REQUIRE(computed.size() == expected.size());
    auto sp_b = std::experimental::mdspan(
        expected.data(), computed.extent(0), computed.extent(1), computed.extent(2));
    for (size_t i = 0; i < computed.extent(0); ++i)
        for (size_t j = 0; j < computed.extent(1); ++j)
            for (size_t k = 0; k < computed.extent(2); ++k)
                REQUIRE_MESSAGE(computed(i, j, k) == doctest::Approx(sp_b(i, j, k)),
                                i << " " << j << " " << k << " " << filebase);
}
