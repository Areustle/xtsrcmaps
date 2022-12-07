#pragma once

#include "doctest/doctest.h"

#include <fstream>
#include <utility>
#include <vector>

#include "xtsrcmaps/tensor_ops.hxx"
#include "xtsrcmaps/tensor_types.hxx"

auto
veq(auto v1, auto v2) -> void
{
    REQUIRE(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); ++i) REQUIRE(v1[i] == doctest::Approx(v2[i]));
}

template <typename T>
auto
et2comp(Tensor2d const& computed, std::vector<T> const& expected) -> void
{
    REQUIRE(computed.size() == expected.size());
    TensorMap<Tensor<T, 2> const> sp_b(
        expected.data(), computed.dimension(0), computed.dimension(1));

    for (long j = 0; j < computed.dimension(1); ++j)
        for (long i = 0; i < computed.dimension(0); ++i)
            REQUIRE_MESSAGE(computed(i, j) == doctest::Approx(sp_b(i, j)),
                            i << " " << j);
}

template <typename T>
auto
et2comp_exprm(Tensor2d const& computed, std::vector<T> const& expected) -> void
{
    REQUIRE(computed.size() == expected.size());
    TensorMap<Tensor<T, 2, Eigen::RowMajor> const> sp_b(
        expected.data(), computed.dimension(1), computed.dimension(0));

    for (long j = 0; j < computed.dimension(1); ++j)
        for (long i = 0; i < computed.dimension(0); ++i)
            REQUIRE_MESSAGE(computed(i, j) == doctest::Approx(sp_b(j, i)),
                            i << " " << j);
}

template <typename T = double>
auto
filecomp2(Tensor2d const& computed, std::string const& filebase) -> void
{
    const size_t sz_exp   = computed.size();
    auto         expected = std::vector<T>(sz_exp);

    Tensor2d const sp_b   = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/" + filebase + ".bin",
        computed.dimension(1),
        computed.dimension(0));

    for (long j = 0; j < computed.dimension(1); ++j)
        for (long i = 0; i < computed.dimension(0); ++i)
            REQUIRE_MESSAGE(computed(i, j) == doctest::Approx(sp_b(i, j)),
                            i << " " << j << " " << filebase);
}


template <typename T = double>
auto
filecomp3(Tensor3d const& computed, std::string const& filebase) -> void
{
    Tensor2d const sp_b = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/" + filebase + ".bin",
        computed.dimension(0),
        computed.dimension(1),
        computed.dimension(2));

    for (long k = 0; k < computed.dimension(2); ++k)
        for (long j = 0; j < computed.dimension(1); ++j)
            for (long i = 0; i < computed.dimension(0); ++i)
                REQUIRE_MESSAGE(computed(i, j, k) == doctest::Approx(sp_b(i, j, k)),
                                i << " " << j << " " << k << " " << filebase);
}
