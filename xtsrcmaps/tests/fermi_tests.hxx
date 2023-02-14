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
    Tensor3d const sp_b = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/" + filebase + ".bin",
        computed.dimension(2),
        computed.dimension(1),
        computed.dimension(0));

    for (long k = 0; k < computed.dimension(2); ++k)
        for (long j = 0; j < computed.dimension(1); ++j)
            for (long i = 0; i < computed.dimension(0); ++i)
                REQUIRE_MESSAGE(computed(i, j, k) == doctest::Approx(sp_b(i, j, k)),
                                i << " " << j << " " << k << " " << filebase);
}

template <typename T, long N>
auto
isfinite(Tensor<T, N> const& x) -> Tensor<bool, N>
{
    return x.unaryExpr([](T v) -> bool { return std::isfinite(v); });
}

template <typename T, long N>
auto
relerr(Tensor<T, N> const& x, Tensor<T, N> const& y) -> Tensor<T, N>
{
    return (x - y).abs() / y.abs();
}


template <typename T, long N>
auto
abserr(Tensor<T, N> const& x, Tensor<T, N> const& y) -> Tensor<T, N>
{
    return (x - y).abs();
}

template <typename T, long N>
auto
within_tol(Tensor<T, N> const& x,
           Tensor<T, N> const& y,
           double const        atol,
           double const        rtol) -> Tensor<bool, N>
{
    Tensor<bool, N> r = (abserr<T, N>(x, y) <= atol + rtol * y.abs());
    return r;
}

template <typename T, long N>
auto
allclose(Tensor<T, N> const& x,
         Tensor<T, N> const& y,
         double const        abstol,
         double const        reltol) -> bool
{
    Tensor0b xfin = isfinite<T, N>(x).all();
    Tensor0b yfin = isfinite<T, N>(y).all();
    if (xfin(0) && yfin(0))
    {

        Tensor<bool, N> istol = within_tol<T, N>(x, y, abstol, reltol);
        Tensor0b        ac    = istol.all();
        return ac(0);
    }
    else { return false; }
}
