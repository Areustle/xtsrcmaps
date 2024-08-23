#pragma once

#include "doctest/doctest.h"

/* #include <fstream> */
/* #include <utility> */
/* #include <vector> */

/* #include "xtsrcmaps/tensor_ops.hxx" */
#include "xtsrcmaps/tensor/read_file_tensor.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

/* using Tensor1b = Fermi::Tensor<bool, 1>; */
/* using Tensor2b = Fermi::Tensor<bool, 2>; */
/* using Tensor3b = Fermi::Tensor<bool, 3>; */
using Tensor1d = Fermi::Tensor<double, 1>;
using Tensor2d = Fermi::Tensor<double, 2>;
using Tensor3d = Fermi::Tensor<double, 3>;
using Tensor4d = Fermi::Tensor<double, 4>;
using Tensor1f = Fermi::Tensor<float, 1>;
using Tensor2f = Fermi::Tensor<float, 2>;
using Tensor3f = Fermi::Tensor<float, 3>;
using Tensor4f = Fermi::Tensor<float, 4>;

auto
veq(auto v1, auto v2) -> void {
    REQUIRE(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); ++i)
        REQUIRE(v1[i] == doctest::Approx(v2[i]));
}

template <typename T1, typename T2>
auto
et2comp(Fermi::Tensor<T1, 2> const& computed,
        std::vector<T2> const&      expected) -> void {

    REQUIRE(computed.size() == expected.size());

    Fermi::Tensor<T2, 2uz> const sp_b(
        expected, computed.extent(0), computed.extent(1));

    for (size_t i = 0; i < computed.extent(0); ++i)
        for (size_t j = 0; j < computed.extent(1); ++j) {
            auto x = computed[i, j];
            REQUIRE_MESSAGE(doctest::Approx(sp_b[i, j]) == x, i << " " << j);
        }
}


template <typename T, size_t Rank>
auto
filecomp(Fermi::Tensor<T, Rank> const& computed,
         std::string const&            filebase) -> void {

    auto const sp_b = Fermi::read_file_tensor(
        "./tests/expected/" + filebase + ".bin", computed.extents());

    auto it1 = computed.begin();
    auto it2 = sp_b.begin();

    for (; it1 != computed.end() && it2 != sp_b.end(); ++it1, ++it2) {
        REQUIRE((*it1 == doctest::Approx(*it2)));
    }
}

/* template <typename T, size_t Rank> */
/* auto */
/* filecomp_reorder(Fermi::Tensor<T, Rank> const&   computed, */
/*                  std::string const&              filebase, */
/*                  std::array<size_t, Rank> const& order) -> void { */
/**/
/**/
/*     auto const sp_b = Fermi::read_file_tensor( */
/*         "./tests/expected/" + filebase + ".bin", computed.extents()); */
/**/
/*     auto it1 = computed.begin(); */
/**/
/*     for (; it1 != computed.end() ; ++it1) { */
/**/
/*         REQUIRE((*it1 == doctest::Approx(*it2))); */
/*     } */
/* } */


/* template <typename T = double> */
/* auto */
/* filecomp3(Tensor3d const& computed, std::string const& filebase) -> void { */
/*     Tensor3d const sp_b = Fermi::row_major_file_to_col_major_tensor( */
/*         "./xtsrcmaps/tests/expected/" + filebase + ".bin", */
/*         computed.dimension(2), */
/*         computed.dimension(1), */
/*         computed.dimension(0)); */
/**/
/*     for (long k = 0; k < computed.dimension(2); ++k) */
/*         for (long j = 0; j < computed.dimension(1); ++j) */
/*             for (long i = 0; i < computed.dimension(0); ++i) */
/*                 REQUIRE_MESSAGE(computed(i, j, k) */
/*                                     == doctest::Approx(sp_b(i, j, k)), */
/*                                 i << " " << j << " " << k << " " <<
 * filebase); */
/* } */
/**/
/* template <typename Derived, int Access> */
/* auto */
/* all_short_circuit(Eigen::TensorBase<Derived, Access> const& x_) -> bool { */
/*     // Derived& tensr = static_cast<Derived&>(tensor) */
/*     Derived const& x           = static_cast<Derived const&>(x_); */
/*     bool           is_all_true = true; */
/*     long           i           = 0; */
/*     while (i < x.size()) { */
/*         is_all_true &= (x.data())[i++]; */
/*         if (!is_all_true) { break; } */
/*     } */
/*     return is_all_true; */
/* } */
/**/
/* template <typename Derived, int Access> */
/* auto */
/* allfinite(Eigen::TensorBase<Derived, Access> const& x) -> bool { */
/*     Eigen::Tensor<bool, Derived::NumIndices> xfint = x.unaryExpr( */
/*         [](typename Derived::Scalar v) -> bool { return std::isfinite(v); });
 */
/*     bool xfin = all_short_circuit(xfint); */
/*     return xfin; */
/* } */
/**/
/* //////////////////////////////////////////////////////////////////////////////////////
 */
/* /// */
/* /// Numerically stable error tolerance. Algebraically equivalent to numpy */
/* /// allclose implementation. Uses no division. Is not symetric: Assumes "y"
 * is */
/* /// the reference value. */
/* //////////////////////////////////////////////////////////////////////////////////////
 */
/* template <typename Derived1, typename Derived2, int Access1, int Access2> */
/* auto */
/* allclose(Eigen::TensorBase<Derived1, Access1> const& x_, */
/*          Eigen::TensorBase<Derived2, Access2> const& y_, */
/*          double const                                abstol = 1e-5, */
/*          double const                                reltol = 1e-5) -> bool {
 */
/**/
/*     Derived1 const& x = static_cast<Derived1 const&>(x_); */
/*     Derived2 const& y = static_cast<Derived2 const&>(y_); */
/*     static_assert(Derived1::NumIndices == Derived2::NumIndices, */
/*                   "Tensor parameters must be of same rank."); */
/**/
/*     if (allfinite(x) && allfinite(y)) { */
/*         Tensor<bool, Derived1::NumIndices> istol */
/*             = (x - y).abs() <= abstol + reltol * y.abs(); */
/*         return all_short_circuit(istol); */
/*     } else { */
/*         return false; */
/*     } */
/* } */

// template <typename T, int N>
// auto
// relerr(Tensor<T, N> const& x, Tensor<T, N> const& y) -> Tensor<T, N>
// {
//     return (x - y).abs() / y.abs();
// }
//
//
// template <typename T, int N>
// auto
// abserr(Tensor<T, N> const& x, Tensor<T, N> const& y) -> Tensor<T, N>
// {
//     return (x - y).abs();
// }
//
// template <typename T, int N>
// auto
// within_tol(Tensor<T, N> const& x,
//            Tensor<T, N> const& y,
//            double const        atol,
//            double const        rtol) -> Tensor<bool, N>
// {
//     Tensor<bool, N> r = ((x - y).abs() <= atol + rtol * y.abs());
//     return r;
// }
