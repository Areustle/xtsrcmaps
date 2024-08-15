#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"

#include <algorithm>
#include <cassert>

namespace Fermi {


template <typename T>
Fermi::Tensor<T, 1>
filter_in(std::vector<T> const& A, std::vector<bool> const& M) {
    std::size_t Nt = std::count(M.begin(), M.end(), true);

    assert(A.size(0) == M.size()); // Ensure the Tensors have the same size
    assert(A.total_size() >= Nt);

    Fermi::Tensor<T, 1> B(Nt);

    long t = 0;
    for (long i = 0; i < A.size(0); ++i) {
        if (M[i]) { B[t++] = A[i]; }
    }

    return B;
}

/**/
/* // auto */
/* // contract210(Tensor2d const& A, Tensor2d const& B) -> Tensor2d; */
/* // */
/* // auto */
/* // contract3210(mdarray3 const& A, mdarray2 const& B) -> mdarray3; */
/* // */
/* // auto */
/* // mul210(mdarray2 const& A, std::vector<double> const& v) -> mdarray2; */
/* // */
/* // auto */
/* // mul310(mdarray3 const& A, std::vector<double> const& v) -> mdarray3; */
/* // */
/* // auto */
/* // mul32_1(mdarray3 const& A, mdarray2 const& B) -> mdarray3; */
/* // */
/* // auto */
/* // mul322(mdarray3 const& A, mdarray2 const& B) -> mdarray3; */
/* // */
/* // auto */
/* // sum2_2(mdarray2 const& A, mdarray2 const& B) -> mdarray2; */
/* // */
/* // auto */
/* // sum3_3(mdarray3 const& A, mdarray3 const& B) -> mdarray3; */
/**/
/* // auto */
/* // safe_reciprocal(mdarray2 const& A) -> mdarray2; */
/**/
/* template <typename T, typename... RMDs> */
/* auto */
/* row_major_file_to_row_major_tensor(std::string const& filename, */
/*                                    RMDs const... row_major_dims) */
/*     -> Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor> */
/* { */
/*     static_assert(sizeof...(RMDs) > 0); */
/*     using VT = std:: */
/*         conditional_t<std::is_same_v<T, bool>, std::vector<char>,
 * std::vector<T>>; */
/*     // using VT        = std::is_same_v<T, bool> ? std::vector<char> :
 * std::vector<T>; */
/*     // if constexpr (std::is_same_v<T, bool>) { using VT = ; } */
/*     // else { using VT = std::vector<T>; } */
/*     VT            v = VT((... * row_major_dims)); */
/*     std::ifstream ifs(filename, std::ios::in | std::ios::binary); */
/*     ifs.read((char*)(&v[0]), sizeof(T) * (... * row_major_dims)); */
/*     ifs.close(); */
/*     Eigen::TensorMap<Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor>> RMt(
 */
/*         reinterpret_cast<T*>(v.data()), row_major_dims...); */
/*     return RMt; */
/* } */
/**/
/* template <typename T, typename... CMDs> */
/* auto */
/* col_major_file_to_col_major_tensor(std::string const& filename, */
/*                                    CMDs const... col_major_dims) */
/*     -> Eigen::Tensor<T, sizeof...(CMDs)> */
/* { */
/*     static_assert(sizeof...(CMDs) > 0); */
/*     auto          v = std::vector<T>((... * col_major_dims)); */
/*     std::ifstream ifs(filename, std::ios::in | std::ios::binary); */
/*     ifs.read((char*)(&v[0]), sizeof(T) * (... * col_major_dims)); */
/*     ifs.close(); */
/*     Eigen::TensorMap<Eigen::Tensor<T, sizeof...(CMDs)>> CMt(v.data(), */
/*                                                             col_major_dims...);
 */
/*     return CMt; */
/* } */
/**/
/**/
/* namespace detail */
/* { */
/**/
/* template <size_t... Is> */
/* constexpr auto */
/* reverse_sequence(std::index_sequence<Is...> const&) */
/*     -> decltype(std::index_sequence<(sizeof...(Is) - 1U - Is)...> {}); */
/**/
/* template <size_t N> */
/* using make_reverse_sequence */
/*     = decltype(reverse_sequence(std::make_index_sequence<N> {})); */
/**/
/* template <typename T, size_t... Rs> */
/* auto */
/* to_colmajor_preserve_order(Eigen::Tensor<T, sizeof...(Rs), Eigen::RowMajor>&
 * rm, */
/*                            std::index_sequence<Rs...>) */
/*     -> Eigen::Tensor<T, sizeof...(Rs), Eigen::ColMajor> */
/* { */
/*     Eigen::array<Eigen::Index, sizeof...(Rs)>        idx { Rs... }; */
/*     Eigen::Tensor<T, sizeof...(Rs), Eigen::ColMajor> cm =
 * rm.swap_layout().shuffle(idx); */
/*     return cm; */
/* } */
/**/
/* // template <typename T, typename I, I Order> */
/* // auto */
/* // to_colmajor(Eigen::Tensor<T, Order, Eigen::RowMajor>& rm) */
/* //     -> Eigen::Tensor<T, Order, Eigen::ColMajor> */
/* // { */
/* //     // Eigen::Tensor<T, Order, Eigen::ColMajor> cm =
 * rm.swap_layout().shuffle(); */
/* //     // return cm; */
/* //     return rm.swap_layout(); */
/* // } */
/**/
/* } // namespace detail */
/**/
/**/
/* template <typename T = double, typename I, I Order> */
/* auto */
/* to_colmajor_preserve_order(Eigen::Tensor<T, Order, Eigen::RowMajor>& rm) */
/*     -> Eigen::Tensor<T, Order, Eigen::ColMajor> */
/* { */
/*     return detail::to_colmajor_preserve_order(rm, */
/*                                               detail::make_reverse_sequence<Order>
 * {}); */
/* } */
/**/
/* template <typename T = double, typename I, I Order> */
/* auto */
/* to_colmajor(Eigen::Tensor<T, Order, Eigen::RowMajor>& rm) */
/*     -> Eigen::Tensor<T, Order, Eigen::ColMajor> */
/* { */
/*     return rm.swap_layout(); */
/* } */
/**/
/* template <typename T = double, typename... RMDs> */
/* auto */
/* row_major_file_to_col_major_tensor_preserve_order(std::string const&
 * filename, */
/*                                                   RMDs const...
 * row_major_dims) */
/*     -> Eigen::Tensor<T, sizeof...(RMDs)> */
/* { */
/*     Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor> rm */
/*         = row_major_file_to_row_major_tensor<T>(filename, row_major_dims...);
 */
/*     return to_colmajor_preserve_order(rm); */
/* } */
/**/
/* template <typename T = double, typename... RMDs> */
/* auto */
/* row_major_file_to_col_major_tensor(std::string const& filename, */
/*                                    RMDs const... row_major_dims) */
/*     -> Eigen::Tensor<T, sizeof...(RMDs)> */
/* { */
/*     Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor> rm */
/*         = row_major_file_to_row_major_tensor<T>(filename, row_major_dims...);
 */
/*     return rm.swap_layout(); */
/* } */
/**/
/* template <typename T = double, typename... RMDs> */
/* auto */
/* row_major_buffer_to_col_major_tensor_preserve_order(T const* buf, */
/*                                                     RMDs const...
 * row_major_dims) */
/*     -> Eigen::Tensor<T, sizeof...(RMDs)> */
/* { */
/*     Eigen::TensorMap<Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor>
 * const> const rm( */
/*         buf, row_major_dims...); */
/*     Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor> rmm = rm; */
/*     return to_colmajor_preserve_order(rmm); */
/* } */
/**/
/* template <typename T = double, typename... RMDs> */
/* auto */
/* row_major_buffer_to_col_major_tensor(T const* buf, RMDs const...
 * row_major_dims) */
/*     -> Eigen::Tensor<T, sizeof...(RMDs)> */
/* { */
/*     Eigen::TensorMap<Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor>
 * const> const rm( */
/*         buf, row_major_dims...); */
/*     Eigen::Tensor<T, sizeof...(RMDs), Eigen::RowMajor> rmm = rm; */
/*     return rmm.swap_layout(); */
/*     // return to_colmajor(rmm); */
/* } */


/*************************************************************************************/
/*                                Should be in Eigen */
/*************************************************************************************/

// Filter In. Given a Tensor and a boolean mask (tensor1b) Return the Tensor
// entries corresponding to true values in the mask.
/* template <typename T> */
/* auto */
/* filter_in(Eigen::Tensor<T, 1> const& A, Eigen::Tensor<bool, 1> const& M) ->
 * Eigen::Tensor<T, 1> */
/* { */
/*     long const Nt = std::count(M.data(), M.data() + M.size(), true); */
/*     assert(A.size() == M.size()); */
/*     assert(A.size() >= Nt); */
/**/
/*     Eigen::Tensor<T, 1> B(Nt); */
/**/
/*     long t = 0; */
/*     for (long i = 0; i < A.size(); ++i) */
/*         if (M(i)) B(t++) = A(i); */
/**/
/*     return B; */
/* } */
} // namespace Fermi
