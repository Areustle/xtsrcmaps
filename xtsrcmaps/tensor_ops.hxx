#pragma once

#include "xtsrcmaps/tensor_types.hxx"

#include <fstream>
#include <vector>

namespace Fermi
{

auto
contract210(mdarray2 const& A, mdarray2 const& B) -> mdarray2;

auto
contract3210(mdarray3 const& A, mdarray2 const& B) -> mdarray3;

auto
mul210(mdarray2 const& A, std::vector<double> const& v) -> mdarray2;

auto
mul310(mdarray3 const& A, std::vector<double> const& v) -> mdarray3;

auto
mul32_1(mdarray3 const& A, mdarray2 const& B) -> mdarray3;

auto
mul322(mdarray3 const& A, mdarray2 const& B) -> mdarray3;

auto
sum2_2(mdarray2 const& A, mdarray2 const& B) -> mdarray2;

auto
sum3_3(mdarray3 const& A, mdarray3 const& B) -> mdarray3;

auto
safe_reciprocal(mdarray2 const& A) -> mdarray2;

template <typename T, typename... Ds>
auto
row_major_file_to_row_major_tensor(std::string const& filename, Ds const... dims)
    -> Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>
{
    static_assert(sizeof...(Ds) > 0);
    auto          v = std::vector<T>((... * dims));
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read((char*)(&v[0]), sizeof(T) * (... * dims));
    ifs.close();
    Eigen::TensorMap<Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>> RMt(v.data(),
                                                                           dims...);
    return RMt;
}

template <typename T, typename... Ds>
auto
col_major_file_to_col_major_tensor(std::string const& filename, Ds const... dims)
    -> Eigen::Tensor<T, sizeof...(Ds)>
{
    static_assert(sizeof...(Ds) > 0);
    auto          v = std::vector<T>((... * dims));
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read((char*)(&v[0]), sizeof(T) * (... * dims));
    ifs.close();
    Eigen::TensorMap<Eigen::Tensor<T, sizeof...(Ds)>> CMt(v.data(), dims...);
    return CMt;
}


namespace detail
{

template <size_t... Is>
constexpr auto
reverse_sequence(std::index_sequence<Is...> const&)
    -> decltype(std::index_sequence<(sizeof...(Is) - 1U - Is)...> {});

template <size_t N>
using make_reverse_sequence
    = decltype(reverse_sequence(std::make_index_sequence<N> {}));

template <typename T, size_t... Rs>
auto
to_colmajor_preserve_order(Eigen::Tensor<T, sizeof...(Rs), Eigen::RowMajor>& rm,
                           std::index_sequence<Rs...>)
    -> Eigen::Tensor<T, sizeof...(Rs), Eigen::ColMajor>
{
    Eigen::array<Eigen::Index, sizeof...(Rs)>        idx { Rs... };
    Eigen::Tensor<T, sizeof...(Rs), Eigen::ColMajor> cm = rm.swap_layout().shuffle(idx);
    return cm;
}

// template <typename T, typename I, I Order>
// auto
// to_colmajor(Eigen::Tensor<T, Order, Eigen::RowMajor>& rm)
//     -> Eigen::Tensor<T, Order, Eigen::ColMajor>
// {
//     // Eigen::Tensor<T, Order, Eigen::ColMajor> cm = rm.swap_layout().shuffle();
//     // return cm;
//     return rm.swap_layout();
// }

} // namespace detail


template <typename T = double, typename I, I Order>
auto
to_colmajor_preserve_order(Eigen::Tensor<T, Order, Eigen::RowMajor>& rm)
    -> Eigen::Tensor<T, Order, Eigen::ColMajor>
{
    return detail::to_colmajor_preserve_order(rm,
                                              detail::make_reverse_sequence<Order> {});
}

template <typename T = double, typename I, I Order>
auto
to_colmajor(Eigen::Tensor<T, Order, Eigen::RowMajor>& rm)
    -> Eigen::Tensor<T, Order, Eigen::ColMajor>
{
    return rm.swap_layout();
}

template <typename T = double, typename... Ds>
auto
row_major_file_to_col_major_tensor_preserve_order(std::string const& filename,
                                                  Ds const... dims)
    -> Eigen::Tensor<T, sizeof...(Ds)>
{
    Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor> rm
        = row_major_file_to_row_major_tensor<T>(filename, dims...);
    return to_colmajor_preserve_order(rm);
}

template <typename T = double, typename... Ds>
auto
row_major_file_to_col_major_tensor(std::string const& filename, Ds const... dims)
    -> Eigen::Tensor<T, sizeof...(Ds)>
{
    Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor> rm
        = row_major_file_to_row_major_tensor<T>(filename, dims...);
    return rm.swap_layout();
    // return to_colmajor(rm);
}

template <typename T = double, typename... Ds>
auto
row_major_buffer_to_col_major_tensor_preserve_order(T* buf, Ds const... dims)
    -> Eigen::Tensor<T, sizeof...(Ds)>
{
    Eigen::TensorMap<Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>> rm(buf, dims...);
    Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>                   rmm = rm;
    return to_colmajor_preserve_order(rmm);
}

template <typename T = double, typename... Ds>
auto
row_major_buffer_to_col_major_tensor(T* buf, Ds const... dims)
    -> Eigen::Tensor<T, sizeof...(Ds)>
{
    Eigen::TensorMap<Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>> rm(buf, dims...);
    Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>                   rmm = rm;
    return rmm.swap_layout();
    // return to_colmajor(rmm);
}

// namespace detail
// {
// template <typename M, size_t... Is>
// auto
// row_major_mdarray_to_col_major_tensor(M const& mdarr, std::index_sequence<Is...>
// const&)
// {
//     Eigen::TensorMap<Eigen::Tensor<double, M::rank, Eigen::RowMajor>> rm(
//         mdarr.data(), M::extent(Is)...);
//     return to_colmajor(rm);
// }
// } // namespace detail
//
// template <typename M>
// auto
// row_major_mdarray_to_col_major_tensor(M const& mdarr)
//     -> Eigen::Tensor<typename M::value_type, M::rank(), Eigen::ColMajor>
// {
//     return detail::row_major_mdarray_to_col_major_tensor(
//         mdarr, std::make_index_sequence<M::rank()> {});
// }



// template <typename T = double, typename... Ds>
// auto
// large_file_rm_to_cm_tensor(std::string const& filename, Ds const... dims)
//     -> Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>
// {
//     static_assert(sizeof...(Ds) > 0);
//     auto          v = std::vector<T>((... * dims));
//     std::ifstream ifs(filename, std::ios::in | std::ios::binary);
//     ifs.read((char*)(&v[0]), sizeof(T) * (... * dims));
//     ifs.close();
//     Eigen::TensorMap<Eigen::Tensor<T, sizeof...(Ds), Eigen::RowMajor>> RMt(v.data(),
//                                                                            dims...);
//     return RMt;
// }

} // namespace Fermi
