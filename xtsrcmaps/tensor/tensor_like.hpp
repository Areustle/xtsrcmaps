#pragma once

#include <concepts>
#include <cstddef>
#include <mdspan>

namespace Fermi {

// Helper traits to check for specific member functions
template <typename T, typename Extent, typename = void>
struct has_data_method : std::false_type {};

template <typename T, typename Extent>
struct has_data_method<T, Extent, std::void_t<decltype(std::declval<T>().data())>> : std::true_type {};

template <typename T, typename Extent, typename = void>
struct has_extent_method : std::false_type {};

template <typename T, typename Extent>
struct has_extent_method<T, Extent, std::void_t<decltype(std::declval<T>().extents())>> : std::true_type {};

template <typename T, typename Extent, typename = void>
struct has_strides_method : std::false_type {};

template <typename T, typename Extent>
struct has_strides_method<T, Extent, std::void_t<decltype(std::declval<T>().strides())>> : std::true_type {};

// Concept to match std::mdspan and Fermi::Tensor
template <typename T, typename... Indices>
concept TensorLike = requires(T t, std::size_t i, Indices... indices) {
    { t[i, indices...] } -> std::convertible_to<typename T::ValueType&>;
};

} // namespace Fermi
