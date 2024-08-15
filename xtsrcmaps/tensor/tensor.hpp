#pragma once

#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "xtsrcmaps/tensor/_tensor_details.hpp"
/* #include "xtsrcmaps/tensor/tensor_like.hpp" */

namespace Fermi {

template <typename T, std::size_t R>
    requires tensor::DataType<T> && tensor::PositiveRank<R>
class Tensor {
  public:
    using ValueType        = T;
    using ExtentsType      = std::array<std::size_t, R>;
    using AlignedVector    = std::vector<T, tensor::AlignedAllocator<T>>;
    using iterator         = typename AlignedVector::iterator;
    using const_iterator   = typename AlignedVector::const_iterator;
    using reverse_iterator = typename AlignedVector::reverse_iterator;
    using const_reverse_iterator =
        typename AlignedVector::const_reverse_iterator;

    static constexpr std::size_t Rank = R;

    static_assert(std::is_default_constructible_v<T>,
                  "Tensor elements must be default constructible");
    static_assert(std::is_copy_constructible_v<T>,
                  "Tensor elements must be copy constructible");
    static_assert(std::is_move_constructible_v<T>,
                  "Tensor elements must be move constructible");

  private:
    ExtentsType   extents_;
    AlignedVector data_;
    ExtentsType   strides_;

    // Helper function to calculate strides based on extents.
    void calculate_strides() {
        strides_[R - 1] = 1;
        for (std::size_t i = R - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * extents_[i];
        }
    }

  public:
    // Initialize the array with a specific size for each dimension.
    Tensor(ExtentsType const& extents)
        : extents_(extents)
        , data_(std::accumulate(
              extents.begin(), extents.end(), 1, std::multiplies<>())) {
        calculate_strides();
    }

    template <typename... SizeTypes>
        requires(sizeof...(SizeTypes) == R
                 && std::conjunction_v<std::is_integral<SizeTypes>...>)
    explicit Tensor(SizeTypes... extents)
        : extents_ { static_cast<std::size_t>(extents)... }
        , data_(std::accumulate(
              extents_.begin(), extents_.end(), 1, std::multiplies<>()))
        , strides_() {
        calculate_strides();
    }
    // Constructor to initialize the tensor with a specific size for each
    // dimension using array of integral types
    template <typename SizeType>
        requires(std::is_integral_v<typename SizeType::value_type>
                 && SizeType::size() == R)
    explicit Tensor(const SizeType& extents)
        : extents_()
        , data_(std::accumulate(
              extents.begin(), extents.end(), 1, std::multiplies<>()))
        , strides_() {
        std::copy(extents.begin(), extents.end(), extents_.begin());
        calculate_strides();
    }

    // Helper function to calculate the linear index in 1D data storage
    std::size_t linear_index(ExtentsType const& indices) const {
        return std::inner_product(
            indices.cbegin(), indices.cend(), strides_.cbegin(), 0UZ);
    }

    T& operator[](const ExtentsType& indices) {
        return data_[linear_index(indices)];
    }

    T const& operator[](const ExtentsType& indices) const {
        return data_[linear_index(indices)];
    }

    // Access element by a multi-index using the new multidimensional operator[]
    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    T& operator[](IndexTypes... indices) {
        ExtentsType idxs = { static_cast<std::size_t>(indices)... };
        return data_[linear_index(idxs)];
    }

    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    T const& operator[](IndexTypes... indices) const {
        ExtentsType idxs = { static_cast<std::size_t>(indices)... };
        return data_[linear_index(idxs)];
    }

    template <typename... IndexTypes>
        requires(sizeof...(IndexTypes) == R
                 && std::conjunction_v<std::is_integral<IndexTypes>...>)
    T* operator&(IndexTypes... indices) {
        ExtentsType idxs = { static_cast<std::size_t>(indices)... };
        return &data_[linear_index(idxs)];
    }

    // Get the extent of a specific dimension
    std::size_t extent(std::size_t dimension) const {
        if (dimension >= R) {
            throw std::out_of_range("Dimension out of range.");
        }
        return extents_[dimension];
    }

    // Get the total number of elements
    std::size_t total_size() const { return data_.size(); }

    // Access to raw data
    T*       data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Access to extents
    ExtentsType extents() const noexcept { return extents_; }

    // Access to strides
    ExtentsType strides() const noexcept { return strides_; }

    // Set the contents of the Tensor to Zero
    void clear() { std::fill(begin(), end(), T {}); }

    // iterator support
    iterator               begin() noexcept { return data_.begin(); }
    const_iterator         begin() const noexcept { return data_.begin(); }
    const_iterator         cbegin() const noexcept { return data_.cbegin(); }
    iterator               end() noexcept { return data_.end(); }
    const_iterator         end() const noexcept { return data_.end(); }
    const_iterator         cend() const noexcept { return data_.cend(); }
    reverse_iterator       rbegin() noexcept { return data_.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return data_.rbegin(); }
    const_reverse_iterator crbegin() const noexcept { return data_.crbegin(); }
    reverse_iterator       rend() noexcept { return data_.rend(); }
    const_reverse_iterator rend() const noexcept { return data_.rend(); }
    const_reverse_iterator crend() const noexcept { return data_.crend(); }

    // Get an iterator to a specific element based on multi-dimensional indices
    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    iterator begin_at(IndexTypes... indices) {
        ExtentsType idxs = { static_cast<std::size_t>(indices)... };
        return data_.begin() + linear_index(idxs);
    }

    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    const_iterator begin_at(IndexTypes... indices) const {
        ExtentsType idxs = { static_cast<std::size_t>(indices)... };
        return data_.begin() + linear_index(idxs);
    }

    // Get an iterator to one past a specific element based on multi-dimensional
    // indices
    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    iterator end_at(IndexTypes... indices) {
        ExtentsType idxs = { static_cast<std::size_t>(indices)... };
        return data_.begin() + linear_index(idxs);
    }

    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    const_iterator end_at(IndexTypes... indices) const {
        ExtentsType idxs = { static_cast<std::size_t>(indices)... };
        return data_.begin() + linear_index(idxs);
    }
};

} // namespace Fermi
