#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"
#include <array>
#include <cstddef>
#include <iterator>

namespace Fermi {

template <typename A, bool IsConst>
class TensorReorderIterator {
  public:
    using ndarray                  = A;
    static constexpr std::size_t R = ndarray::Rank;
    using iterator_category        = std::input_iterator_tag;
    using value_type               = A::ValueType;
    using difference_type          = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference
        = std::conditional_t<IsConst, const value_type&, value_type&>;

    TensorReorderIterator(ndarray                           tensor,
                          const std::array<std::size_t, R>& order,
                          std::size_t                       index = 0)
        : data_(tensor.data())
        , strides_(tensor.strides())
        , sizes_(tensor.sizes())
        , order_(order)
        , index_(index) {}

    reference operator*() const { return data_[compute_index()]; }

    TensorReorderIterator& operator++() {
        ++index_;
        return *this;
    }

    TensorReorderIterator operator++(int) {
        TensorReorderIterator temp = *this;
        ++(*this);
        return temp;
    }

    bool operator==(const TensorReorderIterator& other) const {
        return index_ == other.index_;
    }
    bool operator!=(const TensorReorderIterator& other) const {
        return !(*this == other);
    }

  private:
    pointer                    data_;
    std::array<std::size_t, R> strides_;
    std::array<std::size_t, R> sizes_;
    std::array<std::size_t, R> order_;
    std::size_t                index_;

    std::size_t compute_index() const {
        std::size_t                linear_index = index_;
        std::array<std::size_t, R> indices;
        std::array<std::size_t, R> reordered_indices;

        for (std::size_t i = 0; i < R; ++i) {
            indices[i] = (linear_index / strides_[i]) % sizes_[i];
        }
        for (std::size_t i = 0; i < R; ++i) {
            reordered_indices[i] = indices[order_[i]];
        }
        linear_index = 0;
        for (std::size_t i = 0; i < R; ++i) {
            linear_index += indices[i] * strides_[i];
        }
        return linear_index;
    }
};

// Helper function to create a mutable TensorReorderIterator
template <typename T, std::size_t R>
TensorReorderIterator<Tensor<T, R>, false>
make_tensor_reorder_iterator(Tensor<T, R>&                     tensor,
                             const std::array<std::size_t, R>& order,
                             std::size_t                       index = 0) {
    return TensorReorderIterator<Tensor<T, R>, false>(tensor, order, index);
}

// Helper function to create a const TensorReorderIterator
template <typename T, std::size_t R>
TensorReorderIterator<Tensor<T, R> const, true>
make_tensor_reorder_iterator(Tensor<T, R> const&               tensor,
                             std::array<std::size_t, R> const& order,
                             std::size_t                       index = 0) {
    return TensorReorderIterator<Tensor<T, R>, true>(tensor, order, index);
}

} // namespace Fermi
