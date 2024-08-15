#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"

#include <algorithm>
#include <ranges>

namespace Fermi {
template <typename T, std::size_t R>
Tensor<T, R>
reorder_tensor(Tensor<T, R> const&                       tensor,
               typename Tensor<T, R>::ExtentsType const& order) {

    using ExtentsType = typename Tensor<T, R>::ExtentsType;

    // Ensure the order array is a valid permutation
    ExtentsType check = order;
    std::ranges::sort(check);
    if (!std::ranges::equal(check, std::views::iota(0UZ, R))) {
        throw std::invalid_argument("Invalid permutation order");
    }

    // Calculate new extents based on the permutation order
    typename Tensor<T, R>::ExtentsType new_extents;
    std::transform(order.begin(),
                   order.end(),
                   new_extents.begin(),
                   [&](std::size_t i) { return tensor.extent(i); });

    // Create a new tensor with the reordered layout
    Tensor<T, R> new_tensor(new_extents);

    // Compute strides for the original tensor (assuming row-major order)
    ExtentsType strides;
    strides[R - 1] = 1;
    for (std::size_t i = R - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * tensor.extent(i);
    }

    // Fill the new tensor with reordered data
    std::size_t total_elements = tensor.total_size();
    for (std::size_t lidx = 0; lidx < total_elements; ++lidx) {

        ExtentsType indices;
        for (std::size_t i = 0; i < R; ++i) {
            indices[i] = (lidx / tensor.strides()[i]) % tensor.extents()[i];
        }
        ExtentsType reordered_indices;
        for (std::size_t i = 0; i < R; ++i) {
            reordered_indices[i] = indices[order[i]];
        }

        // Assign the value from the original tensor to the new tensor
        /* new_tensor.data()[lidx] = std::apply( */
        /*     [&](auto&&... args) { return tensor[args...]; },
         * reordered_indices); */
        new_tensor[reordered_indices] = tensor.data()[lidx];
    }

    return new_tensor;
}
} // namespace Fermi
