#pragma once

#include "xtsrcmaps/tensor/_tensor_details.hpp"

#include <algorithm>
#include <iterator>

namespace Fermi {

template <typename T, std::size_t R, bool IsConst = false>
    requires tensor_details::DataType<T> && tensor_details::PositiveRank<R>
class BroadcastIterator {
  public:
    using value_type                  = std::conditional_t<IsConst, T const, T>;
    using pointer                     = value_type*;
    using reference                   = value_type&;
    using ExtentsType                 = std::array<std::size_t, R>;
    using IndicesType                 = std::array<long, R>;
    using difference_type             = std::ptrdiff_t;
    using iterator_category           = std::random_access_iterator_tag;

    static constexpr std::size_t Rank = R;

    ////////////////////////////////////////////////////////////////////////////
    // Constructors
    ////////////////////////////////////////////////////////////////////////////
    // Default constructor
    BroadcastIterator()
        : ptr_(nullptr)
        , broadcast_extents_ {}
        , memory_strides_ {}
        , broadcast_indices_ {} {}

    constexpr BroadcastIterator(pointer            ptr,
                                ExtentsType const& broadcast_extent,
                                ExtentsType const& memory_stride,
                                IndicesType const& broadcast_index = {})
        : ptr_(ptr)
        , broadcast_extents_(broadcast_extent)
        , memory_strides_(memory_stride)
        , broadcast_indices_(broadcast_index) {}

    // Copy constructor
    BroadcastIterator(const BroadcastIterator& other)            = default;

    // Copy assignment operator
    BroadcastIterator& operator=(const BroadcastIterator& other) = default;

    // Destructor
    ~BroadcastIterator()                                         = default;

    //////////////////////////////////////////////////////////////////////////////
    // Pointer Arithmetic Helper Functions
    //////////////////////////////////////////////////////////////////////////////
  private:
    ////
    // Increment the broadcast_indices and return the amount to add to the ptr_
    // member so as to correctly reference the correct data value in a broadcast
    // which manages 'virtual' elements by repeatition.
    //
    // works by incrementing the rightmost index with carry logic, then
    // accumulating memory_stride[i] multiples of the difference between
    // new_index and broadcast_index
    inline difference_type inc_index() noexcept {
        difference_type delta = {};
        bool            carry = true; // A carry of 1, as we're incrementing
        for (int i = R - 1; i >= 0; --i) {
            difference_type new_index = broadcast_indices_[i] + (carry);
            // Calculate carry: if new_index >= extent, carry will be 1, else 0
            carry = (static_cast<size_t>(new_index) >= broadcast_extents_[i]);
            // correct new_index
            new_index -= (broadcast_extents_[i] * carry);
            // Accumulate delta
            delta += (memory_strides_[i]) * (new_index - broadcast_indices_[i]);
            // Update broadcast index
            broadcast_indices_[i] = (new_index);
        }
        // To produce valid end() iterators.
        broadcast_indices_[0] += carry;
        return delta;
    }

    inline difference_type dec_index() {
        difference_type delta  = {};
        bool            borrow = true; // A borrow of 1, as we're decrementing
        for (int i = R - 1; i >= 0; --i) {
            difference_type new_index = broadcast_indices_[i] - (borrow);
            borrow                    = (new_index < 0);
            new_index += broadcast_extents_[i] * borrow;
            delta += memory_strides_[i] * (new_index - broadcast_indices_[i]);
            broadcast_indices_[i] = new_index;
        }
        return delta;
    }

    inline difference_type add_to_index(difference_type carry) {
        difference_type d = {};
        for (int i = R - 1; i >= 0; --i) {
            // Add carry to current index
            difference_type new_index = broadcast_indices_[i] + carry;
            // Calculate the new carry for the next higher dimension
            carry                     = new_index / broadcast_extents_[i];
            // Update the new index & handle wrapping
            new_index %= broadcast_extents_[i];
            // accumulate the new linear offset
            d += memory_strides_[i] * (new_index - broadcast_indices_[i]);
            // update the current index
            broadcast_indices_[i] = new_index;
        }
        return d;
    }

    inline difference_type subtract_from_index(difference_type borrow) {
        difference_type d = {};
        for (int i = R - 1; i >= 0; --i) {
            // Subtract borrow from current index
            difference_type new_index = broadcast_indices_[i] - borrow;
            // Calculate the new borrow for the next higher dimension
            borrow                    = (new_index < 0)
                                            ? (1 + (-new_index - 1) / broadcast_extents_[i])
                                            : 0;

            // Update the current index & handle wrapping
            new_index
                = (new_index % broadcast_extents_[i] + broadcast_extents_[i])
                  % broadcast_extents_[i];

            d += memory_strides_[i] * (new_index - broadcast_indices_[i]);
            broadcast_indices_[i] = new_index;
        }
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Dereference
    ////////////////////////////////////////////////////////////////////////////
  public:
    reference operator*() const { return *ptr_; }
    pointer   operator->() const { return ptr_; }

    // Increment/Decrement
    BroadcastIterator& operator++() {
        ptr_ += inc_index();
        return *this;
    }
    BroadcastIterator operator++(int) {
        BroadcastIterator tmp = *this;
        ptr_ += inc_index();
        return tmp;
    }
    BroadcastIterator& operator--() {
        ptr_ += dec_index();
        return *this;
    }
    BroadcastIterator operator--(int) {
        BroadcastIterator tmp = *this;
        ptr_ += dec_index();
        return tmp;
    }

    // Arithmetic
    BroadcastIterator& operator+=(difference_type n) {
        ptr_ += add_to_index(n);
        return *this;
    }
    BroadcastIterator& operator-=(difference_type n) {
        ptr_ -= subtract_from_index(n);
        return *this;
    }
    BroadcastIterator operator+(difference_type n) const {
        auto tmp = *this;
        tmp.ptr_ += tmp.add_to_index(n);
        return tmp;
    }
    BroadcastIterator operator-(difference_type n) const {
        auto tmp = *this;
        tmp.ptr_ -= tmp.subtract_from_index(n);
        return tmp;
    }
    difference_type operator-(const BroadcastIterator& other) const {
        difference_type delta            = {};
        difference_type broadcast_stride = 1;
        for (int i = R - 1; i >= 0; --i) {
            delta += broadcast_stride
                     * (broadcast_indices_[i] - other.broadcast_indices_[i]);
            broadcast_stride *= broadcast_extents_[i];
        }
        return delta;
    }

    // Comparison
    bool operator==(const BroadcastIterator& other) const {
        return broadcast_indices_ == other.broadcast_indices_;
    }
    bool operator!=(const BroadcastIterator& other) const {
        return broadcast_indices_ != other.broadcast_indices_;
    }
    bool operator<(const BroadcastIterator& other) const {
        return broadcast_indices_ < other.broadcast_indices_;
    }
    bool operator>(const BroadcastIterator& other) const {
        return broadcast_indices_ > other.broadcast_indices_;
    }
    bool operator<=(const BroadcastIterator& other) const {
        return broadcast_indices_ <= other.broadcast_indices_;
    }
    bool operator>=(const BroadcastIterator& other) const {
        return broadcast_indices_ >= other.broadcast_indices_;
    }

    // Subscript
    reference operator[](difference_type n) const {
        BroadcastIterator tmp = *this;
        tmp += n;
        return *tmp;
    }

    ExtentsType extents() const { return broadcast_extents_; }
    ExtentsType strides() const { return memory_strides_; }
    IndicesType indices() const { return broadcast_indices_; }

    IndicesType parent_indices() const {
        IndicesType idx = {};
        std::transform(memory_strides_.begin(),
                       memory_strides_.end(),
                       broadcast_indices_.begin(),
                       idx.begin(),
                       std::multiplies {});
        return idx;
    }

  private:
    pointer     ptr_;
    ExtentsType broadcast_extents_;
    ExtentsType memory_strides_;
    IndicesType broadcast_indices_ = {};
};

} // namespace Fermi
