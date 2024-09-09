#pragma once

#include "xtsrcmaps/tensor/_tensor_details.hpp"
#include "xtsrcmaps/tensor/broadcast_iterator.hpp"

#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Fermi {

/**********************************************************************************
 * Tensor
 *
 * A Multdimensional Array class for large block data operations.
 *********************************************************************************/
template <typename T, std::size_t R, bool IsBcast = false>
    requires tensor_details::DataType<T> && tensor_details::PositiveRank<R>
class Tensor {
    //////////////////////////////////////////////////////////////////////////////
    // Definitions and Traits
    //////////////////////////////////////////////////////////////////////////////
  public:
    using value_type    = T;
    using ExtentsType   = std::array<std::size_t, R>;
    using IndicesType   = std::array<long, R>;
    using AlignedVector = tensor_details::AlignedVector<T>;
    using DataPtrType   = std::shared_ptr<AlignedVector>;
    using iterator      = std::conditional_t<IsBcast,
                                             BroadcastIterator<T, R>,
                                             typename AlignedVector::iterator>;
    using const_iterator
        = std::conditional_t<IsBcast,
                             BroadcastIterator<T, R, true>,
                             typename AlignedVector::const_iterator>;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using iterator_type
        = std::conditional_t<std::is_const_v<std::remove_pointer_t<T*>>,
                             typename AlignedVector::const_iterator,
                             typename AlignedVector::iterator>;

    static constexpr std::size_t Rank = R;

    static_assert(std::is_default_constructible_v<T>,
                  "Tensor elements must be default constructible");
    static_assert(std::is_copy_constructible_v<T>,
                  "Tensor elements must be copy constructible");
    static_assert(std::is_move_constructible_v<T>,
                  "Tensor elements must be move constructible");

    //////////////////////////////////////////////////////////////////////////////
    // Definitions and Traits
    //////////////////////////////////////////////////////////////////////////////
  private:
    DataPtrType   data_;
    ExtentsType   extents_; // Broadcast Extent
    ExtentsType   memory_strides_;
    iterator_type start_;

    //////////////////////////////////////////////////////////////////////////////
    // Constructors
    //////////////////////////////////////////////////////////////////////////////
  public:
    // Default constructor
    Tensor()
        : data_(std::make_shared<AlignedVector>(1, T {}))
        , extents_({ 1 })
        , // Default extent of 1 in each dimension
        memory_strides_({ 1 })
        , start_(data_->begin()) {
        if constexpr (R > 1) {
            std::fill(extents_.begin(), extents_.end(), 1);
            std::exclusive_scan(extents_.rbegin(),
                                extents_.rend(),
                                memory_strides_.rbegin(),
                                1,
                                std::multiplies<>());
        }
    }

    explicit Tensor(std::shared_ptr<AlignedVector> data,
                    ExtentsType const&             extents,
                    ExtentsType const&             memory_strides,
                    iterator_type const&           start)
        : data_(data)
        , extents_(extents)
        , memory_strides_(memory_strides)
        , start_(start) {}

    template <bool IsConst = false>
    explicit Tensor(std::shared_ptr<AlignedVector>          data,
                    ExtentsType const&                      extents,
                    ExtentsType const&                      memory_strides,
                    BroadcastIterator<T, R, IsConst> const& start)
        : data_(data)
        , extents_(extents)
        , memory_strides_(memory_strides)
        , start_(start) {}

    explicit Tensor(AlignedVector        data,
                    ExtentsType const&   extents,
                    ExtentsType const&   memory_strides,
                    iterator_type const& start)
        : data_(std::make_shared<AlignedVector>(data))
        , extents_(extents)
        , memory_strides_(memory_strides)
        , start_(start) {}

    explicit Tensor(ExtentsType const& extents)
        : data_(std::make_shared<AlignedVector>(std::accumulate(
              extents.begin(), extents.end(), 1, std::multiplies<>())))
        , extents_(extents)
        , start_(data_->begin()) {
        std::exclusive_scan(extents_.rbegin(),
                            extents_.rend(),
                            memory_strides_.rbegin(),
                            1UZ,
                            std::multiplies<>());
    }
    // Constructor to initialize a rank-1 tensor with a variadic list of values
    template <typename... Args>
        requires(R == 1 && sizeof...(Args) > 1)
    explicit Tensor(Args... args)
        : Tensor(std::array<std::size_t, 1> { sizeof...(Args) }) {
        data_->assign({ static_cast<T>(args)... });
        start_ = data_->begin();
    }

    // Constructor to build from iterators of a non-tensor object. Can't build
    // from iterators of a tensor as we need to persist the shared_ptr somehow.
    template <typename InputIt>
        requires std::input_iterator<InputIt>
                     && (!std::is_same_v<
                         InputIt,
                         BroadcastIterator<
                             typename std::iterator_traits<InputIt>::value_type,
                             R>>)
    explicit Tensor(InputIt first, InputIt last, const ExtentsType& extents)
        : data_(std::make_shared<AlignedVector>(first, last))
        , extents_(extents)
        , start_(data_->begin()) {
        std::exclusive_scan(extents_.rbegin(),
                            extents_.rend(),
                            memory_strides_.rbegin(),
                            1UZ,
                            std::multiplies<>());
        std::size_t expected_size = std::accumulate(
            extents.begin(), extents.end(), 1, std::multiplies<>());
        if (data_->size() != expected_size) {
            throw std::invalid_argument(
                "Data size does not match the provided tensor extents.");
        }
    }


    template <tensor_details::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    explicit Tensor(IndexTypes... extents)
        : Tensor<T, sizeof...(IndexTypes)>(
              { static_cast<std::size_t>(extents)... }) {}

    Tensor(std::vector<T> const& r, ExtentsType const& extents)
        : Tensor<T, R>(r.begin(), r.end(), extents) {}

    template <typename U, tensor_details::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    Tensor(std::vector<U> const& r, IndexTypes... extents)
        : Tensor(r.begin(), r.end(), { static_cast<std::size_t>(extents)... }) {
    }


    //////////////////////////////////////////////////////////////////////////////
    // Multi-dimensional Subscripting Operations
    //////////////////////////////////////////////////////////////////////////////
  public:
    // Helper function to calculate the linear index in 1D data storage
    inline std::size_t linear_index(IndicesType const& indices) const {
        return std::inner_product(
            indices.cbegin(), indices.cend(), memory_strides_.cbegin(), 0);
    }


    T& operator[](const IndicesType& indices) {
        return start_[linear_index(indices)];
    }
    template <tensor_details::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    T& operator[](IndexTypes... indices) {
        return start_[linear_index({ static_cast<long>(indices)... })];
    }

    T const& operator[](const IndicesType& indices) const {
        return start_[linear_index(indices)];
    }
    template <tensor_details::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    T const& operator[](IndexTypes... indices) const {
        return start_[linear_index({ static_cast<long>(indices)... })];
    }

    //////////////////////////////////////////////////////////////////////////////
    // Utility Methods
    //////////////////////////////////////////////////////////////////////////////
  public:
    // Get the extent of a specific dimension
    std::size_t extent(std::size_t dimension) const {
        if (dimension >= R) {
            throw std::out_of_range("Dimension out of range.");
        }
        return extents_[dimension];
    }

    // Get the total number of elements
    std::size_t size() const {
        return std::accumulate(
            extents_.begin(), extents_.end(), 1UL, std::multiplies<>());
    }

    // Access to raw data
    T*       data() noexcept { return &(*start_); }
    const T* data() const noexcept { return &(*start_); }

    // Access to extents
    ExtentsType extents() const noexcept { return extents_; }

    // Access to strides
    ExtentsType strides() const noexcept { return memory_strides_; }

    // Set the contents of the Tensor to Zero
    void clear() { std::fill(begin(), end(), T {}); }

    //////////////////////////////////////////////////////////////////////////////
    // Iterators
    //////////////////////////////////////////////////////////////////////////////
  public:
    // iterator support
    iterator begin() {
        if constexpr (IsBcast) {
            return iterator(&(*start_), extents_, memory_strides_);
        } else {
            return start_;
        }
    }
    const_iterator begin() const {
        if constexpr (IsBcast) {
            return const_iterator(&(*start_), extents_, memory_strides_);
        } else {
            return start_;
        }
    }
    const_iterator cbegin() const {
        if constexpr (IsBcast) {
            return const_iterator(&(*start_), extents_, memory_strides_);
        } else {
            return start_;
        }
    }
    iterator end() {
        if constexpr (IsBcast) {
            auto end_index = IndicesType {};
            end_index[0]   = extents_[0];
            return iterator(&(*start_), extents_, memory_strides_, end_index);
        } else {
            return start_ + size();
        }
    }
    const_iterator end() const {
        if constexpr (IsBcast) {
            auto end_index = IndicesType {};
            end_index[0]   = extents_[0];
            return const_iterator(
                &(*start_), extents_, memory_strides_, end_index);
        } else {
            return start_ + size();
        }
    }
    const_iterator cend() const {
        if constexpr (IsBcast) {
            auto end_index = IndicesType {};
            end_index[0]   = extents_[0];
            return const_iterator(
                &(*start_), extents_, memory_strides_, end_index);
        } else {
            return start_ + size();
        }
    }
    reverse_iterator       rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(cend());
    }
    const_reverse_iterator crbegin() const {
        return const_reverse_iterator(cend());
    }
    reverse_iterator       rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const {
        return const_reverse_iterator(cbegin());
    }
    const_reverse_iterator crend() const {
        return const_reverse_iterator(cbegin());
    }

    // Get an iterator to a specific element based on multi-dimensional indices
    iterator begin_at(IndicesType const indices) {
        return start_ + linear_index(indices);
    }
    const_iterator begin_at(IndicesType const indices) const {
        return start_ + linear_index(indices);
    }
    template <tensor_details::AllIntegral... IndexTypes>
    iterator begin_at(IndexTypes... indices) {
        IndicesType idxs = { static_cast<long>(indices)... };
        return start_ + linear_index(idxs);
    }
    template <tensor_details::AllIntegral... IndexTypes>
    const_iterator begin_at(IndexTypes... indices) const {
        IndicesType idxs = { static_cast<long>(indices)... };
        return start_ + linear_index(idxs);
    }

    // Get an iterator to one past a specific element based on multi-dimensional
    // indices
    template <tensor_details::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    iterator end_at(IndexTypes... indices) {
        IndicesType idxs = { static_cast<long>(indices)... };
        return start_ + linear_index(idxs);
    }

    template <tensor_details::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R)
    const_iterator end_at(IndexTypes... indices) const {
        IndicesType idxs = { static_cast<long>(indices)... };
        return start_ + linear_index(idxs);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Broadcasting, Reshaping, and Slicing operations
    ////////////////////////////////////////////////////////////////////////////////

  public:
    //////////////////////////////////////////////////////////////////////////////
    // Broadcast
    //////////////////////////////////////////////////////////////////////////////
    Tensor<value_type, Rank, true>
    broadcast(ExtentsType const& broadcast_extents) const {
        // Ensure that the broadcasted extents are compatible
        for (std::size_t i = 0; i < Rank; ++i) {
            if (extents_[i] != broadcast_extents[i] && extents_[i] != 1) {
                throw std::invalid_argument(
                    "Incompatible dimension for broadcasting at index "
                    + std::to_string(i) + ": tensor extent "
                    + std::to_string(extents_[i]) + " cannot be broadcast to "
                    + std::to_string(broadcast_extents[i]) + ".");
            }
        }

        ExtentsType strides = memory_strides_;
        if constexpr (!IsBcast) {
            for (std::size_t i = 0; i < Rank; ++i) {
                strides[i] = (extents_[i] == 1UZ) ? 0UZ : strides[i];
            }
        }

        return Tensor<value_type, Rank, true>(
            data_, broadcast_extents, strides, start_);
    }

    template <tensor_details::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == Rank)
    decltype(auto) broadcast(IndexTypes... new_extents) const {
        return broadcast({ static_cast<std::size_t>(new_extents)... });
    }

    //////////////////////////////////////////////////////////////////////////////
    // Reshape
    //////////////////////////////////////////////////////////////////////////////
    template <std::size_t NewRank>
    Tensor<T, NewRank, IsBcast>
    reshape(const std::array<std::size_t, NewRank>& new_extents) const {
        // Calculate the total size to ensure it's consistent
        std::size_t old_total_size = std::accumulate(
            extents_.begin(), extents_.end(), 1UL, std::multiplies<>());
        std::size_t new_total_size = std::accumulate(
            new_extents.begin(), new_extents.end(), 1UL, std::multiplies<>());

        if (old_total_size != new_total_size) {
            throw std::invalid_argument("Total size of the tensor must remain "
                                        "constant during reshape.");
        }

        typename Tensor<T, NewRank, IsBcast>::ExtentsType new_strides {};
        std::exclusive_scan(new_extents.rbegin(),
                            new_extents.rend(),
                            new_strides.rbegin(),
                            1UZ,
                            std::multiplies<>());


        return Tensor<T, NewRank, IsBcast>(
            data_, new_extents, new_strides, start_);
    }

    template <tensor_details::AllIntegral... IndexTypes>
    decltype(auto) reshape(IndexTypes... new_extents) {
        return reshape<sizeof...(new_extents)>(
            { static_cast<std::size_t>(new_extents)... });
    }

    //////////////////////////////////////////////////////////////////////////////
    // Slice
    //////////////////////////////////////////////////////////////////////////////

    Tensor<T, Rank, IsBcast>
    slice(const IndicesType& offset, const ExtentsType& slice_extents) const {
        // Check if the offset and extent are valid
        for (std::size_t i = 0; i < R; ++i) {
            if (offset[i] + slice_extents[i] > extents_[i]) {
                throw std::out_of_range("Slice exceeds tensor bounds.");
            }
        }

        std::size_t linear_offset = linear_index(offset);

        if (begin() + linear_offset >= end()) {
            throw std::out_of_range("Slice exceeds tensor bounds.");
        }

        // Return a new Tensor that represents the slice
        return Tensor<T, Rank, IsBcast>(
            data_, slice_extents, memory_strides_, start_ + linear_offset);
    }


    ////////////////////////////////////////////////////////////////////////////////
    /// Tensor Arithmetic
    ////////////////////////////////////////////////////////////////////////////////
  private:
    template <typename U,
              std::size_t                              R1,
              bool                                     B,
              tensor_details::ArithmeticBinaryOp<T, U> Op>
    inline Tensor&
    elementwise_arithmetic_helper(const Tensor<U, R1, B>& other, Op&& op) {
        if (this->extents_ != other.extents()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for elementwise operations.");
        }

        std::transform(this->begin(),
                       this->end(),
                       other.begin(),
                       this->begin(),
                       std::forward<Op>(op));
        return *this;
    }
    //
  public:
    template <typename U, std::size_t R1, bool IsBcastR>
        requires tensor_details::SameRankTensor<T, U, R, R1>
    inline decltype(auto) operator+=(const Tensor<U, R1, IsBcastR>& other) {
        return elementwise_arithmetic_helper(other, std::plus<T> {});
    }
    template <typename U, std::size_t R1, bool IsBcastR>
        requires tensor_details::SameRankTensor<T, U, R, R1>
    inline decltype(auto) operator-=(const Tensor<U, R1, IsBcastR>& other) {
        return elementwise_arithmetic_helper(other, std::minus<T> {});
    }
    template <typename U, std::size_t R1, bool IsBcastR>
        requires tensor_details::SameRankTensor<T, U, R, R1>
    inline decltype(auto) operator*=(const Tensor<U, R1, IsBcastR>& other) {
        return elementwise_arithmetic_helper(other, std::multiplies<T> {});
    }
    template <typename U, std::size_t R1, bool IsBcastR>
        requires tensor_details::SameRankTensor<T, U, R, R1>
    inline decltype(auto) operator/=(const Tensor<U, R1, IsBcastR>& other) {
        return elementwise_arithmetic_helper(other, std::divides<T> {});
    }
};

namespace tensor_details {
template <typename T,
          typename U,
          std::size_t              R,
          bool                     B1,
          bool                     B2,
          ArithmeticBinaryOp<T, U> Op>
static inline Tensor<std::common_type_t<T, std::decay_t<U>>, R, false>
elementwise_arithmetic_helper(const Tensor<T, R, B1>& lhs,
                              const Tensor<U, R, B2>& rhs,
                              Op&&                    op) {
    if (lhs.extents() != rhs.extents()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for elementwise addition.");
    }

    using ResultType = std::common_type_t<T, std::decay_t<U>>;

    Tensor<ResultType, R, false> result(lhs.extents());
    std::transform(lhs.begin(),
                   lhs.end(),
                   rhs.begin(),
                   result.begin(),
                   std::forward<Op>(op));
    return result;
}

/* template <typename U, std::size_t R1, bool B, typename Op> */
template <typename T,
          typename U,
          std::size_t              R,
          bool                     IsBcast,
          ArithmeticBinaryOp<T, U> Op>
static inline Tensor<std::common_type_t<T, std::decay_t<U>>, R, false>
elementwise_arithmetic_helper(const Tensor<U, R, IsBcast>& lhs,
                              U&                           rhs,
                              Op&&                         op) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;

    Tensor<ResultType, R, false> result(lhs.extents());
    std::transform(lhs.begin(),
                   lhs.end(),
                   result.begin(),
                   [&](auto& l, auto& r) -> ResultType { return op(l, r); });
    return result;
}
} // namespace tensor_details

////////////////////////////////////////////////////////////////////////
// Tensor, Tensor operations + - * /
//
// Tensor + Tensor
template <typename T, typename U, std::size_t R1, bool IsBcastL, bool IsBcastR>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator+(const Tensor<T, R1, IsBcastL>& lhs,
          const Tensor<U, R1, IsBcastR>& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::plus<ResultType> {});
}

// Tensor - Tensor
template <typename T, typename U, std::size_t R1, bool IsBcastL, bool IsBcastR>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator-(const Tensor<T, R1, IsBcastL>& lhs,
          const Tensor<U, R1, IsBcastR>& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::minus<ResultType> {});
}

//  Tensor * Tensor
template <typename T, typename U, std::size_t R1, bool IsBcastL, bool IsBcastR>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator*(const Tensor<T, R1, IsBcastL>& lhs,
          const Tensor<U, R1, IsBcastR>& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::multiplies<ResultType> {});
}

//  Tensor / Tensor
template <typename T, typename U, std::size_t R1, bool IsBcastL, bool IsBcastR>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator/(const Tensor<T, R1, IsBcastL>& lhs,
          const Tensor<U, R1, IsBcastR>& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::divides<ResultType> {});
}


////////////////////////////////////////////////////////////////////////////////////
// Tensor, Scalar operations + - * /

// Tensor + scaler
template <typename T, typename U, std::size_t R1, bool IsBcastL>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator+(const Tensor<T, R1, IsBcastL>& lhs, const U& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::plus<ResultType> {});
}

// Tensor - scaler
template <typename T, typename U, std::size_t R1, bool IsBcastL>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator-(const Tensor<T, R1, IsBcastL>& lhs, const U& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::minus<ResultType> {});
}

// Tensor * scaler
template <typename T, typename U, std::size_t R1, bool IsBcastL>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator*(const Tensor<T, R1, IsBcastL>& lhs, const U& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::multiplies<ResultType> {});
}

// Tensor / scalar
template <typename T, typename U, std::size_t R1, bool IsBcastL>
    requires tensor_details::SameRankTensor<T, U, R1, R1>
inline decltype(auto)
operator/(const Tensor<T, R1, IsBcastL>& lhs, const U& rhs) {
    using ResultType = std::common_type_t<T, std::decay_t<U>>;
    return tensor_details::elementwise_arithmetic_helper(
        lhs, rhs, std::divides<ResultType> {});
}

////////////////////////////////////////////////////////////////////////////////
// CTAD deduction guide(s)
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename... IndexTypes>
Tensor(std::vector<T> const&,
       IndexTypes...) -> Tensor<T, sizeof...(IndexTypes), false>;

} // namespace Fermi
