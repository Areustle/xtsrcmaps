#pragma once

#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "xtsrcmaps/tensor/_tensor_details.hpp"

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
    explicit Tensor(ExtentsType const& extents)
        : extents_(extents)
        , data_(std::accumulate(
              extents.begin(), extents.end(), 1, std::multiplies<>())) {
        calculate_strides();
    }

    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R
                 && std::conjunction_v<std::is_integral<IndexTypes>...>)
    explicit Tensor(IndexTypes... extents)
        : Tensor({ static_cast<std::size_t>(extents)... }) {}
    // Constructor to initialize the tensor with a specific size for each
    // dimension using array of integral types
    template <typename ExtentsType>
        requires(std::is_integral_v<typename ExtentsType::value_type>
                 && std::tuple_size<ExtentsType>::value == R)
    explicit Tensor(const ExtentsType& extents)
        : extents_()
        , data_(std::accumulate(
              extents.begin(), extents.end(), 1, std::multiplies<>()))
        , strides_() {
        std::copy(extents.begin(), extents.end(), extents_.begin());
        calculate_strides();
    }
    // Constructor to initialize the tensor from an existing container
    // dimension using array of integral types
    template <typename InputIt, typename ExtentsType>
        requires(std::is_integral_v<typename ExtentsType::value_type> //
                 && std::tuple_size<ExtentsType>::value == R
                 && std::input_iterator<InputIt>)
    explicit Tensor(InputIt            other_begin,
                    InputIt            other_end,
                    const ExtentsType& extents)
        : extents_(), data_(other_begin, other_end), strides_() {
        std::copy(extents.begin(), extents.end(), extents_.begin());
        calculate_strides();

        size_t expected_size = std::accumulate(
            extents.begin(), extents.end(), 1, std::multiplies<>());

        if (data_.size() != expected_size) {
            throw std::invalid_argument(
                "Data size does not match the provided tensor extents.");
        }
    }


    template <typename ExtentsType>
        requires(std::is_integral_v<typename ExtentsType::value_type>
                 && std::tuple_size<ExtentsType>::value == R)
    explicit Tensor(std::vector<T> const& data, const ExtentsType& extents)
        : Tensor(data.begin(), data.end(), extents) {}

    template <tensor::AllIntegral... IndexTypes>
        requires(sizeof...(IndexTypes) == R
                 && std::conjunction_v<std::is_integral<IndexTypes>...>)
    explicit Tensor(std::vector<T> const& other, IndexTypes... extents)
        : Tensor(other.begin(),
                 other.end(),
                 ExtentsType { static_cast<std::size_t>(extents)... }) {}



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

    template <tensor::AllIntegral... IndexTypes>
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

    ////////////////////////////////////////////////////////////////////////////////
    /// Broadcasting Reshaping and Slicing operations
    ////////////////////////////////////////////////////////////////////////////////
  private:
    // Special constructor used by the broadcast method
    template <std::size_t NewRank>
    Tensor(const std::array<std::size_t, NewRank>& new_extents,
           const std::array<std::size_t, NewRank>& new_strides,
           const AlignedVector&                    data)
        : extents_(new_extents), strides_(new_strides), data_(data) {}

  public:
    // Broadcasting method to reshape the tensor for broadcasting
    template <std::size_t NewRank>
    Tensor<T, NewRank>
    broadcast(const std::array<std::size_t, NewRank>& new_extents) const {
        static_assert(NewRank >= R,
                      "New rank must be greater than or equal to current rank");

        // Calculate new strides for the broadcasted tensor
        std::array<std::size_t, NewRank> new_strides = {};
        std::size_t                      offset      = NewRank - R;
        for (std::size_t i = 0; i < R; ++i) {
            new_strides[i + offset] = strides_[i];
        }

        // Ensure that the broadcasted extents are compatible
        for (std::size_t i = 0; i < R; ++i) {
            if (extents_[i] != new_extents[i + offset] && extents_[i] != 1) {
                throw std::invalid_argument(
                    "Incompatible dimensions for broadcasting.");
            }
        }

        return Tensor<T, NewRank>(new_extents, new_strides, data_);
    }

    template <std::size_t NewRank, tensor::AllIntegral... IndexTypes>
    Tensor<T, NewRank> broadcast(IndexTypes... new_extents) const {
        return broadcast({ static_cast<std::size_t>(new_extents)... });
    }

    ///// Reshape //////
    template <std::size_t NewRank>
    Tensor<T, NewRank>
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

        // Calculate new strides for the reshaped tensor
        std::array<std::size_t, NewRank> new_strides;
        new_strides[NewRank - 1] = 1;
        for (std::size_t i = NewRank - 1; i > 0; --i) {
            new_strides[i - 1] = new_strides[i] * new_extents[i];
        }

        // Return the new Tensor with adjusted extents and strides
        return Tensor<T, NewRank>(data_, new_extents, new_strides, 0);
    }

    template <tensor::AllIntegral... IndexTypes>
    Tensor<T, sizeof...(IndexTypes)> reshape(IndexTypes... new_extents) {
        return reshape({ static_cast<std::size_t>(new_extents)... });
    }

    /////// Slice //////
  private:
    // Special Constructor for creating a Tensor slice without deep copy
    Tensor(AlignedVector&     data,
           const ExtentsType& extents,
           const ExtentsType& strides,
           std::size_t        offset)
        : extents_(extents)
        , data_(data.begin() + offset, data.end())
        , strides_(strides) {}

  public:
    Tensor
    slice(const ExtentsType& offset, const ExtentsType& slice_extents) const {
        // Check if the offset and extent are valid
        for (std::size_t i = 0; i < R; ++i) {
            if (offset[i] + slice_extents[i] > extents_[i]) {
                throw std::out_of_range("Slice exceeds tensor bounds.");
            }
        }

        // Calculate the new strides and offset
        std::size_t new_offset = std::inner_product(
            offset.begin(), offset.end(), strides_.begin(), 0UL);

        // Return a new Tensor that represents the slice
        return Tensor(data_, slice_extents, strides_, new_offset);
    }
    ////////////////////////////////////////////////////////////////////////////////
    /// Tensor Arithmetic
    ////////////////////////////////////////////////////////////////////////////////
  private:
    template <typename T1, std::size_t R1, typename Op>
    static Tensor elementwise_arithmetic_helper(const Tensor<T1, R1>& lhs,
                                                const Tensor<T1, R1>& rhs,
                                                Op&&                  op) {
        if (lhs.extents_ != rhs.extents_) {
            throw std::invalid_argument(
                "Tensors must have the same shape for elementwise addition.");
        }

        Tensor result(lhs.extents_);
        std::transform(lhs.begin(),
                       lhs.end(),
                       rhs.begin(),
                       result.begin(),
                       std::forward<Op>(op));
        return result;
    }

    template <typename T1, std::size_t R1, typename Op>
    static Tensor
    elementwise_arithmetic_helper(const Tensor<T1, R1>& lhs, T1& rhs, Op&& op) {
        Tensor result(lhs.extents_);
        std::transform(lhs.begin(),
                       lhs.end(),
                       result.begin(),
                       [&](auto l, auto r) -> T { return op(l, r); });
        return result;
    }

    template <typename T1, std::size_t R1, typename Op>
    Tensor&
    elementwise_arithmetic_helper(const Tensor<T1, R1>& other, Op&& op) {
        if (this->extents_ != other.extents_) {
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

  public:
    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::plus<T> {});
    }
    friend Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::minus<T> {});
    }
    friend Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::multiplies<T> {});
    }
    friend Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::divides<T> {});
    }
    friend Tensor operator+(const Tensor& lhs, const T& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::plus<T> {});
    }
    friend Tensor operator-(const Tensor& lhs, const T& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::minus<T> {});
    }
    friend Tensor operator*(const Tensor& lhs, const T& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::multiplies<T> {});
    }
    friend Tensor operator/(const Tensor& lhs, const T& rhs) {
        return elementwise_arithmetic_helper(lhs, rhs, std::divides<T> {});
    }
    Tensor& operator+=(const Tensor& rhs) {
        return elementwise_arithmetic_helper(rhs, std::plus<T> {});
    }
    Tensor& operator-=(const Tensor& rhs) {
        return elementwise_arithmetic_helper(rhs, std::minus<T> {});
    }
    Tensor operator*(const Tensor& rhs) {
        return elementwise_arithmetic_helper(rhs, std::multiplies<T> {});
    }
    Tensor operator/(const Tensor& rhs) {
        return elementwise_arithmetic_helper(rhs, std::divides<T> {});
    }


    ///////////////////////////////////////////////////////////////////////////
    // broadcast multi index
    //
    // If a tensor were broadcast to a new shape for arithmetic purposes what
    // index would need to be applied to access the underlying data correctly?
    //
    // underlying tensor2 a = [1, 4] = {1 2 3 4}
    // broadcast to [3, 4] aka repeat the columns twice.
    // use broadcast rule [3, 1]
    //                     App Data   App Lidx   True Lidx
    //       b = [3, 4] = 1 2 3 4      0 1 2 3     0 1 2 3
    //                    1 2 3 4      4 5 6 7     0 1 2 3
    //                    1 2 3 4      8 9 a b     0 1 2 3
    //  [x y]
    //  [0 1]
    //  l = x0 +y1 = y
    //
    //                     broacast to [4, 3] | copies [1, 3]
    // = [4, 1] Data      App Data    App Lidx   True Lidx
    // for a = {1         1 1 1         0 1 2      0 0 0
    //          2         2 2 2         3 4 5      1 1 1
    //          3         3 3 3         6 7 8      2 2 2
    //          4}        4 4 4         9 a b      3 3 3
    //
    //  [x y]
    //  [1 0]
    //  l = x1 + y0 = x
    //
    //   original shape = [A B]
    //   padded shape = [1 A B]
    //   broadcast shape = [C A B]
    //
    //   original stride = [B 1]
    //   padded stride = [0 B 1]
    //   broadcast stride = [AB B 1]
    //
    //   broacast_multiindex = [c a b]
    //   broacast linear_index = cAB + aB + b1
    //   original linear_index = c0 + aB + b1 = bmix . padded_stride
    //
    //----------------------------------------------------------------
    //
    //   original shape = [A B]
    //   padded shape = [1 A 1 B 1]
    //   broadcast shape = [C A D B E]
    //
    //   original stride = [B 1]
    //   padded stride = [0 B 0 1 0]
    //   broadcast stride = [ADBE DBE BE E 1]
    //
    //   broacast_multiindex = [c a d b e]
    //   original linear_index = [c0 + aB + 0d + b1 + e0]
    //
    //   Generically:
    //      multi_index[i] = (linear_index / strides[i]) % extents[i];
    //
    //   broadcast multi_index[i] =
    //      (broadcast_linear_index / broadcast_stride[i]) % broadcast_extent[i]
    //   original linear_index = broacast_multiindex . padded_stride
    //
    //   memory_index = 0;
    //   memory_index += ps[j] * ((bli / bs[j]) % be[j]);
    //
    // auto broadcast_multi_index(int broadcast_linear_index) -> auto {
    //     std::vector<int> broadcast_extent { 1, 2, 3, 4 };
    //     std::vector<int> broadcast_stride { 24, 12, 4, 1 };
    //     std::vector<int> padded_stride { 0, 4, 0, 1 };
    //     int              memory_index = 0;
    //     for (int i = 0; i < 4; ++i) {
    //         memory_index += padded_stride[i]
    //                         * ((broadcast_linear_index / broadcast_stride[i])
    //                            * broadcast_extent[i]);
    //     }
    // }
    //  =======================================
    //  reshaping operations as broadcast?
    //  original shape = [3 4]    0 1 2 3 |    0 1    6 7
    //  New Shape = [2 3 2]       4 5 6 7 |->  2 3    8 9 
    //  pad shape = [1 3 2]???    8 9 a b |    4 5    a b
    //  original stride = [4 1]
    //  New stride = [6 2 1]
    //  padded_stride = New stride
    //  broacast stride = New stride
    //  linear_index = reshaped_linear_index
    //  ========================================
    //  broadcasting a reshape
    //  broadcast extent to [2 5 3 2 5]
    //  padding shape [2 1 3 2 1]
    //  padding stride [6 0 2 1 0]
    //  broadcast stride [150 30 10 5 1]
    //  memory_index += ps[j] * ((bli / bs[j]) % be[j]);
    //  ========================================
    //  reshapeing a broadcast
    //  Much like reshape, just change broadcast extent and broadcast stride
    //  new extent [2 1 15 2 5]
    //  new stride [150 150 10 5 1]
    //  ========================================
    //  broadcasting a slice
    //  slice maintains memory layout stride, so pad the slice's stride
    //  original shape = [2 3 4]
    //  original stride = [12 4 1]
    //  slice offset = [0, 0, 1]
    //  slice shape = [2 1 3]
    //  slice stride = [12 4 1]
    //  broadcast shape = [2 5 3]
    //  broadcast stride = [15 3 1]
    //  memory stride = [12 0 1]
    //
    //  ========================================
    //  slicing a broadcast: maintain rank
    //  offset added to pointer
    //  new extents
    //  strides remain same.
    //  original shape = [3 4]
    //  original stride = [4 1]
    //  reshape shape [3 1 4]
    //  reshape stride [4 4 1]
    //
    //  broadcast shape = [3 5 4]
    //  broadcast stride = [20 4 1]
    //  padded_stride = [4 0 1]
    //
    //  slice offset = [1 1 1] = update broadcast ptr
    //  slice shape = [2 2 3] = update broadcast stride
    //  slice stride = [20 4 1] = broadcast stride
    //  padded_stride = [4 0 1] = padded stride
    //
    //
};

} // namespace Fermi
