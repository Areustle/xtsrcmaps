#pragma once

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>


// Internal details namespace
namespace Fermi::tensor {

// Concept to ensure T is a valid data type
template <typename T>
concept DataType = std::is_arithmetic_v<T> || std::is_class_v<T>;

// Concept to ensure rank is positive
template <std::size_t R>
concept PositiveRank = (R > 0);

// Concept to ensure all types in the parameter pack are integral
template <typename... Ts>
concept AllIntegral = (std::integral<Ts> && ...);


// Helper to find the maximum alignment between two sizes
template <std::size_t A, std::size_t B>
struct MaxAlignment {
    static constexpr std::size_t value = A > B ? A : B;
};

template <typename T, std::size_t Alignment = alignof(T)>
    requires((Alignment >= alignof(T) || std::is_same_v<T, bool>)
             && std::is_default_constructible_v<T>
             && std::is_copy_constructible_v<T>)
struct AlignedAllocator {
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Ensure alignment is at least the size of a void pointer
    static constexpr std::size_t adjusted_alignment
        = MaxAlignment<Alignment, alignof(std::max_align_t)>::value;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_type n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        size_type aligned_size = n * sizeof(T);
        if (aligned_size % adjusted_alignment != 0) {
            aligned_size
                += adjusted_alignment - (aligned_size % adjusted_alignment);
        }
        void* ptr = std::aligned_alloc(adjusted_alignment, aligned_size);
        if (!ptr) { throw std::bad_alloc(); }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_type) noexcept { std::free(p); }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    // Comparison operators
    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

} // namespace Fermi::tensor

namespace Fermi{

// Forward Declare Broadcast Tensor for use with Tensor.hpp
template <typename T>
class BroadcastTensor{};

} // namespace Fermi


