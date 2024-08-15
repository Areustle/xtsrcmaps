#include <iterator>

namespace Fermi {

template <typename T>
class TensorViewIterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    TensorViewIterator(pointer ptr) : ptr_(ptr) {}

    reference operator*() const { return *ptr_; }
    pointer operator->() { return ptr_; }
    TensorViewIterator& operator++() {
        ++ptr_;
        return *this;
    }
    TensorViewIterator operator++(int) {
        TensorViewIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    TensorViewIterator& operator--() {
        --ptr_;
        return *this;
    }
    TensorViewIterator operator--(int) {
        TensorViewIterator tmp = *this;
        --(*this);
        return tmp;
    }
    TensorViewIterator& operator+=(difference_type offset) {
        ptr_ += offset;
        return *this;
    }
    TensorViewIterator operator+(difference_type offset) const {
        return TensorViewIterator(ptr_ + offset);
    }
    TensorViewIterator& operator-=(difference_type offset) {
        ptr_ -= offset;
        return *this;
    }
    TensorViewIterator operator-(difference_type offset) const {
        return TensorViewIterator(ptr_ - offset);
    }
    difference_type operator-(const TensorViewIterator& other) const {
        return ptr_ - other.ptr_;
    }
    bool operator==(const TensorViewIterator& other) const { return ptr_ == other.ptr_; }
    bool operator!=(const TensorViewIterator& other) const { return ptr_ != other.ptr_; }
    bool operator<(const TensorViewIterator& other) const { return ptr_ < other.ptr_; }
    bool operator<=(const TensorViewIterator& other) const { return ptr_ <= other.ptr_; }
    bool operator>(const TensorViewIterator& other) const { return ptr_ > other.ptr_; }
    bool operator>=(const TensorViewIterator& other) const { return ptr_ >= other.ptr_; }

  private:
    pointer ptr_;
};

} // namespace Fermi
