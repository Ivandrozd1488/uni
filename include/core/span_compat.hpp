#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#if defined(__has_include)
#  if __has_include(<span>)
#    include <span>
#  endif
#endif

#if !defined(__cpp_lib_span) || (__cpp_lib_span < 202002L)
namespace std {

template <class T>
class span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using pointer = T*;
    using reference = T&;
    using iterator = T*;

    constexpr span() noexcept = default;
    constexpr span(pointer ptr, size_type count) noexcept : data_(ptr), size_(count) {}

    template <class U, class Alloc,
              class = std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>>
    constexpr span(std::vector<U, Alloc>& values) noexcept : data_(values.data()), size_(values.size()) {}

    template <class U, class Alloc,
              class = std::enable_if_t<std::is_convertible_v<const U (*)[], T (*)[]>>>
    constexpr span(const std::vector<U, Alloc>& values) noexcept : data_(values.data()), size_(values.size()) {}

    template <class U,
              class = std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>>
    constexpr span(const span<U>& other) noexcept : data_(other.data()), size_(other.size()) {}

    [[nodiscard]] constexpr iterator begin() const noexcept { return data_; }
    [[nodiscard]] constexpr iterator end() const noexcept { return data_ + size_; }
    [[nodiscard]] constexpr reference operator[](size_type index) const noexcept { return data_[index]; }
    [[nodiscard]] constexpr pointer data() const noexcept { return data_; }
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

private:
    pointer data_ = nullptr;
    size_type size_ = 0;
};

} // namespace std
#endif
