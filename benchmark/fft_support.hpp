#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace zlbenchmark {
    // Keep all wrapper sizes representable by the libraries' int interfaces.
    inline int fft_size(const size_t order) {
        if (order > 30) {
            throw std::invalid_argument("FFT order must be in [0, 30]");
        }
        return static_cast<int>(size_t{1} << order);
    }

    class UnsupportedSize : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    template <typename T, size_t Alignment = 64>
    struct AlignedAllocator {
        using value_type = T;
        using is_always_equal = std::true_type;
        template <typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        AlignedAllocator() = default;
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        T* allocate(size_t n) {
            if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
                throw std::bad_alloc();
            }
            return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(Alignment)));
        }

        void deallocate(T* p, size_t) noexcept { ::operator delete(p, std::align_val_t(Alignment)); }
        bool operator==(const AlignedAllocator&) const noexcept { return true; }
    };

    template <typename T>
    using AlignedVector = std::vector<T, AlignedAllocator<T>>;
}
