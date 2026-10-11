#pragma once

#include <kfr/dft.hpp>
#include <complex>
#include "../benchmark/fft_support.hpp"

namespace zlbenchmark {
    template <typename F>
    class KFRFFT final {
        using C = std::complex<F>;

    public:
        explicit KFRFFT(const size_t order) :
            fft_plan_(fft_size(order)) {
            temp_buffer_.resize(fft_plan_.temp_size);
        }

        void forward(C* in_buffer, C* out_buffer) {
            fft_plan_.execute(reinterpret_cast<kfr::complex<F>*>(out_buffer),
                              reinterpret_cast<const kfr::complex<F>*>(in_buffer), temp_buffer_.data());
        }

        void backward(C* in_buffer, C* out_buffer) {
            fft_plan_.execute(reinterpret_cast<kfr::complex<F>*>(out_buffer),
                              reinterpret_cast<const kfr::complex<F>*>(in_buffer), temp_buffer_.data(), true);
        }

    private:
        kfr::dft_plan<F> fft_plan_;
        kfr::univector<kfr::u8> temp_buffer_;
    };

    template <typename F>
    class KFRRFFT final {
        using C = std::complex<F>;

    public:
        explicit KFRRFFT(const size_t order) :
            fft_plan_(fft_size(order)) {
            temp_buffer_.resize(fft_plan_.temp_size);
        }

        void forward(const F* in_buffer, C* out_buffer) {
            fft_plan_.execute(reinterpret_cast<kfr::complex<F>*>(out_buffer), in_buffer, temp_buffer_.data());
        }

        void backward(C* in_buffer, F* out_buffer) {
            fft_plan_.execute(out_buffer, reinterpret_cast<const kfr::complex<F>*>(in_buffer), temp_buffer_.data());
        }

    private:
        kfr::dft_plan_real<F> fft_plan_;
        kfr::univector<kfr::u8> temp_buffer_;
    };
}
