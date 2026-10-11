#pragma once

#include <array>
#include <complex>
#include <cstddef>

#include "zldsp_fft_rfft-inl.hpp"

namespace zldsp::fft {
    template <typename F>
    class RFFT {
        using Complex = std::complex<F>;

        HWY_STATIC_NAMESPACE::RFFT<F> impl_;

    public:
        explicit RFFT(const size_t rfft_order) : impl_(rfft_order) {
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return impl_.get_size();
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return impl_.get_order();
        }

        /**
         * perform forward RFFT from real input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, Complex* out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform forward RFFT from real input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward RFFT from AoS input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, F* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform backward RFFT from SoA input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, F* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward RFFT and write squared magnitudes
         * @param in_buffer
         * @param out_buffer
         */
        void forward_sqr_mag(F* in_buffer, F* out_buffer) noexcept {
            impl_.forward_sqr_mag(in_buffer, out_buffer);
        }

        /**
         * Circular convolution with a complex AoS spectrum in natural bin order.
         * Input/output contain N real samples; spectrum contains N/2+1 bins,
         * including real DC and Nyquist bins. Native RFFT scaling is retained:
         * an all-ones spectrum returns (N/2)*input. Scale the spectrum by 2/N
         * for normalized convolution.
         *
         * Input and spectrum are preserved, except that input == output is
         * supported. Partial input/output overlap and overlap with the spectrum
         * are unsupported. Buffers need only their element type's alignment.
         * No allocation occurs; concurrent calls require separate RFFT instances.
         */
        void convolve(const F* in_buffer, const Complex* spectrum, F* out_buffer) noexcept {
            impl_.convolve(in_buffer, spectrum, out_buffer);
        }

        /**
         * As above, with separate real/imaginary spectrum arrays of N/2+1 values.
         */
        void convolve(const F* in_buffer, std::array<const F*, 2> spectrum, F* out_buffer) noexcept {
            impl_.convolve(in_buffer, spectrum, out_buffer);
        }

        /**
         * As above, with N/2+1 real spectrum multipliers (negative values allowed).
         */
        void convolve(const F* in_buffer, const F* spectrum, F* out_buffer) noexcept {
            impl_.convolve(in_buffer, spectrum, out_buffer);
        }
    };
}
