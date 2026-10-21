#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <span>

#include "zldsp_fft_cfft-inl.hpp"

namespace zldsp::fft {
    /**
     * read-only CFFT plan: every method is const and can run on any number of threads at once,
     * each transform writing only to its in/out buffers and the scratch it is given;
     * CFFT is the same plan with its own scratch, for the scratch-free API
     */
    template <typename F>
    class CFFTPlan {
        using Complex = std::complex<F>;

        HWY_STATIC_NAMESPACE::CFFTPlan<F> impl_;

    public:
        /**
         * byte alignment the scratch passed to a transform must have
         */
        static constexpr size_t kScratchAlignment = HWY_STATIC_NAMESPACE::CFFTPlan<F>::kScratchAlignment;

        explicit CFFTPlan(const size_t cfft_order) : impl_(cfft_order) {
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return impl_.get_size();
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return impl_.get_order();
        }

        /**
         * get the scratch one transform needs
         * @return number of F elements
         */
        [[nodiscard]] size_t get_scratch_size() const noexcept {
            return impl_.get_scratch_size();
        }

        /**
         * perform forward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(Complex* in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            impl_.forward(in_buffer, out_buffer, scratch);
        }

        /**
         * perform backward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(Complex* in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            impl_.backward(in_buffer, out_buffer, scratch);
        }

        /**
         * perform forward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(Complex* in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            impl_.forward(in_buffer, out_buffer, scratch);
        }

        /**
         * perform backward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(Complex* in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            impl_.backward(in_buffer, out_buffer, scratch);
        }

        /**
         * perform forward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(std::array<F*, 2> in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            impl_.forward(in_buffer, out_buffer, scratch);
        }

        /**
         * perform backward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(std::array<F*, 2> in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            impl_.backward(in_buffer, out_buffer, scratch);
        }

        /**
         * perform forward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            impl_.forward(in_buffer, out_buffer, scratch);
        }

        /**
         * perform backward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            impl_.backward(in_buffer, out_buffer, scratch);
        }
    };

    template <typename F>
    class CFFT {
        using Complex = std::complex<F>;

        HWY_STATIC_NAMESPACE::CFFT<F> impl_;

    public:
        explicit CFFT(const size_t cfft_order) : impl_(cfft_order) {
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return impl_.get_size();
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return impl_.get_order();
        }

        /**
         * perform forward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, Complex* out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, Complex* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }
    };
}
