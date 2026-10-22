#if defined(ZLDSP_FFT_CFFT_INL_HPP_) == defined(HWY_TARGET_TOGGLE)
#ifdef ZLDSP_FFT_CFFT_INL_HPP_
#undef ZLDSP_FFT_CFFT_INL_HPP_
#else
#define ZLDSP_FFT_CFFT_INL_HPP_
#endif

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <hwy/highway.h>

#include "common/zldsp_fft_common_execute.hpp"
#include "common/zldsp_fft_common_init.hpp"

HWY_BEFORE_NAMESPACE();

namespace zldsp::fft::HWY_NAMESPACE {
    /**
     * read-only CFFT plan: every method is const and can run on any number of threads at once,
     * each transform writing only to its in/out buffers and the scratch it is given
     */
    template <typename F>
    class CFFTPlan {
        static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>,
                      "zldsp::fft::CFFT supports float and double");
        static_assert(!HWY_HAVE_SCALABLE, "zldsp::fft requires a fixed-width Highway target");

        using TargetTag = hwy::HWY_NAMESPACE::ScalableTag<F>;
        static constexpr size_t kTargetVectorBytes = TargetTag{}.MaxBytes();
        static_assert(kTargetVectorBytes == 16 || kTargetVectorBytes == 32,
                      "zldsp::fft requires a 128-bit or 256-bit Highway target");

        using Complex = std::complex<F>;

    private:
        common::CFFTState<F> state_;

    public:
        /**
         * byte alignment the scratch passed to a transform must have
         */
        static constexpr size_t kScratchAlignment = common::kScratchAlignment;

        explicit CFFTPlan(const size_t cfft_order) {
            state_.cfft_size = static_cast<size_t>(1) << cfft_order;
            state_.cfft_order = cfft_order;
            common::init_cfft_state(cfft_order, state_);
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return state_.cfft_size;
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return state_.cfft_order;
        }

        /**
         * get the scratch one transform needs
         * @return number of F elements, 0 if the transform needs none
         */
        [[nodiscard]] size_t get_scratch_size() const noexcept {
            return state_.scratch_size;
        }

        /**
         * perform forward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(Complex* in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            execute<true>(common::make_aos(in_buffer), common::make_aos(out_buffer), scratch);
        }

        /**
         * perform backward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(Complex* in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            execute<false>(common::make_aos(in_buffer), common::make_aos(out_buffer), scratch);
        }

        /**
         * perform forward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(Complex* in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            execute<true>(common::make_aos(in_buffer), common::make_soa(out_buffer), scratch);
        }

        /**
         * perform backward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(Complex* in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            execute<false>(common::make_aos(in_buffer), common::make_soa(out_buffer), scratch);
        }

        /**
         * perform forward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(std::array<F*, 2> in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            execute<true>(common::make_soa(in_buffer), common::make_aos(out_buffer), scratch);
        }

        /**
         * perform backward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(std::array<F*, 2> in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            execute<false>(common::make_soa(in_buffer), common::make_aos(out_buffer), scratch);
        }

        /**
         * perform forward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            execute<true>(common::make_soa(in_buffer), common::make_soa(out_buffer), scratch);
        }

        /**
         * perform backward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            execute<false>(common::make_soa(in_buffer), common::make_soa(out_buffer), scratch);
        }

    private:
        /**
         * execute CFFT
         * @tparam is_forward
         * @tparam InPtr
         * @tparam OutPtr
         * @param in
         * @param out
         * @param scratch
         */
        template <bool is_forward, typename InPtr, typename OutPtr>
        void execute(InPtr in, OutPtr out, std::span<F> scratch) const noexcept {
            HWY_DASSERT(scratch.size() >= state_.scratch_size);
            HWY_DASSERT(reinterpret_cast<std::uintptr_t>(scratch.data()) % kScratchAlignment == 0);
            common::execute_cfft<is_forward>(state_, in, out, scratch.data());
        }
    };

    /**
     * CFFT plan with its own scratch, allocated in the constructor;
     * one instance must not transform on two threads at once
     */
    template <typename F>
    class CFFT {
        using Complex = std::complex<F>;

    private:
        CFFTPlan<F> plan_;
        hwy::AlignedFreeUniquePtr<F[]> scratch_;

    public:
        explicit CFFT(const size_t cfft_order)
            : plan_(cfft_order), scratch_(common::allocate_aligned<F>(plan_.get_scratch_size())) {
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return plan_.get_size();
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return plan_.get_order();
        }

        /**
         * perform forward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, Complex* out_buffer) noexcept {
            plan_.forward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform backward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, Complex* out_buffer) noexcept {
            plan_.backward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform forward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            plan_.forward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform backward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            plan_.backward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform forward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            plan_.forward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform backward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            plan_.backward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform forward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
            plan_.forward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform backward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
            plan_.backward(in_buffer, out_buffer, get_scratch());
        }

    private:
        [[nodiscard]] std::span<F> get_scratch() noexcept {
            return {scratch_.get(), plan_.get_scratch_size()};
        }
    };
}

HWY_AFTER_NAMESPACE();

#endif
