#if defined(ZLDSP_FFT_RFFT_INL_HPP_) == defined(HWY_TARGET_TOGGLE)
#ifdef ZLDSP_FFT_RFFT_INL_HPP_
#undef ZLDSP_FFT_RFFT_INL_HPP_
#else
#define ZLDSP_FFT_RFFT_INL_HPP_
#endif

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include <hwy/highway.h>

#include "common/zldsp_fft_common_execute.hpp"
#include "common/zldsp_fft_common_init.hpp"

HWY_BEFORE_NAMESPACE();

namespace zldsp::fft::HWY_NAMESPACE {
    /**
     * read-only RFFT plan: every method is const and can run on any number of threads at once,
     * each transform writing only to its in/out buffers and the scratch it is given
     */
    template <typename F>
    class RFFTPlan {
        static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>,
                      "zldsp::fft::RFFT supports float and double");
        static_assert(!HWY_HAVE_SCALABLE, "zldsp::fft requires a fixed-width Highway target");

        using TargetTag = hwy::HWY_NAMESPACE::ScalableTag<F>;
        static constexpr size_t kTargetVectorBytes = TargetTag{}.MaxBytes();
        static_assert(kTargetVectorBytes == 16 || kTargetVectorBytes == 32,
                      "zldsp::fft requires a 128-bit or 256-bit Highway target");

        using Complex = std::complex<F>;

        /**
         * real and imaginary offsets of a SoA buffer inside the scratch, in F elements
         */
        struct ScratchSoA {
            size_t real = 0;
            size_t imag = 0;
        };

    private:
#ifdef ZLDSP_EXPERIMENT_RUNTIME_OUTPUT_REUSE
        bool reuse_outputs_ = true;
#else
        static constexpr bool reuse_outputs_ = true;
#endif
        common::CFFTState<F> state_;
        size_t rfft_size_;
        hwy::AlignedFreeUniquePtr<F[]> rfft_w_;
        ScratchSoA forward_cfft_output_;
        ScratchSoA backward_cfft_input_;
        size_t scratch_size_ = 0;

    public:
        /**
         * byte alignment the scratch passed to a transform must have
         */
        static constexpr size_t kScratchAlignment = common::kScratchAlignment;

#ifdef ZLDSP_EXPERIMENT_RUNTIME_OUTPUT_REUSE
        /**
         * select RFFT output reuse for throughput comparisons
         * @param enabled
         */
        void set_experiment_output_reuse(const bool enabled) noexcept {
            reuse_outputs_ = enabled;
        }
#endif
        explicit RFFTPlan(const size_t rfft_order) : rfft_size_(static_cast<size_t>(1) << rfft_order) {
            state_.cfft_order = rfft_order - 1;
            state_.cfft_size = static_cast<size_t>(1) << state_.cfft_order;
            common::init_cfft_state(state_.cfft_order, state_);
            // the CFFT scratch comes first; the RFFT keeps its CFFT output and backward input either
            // inside the CFFT workspace or in a region of its own after it
            if (state_.cfft_order < 6) {
                forward_cfft_output_ = {0, state_.cfft_size};
                backward_cfft_input_ = forward_cfft_output_;
                scratch_size_ = 2 * state_.cfft_size;
            } else if (state_.num_macro_stages == 0) {
                if (state_.micro_stages.size() % 2 == 0) {
                    forward_cfft_output_ = {0, state_.micro_stride};
                } else {
                    forward_cfft_output_ = {2 * state_.micro_stride, 3 * state_.micro_stride};
                }
                backward_cfft_input_ = {0, state_.micro_stride};
                scratch_size_ = state_.scratch_size;
            } else {
                const size_t rfft_workspace_stride = state_.cfft_size + common::get_cache_color_padding<F>();
                const size_t rfft_workspace =
                    common::align_scratch_count<F>(state_.scratch_size) + common::get_cache_color_padding<F>();
                forward_cfft_output_ = {rfft_workspace, rfft_workspace + rfft_workspace_stride};
                backward_cfft_input_ = forward_cfft_output_;
                scratch_size_ = rfft_workspace + 2 * rfft_workspace_stride;
            }
            common::generate_rfft_pre_post_w(state_.cfft_order, rfft_w_);
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return rfft_size_;
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return state_.cfft_order + 1;
        }

        /**
         * get the scratch one transform needs
         * @return number of F elements
         */
        [[nodiscard]] size_t get_scratch_size() const noexcept {
            return scratch_size_;
        }

        /**
         * perform forward RFFT from real input to AoS output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(F* in_buffer, Complex* out_buffer, std::span<F> scratch) const noexcept {
            execute_forward(in_buffer, common::make_aos(out_buffer), scratch);
        }

        /**
         * perform forward RFFT from real input to SoA output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward(F* in_buffer, std::array<F*, 2> out_buffer, std::span<F> scratch) const noexcept {
            execute_forward(in_buffer, common::make_soa(out_buffer), scratch);
        }

        /**
         * perform backward RFFT from AoS input to real output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(Complex* in_buffer, F* out_buffer, std::span<F> scratch) const noexcept {
            execute_backward(common::make_aos(in_buffer), out_buffer, scratch);
        }

        /**
         * perform backward RFFT from SoA input to real output
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void backward(std::array<F*, 2> in_buffer, F* out_buffer, std::span<F> scratch) const noexcept {
            execute_backward(common::make_soa(in_buffer), out_buffer, scratch);
        }

        /**
         * perform forward RFFT and write squared magnitudes
         * @param in_buffer
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void forward_sqr_mag(F* in_buffer, F* out_buffer, std::span<F> scratch) const noexcept {
            execute_forward_sqr_mag(in_buffer, out_buffer, scratch);
        }

        /**
         * perform RFFT convolution with AoS spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void convolve(const F* in_buffer, const Complex* spectrum, F* out_buffer, std::span<F> scratch) const noexcept {
            execute_convolve(in_buffer, common::make_aos(spectrum), out_buffer, scratch);
        }

        /**
         * perform RFFT convolution with SoA spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void convolve(const F* in_buffer, std::array<const F*, 2> spectrum, F* out_buffer,
                      std::span<F> scratch) const noexcept {
            execute_convolve(in_buffer, common::make_soa(spectrum), out_buffer, scratch);
        }

        /**
         * perform RFFT convolution with real spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         * @param scratch at least get_scratch_size() elements, aligned to kScratchAlignment bytes
         */
        void convolve(const F* in_buffer, const F* spectrum, F* out_buffer, std::span<F> scratch) const noexcept {
            execute_convolve(in_buffer, spectrum, out_buffer, scratch);
        }

    private:
        /**
         * check the scratch passed to a transform
         * @param scratch
         * @return scratch base pointer
         */
        [[nodiscard]] F* check_scratch(const std::span<F> scratch) const noexcept {
            HWY_DASSERT(scratch.size() >= scratch_size_);
            HWY_DASSERT(reinterpret_cast<std::uintptr_t>(scratch.data()) % kScratchAlignment == 0);
            return scratch.data();
        }

        /**
         * make a SoA pointer to a buffer inside the scratch
         * @param scratch
         * @param offsets
         */
        [[nodiscard]] static common::SoAPtr<F> at(F* scratch, const ScratchSoA offsets) noexcept {
            return common::make_soa<F>({scratch + offsets.real, scratch + offsets.imag});
        }

        /**
         * check whether two scalar buffer ranges overlap
         * @param a
         * @param a_count
         * @param b
         * @param b_count
         */
        [[nodiscard]] static bool buffers_overlap(const F* a, const size_t a_count,
                                                 const F* b, const size_t b_count) noexcept {
            constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
            if (a_count > max_address / sizeof(F) || b_count > max_address / sizeof(F)) {
                return true;
            }
            const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
            const auto b_begin = reinterpret_cast<std::uintptr_t>(b);
            const auto a_bytes = static_cast<std::uintptr_t>(a_count) * sizeof(F);
            const auto b_bytes = static_cast<std::uintptr_t>(b_count) * sizeof(F);
            if (a_bytes > max_address - a_begin || b_bytes > max_address - b_begin) {
                return true;
            }
            return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
        }

        /**
         * check whether an RFFT input spectrum overlaps the real output
         * @tparam InPtr
         * @param in
         * @param out
         */
        template <typename InPtr>
        [[nodiscard]] bool backward_input_overlaps_output(InPtr in, const F* out) const noexcept {
            if constexpr (std::is_same_v<InPtr, common::SoAPtr<F>>) {
                return buffers_overlap(in.real, state_.cfft_size + 1, out, rfft_size_) ||
                       buffers_overlap(in.imag, state_.cfft_size + 1, out, rfft_size_);
            } else {
                return buffers_overlap(in.interleaved, rfft_size_ + 2, out, rfft_size_);
            }
        }

        /**
         * execute forward RFFT and write squared magnitudes
         * @param in
         * @param out
         * @param scratch
         */
        void execute_forward_sqr_mag(F* in, F* out, const std::span<F> scratch) const noexcept {
            F* data = check_scratch(scratch);
            const auto forward_cfft_output = at(data, forward_cfft_output_);
            common::execute_cfft<true>(state_, common::make_aos(reinterpret_cast<Complex*>(in)),
                                       forward_cfft_output, data);
            common::execute_rfft_forward_sqr_mag_post(forward_cfft_output, out,
                                                      state_.cfft_order, rfft_w_.get());
        }

        /**
         * execute RFFT convolution
         * @tparam SpectrumPtr
         * @param in
         * @param spectrum
         * @param out
         * @param scratch
         */
        template <typename SpectrumPtr>
        void execute_convolve(const F* in, SpectrumPtr spectrum, F* out, const std::span<F> scratch) const noexcept {
            F* data = check_scratch(scratch);
            const auto forward_cfft_output = at(data, forward_cfft_output_);
            const auto backward_cfft_input = at(data, backward_cfft_input_);
            common::execute_cfft<true>(state_, common::AoSPtr<const F>{in}, forward_cfft_output, data);
            common::execute_rfft_convolve<F>(
                {forward_cfft_output.real, forward_cfft_output.imag}, spectrum,
                {backward_cfft_input.real, backward_cfft_input.imag}, state_.cfft_order, rfft_w_.get());
            common::execute_cfft<false>(state_, backward_cfft_input,
                                       common::make_aos(reinterpret_cast<Complex*>(out)), data);
        }

        /**
         * execute forward RFFT
         * @tparam OutPtr
         * @param in
         * @param out
         * @param scratch
         */
        template <typename OutPtr>
        void execute_forward(F* in, OutPtr out, const std::span<F> scratch) const noexcept {
            F* data = check_scratch(scratch);
            if constexpr (std::is_same_v<OutPtr, common::SoAPtr<F>>) {
                const size_t bins = state_.cfft_size + 1;
                if (state_.num_macro_stages != 0 && reuse_outputs_ &&
                    !buffers_overlap(in, rfft_size_, out.real, bins) &&
                    !buffers_overlap(in, rfft_size_, out.imag, bins) &&
                    !buffers_overlap(out.real, bins, out.imag, bins)) {
                    common::execute_cfft<true>(state_, common::make_aos(reinterpret_cast<Complex*>(in)), out, data);
                    common::execute_rfft_forward_post_inplace(out, state_.cfft_order, rfft_w_.get());
                    return;
                }
            } else {
                if (state_.num_macro_stages != 0 && reuse_outputs_ &&
                    !buffers_overlap(in, rfft_size_, out.interleaved, rfft_size_ + 2)) {
                    common::execute_cfft<true>(state_, common::make_aos(reinterpret_cast<Complex*>(in)), out, data);
                    common::execute_rfft_forward_post_inplace(out, state_.cfft_order, rfft_w_.get());
                    return;
                }
            }
            const auto forward_cfft_output = at(data, forward_cfft_output_);
            common::execute_cfft<true>(state_, common::make_aos(reinterpret_cast<Complex*>(in)),
                                       forward_cfft_output, data);
            common::execute_rfft_forward_post(forward_cfft_output, out,
                                              state_.cfft_order, rfft_w_.get());
        }

        /**
         * execute backward RFFT
         * @tparam InPtr
         * @param in
         * @param out
         * @param scratch
         */
        template <typename InPtr>
        void execute_backward(InPtr in, F* out, const std::span<F> scratch) const noexcept {
            F* data = check_scratch(scratch);
            if (state_.num_macro_stages != 0 && reuse_outputs_ && !backward_input_overlaps_output(in, out)) {
                // reuse the real output as SoA scratch, ending each restricted view before changing layouts
                common::execute_rfft_backward_pre(in, common::make_soa<F>({out, out + state_.cfft_size}),
                                                   state_.cfft_order, rfft_w_.get());
                common::execute_hybrid_macro<false>(state_, common::make_soa<F>({out, out + state_.cfft_size}),
                                                    data);
                common::dispatch_hybrid_tail<false>(state_, common::make_aos<F>(reinterpret_cast<Complex*>(out)),
                                                    data);
                return;
            }
            const auto backward_cfft_input = at(data, backward_cfft_input_);
            common::execute_rfft_backward_pre(in, backward_cfft_input,
                                              state_.cfft_order, rfft_w_.get());
            common::execute_cfft<false>(state_, backward_cfft_input,
                                        common::make_aos<F>(reinterpret_cast<Complex*>(out)), data);
        }
    };

    /**
     * RFFT plan with its own scratch, allocated in the constructor;
     * one instance must not transform on two threads at once
     */
    template <typename F>
    class RFFT {
        using Complex = std::complex<F>;

    private:
        RFFTPlan<F> plan_;
        hwy::AlignedFreeUniquePtr<F[]> scratch_;

    public:
#ifdef ZLDSP_EXPERIMENT_RUNTIME_OUTPUT_REUSE
        /**
         * select RFFT output reuse for throughput comparisons
         * @param enabled
         */
        void set_experiment_output_reuse(const bool enabled) noexcept {
            plan_.set_experiment_output_reuse(enabled);
        }
#endif
        explicit RFFT(const size_t rfft_order)
            : plan_(rfft_order), scratch_(common::allocate_aligned<F>(plan_.get_scratch_size())) {
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return plan_.get_size();
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return plan_.get_order();
        }

        /**
         * perform forward RFFT from real input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, Complex* out_buffer) noexcept {
            plan_.forward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform forward RFFT from real input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            plan_.forward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform backward RFFT from AoS input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, F* out_buffer) noexcept {
            plan_.backward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform backward RFFT from SoA input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, F* out_buffer) noexcept {
            plan_.backward(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform forward RFFT and write squared magnitudes
         * @param in_buffer
         * @param out_buffer
         */
        void forward_sqr_mag(F* in_buffer, F* out_buffer) noexcept {
            plan_.forward_sqr_mag(in_buffer, out_buffer, get_scratch());
        }

        /**
         * perform RFFT convolution with AoS spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         */
        void convolve(const F* in_buffer, const Complex* spectrum, F* out_buffer) noexcept {
            plan_.convolve(in_buffer, spectrum, out_buffer, get_scratch());
        }

        /**
         * perform RFFT convolution with SoA spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         */
        void convolve(const F* in_buffer, std::array<const F*, 2> spectrum, F* out_buffer) noexcept {
            plan_.convolve(in_buffer, spectrum, out_buffer, get_scratch());
        }

        /**
         * perform RFFT convolution with real spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         */
        void convolve(const F* in_buffer, const F* spectrum, F* out_buffer) noexcept {
            plan_.convolve(in_buffer, spectrum, out_buffer, get_scratch());
        }

    private:
        [[nodiscard]] std::span<F> get_scratch() noexcept {
            return {scratch_.get(), plan_.get_scratch_size()};
        }
    };
}

HWY_AFTER_NAMESPACE();

#endif
