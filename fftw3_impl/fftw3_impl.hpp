#pragma once

#include <complex>
#include <memory>
#include <fftw3.h>

#include "../benchmark/fft_support.hpp"

namespace zlbenchmark {
    namespace detail {
        template <typename F>
        struct FFTWApi;

        template <>
        struct FFTWApi<float> {
            using Complex = fftwf_complex;
            using Plan = fftwf_plan;
            static Complex* alloc_complex(size_t n) { return fftwf_alloc_complex(n); }
            static float* alloc_real(size_t n) { return fftwf_alloc_real(n); }
            static void free(void* p) { fftwf_free(p); }
            static void destroy(Plan p) { fftwf_destroy_plan(p); }
            static Plan cfft(int n, Complex* in, Complex* out, int direction, unsigned flags) {
                return fftwf_plan_dft_1d(n, in, out, direction, flags);
            }
            static Plan rfft(int n, float* in, Complex* out, unsigned flags) {
                return fftwf_plan_dft_r2c_1d(n, in, out, flags);
            }
            static Plan irfft(int n, Complex* in, float* out, unsigned flags) {
                return fftwf_plan_dft_c2r_1d(n, in, out, flags);
            }
            static void cfft(Plan p, Complex* in, Complex* out) { fftwf_execute_dft(p, in, out); }
            static void rfft(Plan p, float* in, Complex* out) { fftwf_execute_dft_r2c(p, in, out); }
            static void irfft(Plan p, Complex* in, float* out) { fftwf_execute_dft_c2r(p, in, out); }
        };

        template <>
        struct FFTWApi<double> {
            using Complex = fftw_complex;
            using Plan = fftw_plan;
            static Complex* alloc_complex(size_t n) { return fftw_alloc_complex(n); }
            static double* alloc_real(size_t n) { return fftw_alloc_real(n); }
            static void free(void* p) { fftw_free(p); }
            static void destroy(Plan p) { fftw_destroy_plan(p); }
            static Plan cfft(int n, Complex* in, Complex* out, int direction, unsigned flags) {
                return fftw_plan_dft_1d(n, in, out, direction, flags);
            }
            static Plan rfft(int n, double* in, Complex* out, unsigned flags) {
                return fftw_plan_dft_r2c_1d(n, in, out, flags);
            }
            static Plan irfft(int n, Complex* in, double* out, unsigned flags) {
                return fftw_plan_dft_c2r_1d(n, in, out, flags);
            }
            static void cfft(Plan p, Complex* in, Complex* out) { fftw_execute_dft(p, in, out); }
            static void rfft(Plan p, double* in, Complex* out) { fftw_execute_dft_r2c(p, in, out); }
            static void irfft(Plan p, Complex* in, double* out) { fftw_execute_dft_c2r(p, in, out); }
        };

        template <typename F>
        struct FFTWPlans {
            using API = FFTWApi<F>;
            typename API::Plan forward = nullptr, backward = nullptr;
            FFTWPlans() = default;
            FFTWPlans(const FFTWPlans&) = delete;
            FFTWPlans& operator=(const FFTWPlans&) = delete;
            ~FFTWPlans() {
                if (forward) API::destroy(forward);
                if (backward) API::destroy(backward);
            }
            void check() const {
                if (!forward || !backward) throw std::runtime_error("FFTW-compatible plan creation failed");
            }
        };
    }

    // ArmPL uses these same FFTW-compatible entry points. Plans are out of place
    // and preserve their input in both directions, including the 1-D c2r plan.
    template <typename F, unsigned Flags = FFTW_MEASURE>
    class FFTW3FFT final {
        using C = std::complex<F>;
        using API = detail::FFTWApi<F>;
        using NativeC = typename API::Complex;
        detail::FFTWPlans<F> plans_;

    public:
        explicit FFTW3FFT(size_t order) {
            const auto n = fft_size(order);
            std::unique_ptr<NativeC, decltype(&API::free)> in(API::alloc_complex(n), API::free);
            std::unique_ptr<NativeC, decltype(&API::free)> out(API::alloc_complex(n), API::free);
            if (!in || !out) throw std::bad_alloc();
            plans_.forward = API::cfft(n, in.get(), out.get(), FFTW_FORWARD, Flags | FFTW_PRESERVE_INPUT);
            plans_.backward = API::cfft(n, in.get(), out.get(), FFTW_BACKWARD, Flags | FFTW_PRESERVE_INPUT);
            plans_.check();
        }
        void forward(C* in, C* out) {
            API::cfft(plans_.forward, reinterpret_cast<NativeC*>(in), reinterpret_cast<NativeC*>(out));
        }
        void backward(C* in, C* out) {
            API::cfft(plans_.backward, reinterpret_cast<NativeC*>(in), reinterpret_cast<NativeC*>(out));
        }
    };

    template <typename F, unsigned Flags = FFTW_MEASURE>
    class FFTW3RFFT final {
        using C = std::complex<F>;
        using API = detail::FFTWApi<F>;
        using NativeC = typename API::Complex;
        detail::FFTWPlans<F> plans_;

    public:
        explicit FFTW3RFFT(size_t order) {
            const auto n = fft_size(order);
            std::unique_ptr<F, decltype(&API::free)> real(API::alloc_real(n), API::free);
            std::unique_ptr<NativeC, decltype(&API::free)> complex(API::alloc_complex(n / 2 + 1), API::free);
            if (!real || !complex) throw std::bad_alloc();
            plans_.forward = API::rfft(n, real.get(), complex.get(), Flags | FFTW_PRESERVE_INPUT);
            plans_.backward = API::irfft(n, complex.get(), real.get(), Flags | FFTW_PRESERVE_INPUT);
            plans_.check();
        }
        void forward(F* in, C* out) {
            API::rfft(plans_.forward, in, reinterpret_cast<NativeC*>(out));
        }
        void backward(C* in, F* out) {
            API::irfft(plans_.backward, reinterpret_cast<NativeC*>(in), out);
        }
    };
}
