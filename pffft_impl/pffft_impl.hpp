#pragma once

#include <complex>
#include <memory>

#include "pffft/include/pffft/pffft.h"
#include "pffft/include/pffft/pffft_double.h"
#include "../benchmark/fft_support.hpp"

namespace zlbenchmark {
    namespace detail {
        template <typename F>
        struct PffftApi;

        template <>
        struct PffftApi<float> {
            using Setup = PFFFT_Setup;
            static bool valid(int n, pffft_transform_t kind) { return pffft_is_valid_size(n, kind); }
            static Setup* create(int n, pffft_transform_t kind) { return pffft_new_setup(n, kind); }
            static void destroy(Setup* p) { pffft_destroy_setup(p); }
            static void execute(Setup* p, const float* in, float* out, float* work,
                                pffft_direction_t direction) {
                pffft_transform_ordered(p, in, out, work, direction);
            }
        };

        template <>
        struct PffftApi<double> {
            using Setup = PFFFTD_Setup;
            static bool valid(int n, pffft_transform_t kind) { return pffftd_is_valid_size(n, kind); }
            static Setup* create(int n, pffft_transform_t kind) { return pffftd_new_setup(n, kind); }
            static void destroy(Setup* p) { pffftd_destroy_setup(p); }
            static void execute(Setup* p, const double* in, double* out, double* work,
                                pffft_direction_t direction) {
                pffftd_transform_ordered(p, in, out, work, direction);
            }
        };

        template <typename F, pffft_transform_t Kind>
        class PffftPlan {
            using API = PffftApi<F>;
            std::unique_ptr<typename API::Setup, decltype(&API::destroy)> setup_{nullptr, API::destroy};
            AlignedVector<F> work_;

        public:
            explicit PffftPlan(size_t order) {
                const int n = fft_size(order);
                if (!API::valid(n, Kind)) throw UnsupportedSize("PFFFT does not support this transform size");
                setup_.reset(API::create(n, Kind));
                if (!setup_) throw std::runtime_error("PFFFT setup failed");
                work_.resize(Kind == PFFFT_REAL ? n : 2 * size_t(n));
            }
            void execute(const F* in, F* out, pffft_direction_t direction) {
                API::execute(setup_.get(), in, out, work_.data(), direction);
            }
        };
    }

    template <typename F>
    class PffftFFT final {
        using C = std::complex<F>;
        detail::PffftPlan<F, PFFFT_COMPLEX> plan_;

    public:
        explicit PffftFFT(size_t order) : plan_(order) {}
        void forward(const C* in, C* out) {
            plan_.execute(reinterpret_cast<const F*>(in), reinterpret_cast<F*>(out), PFFFT_FORWARD);
        }
        void backward(const C* in, C* out) {
            plan_.execute(reinterpret_cast<const F*>(in), reinterpret_cast<F*>(out), PFFFT_BACKWARD);
        }
    };

    template <typename F>
    class PffftRFFT final {
        using C = std::complex<F>;
        detail::PffftPlan<F, PFFFT_REAL> plan_;
        AlignedVector<F> packed_;

    public:
        explicit PffftRFFT(size_t order) : plan_(order), packed_(fft_size(order)) {}
        void forward(const F* in, C* out) {
            plan_.execute(in, packed_.data(), PFFFT_FORWARD);
            const size_t half = packed_.size() / 2;
            out[0] = C(packed_[0], 0);
            out[half] = C(packed_[1], 0);
            for (size_t k = 1; k < half; ++k) out[k] = C(packed_[2 * k], packed_[2 * k + 1]);
        }
        void backward(const C* in, F* out) {
            const size_t half = packed_.size() / 2;
            packed_[0] = in[0].real();
            packed_[1] = in[half].real();
            for (size_t k = 1; k < half; ++k) {
                packed_[2 * k] = in[k].real();
                packed_[2 * k + 1] = in[k].imag();
            }
            plan_.execute(packed_.data(), out, PFFFT_BACKWARD);
        }
    };
}
