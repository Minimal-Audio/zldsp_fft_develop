#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>

#include "benchmark_include.hpp"

namespace zlbenchmark {
    inline bool finite(F x) { return std::isfinite(x); }
    inline bool finite(C x) { return finite(x.real()) && finite(x.imag()); }
    template <typename T>
    bool all_finite(const AlignedVector<T>& values) {
        return std::all_of(values.begin(), values.end(), [](T x) { return finite(x); });
    }

    inline int parse_integer(const char* text, int low, int high) {
        size_t end = 0;
        const int value = std::stoi(text, &end);
        if (end != std::string_view(text).size() || value < low || value > high) {
            throw std::invalid_argument("integer argument outside the supported range");
        }
        return value;
    }

    // A single seed describes identical data for every library and layout.
    // Structured inputs cover zero, DC, Nyquist, phase/sign and bin ordering.
    inline AlignedVector<C> make_input(size_t n, bool real, bool backward, int pattern, unsigned seed) {
        const size_t count = real && backward ? n / 2 + 1 : n;
        AlignedVector<C> values(count);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<F> random(F(-1), F(1));
        for (size_t i = 0; i < count; ++i) {
            switch (pattern) {
            case 0: values[i] = C(0, 0); break;
            case 1: values[i] = C(i == 1 ? 1 : 0, 0); break;
            case 2: values[i] = C(1, 0); break;
            case 3: values[i] = C(i % 2 ? -1 : 1, 0); break;
            case 4: {
                const long double angle = 2 * std::numbers::pi_v<long double> * 3 * i / n;
                values[i] = C(static_cast<F>(std::cos(angle)), static_cast<F>(std::sin(angle)));
                break;
            }
            default: {
                const F re = random(rng), im = random(rng);
                values[i] = C(re, im);
                break;
            }
            }
            if (real && !backward) values[i].imag(0);
        }
        if (real && backward) {
            values.front().imag(0);
            values.back().imag(0);
        }
        return values;
    }

    template <bool SoA>
    class ComplexBuffer {
        AlignedVector<C> aos_;
        AlignedVector<F> re_, im_;

    public:
        explicit ComplexBuffer(size_t n) : aos_(SoA ? 0 : n), re_(SoA ? n : 0), im_(SoA ? n : 0) {
            const F nan = std::numeric_limits<F>::quiet_NaN();
            std::fill(aos_.begin(), aos_.end(), C(nan, nan));
            std::fill(re_.begin(), re_.end(), nan);
            std::fill(im_.begin(), im_.end(), nan);
        }
        auto data() {
            if constexpr (SoA) return std::array<F*, 2>{re_.data(), im_.data()};
            else return aos_.data();
        }
        auto data() const {
            if constexpr (SoA) return std::array<const F*, 2>{re_.data(), im_.data()};
            else return aos_.data();
        }
        size_t size() const { return SoA ? re_.size() : aos_.size(); }
        C operator[](size_t i) const {
            if constexpr (SoA) return C(re_[i], im_[i]);
            else return aos_[i];
        }
        void assign(const AlignedVector<C>& values) {
            if constexpr (SoA) {
                for (size_t i = 0; i < size(); ++i) {
                    re_[i] = values[i].real();
                    im_[i] = values[i].imag();
                }
            } else std::copy(values.begin(), values.end(), aos_.begin());
        }
        bool is_finite() const {
            if constexpr (SoA) return all_finite(re_) && all_finite(im_);
            else return all_finite(aos_);
        }
    };

    template <bool Backward, bool InputSoA = false, bool OutputSoA = false, typename FFT = CFFTClass>
    struct CfftCase {
        static constexpr bool backward = Backward, real = false, magnitude = false;
        static constexpr bool convolution = false;
        static constexpr const char* direction = Backward ? "backward" : "forward";
        static constexpr F output_scale = 1;
        static constexpr const char* api = InputSoA ? (OutputSoA ? "soa_soa" : "soa_aos")
                                                    : (OutputSoA ? "aos_soa" : "aos_aos");
        FFT fft;
        ComplexBuffer<InputSoA> in;
        ComplexBuffer<OutputSoA> out;
        AlignedVector<C> original;

        explicit CfftCase(int order) : fft(order), in(fft_size(order)), out(fft_size(order)) {
            original = make_input(in.size(), false, Backward, 5, 42);
            reset();
        }
        void reset() { in.assign(original); }
        void execute() {
            if constexpr (Backward) fft.backward(in.data(), out.data());
            else fft.forward(in.data(), out.data());
        }
        bool input_unchanged() const {
            for (size_t i = 0; i < in.size(); ++i) if (in[i] != original[i]) return false;
            return true;
        }
        auto output_data() { return out.data(); }
        bool output_finite() const { return out.is_finite(); }
        size_t output_size() const { return out.size(); }
        C output(size_t i) const { return out[i]; }
    };

    template <bool Backward, bool SoA = false, bool Magnitude = false, typename FFT = RFFTClass>
    struct RfftCase {
        static_assert(!Magnitude || !Backward);
        static constexpr bool backward = Backward, real = true, magnitude = Magnitude;
        static constexpr bool convolution = false;
        static constexpr const char* direction = Backward ? "backward" : "forward";
        static constexpr F output_scale = Backward ? rfft_backward_scale : F(1);
        static constexpr const char* api = Magnitude ? "real_power"
            : Backward ? (SoA ? "soa_real" : "aos_real") : (SoA ? "real_soa" : "real_aos");
        FFT fft;
        AlignedVector<F> time;
        ComplexBuffer<SoA> spectrum;
        AlignedVector<F> power;
        AlignedVector<C> original;

        explicit RfftCase(int order) : fft(order), time(fft_size(order), std::numeric_limits<F>::quiet_NaN()),
            spectrum(time.size() / 2 + 1), power(Magnitude ? spectrum.size() : 0, std::numeric_limits<F>::quiet_NaN()) {
            original = make_input(time.size(), true, Backward, 5, 42);
            reset();
        }
        void reset() {
            if constexpr (Backward) spectrum.assign(original);
            else for (size_t i = 0; i < time.size(); ++i) time[i] = original[i].real();
        }
        void execute() {
            if constexpr (Magnitude) fft.forward_sqr_mag(time.data(), power.data());
            else if constexpr (Backward) fft.backward(spectrum.data(), time.data());
            else fft.forward(time.data(), spectrum.data());
        }
        bool input_unchanged() const {
            for (size_t i = 0; i < original.size(); ++i) {
                if constexpr (Backward) { if (spectrum[i] != original[i]) return false; }
                else if (time[i] != original[i].real()) return false;
            }
            return true;
        }
        auto output_data() {
            if constexpr (Magnitude) return power.data();
            else if constexpr (Backward) return time.data();
            else return spectrum.data();
        }
        bool output_finite() const {
            if constexpr (Magnitude) return all_finite(power);
            else if constexpr (Backward) return all_finite(time);
            else return spectrum.is_finite();
        }
        size_t output_size() const { return Backward ? time.size() : spectrum.size(); }
        C output(size_t i) const {
            if constexpr (Magnitude) return C(power[i], 0);
            else if constexpr (Backward) return C(time[i], 0);
            else return spectrum[i];
        }
    };

    enum class ConvolveLayout { kComplexAoS, kComplexSoA, kReal };

    template <ConvolveLayout Layout, bool Fused = true, typename FFT = RFFTClass>
    struct ConvolveCase {
        static constexpr bool convolution = true;
        static constexpr bool real_kernel = Layout == ConvolveLayout::kReal;
        // The real-kernel baseline uses SoA intermediates for componentwise multiplication.
        static constexpr bool soa = Layout != ConvolveLayout::kComplexAoS;
        static constexpr ConvolveLayout layout = Layout;
        static constexpr F output_scale = 1;
        static constexpr const char* direction = "convolve";
        static constexpr const char* api = real_kernel ? (Fused ? "real" : "real_separate")
            : soa ? (Fused ? "complex_soa" : "complex_soa_separate")
                  : (Fused ? "complex_aos" : "complex_aos_separate");

        FFT fft;
        AlignedVector<F> time, out, real_spectrum;
        ComplexBuffer<soa> spectrum, work;
        AlignedVector<C> original, original_spectrum;

        explicit ConvolveCase(int order) : fft(order), time(fft_size(order)), out(time.size()),
            real_spectrum(real_kernel ? time.size() / 2 + 1 : 0),
            spectrum(real_kernel ? 0 : time.size() / 2 + 1), work(Fused ? 0 : time.size() / 2 + 1) {
            original = make_input(time.size(), true, false, 5, 42);
            original_spectrum = make_input(time.size(), true, true, 5, 43);
            if constexpr (real_kernel) for (auto& h : original_spectrum) h.imag(0);
            reset();
        }
        void reset() {
            for (size_t i = 0; i < time.size(); ++i) time[i] = original[i].real();
            if constexpr (real_kernel) {
                for (size_t k = 0; k < real_spectrum.size(); ++k) real_spectrum[k] = original_spectrum[k].real();
            } else spectrum.assign(original_spectrum);
        }
        auto kernel_data() const {
            if constexpr (real_kernel) return real_spectrum.data();
            else return spectrum.data();
        }
        void execute_to(F* output) {
            if constexpr (Fused) {
                // Keep historical revisions buildable by the comparison harness;
                // requesting a new API from an old revision fails explicitly.
                if constexpr (requires { fft.convolve(time.data(), kernel_data(), output); }) {
                    fft.convolve(time.data(), kernel_data(), output);
                } else {
                    throw std::runtime_error("convolve API is unavailable in this revision");
                }
            } else {
                auto values = work.data();
                const auto kernel = kernel_data();
                fft.forward(time.data(), values);
                for (size_t k = 0; k < work.size(); ++k) {
                    if constexpr (real_kernel) {
                        values[0][k] *= kernel[k];
                        values[1][k] *= kernel[k];
                    } else if constexpr (soa) {
                        const F r = values[0][k], i = values[1][k];
                        values[0][k] = r * kernel[0][k] - i * kernel[1][k];
                        values[1][k] = r * kernel[1][k] + i * kernel[0][k];
                    } else {
                        const F r = values[k].real(), i = values[k].imag();
                        values[k] = C(r * kernel[k].real() - i * kernel[k].imag(),
                                      r * kernel[k].imag() + i * kernel[k].real());
                    }
                }
                fft.backward(values, output);
            }
        }
        void execute() { execute_to(out.data()); }
        bool input_unchanged() const {
            for (size_t i = 0; i < time.size(); ++i) if (time[i] != original[i].real()) return false;
            return true;
        }
        bool kernel_unchanged() const {
            for (size_t k = 0; k < original_spectrum.size(); ++k) {
                if constexpr (real_kernel) {
                    if (real_spectrum[k] != original_spectrum[k].real()) return false;
                } else if (spectrum[k] != original_spectrum[k]) return false;
            }
            return true;
        }
        auto output_data() { return out.data(); }
        bool output_finite() const { return all_finite(out); }
        size_t output_size() const { return out.size(); }
        C output(size_t i) const { return C(out[i], 0); }
    };

    // Accuracy and throughput share all public API cases. Throughput additionally
    // measures three separate forward/multiply/backward convolution baselines.
    template <typename Visit>
    void for_each_case(Visit visit) {
#ifdef ZLB_REAL_FFT
        visit.template operator()<RfftCase<false>>();
        visit.template operator()<RfftCase<true>>();
#ifdef ZLB_ALL_APIS
        visit.template operator()<RfftCase<false, true>>();
        visit.template operator()<RfftCase<true, true>>();
        visit.template operator()<RfftCase<false, false, true>>();
        visit.template operator()<ConvolveCase<ConvolveLayout::kComplexAoS>>();
        visit.template operator()<ConvolveCase<ConvolveLayout::kComplexSoA>>();
        visit.template operator()<ConvolveCase<ConvolveLayout::kReal>>();
#endif
#else
        visit.template operator()<CfftCase<false>>();
        visit.template operator()<CfftCase<true>>();
#ifdef ZLB_ALL_APIS
        visit.template operator()<CfftCase<false, false, true>>();
        visit.template operator()<CfftCase<true, false, true>>();
        visit.template operator()<CfftCase<false, true, false>>();
        visit.template operator()<CfftCase<true, true, false>>();
        visit.template operator()<CfftCase<false, true, true>>();
        visit.template operator()<CfftCase<true, true, true>>();
#endif
#endif
    }

    template <typename Visit>
    void for_each_convolve_baseline(Visit visit) {
#if defined(ZLB_REAL_FFT) && defined(ZLB_ALL_APIS)
        visit.template operator()<ConvolveCase<ConvolveLayout::kComplexAoS, false>>();
        visit.template operator()<ConvolveCase<ConvolveLayout::kComplexSoA, false>>();
        visit.template operator()<ConvolveCase<ConvolveLayout::kReal, false>>();
#endif
    }
}
