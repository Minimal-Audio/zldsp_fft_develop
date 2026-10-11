#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include "benchmark_cases.hpp"

#ifndef ZLB_ALL_APIS
#include "../fftw3_impl/fftw3_impl.hpp"
#include "../kfr_impl/kfr_impl.hpp"
#include "../zldsp_fft/zldsp_fft_cfft.hpp"
#include "../zldsp_fft/zldsp_fft_rfft.hpp"
#ifdef ZLB_ACCURACY_VDSP
#include "../vdsp_impl/vdsp_impl.hpp"
#endif
#ifdef ZLB_ACCURACY_IPP
#include "../ipp_impl/ipp_impl.hpp"
#endif
#endif

namespace zlbenchmark {
#ifdef ZLB_ACCURACY_IPP
#ifdef IPP_USE_AVX2
    inline constexpr auto accuracy_ipp_simd = IPPSimdLevel::AVX2;
#else
    inline constexpr auto accuracy_ipp_simd = IPPSimdLevel::SSE42;
#endif
#endif
    using WideComplex = std::complex<long double>;
    using WideVector = std::vector<WideComplex>;

    struct Errors {
        long double relative_rmse = 0, max_abs = 0;
        void add(const WideVector& expected, const WideVector& actual) {
            if (expected.size() != actual.size()) throw std::runtime_error("accuracy output size mismatch");
            long double error_energy = 0, signal_energy = 0;
            for (size_t i = 0; i < expected.size(); ++i) {
                if (!std::isfinite(actual[i].real()) || !std::isfinite(actual[i].imag()) ||
                    !std::isfinite(expected[i].real()) || !std::isfinite(expected[i].imag())) {
                    throw std::runtime_error("non-finite accuracy result");
                }
                const auto difference = actual[i] - expected[i];
                max_abs = std::max(max_abs, std::abs(difference));
                error_energy += std::norm(difference);
                signal_energy += std::norm(expected[i]);
            }
            // For an all-zero reference, use absolute RMS instead.
            const auto error = std::sqrt(error_energy / (signal_energy > 0 ? signal_energy : expected.size()));
            if (!std::isfinite(error)) throw std::runtime_error("non-finite accuracy error");
            relative_rmse = std::max(relative_rmse, error);
        }
    };

    inline WideVector widen(const AlignedVector<C>& values, long double scale = 1) {
        WideVector out(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            if (!finite(values[i])) throw std::runtime_error("non-finite accuracy result");
            out[i] = WideComplex(values[i].real(), values[i].imag()) * scale;
        }
        return out;
    }

    class Reporter {
        int order_;
        long double threshold_;
        bool passed_ = true;

    public:
        Reporter(int order, double threshold) : order_(order), threshold_(threshold) {}
        void report(const char* library, const char* direction, const char* api, const Errors& error) {
            const bool ok = error.relative_rmse <= threshold_;
            passed_ &= ok;
            std::cout << "{\"order\":" << order_ << ",\"library\":\"" << library
                      << "\",\"direction\":\"" << direction << "\",\"api\":\"" << api
                      << "\",\"relative_rmse\":" << error.relative_rmse << ",\"max_abs\":" << error.max_abs
                      << ",\"status\":\"" << (ok ? "passed" : "failed") << "\"}\n";
        }
        bool passed() const { return passed_; }
    };

#ifdef ZLB_ALL_APIS
    template <typename Case>
    WideVector case_output(const Case& test) {
        WideVector out(test.output_size());
        for (size_t i = 0; i < out.size(); ++i) {
            const C value = test.output(i);
            out[i] = WideComplex(value.real(), value.imag()) * static_cast<long double>(Case::output_scale);
        }
        return out;
    }

    template <typename Case>
    Errors check_api(int order, int repetitions, unsigned seed) {
        const size_t n = fft_size(order);
        Case test(order);
        using Baseline = std::conditional_t<Case::real, RfftCase<Case::backward>, CfftCase<Case::backward>>;
        Baseline baseline(order);
        Errors errors;
        for (int pattern = 0; pattern < 5 + repetitions; ++pattern) {
            test.original = make_input(n, Case::real, Case::backward, pattern, seed + pattern);
            baseline.original = test.original;
            test.reset();
            baseline.reset();
            test.execute();
            baseline.execute();
            auto expected = case_output(baseline);
            if constexpr (Case::magnitude) for (auto& value : expected) value = std::norm(value);
            errors.add(expected, case_output(test));
        }
        return errors;
    }

    inline Errors check_roundtrip(int order, int repetitions, unsigned seed) {
        const size_t n = fft_size(order);
#ifdef ZLB_REAL_FFT
        RfftCase<false> test(order);
        AlignedVector<F> recovered(n);
#else
        CfftCase<false> test(order);
        AlignedVector<C> recovered(n);
#endif
        Errors errors;
        for (int pattern = 0; pattern < 5 + repetitions; ++pattern) {
            test.original = make_input(n, test.real, false, pattern, seed + pattern);
            test.reset();
            test.execute();
#ifdef ZLB_REAL_FFT
            test.fft.backward(test.spectrum.data(), recovered.data());
            const long double scale = static_cast<long double>(rfft_backward_scale) / n;
#else
            test.fft.backward(test.out.data(), recovered.data());
            const long double scale = 1.0L / n;
#endif
            WideVector actual(n);
            for (size_t i = 0; i < n; ++i) {
                const C value(recovered[i]);
                actual[i] = WideComplex(value.real(), value.imag()) * scale;
            }
            errors.add(widen(test.original), actual);
        }
        return errors;
    }
#else
    struct Library {
        const char* name;
        long double inverse_scale;
        std::function<AlignedVector<C>(const AlignedVector<C>&, bool)> run;
        Errors forward, backward, roundtrip;
    };

    template <typename FFT, bool Real>
    Library make_library(const char* name, int order, long double inverse_scale = 1) {
        const size_t n = fft_size(order);
        auto fft = std::make_shared<FFT>(order);
        return {name, inverse_scale, [fft, n](const AlignedVector<C>& input, bool backward) {
            const F nan = std::numeric_limits<F>::quiet_NaN();
            AlignedVector<C> out(Real && !backward ? n / 2 + 1 : n, C(nan, nan));
            // Each call gets a fresh copy, including for destructive library APIs.
            AlignedVector<C> in = input;
            if constexpr (Real) {
                AlignedVector<F> time(n, nan);
                if (backward) {
                    fft->backward(in.data(), time.data());
                    for (size_t i = 0; i < n; ++i) out[i] = C(time[i], 0);
                } else {
                    for (size_t i = 0; i < n; ++i) time[i] = input[i].real();
                    fft->forward(time.data(), out.data());
                }
            } else {
                if (backward) fft->backward(in.data(), out.data());
                else fft->forward(in.data(), out.data());
            }
            return out;
        }, {}, {}, {}};
    }

    inline WideVector component_median(const std::vector<WideVector>& outputs) {
        if (outputs.size() < 3) throw std::runtime_error("accuracy requires at least three libraries");
        WideVector median(outputs.front().size());
        std::vector<long double> re(outputs.size()), im(outputs.size());
        for (size_t i = 0; i < median.size(); ++i) {
            for (size_t j = 0; j < outputs.size(); ++j) {
                re[j] = outputs[j][i].real();
                im[j] = outputs[j][i].imag();
            }
            std::sort(re.begin(), re.end());
            std::sort(im.begin(), im.end());
            const size_t lo = (outputs.size() - 1) / 2, hi = outputs.size() / 2;
            median[i] = WideComplex((re[lo] + re[hi]) / 2, (im[lo] + im[hi]) / 2);
        }
        return median;
    }

    inline void check_libraries(int order, int repetitions, unsigned seed, Reporter& reporter) {
        const size_t n = fft_size(order);
        std::vector<Library> libraries;
#ifdef ZLB_REAL_FFT
        constexpr bool real = true;
        libraries.push_back(make_library<FFTW3RFFT<F, FFTW_ESTIMATE>, true>("fftw3_estimate", order));
        libraries.push_back(make_library<KFRRFFT<F>, true>("kfr", order));
        libraries.push_back(make_library<zldsp::fft::RFFT<F>, true>("zldsp", order, 2));
#ifdef ZLB_ACCURACY_VDSP
        libraries.push_back(make_library<VDSPRFFT<F>, true>("vdsp", order));
#endif
#ifdef ZLB_ACCURACY_IPP
        libraries.push_back(make_library<IPPRFFT<F, accuracy_ipp_simd>, true>("ipp", order));
#endif
#else
        constexpr bool real = false;
        libraries.push_back(make_library<FFTW3FFT<F, FFTW_ESTIMATE>, false>("fftw3_estimate", order));
        libraries.push_back(make_library<KFRFFT<F>, false>("kfr", order));
        libraries.push_back(make_library<zldsp::fft::CFFT<F>, false>("zldsp", order));
#ifdef ZLB_ACCURACY_VDSP
        libraries.push_back(make_library<VDSPFFT<F>, false>("vdsp", order));
#endif
#ifdef ZLB_ACCURACY_IPP
        libraries.push_back(make_library<IPPFFT<F, accuracy_ipp_simd>, false>("ipp", order));
#endif
#endif
        for (int pattern = 0; pattern < 5 + repetitions; ++pattern) {
            for (bool backward : {false, true}) {
                const auto input = make_input(n, real, backward, pattern, seed + pattern);
                std::vector<WideVector> outputs;
                for (auto& library : libraries) {
                    auto output = library.run(input, backward);
                    outputs.push_back(widen(output, backward ? library.inverse_scale : 1));
                    if (!backward) {
                        library.roundtrip.add(widen(input), widen(library.run(output, true), library.inverse_scale / n));
                    }
                }
                const auto median = component_median(outputs);
                for (size_t i = 0; i < libraries.size(); ++i) {
                    (backward ? libraries[i].backward : libraries[i].forward).add(median, outputs[i]);
                }
            }
        }
        for (const auto& library : libraries) {
            reporter.report(library.name, "forward", "median", library.forward);
            reporter.report(library.name, "backward", "median", library.backward);
            reporter.report(library.name, "roundtrip", "aos", library.roundtrip);
        }
    }
#endif
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) throw std::invalid_argument("usage: executable <order> [repetitions] [seed] [relative-rmse-limit]");
        const int order = zlbenchmark::parse_integer(argv[1], 1, 30);
        const int repetitions = argc > 2 ? zlbenchmark::parse_integer(argv[2], 1, 1000) : 1;
        const unsigned seed = argc > 3 ? zlbenchmark::parse_integer(argv[3], 0, std::numeric_limits<int>::max()) : 42;
        size_t parsed = 0;
        const double threshold = argc > 4 ? std::stod(argv[4], &parsed) : (sizeof(F) == 4 ? 2e-5 : 2e-12);
        if (!std::isfinite(threshold) || threshold <= 0 || (argc > 4 && parsed != std::string_view(argv[4]).size())) {
            throw std::invalid_argument("threshold must be finite and positive");
        }
        std::cout << std::setprecision(12);
        zlbenchmark::Reporter reporter(order, threshold);
#ifdef ZLB_ALL_APIS
        zlbenchmark::for_each_case([&]<typename Case>() {
            reporter.report("zldsp", Case::backward ? "backward" : "forward", Case::api,
                            zlbenchmark::check_api<Case>(order, repetitions, seed));
        });
        reporter.report("zldsp", "roundtrip", "aos", zlbenchmark::check_roundtrip(order, repetitions, seed));
#else
        zlbenchmark::check_libraries(order, repetitions, seed, reporter);
#endif
        return reporter.passed() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
