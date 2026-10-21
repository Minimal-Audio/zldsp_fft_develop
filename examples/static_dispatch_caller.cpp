#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "zldsp_fft_cfft.hpp"
#include "zldsp_fft_rfft.hpp"

template <typename F, std::size_t alignment>
struct AlignedDelete {
    void operator()(F* ptr) const noexcept {
        ::operator delete(ptr, std::align_val_t{alignment});
    }
};

// scratch for one plan transform at a time: get_scratch_size() elements, aligned to kScratchAlignment bytes
template <typename F, typename Plan>
std::unique_ptr<F[], AlignedDelete<F, Plan::kScratchAlignment>> make_scratch(const Plan& plan) {
    const std::size_t bytes = std::max<std::size_t>(1, plan.get_scratch_size()) * sizeof(F);
    return std::unique_ptr<F[], AlignedDelete<F, Plan::kScratchAlignment>>(
        static_cast<F*>(::operator new(bytes, std::align_val_t{Plan::kScratchAlignment})));
}

// runs every API of a CFFT, or of a CFFTPlan when given its scratch
template <typename F, typename FFT, typename... Scratch>
void run_cfft(FFT& cfft, const std::size_t order, Scratch... scratch) {
    const std::size_t fft_size = std::size_t{1} << order;
    using Complex = std::complex<F>;
    using SoA = std::array<F*, 2>;

    std::vector<Complex> complex_input(fft_size, Complex{F{1}, F{0}});
    std::vector<Complex> complex_output(fft_size);
    std::vector<F> complex_input_real(fft_size, F{1});
    std::vector<F> complex_input_imag(fft_size, F{0});
    std::vector<F> complex_output_real(fft_size);
    std::vector<F> complex_output_imag(fft_size);
    SoA complex_input_soa{complex_input_real.data(), complex_input_imag.data()};
    SoA complex_output_soa{complex_output_real.data(), complex_output_imag.data()};

    // CFFT forward AoS to AoS
    cfft.forward(complex_input.data(), complex_output.data(), scratch...);
    // CFFT forward AoS to SoA
    cfft.forward(complex_input.data(), complex_output_soa, scratch...);
    // CFFT forward SoA to AoS
    cfft.forward(complex_input_soa, complex_output.data(), scratch...);
    // CFFT forward SoA to SoA
    cfft.forward(complex_input_soa, complex_output_soa, scratch...);

    // CFFT backward AoS to AoS
    cfft.backward(complex_output.data(), complex_input.data(), scratch...);
    // CFFT backward AoS to SoA
    cfft.backward(complex_output.data(), complex_input_soa, scratch...);
    // CFFT backward SoA to AoS
    cfft.backward(complex_output_soa, complex_input.data(), scratch...);
    // CFFT backward SoA to SoA
    cfft.backward(complex_output_soa, complex_input_soa, scratch...);
}

// runs every API of an RFFT, or of an RFFTPlan when given its scratch
template <typename F, typename FFT, typename... Scratch>
void run_rfft(FFT& rfft, const std::size_t order, Scratch... scratch) {
    const std::size_t fft_size = std::size_t{1} << order;
    using Complex = std::complex<F>;
    using SoA = std::array<F*, 2>;

    std::vector<F> real_input(fft_size, F{1});
    std::vector<Complex> real_output(fft_size / 2 + 1);
    std::vector<F> real_output_real(fft_size / 2 + 1);
    std::vector<F> real_output_imag(fft_size / 2 + 1);
    SoA real_output_soa{real_output_real.data(), real_output_imag.data()};

    // RFFT forward AoS
    rfft.forward(real_input.data(), real_output.data(), scratch...);
    // RFFT forward SoA
    rfft.forward(real_input.data(), real_output_soa, scratch...);

    // RFFT backward AoS
    rfft.backward(real_output.data(), real_input.data(), scratch...);
    // RFFT backward SoA
    rfft.backward(real_output_soa, real_input.data(), scratch...);
    // RFFT forward squared magnitude
    rfft.forward_sqr_mag(real_input.data(), real_output_real.data(), scratch...);

    const F scale = F(2) / static_cast<F>(fft_size);
    const std::vector<Complex> kernel_aos(fft_size / 2 + 1, Complex{scale, F(0)});
    const std::vector<F> kernel_real(fft_size / 2 + 1, scale);
    const std::vector<F> kernel_imag(fft_size / 2 + 1, F(0));
    const std::array<const F*, 2> kernel_soa{kernel_real.data(), kernel_imag.data()};
    std::vector<F> convolved(fft_size);
    // RFFT convolve AoS
    rfft.convolve(real_input.data(), kernel_aos.data(), convolved.data(), scratch...);
    // RFFT convolve SoA
    rfft.convolve(real_input.data(), kernel_soa, convolved.data(), scratch...);
    // RFFT convolve real
    rfft.convolve(real_input.data(), kernel_real.data(), convolved.data(), scratch...);
    // RFFT convolve in-place
    rfft.convolve(real_input.data(), kernel_real.data(), real_input.data(), scratch...);
}

// one read-only plan shared by several threads, each with its own scratch; true if every thread
// reproduces the single-threaded result exactly
template <typename F>
bool check_shared_rfft_plan(const std::size_t order) {
    using Complex = std::complex<F>;
    const std::size_t fft_size = std::size_t{1} << order;
    const zldsp::fft::RFFTPlan<F> plan(order);

    std::vector<F> input(fft_size);
    for (std::size_t i = 0; i < fft_size; ++i) {
        input[i] = static_cast<F>((i * 7919) % 1024) / F(512) - F(1);
    }
    auto transform = [&](std::vector<Complex>& spectrum, std::vector<F>& output, std::span<F> scratch) {
        std::vector<F> time = input;
        plan.forward(time.data(), spectrum.data(), scratch);
        plan.backward(spectrum.data(), output.data(), scratch);
    };

    std::vector<Complex> reference_spectrum(fft_size / 2 + 1);
    std::vector<F> reference_output(fft_size);
    {
        auto scratch = make_scratch<F>(plan);
        transform(reference_spectrum, reference_output, {scratch.get(), plan.get_scratch_size()});
    }

    static constexpr std::size_t kNumThreads = 4;
    static constexpr std::size_t kNumRepeats = 8;
    std::array<bool, kNumThreads> matches{};
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t] {
            auto scratch = make_scratch<F>(plan);
            std::vector<Complex> spectrum(fft_size / 2 + 1);
            std::vector<F> output(fft_size);
            bool match = true;
            for (std::size_t r = 0; r < kNumRepeats; ++r) {
                transform(spectrum, output, {scratch.get(), plan.get_scratch_size()});
                match = match &&
                    std::memcmp(spectrum.data(), reference_spectrum.data(), spectrum.size() * sizeof(Complex)) == 0 &&
                    std::memcmp(output.data(), reference_output.data(), output.size() * sizeof(F)) == 0;
            }
            matches[t] = match;
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return std::all_of(matches.begin(), matches.end(), [](const bool match) { return match; });
}

int main(const int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <start-order> <end-order>" << std::endl;
        return 2;
    }

    try {
        const std::size_t start_order = std::stoul(argv[1]);
        const std::size_t end_order = std::stoul(argv[2]);
        // CFFT float
        // for static dispatch, you don't need std::unique_ptr if the order is fixed
        for (std::size_t order = start_order; order <= end_order; ++order) {
            std::unique_ptr<zldsp::fft::CFFT<float>> cfft =
                std::make_unique<zldsp::fft::CFFT<float>>(order);
            run_cfft<float>(*cfft, order);
        }
        // CFFT double
        for (std::size_t order = start_order; order <= end_order; ++order) {
            std::unique_ptr<zldsp::fft::CFFT<double>> cfft =
                std::make_unique<zldsp::fft::CFFT<double>>(order);
            run_cfft<double>(*cfft, order);
        }
        // RFFT float
        for (std::size_t order = start_order; order <= end_order; ++order) {
            std::unique_ptr<zldsp::fft::RFFT<float>> rfft =
                std::make_unique<zldsp::fft::RFFT<float>>(order);
            run_rfft<float>(*rfft, order);
        }
        // RFFT double
        for (std::size_t order = start_order; order <= end_order; ++order) {
            std::unique_ptr<zldsp::fft::RFFT<double>> rfft =
                std::make_unique<zldsp::fft::RFFT<double>>(order);
            run_rfft<double>(*rfft, order);
        }
        // CFFT and RFFT plans: read-only, so one can be shared; each caller passes its own scratch
        for (std::size_t order = start_order; order <= end_order; ++order) {
            const zldsp::fft::CFFTPlan<float> cfft_float(order);
            const zldsp::fft::CFFTPlan<double> cfft_double(order);
            const zldsp::fft::RFFTPlan<float> rfft_float(order);
            const zldsp::fft::RFFTPlan<double> rfft_double(order);
            auto cfft_float_scratch = make_scratch<float>(cfft_float);
            auto cfft_double_scratch = make_scratch<double>(cfft_double);
            auto rfft_float_scratch = make_scratch<float>(rfft_float);
            auto rfft_double_scratch = make_scratch<double>(rfft_double);
            run_cfft<float>(cfft_float, order,
                            std::span<float>(cfft_float_scratch.get(), cfft_float.get_scratch_size()));
            run_cfft<double>(cfft_double, order,
                             std::span<double>(cfft_double_scratch.get(), cfft_double.get_scratch_size()));
            run_rfft<float>(rfft_float, order,
                            std::span<float>(rfft_float_scratch.get(), rfft_float.get_scratch_size()));
            run_rfft<double>(rfft_double, order,
                             std::span<double>(rfft_double_scratch.get(), rfft_double.get_scratch_size()));
        }
        // one RFFT plan shared by several threads
        for (std::size_t order = start_order; order <= end_order; ++order) {
            if (!check_shared_rfft_plan<float>(order) || !check_shared_rfft_plan<double>(order)) {
                std::cerr << "Shared RFFT plan mismatch at order " << order << std::endl;
                return 1;
            }
        }
    }
    catch (...) {
        std::cerr << "Invalid order range: " << argv[1] << " " << argv[2] << std::endl;
        return 2;
    }
    return 0;
}
