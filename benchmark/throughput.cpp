#include <iostream>
#include <benchmark/benchmark.h>
#include "benchmark_cases.hpp"

namespace zlbenchmark {
    inline bool throughput_failed = false;
    template <typename Case>
    void measure(benchmark::State& state) {
        try {
            Case test(static_cast<int>(state.range(0)));
            test.execute();
            const bool destructive = !test.input_unchanged();
            if (!test.output_finite()) {
                throughput_failed = true;
                state.SkipWithError("non-finite output during warmup");
                return;
            }
            for (int i = 0; i < 3; ++i) {
                test.reset();
                test.execute();
            }
            test.reset();
            auto output = test.output_data();
            benchmark::DoNotOptimize(output);
            for (auto _ : state) {
                // Never feed an unscaled transform's output into the next call.
                // Destructive APIs get fresh input outside the measured region.
                if (destructive) {
                    state.PauseTiming();
                    test.reset();
                    state.ResumeTiming();
                }
                test.execute();
                benchmark::ClobberMemory();
            }
            if (!test.output_finite()) {
                throughput_failed = true;
                state.SkipWithError("non-finite output after measurement");
            }
            state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(fft_size(state.range(0))));
            state.counters["transforms_per_iteration"] = 1;
            state.counters["input_reset_required"] = destructive ? 1 : 0;
        } catch (const UnsupportedSize& error) {
            state.SkipWithError((std::string("unsupported size: ") + error.what()).c_str());
        } catch (const std::exception& error) {
            throughput_failed = true;
            state.SkipWithError(error.what());
        }
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::invalid_argument("usage: executable <n0> <n1> [google-benchmark-flags]");
        const int n0 = zlbenchmark::parse_integer(argv[1], 1, 30);
        const int n1 = zlbenchmark::parse_integer(argv[2], n0, 30);
        // Remove our positional arguments before letting Google Benchmark parse.
        for (int i = 3; i < argc; ++i) argv[i - 2] = argv[i];
        argc -= 2;
        argv[argc] = nullptr;
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
        zlbenchmark::for_each_case([&]<typename Case>() {
            const std::string name = std::string(Case::backward ? "backward/" : "forward/") + Case::api;
            benchmark::RegisterBenchmark(name.c_str(), zlbenchmark::measure<Case>)
                ->DenseRange(n0, n1)->Unit(benchmark::kMicrosecond);
        });
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return zlbenchmark::throughput_failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
