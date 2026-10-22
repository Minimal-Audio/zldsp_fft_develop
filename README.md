# zldsp::fft (develop)

Development, benchmarking, and analysis repository for `zldsp::fft`.

If you want to use `zldsp::fft` in your project, please refer to [zldsp_fft](https://github.com/ZL-Audio/zldsp_fft).

## Usage

Requirements: `Clang` (`AppleClang 16+` or `LLVM/Clang 17+`), `CMake` (`3.20+`), and `Ninja`. On Windows, run from a Visual Studio developer shell.

```console
git clone https://github.com/ZL-Audio/zldsp_fft_develop
cd zldsp_fft_develop
git submodule update --init
```

All benchmarks run through a single CLI (`1 <= n0 <= n1 <= 30`, FFT size `N = 2^order`):

```console
python3 benchmark/run.py <accuracy|throughput|compare> <cfft|rfft> <n0> <n1> [options]
```

### Common Options

- `--double`: Use `double` precision (default: `float`).
- `--avx2` / `--sse2`: Select x86 SIMD target (default: SSE4.2 on x86; ARM uses NEON).
- `--apis`: Test all `zldsp::fft` layouts and APIs (AoS/SoA, `forward_sqr_mag`, RFFT `convolve`).
- `--repeats <k>`: Repetition count for accuracy inputs (`3`), throughput measurements (`3`), or paired comparison runs (`6`).
- `--artifacts-dir <dir>`: Directory for build artifacts and temporary files (default: `<repo>/build_fft` for accuracy/throughput, system temporary directory for comparisons).
- `--json <file>`: Save detailed results to a JSON file.

### Accuracy

```console
python3 benchmark/run.py accuracy cfft 2 20
python3 benchmark/run.py accuracy rfft 2 20 --double
python3 benchmark/run.py accuracy cfft 2 20 --apis
python3 benchmark/run.py accuracy rfft 2 20 --apis --double
```

- **Library agreement (default)**: Compares `fftw3_estimate`, `kfr`, and `zldsp` (plus `vdsp` on macOS and `ipp` if installed) against their component-wise median, and verifies normalized round trips.
- **API consistency (`--apis`)**: Validates all `zldsp` AoS/SoA layouts, `forward_sqr_mag`, and `convolve` overloads against the reference AoS path and direct DFT.
- **Metrics**: Reports worst relative RMSE and maximum absolute error across deterministic and random test signals. Fails if relative RMSE exceeds `--threshold` (`2e-5` for `float`, `2e-12` for `double`).
- **Scaling**: `zldsp` backward RFFT yields `(N/2) * input`, so RFFT round trips scale by `2/N` (`1/N` for CFFT).

### Throughput

```console
python3 benchmark/run.py throughput cfft 5 20
python3 benchmark/run.py throughput rfft 5 20 --libraries zldsp kfr pffft vdsp
python3 benchmark/run.py throughput rfft 5 20 --standard-libraries
python3 benchmark/run.py throughput cfft 5 20 --all-libraries --double
python3 benchmark/run.py throughput rfft 5 20 --apis
```

- **Target selection** (default: `zldsp`):
  - `--libraries <lib ...>`: Benchmark specific backends.
  - `--standard-libraries`: Benchmark available standard backends (`zldsp`, `kfr`, `pffft`, plus `vdsp`, `ipp`, `armpl`, `armpl_estimate` when installed).
  - `--all-libraries`: `--standard-libraries` plus `fftw3` and `fftw3_estimate`.
  - `--apis`: Benchmark all `zldsp` AoS/SoA layouts, `forward_sqr_mag`, and fused vs. separate convolution.
- **Measurement**: Plans and buffers are reused across repetitions, timing only transform execution. Memory is freed between transform sizes to keep peak usage low.
- **Duration**: By default, measurement time scales exponentially from `--min-time0` (0.5s at `n0`) to `--min-time1` (5s at `n1`) per repetition. Use `--min-time <seconds>` for a fixed duration across all orders.
- **Builds**: Selected backends are compiled before benchmarking. Use `--build-only` to compile without measuring, or `--skip-build` to benchmark existing binaries.
- **Metrics**: Reports nominal **MFLOPS** from median single-call times across `--repeats` runs (`5*N*log2(N)` for CFFT, `2.5*N*log2(N)` for RFFT).

### Comparing Git Revisions

```console
python3 benchmark/run.py compare cfft 5 20 --repeats 6
python3 benchmark/run.py compare rfft 5 20 --apis --current HEAD --previous HEAD^
```

Compares performance between two git revisions (`--current`, default `HEAD`; `--previous`, default `HEAD^`). Alternates execution across `--repeats` paired runs and reports median times, speedup ratios (`> 1` is faster), percentage changes, and MAD.

### Supported Backends

`zldsp`, `fftw3`, `fftw3_estimate`, `kfr`, `pffft`, `vdsp`, `vdsp_stride_2`, `ipp`, `armpl`, `armpl_estimate`.

- `fftw3` and `armpl` use `FFTW_MEASURE`; `*_estimate` variants use `FFTW_ESTIMATE`.
- Set `ARMPL_DIR` for Arm Performance Libraries, and `IPP_ROOT` for Intel IPP installations outside default paths.

## Use `zldsp::fft` in another project

### Static Dispatch

```cmake
target_compile_definitions(my_static_target PRIVATE HWY_COMPILE_ONLY_STATIC)
```

Use compiler flags to select a static SIMD target:

| SIMD Target | GCC/Clang                         | MSVC             |
| ----------- | --------------------------------- | ---------------- |
| SSE2        | `-march=x86-64`                   | no flag required |
| SSE4        | `-march=x86-64-v2 -maes -mpclmul` | not supported    |
| AVX2        | `-march=x86-64-v3 -maes -mpclmul` | `/arch:AVX2`     |
| NEON        | `-march=armv8-a+simd`             | `/arch:armv8.0`  |

See [`static_dispatch_caller`](examples/static_dispatch_caller.cpp) for CFFT and RFFT examples.

### Caller-owned Dynamic Dispatch

An x86 application can restrict dynamic targets (for example, enabling only SSE2 and AVX2):

```cmake
target_compile_definitions(my_dynamic_target PRIVATE "HWY_DISABLED_TARGETS=~(HWY_SSE2|HWY_AVX2)")
```

Compile this target for the oldest supported baseline (e.g. `-march=x86-64` for SSE2).

See [`wrapper`](examples/dynamic_dispatch_wrapper.cpp), [`interface`](examples/dynamic_dispatch_wrapper.hpp), and [`dynamic_dispatch_caller`](examples/dynamic_dispatch_caller.cpp) for complete examples.

### Shared Plans and Caller-provided Scratch

`CFFT` and `RFFT` own the buffers a transform writes to, so one instance transforms on one thread at a time. `CFFTPlan` and `RFFTPlan` hold only the read-only part (twiddles and stage tables): every method is `const`, and any number of threads can transform with one plan at once, each passing its own scratch.

```cpp
const zldsp::fft::RFFTPlan<float> plan(order);
// plan.get_scratch_size() elements, aligned to RFFTPlan<float>::kScratchAlignment bytes
std::span<float> scratch = ...;
plan.forward(input, output, scratch);
```

The scratch can live anywhere, including the stack. Its size depends on the SIMD target and on cache sizes read at run time, so always ask the plan for it. The plans take the same layouts as `CFFT` and `RFFT`, and produce bit-identical results. See [`static_dispatch_caller`](examples/static_dispatch_caller.cpp) for an example with one plan shared by several threads.

## License

zldsp_fft is licensed under the Apache-2.0 license ([LICENSE.md](LICENSE.md)).

External libraries (submodules) retain their respective licenses:

- FFTW3: `fftw3_impl/fftw3` (GPL-2.0 license)
- KFR: `kfr_impl/kfr` (GPL-2.0 license)
- pffft: `pffft_impl/pffft` (BSD-like license)
- Google benchmark: `google/benchmark` (Apache-2.0 license)
- Google highway: `google/highway` (Apache-2.0 license)
