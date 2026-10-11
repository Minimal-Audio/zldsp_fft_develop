# zldsp::fft(develop)

zldsp::fft(develop) aims at FFT implementation and analysis.

If you want to use zldsp::fft, please refer to [zldsp_fft](https://github.com/ZL-Audio/zldsp_fft).

## Usage

Please make sure `Clang` (`AppleClang 16+` or `LLVM/Clang 17+`), `CMake` (`3.20+`), and `Ninja` are installed and configured on your OS. On Windows, run from a Visual Studio developer shell.

```console
git clone https://github.com/ZL-Audio/zldsp_fft_develop
cd zldsp_fft_develop
git submodule update --init
```

All benchmarks run through a single CLI (`1 <= n0 <= n1 <= 30`, FFT size `N = 2^order`):

```console
python3 benchmark/run.py <accuracy|throughput|compare> <cfft|rfft> <n0> <n1> [options]
```

Common options across subcommands:
- `--double`: Use `double` precision (default: `float`).
- `--avx2` / `--sse2`: Select x86 SIMD target (x86 default: SSE4.2; ARM uses NEON).
- `--apis`: Test all `zldsp::fft` APIs and layouts (AoS/SoA, `forward_sqr_mag`, and RFFT `convolve`).
- `--repeats <k>`: Random input count for `accuracy` (default: `3`), timing repetitions for `throughput` (default: `3`), or paired runs for `compare` (default: `6`).
- `--json <file>`: Save full structured results to a JSON file (a compact JSON summary is always printed to stdout).

### Accuracy

```console
python3 benchmark/run.py accuracy cfft 2 20
python3 benchmark/run.py accuracy rfft 2 20 --double
python3 benchmark/run.py accuracy cfft 2 20 --apis
python3 benchmark/run.py accuracy rfft 2 20 --apis --double
```

- **Library agreement (default)**: Compares `fftw3_estimate`, `kfr`, and `zldsp` (plus `vdsp` on macOS and `ipp` when installed) against their component-wise median, and verifies normalized forward/backward round trips.
- **API consistency (`--apis`)**: Compares all `zldsp` AoS/SoA layouts, `forward_sqr_mag`, and `convolve` overloads against the AoS API, separate forward/multiply/backward execution, and a small-size direct DFT reference.
- **Metrics**: Reports worst **relative RMSE** and **maximum absolute error** across five deterministic signals and `--repeats` random inputs. Fails if relative RMSE exceeds `--threshold` (default: `2e-5` for `float`, `2e-12` for `double`).
- **Scaling**: `zldsp` backward RFFT returns half the conventional unscaled inverse (`(N/2) * input`), so `zldsp` RFFT round trips scale by `2/N` (`1/N` for CFFT and external wrappers).

### Throughput

```console
python3 benchmark/run.py throughput cfft 5 20
python3 benchmark/run.py throughput rfft 5 20 --libraries zldsp kfr pffft vdsp
python3 benchmark/run.py throughput rfft 5 20 --standard-libraries
python3 benchmark/run.py throughput cfft 5 20 --all-libraries --double
python3 benchmark/run.py throughput rfft 5 20 --apis
```

- **Target selection** (mutually exclusive, default: `zldsp`):
  - `--libraries <lib ...>`: Measure specific backends.
  - `--standard-libraries`: Measure available backends (`zldsp`, `kfr`, `pffft`, plus `vdsp`, `ipp`, `armpl`, `armpl_estimate` when available).
  - `--all-libraries`: `--standard-libraries` plus `fftw3` and `fftw3_estimate`.
  - `--apis`: Measure all `zldsp` forward/backward AoS/SoA layouts, `forward_sqr_mag`, and fused vs. separate `convolve`.
- **Metrics**: Reports nominal **MFLOPS** from the median single-call time across `--repeats` runs (`--min-time` defaults to `0.1`s; `5*N*log2(N)` for CFFT, `2.5*N*log2(N)` for RFFT, and `5*N*log2(N) + 3*N` or `5*N*log2(N) + N` for complex or real RFFT convolution).

### Before/after a commit

```console
python3 benchmark/run.py compare cfft 5 20 --repeats 6
python3 benchmark/run.py compare rfft 5 20 --apis --current HEAD --previous HEAD^
```

Exports `--current` (default: `HEAD`) and `--previous` (default: `HEAD^`) into temporary directories using the current checkout's benchmark harness, alternates execution order across `--repeats` paired runs, and reports median times, geometric-mean throughput ratio (`current / previous`, `> 1` is faster), percentage change, and MAD.

### Supported Backends

`zldsp`, `fftw3`, `fftw3_estimate`, `kfr`, `pffft`, `vdsp`, `vdsp_stride_2`, `ipp`, `armpl`, `armpl_estimate`.

- `fftw3` and `armpl` use `FFTW_MEASURE`; `*_estimate` variants use `FFTW_ESTIMATE`.
- Set `ARMPL_DIR` for Arm Performance Libraries, and `IPP_ROOT` for Intel IPP installations outside default NuGet paths.

## Use `zldsp::fft` in another project

### Static Dispatch

```cmake
target_compile_definitions(my_static_target PRIVATE HWY_COMPILE_ONLY_STATIC)
```

Use the compiler's architecture option to select an SSE, AVX2, or NEON static target.

| SIMD Target | GCC/Clang                         | MSVC             |
| ----------- | --------------------------------- | ---------------- |
| SSE2        | `-march=x86-64`                   | no flag required |
| SSE4        | `-march=x86-64-v2 -maes -mpclmul` | not supported    |
| AVX2        | `-march=x86-64-v3 -maes -mpclmul` | `/arch:AVX2`     |
| NEON        | `-march=armv8-a+simd`             | `/arch:armv8.0`  |

See [`static_dispatch_caller`](examples/static_dispatch_caller.cpp) for CFFT and RFFT static dispatch examples.

### Caller-owned Dynamic Dispatch

For example, an x86 application can enable only SSE2 and AVX2:

```cmake
target_compile_definitions(my_dynamic_target PRIVATE "HWY_DISABLED_TARGETS=~(HWY_SSE2|HWY_AVX2)")
```

Compile this target for the oldest supported baseline (for example, `-march=x86-64` for SSE2).

See [`wrapper`](examples/dynamic_dispatch_wrapper.cpp), its [`interface`](examples/dynamic_dispatch_wrapper.hpp), and [`dynamic_dispatch_caller`](examples/dynamic_dispatch_caller.cpp) for CFFT and RFFT dynamic dispatch examples.

## License

zldsp_fft is licensed under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

All external libraries (submodules) are not covered by this license. All trademarks, product names, and company names are the property of their respective owners and are used for identification purposes only. Please refer to the individual licenses within each submodule:

- FFTW3: `fftw3_impl/fftw3` (GPL-2.0 license)
- KFR: `kfr_impl/kfr` (GPL-2.0 license)
- pffft: `pffft_impl/pffft` (BSD-like license)
- Google benchmark: `google/benchmark` (Apache-2.0 license)
- Google highway: `google/highway` (Apache-2.0 license)
