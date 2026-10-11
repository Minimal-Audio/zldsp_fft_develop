# zldsp::fft(develop)

zldsp::fft(develop) aims at FFT implementation and analysis.

If you want to use zldsp::fft, please refer to [zldsp_fft](https://github.com/ZL-Audio/zldsp_fft).

## Usage

Please make sure `Clang` (`AppleClang 16+` or `LLVM/Clang 17+`), `cmake` (minimum 3.20) are installed and configured on your OS.

```console
git clone https://github.com/ZL-Audio/zldsp_fft_develop
cd zldsp_fft_develop
git submodule update --init
```

All benchmarks use one command: `python3 benchmark/run.py <accuracy|throughput|compare> <cfft|rfft> <n0> <n1>`.
It works from any directory when invoked by its path, and caches separate builds by library, precision, and ISA under `build_fft/`.
On Windows, run from a Visual Studio developer shell with Clang and Ninja available.
Use `python3 benchmark/run.py --help` or a subcommand's `--help` for options.

### Accuracy

Library agreement uses the component-wise median of **FFTW (`FFTW_ESTIMATE`), KFR, and zldsp::fft**,
plus vDSP on Apple platforms and Intel IPP when its headers and libraries are found.
PFFFT is excluded from accuracy comparisons. These checks establish agreement within tolerance;
the median is not an exact mathematical oracle.

```console
python3 benchmark/run.py accuracy cfft 2 20
python3 benchmark/run.py accuracy rfft 2 20 --double
```

The API checks compare every zldsp layout with its AoS API, in both directions. RFFT also checks
`forward_sqr_mag` separately. RFFT `--apis` checks all three `convolve` overloads against separate
forward/multiply/backward execution, plus an independent direct DFT reference for small sizes.
Convolution checks cover changing kernels, native and caller-normalized scaling, zero/negative/sparse
kernels, DC/Nyquist and mirrored bins, preserved inputs/kernels, unaligned buffers, and in-place output.
Both accuracy suites verify normalized forward/backward round trips.

```console
python3 benchmark/run.py accuracy cfft 2 20 --apis
python3 benchmark/run.py accuracy rfft 2 20 --apis --double
```

Each order uses zero, shifted impulse, constant, alternating-sign, and tone inputs, followed by
three deterministic random inputs (`--repeats` changes the random count). Backward RFFT inputs
have real DC and Nyquist bins. Results report the worst **relative RMSE** and maximum absolute
error across inputs, separately for each library/API and direction. For an all-zero reference,
absolute RMS replaces relative RMSE. `--threshold` sets the relative RMSE limit (default `2e-5`
for float, `2e-12` for double); this replaces the old, size-dependent MSE threshold. Every library
must pass, and any non-finite result or missing case fails the run.

Scaling is reconciled for accuracy: zldsp RFFT backward returns half the conventional unscaled
inverse, so its result is multiplied by 2 before comparison, and round trips use `2/N`. The other
wrappers use `1/N` for round trips. The vDSP wrapper unpacks DC/Nyquist and removes its native
real-forward factor of 2. Error accumulation widens values before subtraction and squaring.

### Throughput

Compare one library, the available libraries, or all zldsp APIs:

```console
python3 benchmark/run.py throughput cfft 5 20
python3 benchmark/run.py throughput rfft 5 20 --libraries fftw3_estimate --double
python3 benchmark/run.py throughput rfft 5 20 --standard-libraries
python3 benchmark/run.py throughput cfft 5 20 --all-libraries
python3 benchmark/run.py throughput rfft 5 20 --libraries zldsp kfr pffft vdsp
python3 benchmark/run.py throughput cfft 5 20 --apis
python3 benchmark/run.py throughput rfft 5 20 --apis
```

Throughput defaults to zldsp. `--libraries` selects one or more backends; `--standard-libraries`
selects the available vDSP, IPP, ArmPL (measure and estimate), KFR, PFFFT, and zldsp backends.
`--all-libraries` adds FFTW measure and estimate. `--apis` measures all zldsp APIs.
These four options are mutually exclusive. The regular CFFT/RFFT throughput workflows use the
standard set; the full RFFT workflow includes FFTW, matching the original workflow split.

Every transform is measured and reported separately as `forward` or `backward`. CFFT API coverage
includes all four AoS/SoA input/output combinations in each direction. RFFT covers real/AoS and
real/SoA in each direction plus forward squared magnitude. RFFT `--apis` additionally reports
`convolve` timings for complex AoS, complex SoA, and real kernels, alongside matching `_separate`
forward/multiply/backward baselines. The real-kernel baseline uses SoA intermediate spectra.

One Google Benchmark iteration performs **one transform**, or **one complete convolution** (two
transforms plus multiplication) for a `convolve` case. Plans and allocations are outside
the timing region. Inputs are bounded and reused, never fed from the previous output. If an API
modifies its input, restoration is outside the timing region. Outputs are checked for finiteness
before and after timing. Packing conversions and scaling performed by external wrappers are part
of the measured API cost; zldsp's native inverse scale is preserved in throughput runs.

The reported time is the median CPU time across `--repeats` runs (default 3).
`--min-time` controls minimum seconds per case/repetition (default 0.1). MFLOPS uses the conventional
estimate `5*N*log2(N)` for CFFT or `2.5*N*log2(N)` for RFFT; it is not an instruction count.
For squared magnitude, the same RFFT estimate is used, so its time is the more useful comparison.
Convolution uses the same nominal work for fused and separate paths: `5*N*log2(N) + 3*N` for
complex kernels or `5*N*log2(N) + N` for real kernels. Use CPU time to assess the fusion's speedup;
this MFLOPS estimate does not count the fused implementation's actual arithmetic.
Unsupported PFFFT sizes are explicitly marked as skipped, never reported as zero-time work.
Other build/execution failures stop the run.

### Before/after a commit

```console
python3 benchmark/run.py compare cfft 5 20 --repeats 6
python3 benchmark/run.py compare rfft 5 20 --apis --double --json comparison.json
```

By default these compare `HEAD` with `HEAD^`; use `--current <ref>` and `--previous <ref>` to choose
other commits. Both revisions are exported into temporary directories, with each revision's pinned
Highway and Google Benchmark dependencies. The **same harness and CMake configuration from the current
checkout** run against both FFT implementations, preventing changes in measurement methodology from
appearing as speedups. Uncommitted FFT changes are excluded. No checkout, reset, or fetch is performed;
the requested commits and submodule objects must already exist locally.
RFFT `--apis` comparisons now include convolution, so both revisions must provide `convolve`;
an older revision reports the missing API explicitly. Comparisons without `--apis` still measure
only the original transform APIs.

Runs alternate which revision is measured first. The summary reports each direction/API independently,
with the geometric mean of paired throughput ratios (`current / previous`, greater than 1 is faster),
percentage change, and the median absolute deviation of log ratios as a percentage. Results remain
sensitive to system load and frequency changes; small differences need repeated measurements.

### Options and dependencies

Accuracy and throughput tables use FFT orders as rows and libraries as columns. With `--apis`,
the columns are zldsp APIs instead. Forward, backward, convolution, and round-trip results are separate, with
one metric per table: CPU time in microseconds and MFLOPS for throughput; relative RMSE and maximum
absolute error for accuracy. Throughput values use six significant digits; accuracy errors use
scientific notation so small errors remain visible. Unsupported cases display `SKIP` with a reason.

Every successful command ends with a valid, compact `JSON summary:` block in the console, including
CI logs. It groups each metric by `direction/API` and library, with arrays aligned to the explicit
`orders` list. Skipped values are `null`, with reasons in `skipped`. JSON values retain their full
precision; rounding applies only to the tables. Commit summaries also include the compared revisions.

All three commands accept `--double`, `--avx2` or `--sse2`, and `--json <file>`. The optional JSON file
retains the detailed per-case `results` format, including raw paired runs for commit comparisons;
the console summary is printed whether or not a file is requested.
Orders must satisfy `1 <= n0 <= n1 <= 30`; the practical maximum depends on available memory.

Supported throughput backends: `zldsp`, `fftw3`, `fftw3_estimate`, `kfr`, `pffft`, `vdsp`,
`vdsp_stride_2` (CFFT uses interleaved stride 2), `ipp`, `armpl`, `armpl_estimate`.
Set `ARMPL_DIR` for Arm Performance Libraries and `IPP_ROOT` for an IPP installation outside the
existing NuGet locations. `armpl`/`fftw3` use measure plans; the `_estimate` variants use estimate plans.
The existing CMake `ACCURACY_*_TEST` and `THROUGHPUT_*_TEST` target options remain available. A build
selects at most one `ENABLE_<library>` throughput backend; library accuracy always uses the set above.
CMake builds the variants from just `benchmark/accuracy.cpp` and `benchmark/throughput.cpp`, selecting
the FFT type and API coverage with compile definitions.
CMake options `ACCURACY_ENABLE_IPP=OFF` and `ACCURACY_ENABLE_VDSP=OFF` exclude optional SDKs from accuracy.
`BENCHMARK_SELF_TESTS=ON` registers enabled accuracy executables and Python result-parser checks in CTest.

## Use `zldsp::fft` in another project

### RFFT convolution

`RFFT<F>::convolve(input, spectrum, output)` performs circular convolution of an `N = 2^order`
sample real block with a supplied frequency response. It fuses RFFT postprocessing, multiplication,
and inverse preprocessing into one pass between the two `N/2`-point complex FFTs, using the existing
RFFT workspace. There is no intermediate RFFT spectrum allocation or kernel preparation step.

The spectrum contains `N/2 + 1` bins in natural order, from DC to Nyquist:

```cpp
// input/output: N floats; each spectrum array: N/2 + 1 elements.
zldsp::fft::RFFT<float> fft(order);
fft.convolve(input, complex_spectrum, output); // const std::complex<float>* (AoS)
fft.convolve(input, std::array<const float*, 2>{spectrum_real, spectrum_imag}, output); // SoA
fft.convolve(input, real_spectrum, output); // const float*: real multipliers, including negative values
fft.convolve(input, real_spectrum, input);  // Exact in-place output is supported.
```

Complex DC and Nyquist bins must be real; negative-frequency bins are implied by conjugate symmetry.
Native scaling matches separate `forward`, spectral multiplication, and `backward`: an all-ones
spectrum produces `(N/2) * input`. Pre-scale the spectrum by `2/N` for normalized convolution.
Input and spectrum are preserved for disjoint output. Partial input/output overlap and spectrum
overlap with either time buffer are unsupported; only ordinary element alignment is required.
Calls allocate no memory and borrow the spectrum only for the call, so kernels may change freely.
One instance owns mutable scratch space; use separate instances for concurrent execution.
Zero padding and overlap-add/save for linear or streaming convolution remain caller responsibilities.

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

See [`static_dispatch_caller`](examples/static_dispatch_caller.cpp) for CFFT and RFFT static dispatch examples,
including all convolution overloads and normalized identity kernels.

### Caller-owned Dynamic Dispatch

For example, an x86 application can enable only SSE2 and AVX2:

```cmake
target_compile_definitions(my_dynamic_target PRIVATE "HWY_DISABLED_TARGETS=~(HWY_SSE2|HWY_AVX2)")
```

Compile this target for the oldest supported baseline (for example, `-march=x86-64` for SSE2).

See [`wrapper`](examples/dynamic_dispatch_wrapper.cpp), its [`interface`](examples/dynamic_dispatch_wrapper.hpp),
and [`dynamic_dispatch_caller`](examples/dynamic_dispatch_caller.cpp) for CFFT and RFFT dynamic dispatch examples,
including the three convolution overloads.

## License

zldsp_fft is licensed under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

All external libraries (submodules) are not covered by this license. All trademarks, product names, and company names are the property of their respective owners and are used for identification purposes only. Please refer to the individual licenses within each submodule:

- FFTW3: `fftw3_impl/fftw3` (GPL-2.0 license)
- KFR: `kfr_impl/kfr` (GPL-2.0 license)
- pffft: `pffft_impl/pffft` (BSD-like license)
- Google benchmark: `google/benchmark` (Apache-2.0 license)
- Google highway: `google/highway` (Apache-2.0 license)
