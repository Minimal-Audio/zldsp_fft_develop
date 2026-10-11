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
`forward_sqr_mag` separately. Both accuracy suites verify normalized forward/backward round trips.

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
real/SoA in each direction plus forward squared magnitude.

One Google Benchmark iteration performs **one transform**. Plans and allocations are outside
the timing region. Inputs are bounded and reused, never fed from the previous output. If an API
modifies its input, restoration is outside the timing region. Outputs are checked for finiteness
before and after timing. Packing conversions and scaling performed by external wrappers are part
of the measured API cost; zldsp's native inverse scale is preserved in throughput runs.

The reported time is the median CPU time across `--repeats` runs (default 3).
`--min-time` controls minimum seconds per case/repetition (default 0.1). MFLOPS uses the conventional
estimate `5*N*log2(N)` for CFFT or `2.5*N*log2(N)` for RFFT; it is not an instruction count.
For squared magnitude, the same RFFT estimate is used, so its time is the more useful comparison.
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

Runs alternate which revision is measured first. The summary reports each direction/API independently,
with the geometric mean of paired throughput ratios (`current / previous`, greater than 1 is faster),
percentage change, and the median absolute deviation of log ratios as a percentage. Results remain
sensitive to system load and frequency changes; small differences need repeated measurements.

### Options and dependencies

Accuracy and throughput tables use FFT orders as rows and libraries as columns. With `--apis`,
the columns are zldsp APIs instead. Forward, backward, and round-trip results are separate, with
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

See [`static_dispatch_caller`](examples/static_dispatch_caller.cpp) for CFFT and RFFT static dispatch examples with AoS and SoA layouts.

### Caller-owned Dynamic Dispatch

For example, an x86 application can enable only SSE2 and AVX2:

```cmake
target_compile_definitions(my_dynamic_target PRIVATE "HWY_DISABLED_TARGETS=~(HWY_SSE2|HWY_AVX2)")
```

Compile this target for the oldest supported baseline (for example, `-march=x86-64` for SSE2).

See [`wrapper`](examples/dynamic_dispatch_wrapper.cpp), its [`interface`](examples/dynamic_dispatch_wrapper.hpp), and [`dynamic_dispatch_caller`](examples/dynamic_dispatch_caller.cpp) for CFFT and RFFT dynamic dispatch examples with AoS and SoA layouts.

## License

zldsp_fft is licensed under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

All external libraries (submodules) are not covered by this license. All trademarks, product names, and company names are the property of their respective owners and are used for identification purposes only. Please refer to the individual licenses within each submodule:

- FFTW3: `fftw3_impl/fftw3` (GPL-2.0 license)
- KFR: `kfr_impl/kfr` (GPL-2.0 license)
- pffft: `pffft_impl/pffft` (BSD-like license)
- Google benchmark: `google/benchmark` (Apache-2.0 license)
- Google highway: `google/highway` (Apache-2.0 license)
