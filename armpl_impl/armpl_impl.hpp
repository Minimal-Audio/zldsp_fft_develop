#pragma once

#include <armpl.h>
#include "../fftw3_impl/fftw3_impl.hpp"

namespace zlbenchmark {
    template <typename F, unsigned Flags = FFTW_MEASURE>
    using ArmPLFFT = FFTW3FFT<F, Flags>;
    template <typename F, unsigned Flags = FFTW_MEASURE>
    using ArmPLRFFT = FFTW3RFFT<F, Flags>;
    template <typename F, unsigned Flags = FFTW_MEASURE>
    using ARMPLFFT = ArmPLFFT<F, Flags>;
    template <typename F, unsigned Flags = FFTW_MEASURE>
    using ARMPLRFFT = ArmPLRFFT<F, Flags>;
}
