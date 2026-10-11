#pragma once

#include <Accelerate/Accelerate.h>
#include <complex>
#include <span>
#include <stdexcept>
#include <vector>
#include "../benchmark/fft_support.hpp"

namespace zlbenchmark {
    template <typename F>
    class VDSPFFT;

    template <>
    class VDSPFFT<float> final {
        using C = std::complex<float>;

    public:
        explicit VDSPFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetup(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetup failed");
            }
#ifndef ENABLE_VDSP_STRIDE_2
            temp_split_real_.resize(n_);
            temp_split_imag_.resize(n_);
            split_complex_.realp = temp_split_real_.data();
            split_complex_.imagp = temp_split_imag_.data();
#endif
        }

        VDSPFFT(const VDSPFFT&) = delete;
        VDSPFFT& operator=(const VDSPFFT&) = delete;

        ~VDSPFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetup(fft_setup_);
            }
        }

        void forward(C* in_buffer, C* out_buffer) { transform(in_buffer, out_buffer, FFT_FORWARD); }
        void backward(C* in_buffer, C* out_buffer) { transform(in_buffer, out_buffer, FFT_INVERSE); }

    private:
        void transform(C* in_buffer, C* out_buffer, FFTDirection direction) {
#ifndef ENABLE_VDSP_STRIDE_2
            vDSP_ctoz(reinterpret_cast<DSPComplex*>(in_buffer), 2, &split_complex_, 1, n_);
            vDSP_fft_zip(fft_setup_, &split_complex_, 1, order_, direction);
            vDSP_ztoc(&split_complex_, 1, reinterpret_cast<DSPComplex*>(out_buffer), 2, n_);
#else
            DSPSplitComplex inline_split_in;
            inline_split_in.realp = reinterpret_cast<float*>(in_buffer);
            inline_split_in.imagp = reinterpret_cast<float*>(in_buffer) + 1;

            DSPSplitComplex inline_split_out;
            inline_split_out.realp = reinterpret_cast<float*>(out_buffer);
            inline_split_out.imagp = reinterpret_cast<float*>(out_buffer) + 1;

            vDSP_fft_zop(fft_setup_, &inline_split_in, 2, &inline_split_out, 2, order_, direction);
#endif
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetup fft_setup_;
        std::vector<float> temp_split_real_;
        std::vector<float> temp_split_imag_;
        DSPSplitComplex split_complex_;
    };

    template <>
    class VDSPFFT<double> final {
        using C = std::complex<double>;

    public:
        explicit VDSPFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetupD(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetupD failed");
            }
#ifndef ENABLE_VDSP_STRIDE_2
            temp_split_real_.resize(n_);
            temp_split_imag_.resize(n_);
            split_complex_.realp = temp_split_real_.data();
            split_complex_.imagp = temp_split_imag_.data();
#endif
        }

        VDSPFFT(const VDSPFFT&) = delete;
        VDSPFFT& operator=(const VDSPFFT&) = delete;

        ~VDSPFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetupD(fft_setup_);
            }
        }

        void forward(C* in_buffer, C* out_buffer) { transform(in_buffer, out_buffer, FFT_FORWARD); }
        void backward(C* in_buffer, C* out_buffer) { transform(in_buffer, out_buffer, FFT_INVERSE); }

    private:
        void transform(C* in_buffer, C* out_buffer, FFTDirection direction) {
#ifndef ENABLE_VDSP_STRIDE_2
            vDSP_ctozD(reinterpret_cast<DSPDoubleComplex*>(in_buffer), 2, &split_complex_, 1, n_);
            vDSP_fft_zipD(fft_setup_, &split_complex_, 1, order_, direction);
            vDSP_ztocD(&split_complex_, 1, reinterpret_cast<DSPDoubleComplex*>(out_buffer), 2, n_);
#else
            DSPDoubleSplitComplex inline_split_in;
            inline_split_in.realp = reinterpret_cast<double*>(in_buffer);
            inline_split_in.imagp = reinterpret_cast<double*>(in_buffer) + 1;

            DSPDoubleSplitComplex inline_split_out;
            inline_split_out.realp = reinterpret_cast<double*>(out_buffer);
            inline_split_out.imagp = reinterpret_cast<double*>(out_buffer) + 1;

            vDSP_fft_zopD(fft_setup_, &inline_split_in, 2, &inline_split_out, 2, order_, direction);
#endif
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetupD fft_setup_;
        std::vector<double> temp_split_real_;
        std::vector<double> temp_split_imag_;
        DSPDoubleSplitComplex split_complex_;
    };
    template <typename F>
    class VDSPRFFT;

    template <>
    class VDSPRFFT<float> final {
        using C = std::complex<float>;

    public:
        explicit VDSPRFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetup(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetup failed");
            }
            temp_split_real_.resize(n_ / 2);
            temp_split_imag_.resize(n_ / 2);
            split_complex_.realp = temp_split_real_.data();
            split_complex_.imagp = temp_split_imag_.data();
        }

        VDSPRFFT(const VDSPRFFT&) = delete;
        VDSPRFFT& operator=(const VDSPRFFT&) = delete;

        ~VDSPRFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetup(fft_setup_);
            }
        }

        void forward(const float* in_buffer, C* out_buffer) {
            vDSP_ctoz(reinterpret_cast<const DSPComplex*>(in_buffer), 2, &split_complex_, 1, n_ / 2);
            vDSP_fft_zrip(fft_setup_, &split_complex_, 1, order_, FFT_FORWARD);
            // vDSP packs Nyquist in imagp[0] and returns twice the usual DFT.
            out_buffer[0] = C(split_complex_.realp[0] * 0.5, 0);
            out_buffer[n_ / 2] = C(split_complex_.imagp[0] * 0.5, 0);
            for (size_t k = 1; k < n_ / 2; ++k) {
                out_buffer[k] = C(split_complex_.realp[k] * 0.5, split_complex_.imagp[k] * 0.5);
            }
        }

        void backward(const C* in_buffer, float* out_buffer) {
            split_complex_.realp[0] = in_buffer[0].real();
            split_complex_.imagp[0] = in_buffer[n_ / 2].real();
            for (size_t k = 1; k < n_ / 2; ++k) {
                split_complex_.realp[k] = in_buffer[k].real();
                split_complex_.imagp[k] = in_buffer[k].imag();
            }
            vDSP_fft_zrip(fft_setup_, &split_complex_, 1, order_, FFT_INVERSE);
            vDSP_ztoc(&split_complex_, 1, reinterpret_cast<DSPComplex*>(out_buffer), 2, n_ / 2);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetup fft_setup_;
        std::vector<float> temp_split_real_;
        std::vector<float> temp_split_imag_;
        DSPSplitComplex split_complex_;
    };

    template <>
    class VDSPRFFT<double> final {
        using C = std::complex<double>;

    public:
        explicit VDSPRFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetupD(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetupD failed");
            }
            temp_split_real_.resize(n_ / 2);
            temp_split_imag_.resize(n_ / 2);
            split_complex_.realp = temp_split_real_.data();
            split_complex_.imagp = temp_split_imag_.data();
        }

        VDSPRFFT(const VDSPRFFT&) = delete;
        VDSPRFFT& operator=(const VDSPRFFT&) = delete;

        ~VDSPRFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetupD(fft_setup_);
            }
        }

        void forward(const double* in_buffer, C* out_buffer) {
            vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(in_buffer), 2, &split_complex_, 1, n_ / 2);
            vDSP_fft_zripD(fft_setup_, &split_complex_, 1, order_, FFT_FORWARD);
            // vDSP packs Nyquist in imagp[0] and returns twice the usual DFT.
            out_buffer[0] = C(split_complex_.realp[0] * 0.5, 0);
            out_buffer[n_ / 2] = C(split_complex_.imagp[0] * 0.5, 0);
            for (size_t k = 1; k < n_ / 2; ++k) {
                out_buffer[k] = C(split_complex_.realp[k] * 0.5, split_complex_.imagp[k] * 0.5);
            }
        }

        void backward(const C* in_buffer, double* out_buffer) {
            split_complex_.realp[0] = in_buffer[0].real();
            split_complex_.imagp[0] = in_buffer[n_ / 2].real();
            for (size_t k = 1; k < n_ / 2; ++k) {
                split_complex_.realp[k] = in_buffer[k].real();
                split_complex_.imagp[k] = in_buffer[k].imag();
            }
            vDSP_fft_zripD(fft_setup_, &split_complex_, 1, order_, FFT_INVERSE);
            vDSP_ztocD(&split_complex_, 1, reinterpret_cast<DSPDoubleComplex*>(out_buffer), 2, n_ / 2);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetupD fft_setup_;
        std::vector<double> temp_split_real_;
        std::vector<double> temp_split_imag_;
        DSPDoubleSplitComplex split_complex_;
    };
}
