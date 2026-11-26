/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE void Average16(__m256i & a, const __m256i & b)
        {
            a = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(a, b), K16_0002), 2);
        }

        SIMD_INLINE void Average16(__m256i a[2][2])
        {
            a[0][0] = _mm256_srli_epi16(_mm256_add_epi16(a[0][0], K16_0001), 1);
            a[0][1] = _mm256_srli_epi16(_mm256_add_epi16(a[0][1], K16_0001), 1);
            a[1][0] = _mm256_srli_epi16(_mm256_add_epi16(a[1][0], K16_0001), 1);
            a[1][1] = _mm256_srli_epi16(_mm256_add_epi16(a[1][1], K16_0001), 1);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE __m256i BgrToY16(__m256i b16_r16[2], __m256i g16_1[2])
        {
            static const __m256i Y_LO = SIMD_MM256_SET1_EPI16(T::Y_LO);
            return SaturateI16ToU8(_mm256_add_epi16(Y_LO, PackI32ToI16(BgrToY32<T>(b16_r16[0], g16_1[0]), BgrToY32<T>(b16_r16[1], g16_1[1]))));
        }

        template <class T> SIMD_INLINE __m256i BgrToU16(__m256i b16_r16[2], __m256i g16_1[2])
        {
            static const __m256i UV_Z = SIMD_MM256_SET1_EPI16(T::UV_Z);
            return SaturateI16ToU8(_mm256_add_epi16(UV_Z, PackI32ToI16(BgrToU32<T>(b16_r16[0], g16_1[0]), BgrToU32<T>(b16_r16[1], g16_1[1]))));
        }

        template <class T> SIMD_INLINE __m256i BgrToV16(__m256i b16_r16[2], __m256i g16_1[2])
        {
            static const __m256i UV_Z = SIMD_MM256_SET1_EPI16(T::UV_Z);
            return SaturateI16ToU8(_mm256_add_epi16(UV_Z, PackI32ToI16(BgrToV32<T>(b16_r16[0], g16_1[0]), BgrToV32<T>(b16_r16[1], g16_1[1]))));
        }

        template <class T> SIMD_INLINE void BgraToYuv444pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            __m256i _b16_r16[2][2], _g16_1[2][2];
            LoadPreparedBgra16<false>((__m256i*)bgra + 0, _b16_r16[0][0], _g16_1[0][0]);
            LoadPreparedBgra16<false>((__m256i*)bgra + 1, _b16_r16[0][1], _g16_1[0][1]);
            LoadPreparedBgra16<false>((__m256i*)bgra + 2, _b16_r16[1][0], _g16_1[1][0]);
            LoadPreparedBgra16<false>((__m256i*)bgra + 3, _b16_r16[1][1], _g16_1[1][1]);

            Store<false>((__m256i*)y, PackI16ToU8(BgrToY16<T>(_b16_r16[0], _g16_1[0]), BgrToY16<T>(_b16_r16[1], _g16_1[1])));
            Store<false>((__m256i*)u, PackI16ToU8(BgrToU16<T>(_b16_r16[0], _g16_1[0]), BgrToU16<T>(_b16_r16[1], _g16_1[1])));
            Store<false>((__m256i*)v, PackI16ToU8(BgrToV16<T>(_b16_r16[0], _g16_1[0]), BgrToV16<T>(_b16_r16[1], _g16_1[1])));
        }

        template <class T> void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert(width >= A);

            size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    BgraToYuv444pV2<T>(bgra + col * 4, y + col, u + col, v + col);
                if (width != widthA)
                {
                    size_t col = width - A;
                    BgraToYuv444pV2<T>(bgra + col * 4, y + col, u + col, v + col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuv444pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv444pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv444pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv444pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv444pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE __m256i LoadAndBgrToY16(const __m256i* bgra, __m256i& b16_r16, __m256i& g16_1)
        {
            static const __m256i Y_LO = SIMD_MM256_SET1_EPI16(T::Y_LO);
            __m256i _b16_r16[2], _g16_1[2];
            LoadPreparedBgra16<false>(bgra + 0, _b16_r16[0], _g16_1[0]);
            LoadPreparedBgra16<false>(bgra + 1, _b16_r16[1], _g16_1[1]);
            b16_r16 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_b16_r16[0], _b16_r16[1]), 0xD8);
            g16_1 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_g16_1[0], _g16_1[1]), 0xD8);
            return BgrToY16<T>(_b16_r16, _g16_1);
        }

        template <class T> SIMD_INLINE __m256i LoadAndBgrToY8(const __m256i* bgra, __m256i b16_r16[2], __m256i g16_1[2])
        {
            return PackI16ToU8(LoadAndBgrToY16<T>(bgra + 0, b16_r16[0], g16_1[0]), LoadAndBgrToY16<T>(bgra + 2, b16_r16[1], g16_1[1]));
        }

        template <class T> SIMD_INLINE void BgraToYuv422pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            __m256i _b16_r16[2][2], _g16_1[2][2];
            Store<false>((__m256i*)y + 0, LoadAndBgrToY8<T>((__m256i*)bgra + 0, _b16_r16[0], _g16_1[0]));
            Store<false>((__m256i*)y + 1, LoadAndBgrToY8<T>((__m256i*)bgra + 4, _b16_r16[1], _g16_1[1]));

            Average16(_b16_r16);
            Average16(_g16_1);

            Store<false>((__m256i*)u, PackI16ToU8(BgrToU16<T>(_b16_r16[0], _g16_1[0]), BgrToU16<T>(_b16_r16[1], _g16_1[1])));
            Store<false>((__m256i*)v, PackI16ToU8(BgrToV16<T>(_b16_r16[0], _g16_1[0]), BgrToV16<T>(_b16_r16[1], _g16_1[1])));
        }

        template <class T>  void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));

            size_t widthDA = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv422pV2<T>(bgra + colBgra, y + colY, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t colY = width - DA;
                    BgraToYuv422pV2<T>(bgra + colY * 4, y + colY, u + colY / 2, v + colY / 2);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuv422pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv422pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv422pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv422pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv422pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE __m256i LoadAndRgbaToY16(const __m256i* bgra, __m256i& b16_r16, __m256i& g16_1)
        {
            __m256i _b16_r16[2], _g16_1[2];
            LoadPreparedBgra16<false>(bgra + 0, _b16_r16[0], _g16_1[0]);
            LoadPreparedBgra16<false>(bgra + 1, _b16_r16[1], _g16_1[1]);
            // For RGBA input, actual output from LoadPreparedBgra16 is r16_b16,
            // so we need to swap the R and B to get correct b16_r16 for calculation.
            static const __m256i K8_SWAP_BR = SIMD_MM256_SETR_EPI8(
                0x2, 0x3, 0x0, 0x1, 0x6, 0x7, 0x4, 0x5, 0xA, 0xB, 0x8, 0x9, 0xE, 0xF, 0xC, 0xD,
                0x2, 0x3, 0x0, 0x1, 0x6, 0x7, 0x4, 0x5, 0xA, 0xB, 0x8, 0x9, 0xE, 0xF, 0xC, 0xD);
            _b16_r16[0] = _mm256_shuffle_epi8(_b16_r16[0], K8_SWAP_BR);
            _b16_r16[1] = _mm256_shuffle_epi8(_b16_r16[1], K8_SWAP_BR);
            b16_r16 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_b16_r16[0], _b16_r16[1]), 0xD8);
            g16_1 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_g16_1[0], _g16_1[1]), 0xD8);
            return BgrToY16<T>(_b16_r16, _g16_1);
        }

        template <bool rgba, class T> SIMD_INLINE __m256i LoadAndRgbaToY8(const __m256i* bgra, __m256i b16_r16[2], __m256i g16_1[2])
        {
            if (!rgba)
                return LoadAndBgrToY8<T>(bgra, b16_r16, g16_1);

            return PackI16ToU8(LoadAndRgbaToY16<T>(bgra + 0, b16_r16[0], g16_1[0]), LoadAndRgbaToY16<T>(bgra + 2, b16_r16[1], g16_1[1]));
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool nv12 = false, bool rgba = false> SIMD_INLINE void BgraToYuv420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            __m256i _b16_r16[2][2][2], _g16_1[2][2][2];
            Store<false>((__m256i*)y0 + 0, LoadAndRgbaToY8<rgba, T>((__m256i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0]));
            Store<false>((__m256i*)y0 + 1, LoadAndRgbaToY8<rgba, T>((__m256i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1]));
            Store<false>((__m256i*)y1 + 0, LoadAndRgbaToY8<rgba, T>((__m256i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0]));
            Store<false>((__m256i*)y1 + 1, LoadAndRgbaToY8<rgba, T>((__m256i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1]));

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            __m256i chromaU = PackI16ToU8(BgrToU16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToU16<T>(_b16_r16[0][1], _g16_1[0][1]));
            __m256i chromaV = PackI16ToU8(BgrToV16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToV16<T>(_b16_r16[0][1], _g16_1[0][1]));
            if (nv12)
            {
                chromaU = _mm256_permute4x64_epi64(chromaU, 0xD8);
                chromaV = _mm256_permute4x64_epi64(chromaV, 0xD8);
                Store<false>((__m256i*)u + 0, UnpackU8<0>(chromaU, chromaV));
                Store<false>((__m256i*)u + 1, UnpackU8<1>(chromaU, chromaV));
            }
            else
            {
                Store<false>((__m256i*)u, chromaU);
                Store<false>((__m256i*)v, chromaV);
            }
        }

        template <class T>  void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));

            size_t widthDA = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t colY = width - DA;
                    BgraToYuv420pV2<T>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colY / 2, v + colY / 2);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuv420pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv420pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv420pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv420pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv420pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool rgba> void BgraToNv12(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* uv, size_t uvStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0; col < widthDA; col += DA)
                    BgraToYuv420pV2<T, true, rgba>(bgra + col * 4, bgraStride, y + col, yStride, uv + col, NULL);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgraToYuv420pV2<T, true, rgba>(bgra + col * 4, bgraStride, y + col, yStride, uv + col, NULL);
                }
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                uv += uvStride;
            }
        }

        template <bool rgba> void BgraToNv12(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* uv, size_t uvStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToNv12<Base::Bt601, rgba>(bgra, bgraStride, width, height, y, yStride, uv, uvStride); break;
            case SimdYuvBt709: BgraToNv12<Base::Bt709, rgba>(bgra, bgraStride, width, height, y, yStride, uv, uvStride); break;
            case SimdYuvBt2020: BgraToNv12<Base::Bt2020, rgba>(bgra, bgraStride, width, height, y, yStride, uv, uvStride); break;
            case SimdYuvTrect871: BgraToNv12<Base::Trect871, rgba>(bgra, bgraStride, width, height, y, yStride, uv, uvStride); break;
            default:
                assert(0);
            }
        }

        void BgraToNv12(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* uv, size_t uvStride, SimdRgbOrderType rgbOrder, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToNv12(bgra, bgraStride, width, height, y, yStride, uv, uvStride, rgbOrder, yuvType);
#else
            assert(rgbOrder == SimdBgr || rgbOrder == SimdRgb);
            if (rgbOrder == SimdRgb)
                BgraToNv12<true>(bgra, bgraStride, width, height, y, yStride, uv, uvStride, yuvType);
            else
                BgraToNv12<false>(bgra, bgraStride, width, height, y, yStride, uv, uvStride, yuvType);
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void LoadAndConvertYA16(const __m256i* bgra, __m256i& b16_r16, __m256i& g16_1, __m256i& y16, __m256i& a16)
        {
            __m256i _b16_r16[2], _g16_1[2], a32[2];
            LoadPreparedBgra16<false>(bgra + 0, _b16_r16[0], _g16_1[0], a32[0]);
            LoadPreparedBgra16<false>(bgra + 1, _b16_r16[1], _g16_1[1], a32[1]);
            b16_r16 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_b16_r16[0], _b16_r16[1]), 0xD8);
            g16_1 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_g16_1[0], _g16_1[1]), 0xD8);
            static const __m256i Y_LO = SIMD_MM256_SET1_EPI16(T::Y_LO);
            y16 = SaturateI16ToU8(_mm256_add_epi16(Y_LO, PackI32ToI16(BgrToY32<T>(_b16_r16[0], _g16_1[0]), BgrToY32<T>(_b16_r16[1], _g16_1[1]))));
            a16 = PackI32ToI16(a32[0], a32[1]);
        }

        template <class T> SIMD_INLINE void LoadAndStoreYA(const __m256i* bgra, __m256i b16_r16[2], __m256i g16_1[2], __m256i* y, __m256i* a)
        {
            __m256i y16[2], a16[2];
            LoadAndConvertYA16<T>(bgra + 0, b16_r16[0], g16_1[0], y16[0], a16[0]);
            LoadAndConvertYA16<T>(bgra + 2, b16_r16[1], g16_1[1], y16[1], a16[1]);
            _mm256_storeu_si256(y, PackI16ToU8(y16[0], y16[1]));
            _mm256_storeu_si256(a, PackI16ToU8(a16[0], a16[1]));
        }

        template <class T> SIMD_INLINE void BgraToYuva420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v, uint8_t* a0, size_t aStride)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;
            uint8_t* a1 = a0 + aStride;

            __m256i _b16_r16[2][2][2], _g16_1[2][2][2];
            LoadAndStoreYA<T>((__m256i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0], (__m256i*)y0 + 0, (__m256i*)a0 + 0);
            LoadAndStoreYA<T>((__m256i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1], (__m256i*)y0 + 1, (__m256i*)a0 + 1);
            LoadAndStoreYA<T>((__m256i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0], (__m256i*)y1 + 0, (__m256i*)a1 + 0);
            LoadAndStoreYA<T>((__m256i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1], (__m256i*)y1 + 1, (__m256i*)a1 + 1);

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            _mm256_storeu_si256((__m256i*)u, PackI16ToU8(BgrToU16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToU16<T>(_b16_r16[0][1], _g16_1[0][1])));
            _mm256_storeu_si256((__m256i*)v, PackI16ToU8(BgrToV16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToV16<T>(_b16_r16[0][1], _g16_1[0][1])));
        }

        template <class T>  void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA));

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgraToYuva420pV2<T>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colUV, v + colUV, a + colY, aStride);
                if (width != widthDA)
                {
                    size_t colY = width - DA;
                    BgraToYuva420pV2<T>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colY / 2, v + colY / 2, a + colY, aStride);
                }
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
            }
        }

        void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuva420pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuva420pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt709: BgraToYuva420pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt2020: BgraToYuva420pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvTrect871: BgraToYuva420pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align, bool rgba, class T> SIMD_INLINE void BgraToYuv422Packed(const uint8_t* bgra, uint8_t* yuv, SimdYuv422PackPatternType yuvPattern)
        {

            // input 64 BGRA/RGBA pixels
            __m256i _b16_r16[2][2], _g16_1[2][2];

            // LoadAndRgbaToY8 gives pixel-ordered 32 Y values, need to permute so that 
            // UnpackU8<0> and UnpackU8<1> can interleave them correctly with the UV values.
            __m256i y0 = _mm256_permute4x64_epi64(LoadAndRgbaToY8<rgba, T>((__m256i*)bgra + 0, _b16_r16[0], _g16_1[0]), 0xD8);
            __m256i y1 = _mm256_permute4x64_epi64(LoadAndRgbaToY8<rgba, T>((__m256i*)bgra + 4, _b16_r16[1], _g16_1[1]), 0xD8);

            Average16(_b16_r16);
            Average16(_g16_1);

            // Use K32_PERMUTE_UV to rearrange U and V to ensure following UnpackU8<0> and UnpackU8<1> interleave them correctly with the Y values.
            static const __m256i K32_PERMUTE_UV = SIMD_MM256_SETR_EPI32(0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7);
            __m256i u = _mm256_permutevar8x32_epi32(PackI16ToU8(BgrToU16<T>(_b16_r16[0], _g16_1[0]), BgrToU16<T>(_b16_r16[1], _g16_1[1])), K32_PERMUTE_UV);
            __m256i v = _mm256_permutevar8x32_epi32(PackI16ToU8(BgrToV16<T>(_b16_r16[0], _g16_1[0]), BgrToV16<T>(_b16_r16[1], _g16_1[1])), K32_PERMUTE_UV);

            // Interleave luma and chroma into the packed YUV422 output.
            switch (yuvPattern)
            {
            case SimdYuyv422:
            {
                __m256i uv0 = UnpackU8<0>(u, v);
                __m256i uv1 = UnpackU8<1>(u, v);
                Store<align>((__m256i*)yuv + 0, UnpackU8<0>(y0, uv0));
                Store<align>((__m256i*)yuv + 1, UnpackU8<1>(y0, uv0));
                Store<align>((__m256i*)yuv + 2, UnpackU8<0>(y1, uv1));
                Store<align>((__m256i*)yuv + 3, UnpackU8<1>(y1, uv1));
                break;
            }
            case SimdYvyu422:
            {
                __m256i vu0 = UnpackU8<0>(v, u);
                __m256i vu1 = UnpackU8<1>(v, u);
                Store<align>((__m256i*)yuv + 0, UnpackU8<0>(y0, vu0));
                Store<align>((__m256i*)yuv + 1, UnpackU8<1>(y0, vu0));
                Store<align>((__m256i*)yuv + 2, UnpackU8<0>(y1, vu1));
                Store<align>((__m256i*)yuv + 3, UnpackU8<1>(y1, vu1));
                break;
            }
            case SimdUyvy422:
            {
                __m256i uv0 = UnpackU8<0>(u, v);
                __m256i uv1 = UnpackU8<1>(u, v);
                Store<align>((__m256i*)yuv + 0, UnpackU8<0>(uv0, y0));
                Store<align>((__m256i*)yuv + 1, UnpackU8<1>(uv0, y0));
                Store<align>((__m256i*)yuv + 2, UnpackU8<0>(uv1, y1));
                Store<align>((__m256i*)yuv + 3, UnpackU8<1>(uv1, y1));
                break;
            }
            case SimdVyuy422:
            {
                __m256i vu0 = UnpackU8<0>(v, u);
                __m256i vu1 = UnpackU8<1>(v, u);
                Store<align>((__m256i*)yuv + 0, UnpackU8<0>(vu0, y0));
                Store<align>((__m256i*)yuv + 1, UnpackU8<1>(vu0, y0));
                Store<align>((__m256i*)yuv + 2, UnpackU8<0>(vu1, y1));
                Store<align>((__m256i*)yuv + 3, UnpackU8<1>(vu1, y1));
                break;
            }
            default:
                assert(0);
            }
        }

        template <bool align, bool rgba, class T> void BgraToYuv422Packed(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* yuv, size_t yuvStride,
            SimdYuv422PackPatternType yuvPattern)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
                assert(Aligned(yuv) && Aligned(yuvStride));

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t col = 0; col < widthDA; col += DA)
                    BgraToYuv422Packed<align, rgba, T>(bgra + col * 4, yuv + col * 2, yuvPattern);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgraToYuv422Packed<false, rgba, T>(bgra + col * 4, yuv + col * 2, yuvPattern);
                }
                yuv += yuvStride;
                bgra += bgraStride;
            }
        }

        template <bool align, class T> void BgraToYuv422Packed(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* yuv, size_t yuvStride,
            SimdRgbOrderType rgbOrder, SimdYuv422PackPatternType yuvPattern)
        {
            switch (rgbOrder)
            {
            case SimdBgr: BgraToYuv422Packed<align, false, T>(bgra, bgraStride, width, height, yuv, yuvStride, yuvPattern); break;
            case SimdRgb: BgraToYuv422Packed<align, true, T>(bgra, bgraStride, width, height, yuv, yuvStride, yuvPattern); break;
            default:
                assert(0);
            }
        }

        template <class T> void BgraToYuv422Packed(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* yuv, size_t yuvStride,
            SimdRgbOrderType rgbOrder, SimdYuv422PackPatternType yuvPattern)
        {
            if (Aligned(yuv) && Aligned(yuvStride))
                BgraToYuv422Packed<true, T>(bgra, bgraStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern);
            else
                BgraToYuv422Packed<false, T>(bgra, bgraStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern);
        }

        void BgraToYuv422Packed(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* yuv, size_t yuvStride,
            SimdRgbOrderType rgbOrder, SimdYuvType yuvType, SimdYuv422PackPatternType yuvPattern)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuv422Packed(bgra, bgraStride, width, height, yuv, yuvStride, rgbOrder, yuvType, yuvPattern);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv422Packed<Base::Bt601>(bgra, bgraStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            case SimdYuvBt709: BgraToYuv422Packed<Base::Bt709>(bgra, bgraStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            case SimdYuvBt2020: BgraToYuv422Packed<Base::Bt2020>(bgra, bgraStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            case SimdYuvTrect871: BgraToYuv422Packed<Base::Trect871>(bgra, bgraStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            default:
                assert(0);
            }
#endif
        }
    }
#endif// SIMD_AVX2_ENABLE
}
