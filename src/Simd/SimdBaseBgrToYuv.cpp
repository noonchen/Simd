/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
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
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
    namespace Base
    {
        template <class YuvType> SIMD_INLINE void BgrToYuv420pV2(const uint8_t* bgr0, size_t bgrStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v,
            size_t blueIndex = 0, size_t redIndex = 2)
        {
            const uint8_t* bgr1 = bgr0 + bgrStride;
            uint8_t* y1 = y0 + yStride;

            y0[0] = BgrToY<YuvType>(bgr0[blueIndex], bgr0[1], bgr0[redIndex]);
            y0[1] = BgrToY<YuvType>(bgr0[3 + blueIndex], bgr0[4], bgr0[3 + redIndex]);
            y1[0] = BgrToY<YuvType>(bgr1[blueIndex], bgr1[1], bgr1[redIndex]);
            y1[1] = BgrToY<YuvType>(bgr1[3 + blueIndex], bgr1[4], bgr1[3 + redIndex]);

            int blue = Average(bgr0[blueIndex], bgr0[3 + blueIndex], bgr1[blueIndex], bgr1[3 + blueIndex]);
            int green = Average(bgr0[1], bgr0[4], bgr1[1], bgr1[4]);
            int red = Average(bgr0[redIndex], bgr0[3 + redIndex], bgr1[redIndex], bgr1[3 + redIndex]);

            u[0] = BgrToU<YuvType>(blue, green, red);
            v[0] = BgrToV<YuvType>(blue, green, red);
        }

        template <class YuvType> void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
                    BgrToYuv420pV2<YuvType>(bgr + colBgr, bgrStride, y + colY, yStride, u + colUV, v + colUV);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv420pV2<Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv420pV2<Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv420pV2<Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv420pV2<Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class YuvType> void BgrToNv12(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* uv, size_t uvStride, SimdRgbOrderType rgbOrder)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));
            assert(rgbOrder == SimdBgr || rgbOrder == SimdRgb);
            size_t blueIndex = rgbOrder == SimdRgb ? 2 : 0;
            size_t redIndex = 2 - blueIndex;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0; col < width; col += 2)
                    BgrToYuv420pV2<YuvType>(bgr + col * 3, bgrStride, y + col, yStride, uv + col, uv + col + 1, blueIndex, redIndex);
                bgr += 2 * bgrStride;
                y += 2 * yStride;
                uv += uvStride;
            }
        }

        void BgrToNv12(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* uv, size_t uvStride, SimdRgbOrderType rgbOrder, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToNv12<Bt601>(bgr, bgrStride, width, height, y, yStride, uv, uvStride, rgbOrder); break;
            case SimdYuvBt709: BgrToNv12<Bt709>(bgr, bgrStride, width, height, y, yStride, uv, uvStride, rgbOrder); break;
            case SimdYuvBt2020: BgrToNv12<Bt2020>(bgr, bgrStride, width, height, y, yStride, uv, uvStride, rgbOrder); break;
            case SimdYuvTrect871: BgrToNv12<Trect871>(bgr, bgrStride, width, height, y, yStride, uv, uvStride, rgbOrder); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class YuvType> SIMD_INLINE void BgrToYuv422pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            y[0] = BgrToY<YuvType>(bgr[0], bgr[1], bgr[2]);
            y[1] = BgrToY<YuvType>(bgr[3], bgr[4], bgr[5]);

            int blue = Average(bgr[0], bgr[3]);
            int green = Average(bgr[1], bgr[4]);
            int red = Average(bgr[2], bgr[5]);

            u[0] = BgrToU<YuvType>(blue, green, red);
            v[0] = BgrToV<YuvType>(blue, green, red);
        }

        template <class YuvType> void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
                    BgrToYuv422pV2<YuvType>(bgr + colBgr, y + colY, u + colUV, v + colUV);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv422pV2<Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv422pV2<Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv422pV2<Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv422pV2<Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class YuvType> SIMD_INLINE void BgrToYuv444pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            const int blue = bgr[0], green = bgr[1], red = bgr[2];
            y[0] = BgrToY<YuvType>(blue, green, red);
            u[0] = BgrToU<YuvType>(blue, green, red);
            v[0] = BgrToV<YuvType>(blue, green, red);
        }

        template <class YuvType> void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t col = 0; col < width; col++)
                    BgrToYuv444pV2<YuvType>(bgr + col * 3, y + col, u + col, v + col);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv444pV2<Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv444pV2<Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv444pV2<Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv444pV2<Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class YuvType> SIMD_INLINE void BgrToYuv422Packed(const uint8_t* bgr, uint8_t* yuv, SimdRgbOrderType rgbOrder, SimdYuv422PackPatternType yuvPattern)
        {
            int b0, g0, r0, b1, g1, r1;
            if (rgbOrder == SimdBgr)
            {
                b0 = bgr[0], g0 = bgr[1], r0 = bgr[2];
                b1 = bgr[3], g1 = bgr[4], r1 = bgr[5];
            }
            else
            {
                r0 = bgr[0], g0 = bgr[1], b0 = bgr[2];
                r1 = bgr[3], g1 = bgr[4], b1 = bgr[5];
            }

            int y0 = BgrToY<YuvType>(b0, g0, r0);
            int y1 = BgrToY<YuvType>(b1, g1, r1);

            int blue = Average(b0, b1);
            int green = Average(g0, g1);
            int red = Average(r0, r1);
            int u = BgrToU<YuvType>(blue, green, red);
            int v = BgrToV<YuvType>(blue, green, red);

            switch (yuvPattern)
            {
            case SimdYuyv422: yuv[0] = y0; yuv[1] = u;  yuv[2] = y1; yuv[3] = v;  break;
            case SimdYvyu422: yuv[0] = y0; yuv[1] = v;  yuv[2] = y1; yuv[3] = u;  break;
            case SimdUyvy422: yuv[0] = u;  yuv[1] = y0; yuv[2] = v;  yuv[3] = y1; break;
            case SimdVyuy422: yuv[0] = v;  yuv[1] = y0; yuv[2] = u;  yuv[3] = y1; break;
            default:
                assert(0);
            }
        }

        template <class YuvType> void BgrToYuv422Packed(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* yuv, size_t yuvStride, SimdRgbOrderType rgbOrder, SimdYuv422PackPatternType yuvPattern)
        {
            assert((width % 2 == 0) && (width >= 2));

            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t colBgr = 0, colYuv = 0; colBgr < width * 3; colBgr += 6, colYuv += 4)
                    BgrToYuv422Packed<YuvType>(bgr + colBgr, yuv + colYuv, rgbOrder, yuvPattern);
                yuv += yuvStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv422Packed(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* yuv, size_t yuvStride,
            SimdRgbOrderType rgbOrder, SimdYuvType yuvType, SimdYuv422PackPatternType yuvPattern)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv422Packed<Bt601>(bgr, bgrStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            case SimdYuvBt709: BgrToYuv422Packed<Bt709>(bgr, bgrStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            case SimdYuvBt2020: BgrToYuv422Packed<Bt2020>(bgr, bgrStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            case SimdYuvTrect871: BgrToYuv422Packed<Trect871>(bgr, bgrStride, width, height, yuv, yuvStride, rgbOrder, yuvPattern); break;
            default:
                assert(0);
            }
        }
    }
}
