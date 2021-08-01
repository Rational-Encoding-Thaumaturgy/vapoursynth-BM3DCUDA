// Implementation of BM3DCPU
// Copyright (c) 2003, 2007-14 Matteo Frigo
// Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
// Copyright (c) 2021 WolframRhodium
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

// Reference: 
// [1] K. Dabov, A. Foi, V. Katkovnik and K. Egiazarian, 
//     "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering," 
//     in IEEE Transactions on Image Processing, vol. 16, no. 8, pp. 2080-2095, 
//     Aug. 2007, doi: 10.1109/TIP.2007.901238.
// [2] K. Dabov, A. Foi and K. Egiazarian, 
//     "Video denoising by sparse 3D transform-domain collaborative filtering," 
//     proceedings of the 15th European Signal Processing Conference, 2007, pp. 145-149.

// function "dct" is modified from code generated by fftw-3.3.9
// WolframRhodium, 8 May 2021

// Wordings:
// The coordinate of a block is denoted by the coordinate of its top-left pixel.
//
// Algorithm details:
// 1. The DC element of the transform coefficients of 3D group is always untouched.
// 2. Coarse prefiltering and Kaiser window are not implemented.
// 3. `group_size` is fixed to 8.
// 4. Predictive search is only implemented for V-BM3D, and the spatial coordinates
//    of the previously found locations are restricted to the top `ps_num` coordinates.
//
// Implementation details:
// 1. The spectra of 3D group is computed online.
// 2. The DCT implementation uses a modified FFTW subroutine that is normalized
//    and scaled, i.e. each inverse results in the original array multiplied by N.

#pragma once

#include <array>
#include <limits>
#include <immintrin.h>

// shuffle_up({0, 1, ..., 7}) => {0, 0, 1, ..., 6}
static inline __m256i shuffle_up(__m256i x) noexcept { 
    __m256i pre_mask { _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6) };
    return _mm256_permutevar8x32_epi32(x, pre_mask);
}

// Reduction operation of YMM lanes
static inline __m256 reduce_add(__m256 x) noexcept {
    x = _mm256_add_ps(x, _mm256_permute_ps(x, 0b10110001));
    x = _mm256_add_ps(x, _mm256_permute_ps(x, 0b01001110));
    x = _mm256_add_ps(x, _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x), 0b01001110)));
    return x;
}

// Reduction operation of YMM lanes
static inline __m256i reduce_add(__m256i x) noexcept {
    x = _mm256_add_epi32(x, _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(x), 0b10110001)));
    x = _mm256_add_epi32(x, _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(x), 0b01001110)));
    x = _mm256_add_epi32(x, _mm256_permute4x64_epi64(x, 0b01001110));
    return x;
}

static inline void load_block(
    __m256 dst[8], const float * srcp, int stride
) noexcept {

    for (int i = 0; i < 8; ++i) {
        dst[i] = _mm256_loadu_ps(&srcp[i * stride]);
    }
}

// Returns the sum of square distance of input blocks
static inline __m256 compute_distance(
    const __m256 reference_block[8], const __m256 candidate_block[8]
) noexcept {

    // manual unroll
    __m256 errors[2] {};

    for (int i = 0; i < 8; ++i) {
        __m256 row_diff = _mm256_sub_ps(reference_block[i], candidate_block[i]);
        errors[i % 2] = _mm256_fmadd_ps(row_diff, row_diff, errors[i % 2]);
    }

    return reduce_add(_mm256_add_ps(errors[0], errors[1]));
}

// Given a `reference_block`, finds 8 most similar blocks
// whose coordinates are within a local neighborhood of (2 * `bm_range` + 1)^2 
// centered at coordinates (`x`, `y`) in an input plane denoted by 
// (`srcp`, `stride`, `width`, `height`), and updates the 
// matched coordinates and distances in (`index_x`, `index_y`) and `errors`.
static inline void block_matching(
    std::array<float, 8> & errors, 
    std::array<int, 8> & index_x, 
    std::array<int, 8> & index_y, 
    const __m256 reference_block[8], 
    const float * srcp, int stride, 
    int width, int height, 
    int bm_range, int x, int y
) noexcept {

    // helper data
    constexpr int blend[] = {
        0, 
        0, 0, 0, 0, 0, 0, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0 };
    __m256i shift_base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    // clamps candidate locations to be within the plane
    int left = std::max(x - bm_range, 0);
    int right = std::min(x + bm_range, width - 8);
    int top = std::max(y - bm_range, 0);
    int bottom = std::min(y + bm_range, height - 8);

    __m256 errors8 { _mm256_loadu_ps(errors.data()) };
    __m256i index8_x { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index_x.data())) };
    __m256i index8_y { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index_y.data())) };

    const float * srcp_row = &srcp[top * stride + left];
    for (int row = top; row <= bottom; ++row) {
        const float * srcp = srcp_row; // pointer to 2D neighborhoods
        for (int col = left; col <= right; ++col) {
            __m256 candidate_block[8];
            load_block(candidate_block, srcp, stride);

            __m256 error = compute_distance(reference_block, candidate_block);

            __m256 flag { _mm256_cmp_ps(error, errors8, _CMP_LT_OQ) };

            if (int imask = _mm256_movemask_ps(flag); imask) {
                __m256i shuffle_mask = _mm256_add_epi32(
                    shift_base, _mm256_castps_si256(flag));
                __m256 pre_error = _mm256_permutevar8x32_ps(
                    errors8, shuffle_mask);
                __m256i pre_index_x = _mm256_permutevar8x32_epi32(
                    index8_x, shuffle_mask);
                __m256i pre_index_y = _mm256_permutevar8x32_epi32(
                    index8_y, shuffle_mask);

                int count = _mm_popcnt_u32(static_cast<unsigned int>(imask));
                __m256 blend_mask = _mm256_castsi256_ps(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(&blend[count])));
                errors8 = _mm256_blendv_ps(
                    pre_error, error, blend_mask);
                index8_x = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(pre_index_x), 
                    _mm256_castsi256_ps(_mm256_set1_epi32(col)), 
                    blend_mask));
                index8_y = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(pre_index_y), 
                    _mm256_castsi256_ps(_mm256_set1_epi32(row)), 
                    blend_mask));
            }

            ++srcp;
        }

        srcp_row += stride;
    }

    _mm256_storeu_ps(errors.data(), errors8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index_x.data()), index8_x);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index_y.data()), index8_y);
}

// Similar to function `block_matching`, but with candidate locations 
// extended to other planes on the temporal axis 
// and using predictive search instead of exhaustive search.
static inline void block_matching_temporal(
    std::array<float, 8> & errors, 
    std::array<int, 8> & index_x, 
    std::array<int, 8> & index_y, 
    std::array<int, 8> & index_z, 
    const __m256 reference_block[8], 
    const float * __restrict global_srcps[/* 2 * radius + 1 */], 
    int stride, int width, int height, int bm_range, 
    int x, int y, int radius, int ps_num, int ps_range
) noexcept {

    // helper data
    constexpr int blend[] = {
        0, 
        0, 0, 0, 0, 0, 0, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0 };
    __m256i shift_base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    int center = radius;

    block_matching(
        errors, index_x, index_y, 
        reference_block, 
        global_srcps[center], stride, 
        width, height, 
        bm_range, x, y);

    index_z.fill(center);

    __m256 errors8 { _mm256_loadu_ps(errors.data()) };
    __m256i index8_x { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index_x.data())) };
    __m256i index8_y { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index_y.data())) };
    __m256i index8_z { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index_z.data())) };

    std::array<int, 8> center_index8_x { index_x };
    std::array<int, 8> center_index8_y { index_y };

    for (int direction = -1; direction <= 1; direction += 2) {
        std::array<int, 8> last_index8_x { center_index8_x };
        std::array<int, 8> last_index8_y { center_index8_y };
        for (int t = 1; t <= radius; ++t) {
            int z = center + direction * t;

            std::array<float, 8> frame_errors8;
            frame_errors8.fill(std::numeric_limits<float>::max());
            std::array<int, 8> frame_index8_x;
            std::array<int, 8> frame_index8_y;
            for (int i = 0; i < ps_num; ++i) {
                block_matching(
                    frame_errors8, frame_index8_x, frame_index8_y, 
                    reference_block, 
                    global_srcps[z], stride, 
                    width, height, 
                    ps_range, last_index8_x[i], last_index8_y[i]);
            }
            for (int i = 0; i < ps_num; ++i) {
                __m256 error = _mm256_set1_ps(frame_errors8[i]);

                __m256 flag { _mm256_cmp_ps(error, errors8, _CMP_LT_OQ) };

                if (int imask = _mm256_movemask_ps(flag); imask) {
                    __m256i shuffle_mask = _mm256_add_epi32(
                        shift_base, _mm256_castps_si256(flag));
                    __m256 pre_error = _mm256_permutevar8x32_ps(
                        errors8, shuffle_mask);
                    __m256i pre_index_x = _mm256_permutevar8x32_epi32(
                        index8_x, shuffle_mask);
                    __m256i pre_index_y = _mm256_permutevar8x32_epi32(
                        index8_y, shuffle_mask);
                    __m256i pre_index_z = _mm256_permutevar8x32_epi32(
                        index8_z, shuffle_mask);

                    int count = _mm_popcnt_u32(static_cast<unsigned int>(imask));
                    __m256 blend_mask = _mm256_castsi256_ps(_mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(&blend[count])));
                    errors8 = _mm256_blendv_ps(
                        pre_error, error, blend_mask);
                    index8_x = _mm256_castps_si256(_mm256_blendv_ps(
                        _mm256_castsi256_ps(pre_index_x), 
                        _mm256_castsi256_ps(_mm256_set1_epi32(frame_index8_x[i])), 
                        blend_mask));
                    index8_y = _mm256_castps_si256(_mm256_blendv_ps(
                        _mm256_castsi256_ps(pre_index_y), 
                        _mm256_castsi256_ps(_mm256_set1_epi32(frame_index8_y[i])), 
                        blend_mask));
                    index8_z = _mm256_castps_si256(_mm256_blendv_ps(
                        _mm256_castsi256_ps(pre_index_z), 
                        _mm256_castsi256_ps(_mm256_set1_epi32(z)), 
                        blend_mask));
                }
            }

            last_index8_x = frame_index8_x;
            last_index8_y = frame_index8_y;
        }
    }

    _mm256_storeu_ps(errors.data(), errors8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index_x.data()), index8_x);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index_y.data()), index8_y);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index_z.data()), index8_z);
}

// Set the first element in the arrays of coordinates to be (`x`, `y`)
// if the coordinate is not in the array
static inline void insert_if_not_in(
    std::array<int, 8> &index8_x_data,
    std::array<int, 8> &index8_y_data,
    int x, int y
) noexcept {

    const __m256i first_mask { _mm256_setr_epi32(0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0) };

    __m256i index8_x { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index8_x_data.data())) };
    __m256i index8_y { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index8_y_data.data())) };

    __m256i current_index_x { _mm256_set1_epi32(x) };
    __m256i current_index_y { _mm256_set1_epi32(y) };
    __m256i flag { 
        _mm256_and_si256(
            _mm256_cmpeq_epi32(index8_x, current_index_x), 
            _mm256_cmpeq_epi32(index8_y, current_index_y)) 
    };

    if (!_mm256_movemask_ps(_mm256_castsi256_ps(flag))) {
        __m256i pre_index_x { shuffle_up(index8_x) };
        __m256i pre_index_y { shuffle_up(index8_y) };
        index8_x = _mm256_blendv_epi8(pre_index_x, current_index_x, first_mask);
        index8_y = _mm256_blendv_epi8(pre_index_y, current_index_y, first_mask);
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index8_x_data.data()), index8_x);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index8_y_data.data()), index8_y);
}

// Temporal version of function `insert_if_not_in`
static inline void insert_if_not_in_temporal(
    std::array<int, 8> &index8_x_data, 
    std::array<int, 8> &index8_y_data, 
    std::array<int, 8> &index8_z_data, 
    int x, int y, int z
) noexcept {

    const __m256i first_mask { _mm256_setr_epi32(0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0) };

    __m256i index8_x { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index8_x_data.data())) };
    __m256i index8_y { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index8_y_data.data())) };
    __m256i index8_z { _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index8_z_data.data())) };

    __m256i current_index_x { _mm256_set1_epi32(x) };
    __m256i current_index_y { _mm256_set1_epi32(y) };
    __m256i current_index_z { _mm256_set1_epi32(z) };
    __m256i flag { 
        _mm256_and_si256(_mm256_and_si256(
            _mm256_cmpeq_epi32(index8_x, current_index_x), 
            _mm256_cmpeq_epi32(index8_y, current_index_y)), 
            _mm256_cmpeq_epi32(index8_z, current_index_z))
    };

    if (!_mm256_movemask_ps(_mm256_castsi256_ps(flag))) {
        __m256i pre_index_x { shuffle_up(index8_x) };
        __m256i pre_index_y { shuffle_up(index8_y) };
        __m256i pre_index_z { shuffle_up(index8_z) };
        index8_x = _mm256_blendv_epi8(pre_index_x, current_index_x, first_mask);
        index8_y = _mm256_blendv_epi8(pre_index_y, current_index_y, first_mask);
        index8_z = _mm256_blendv_epi8(pre_index_z, current_index_z, first_mask);
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index8_x_data.data()), index8_x);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index8_y_data.data()), index8_y);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(index8_z_data.data()), index8_z);
}

static inline void load_3d_group(
    __m256 dst[64], const float * __restrict srcp, int stride, 
    const std::array<int, 8> &index_x, const std::array<int, 8> &index_y
) noexcept {

    for (int i = 0; i < 8; ++i) {
        int x { index_x[i] };
        int y { index_y[i] };

        load_block(&dst[i * 8], &srcp[y * stride + x], stride);
    }
}

// Temporal version of function `load_3d_group`
static inline void load_3d_group_temporal(__m256 dst[64], 
    const float * __restrict srcps[/* 2 * radius + 1 */], int stride, 
    const std::array<int, 8> &index_x, 
    const std::array<int, 8> &index_y, 
    const std::array<int, 8> &index_z
) noexcept {

    for (int i = 0; i < 8; ++i) {
        int x { index_x[i] };
        int y { index_y[i] };
        int z { index_z[i] };

        load_block(&dst[i * 8], &srcps[z][y * stride + x], stride);
    }
}

// FFTW-style 1D transform
template <auto transform_impl, int stride=1, int howmany=8, int howmany_stride=8>
static inline void transform_pack8(__m256 data[64]) noexcept {
    for (int iter = 0; iter < howmany; ++iter, data += howmany_stride) {
        __m256 v[8];

        for (int i = 0; i < 8; ++i) {
            v[i] = data[i * stride];
        }

        transform_impl(v);

        for (int i = 0; i < 8; ++i) {
            data[i * stride] = v[i];
        }
    }
}

// (normalized, scaled, in-place) DCT-II/DCT-III
// Modified from fftw-3.3.9 generated code:
// fftw-3.3.9/rdft/scalar/r2r/e10_8.c and e01_8.c
template <bool forward>
static inline void dct(__m256 block[8]) noexcept {
    if constexpr (forward) {
        __m256 KP414213562 { _mm256_set1_ps(+0.414213562373095048801688724209698078569671875) };
        __m256 KP1_847759065 { _mm256_set1_ps(+1.847759065022573512256366378793576573644833252) };
        __m256 KP198912367 { _mm256_set1_ps(+0.198912367379658006911597622644676228597850501) };
        __m256 KP1_961570560 { _mm256_set1_ps(1.961570560806460898252364472268478073947867462) };
        __m256 KP1_414213562 { _mm256_set1_ps(+1.414213562373095048801688724209698078569671875) };
        __m256 KP668178637 { _mm256_set1_ps(+0.668178637919298919997757686523080761552472251) };
        __m256 KP1_662939224 { _mm256_set1_ps(+1.662939224605090474157576755235811513477121624) };
        __m256 KP707106781 { _mm256_set1_ps(+0.707106781186547524400844362104849039284835938) };
        __m256 neg_mask { _mm256_set1_ps(-0.0f) };

        auto T1 = block[0];
        auto T2 = block[7];
        auto T3 = _mm256_sub_ps(T1, T2);
        auto Tj = _mm256_add_ps(T1, T2);
        auto Tc = block[4];
        auto Td = block[3];
        auto Te = _mm256_sub_ps(Tc, Td);
        auto Tk = _mm256_add_ps(Tc, Td);
        auto T4 = block[2];
        auto T5 = block[5];
        auto T6 = _mm256_sub_ps(T4, T5);
        auto T7 = block[1];
        auto T8 = block[6];
        auto T9 = _mm256_sub_ps(T7, T8);
        auto Ta = _mm256_add_ps(T6, T9);
        auto Tn = _mm256_add_ps(T7, T8);
        auto Tf = _mm256_sub_ps(T6, T9);
        auto Tm = _mm256_add_ps(T4, T5);
        auto Tb = _mm256_fnmadd_ps(KP707106781, Ta, T3);
        auto Tg = _mm256_fnmadd_ps(KP707106781, Tf, Te);
        block[3] = _mm256_mul_ps(KP1_662939224, _mm256_fmadd_ps(KP668178637, Tg, Tb));
        block[5] = _mm256_xor_ps(neg_mask, _mm256_mul_ps(KP1_662939224, _mm256_fnmadd_ps(KP668178637, Tb, Tg)));
        auto Tp = _mm256_add_ps(Tj, Tk);
        auto Tq = _mm256_add_ps(Tm, Tn);
        block[4] = _mm256_mul_ps(KP1_414213562, _mm256_sub_ps(Tp, Tq));
        block[0] = _mm256_mul_ps(KP1_414213562, _mm256_add_ps(Tp, Tq));
        auto Th = _mm256_fmadd_ps(KP707106781, Ta, T3);
        auto Ti = _mm256_fmadd_ps(KP707106781, Tf, Te);
        block[1] = _mm256_mul_ps(KP1_961570560, _mm256_fnmadd_ps(KP198912367, Ti, Th));
        block[7] = _mm256_mul_ps(KP1_961570560, _mm256_fmadd_ps(KP198912367, Th, Ti));
        auto Tl = _mm256_sub_ps(Tj, Tk);
        auto To = _mm256_sub_ps(Tm, Tn);
        block[2] = _mm256_mul_ps(KP1_847759065, _mm256_fnmadd_ps(KP414213562, To, Tl));
        block[6] = _mm256_mul_ps(KP1_847759065, _mm256_fmadd_ps(KP414213562, Tl, To));
    } else {
        __m256 KP1_662939224 { _mm256_set1_ps(+1.662939224605090474157576755235811513477121624) };
        __m256 KP668178637 { _mm256_set1_ps(+0.668178637919298919997757686523080761552472251) };
        __m256 KP1_961570560 { _mm256_set1_ps(+1.961570560806460898252364472268478073947867462) };
        __m256 KP198912367 { _mm256_set1_ps(+0.198912367379658006911597622644676228597850501) };
        __m256 KP1_847759065 { _mm256_set1_ps(+1.847759065022573512256366378793576573644833252) };
        __m256 KP707106781 { _mm256_set1_ps(+0.707106781186547524400844362104849039284835938) };
        __m256 KP414213562 { _mm256_set1_ps(+0.414213562373095048801688724209698078569671875) };
        __m256 KP1_414213562 { _mm256_set1_ps(+1.414213562373095048801688724209698078569671875) };

        auto T1 = _mm256_mul_ps(KP1_414213562, block[0]);
        auto T2 = block[4];
        auto T3 = _mm256_fmadd_ps(KP1_414213562, T2, T1);
        auto Tj = _mm256_fnmadd_ps(KP1_414213562, T2, T1);
        auto T4 = block[2];
        auto T5 = block[6];
        auto T6 = _mm256_fmadd_ps(KP414213562, T5, T4);
        auto Tk = _mm256_fmsub_ps(KP414213562, T4, T5);
        auto T8 = block[1];
        auto Td = block[7];
        auto T9 = block[5];
        auto Ta = block[3];
        auto Tb = _mm256_add_ps(T9, Ta);
        auto Te = _mm256_sub_ps(Ta, T9);
        auto Tc = _mm256_fmadd_ps(KP707106781, Tb, T8);
        auto Tn = _mm256_fnmadd_ps(KP707106781, Te, Td);
        auto Tf = _mm256_fmadd_ps(KP707106781, Te, Td);
        auto Tm = _mm256_fnmadd_ps(KP707106781, Tb, T8);
        auto T7 = _mm256_fmadd_ps(KP1_847759065, T6, T3);
        auto Tg = _mm256_fmadd_ps(KP198912367, Tf, Tc);
        block[7] = _mm256_fnmadd_ps(KP1_961570560, Tg, T7);
        block[0] = _mm256_fmadd_ps(KP1_961570560, Tg, T7);
        auto Tp = _mm256_fnmadd_ps(KP1_847759065, Tk, Tj);
        auto Tq = _mm256_fmadd_ps(KP668178637, Tm, Tn);
        block[5] = _mm256_fnmadd_ps(KP1_662939224, Tq, Tp);
        block[2] = _mm256_fmadd_ps(KP1_662939224, Tq, Tp);
        auto Th = _mm256_fnmadd_ps(KP1_847759065, T6, T3);
        auto Ti = _mm256_fnmadd_ps(KP198912367, Tc, Tf);
        block[3] = _mm256_fnmadd_ps(KP1_961570560, Ti, Th);
        block[4] = _mm256_fmadd_ps(KP1_961570560, Ti, Th);
        auto Tl = _mm256_fmadd_ps(KP1_847759065, Tk, Tj);
        auto To = _mm256_fnmadd_ps(KP668178637, Tn, Tm);
        block[6] = _mm256_fnmadd_ps(KP1_662939224, To, Tl);
        block[1] = _mm256_fmadd_ps(KP1_662939224, To, Tl);
    }
}

// Transposition of a 8x8 block.
static inline void transpose(__m256 block[8]) noexcept {
    for (int i = 0; i < 4; ++i) {
        __m256 temp1 = _mm256_shuffle_ps(block[i * 2], block[i * 2 + 1], 0b10001000);
        __m256 temp2 = _mm256_shuffle_ps(block[i * 2], block[i * 2 + 1], 0b11011101);
        block[i * 2] = temp1;
        block[i * 2 + 1] = temp2;
    }

    for (int i = 0; i < 4; ++i) {
        __m256 temp1 = _mm256_shuffle_ps(block[i + (i & -2)], block[i + (i & -2) + 2], 0b10001000);
        __m256 temp2 = _mm256_shuffle_ps(block[i + (i & -2)], block[i + (i & -2) + 2], 0b11011101);
        block[i + (i & -2)] = temp1;
        block[i + (i & -2) + 2] = temp2;
    }

    for (int i = 0; i < 4; ++i) {
        __m256 temp1 = _mm256_permute2f128_ps(block[i], block[i + 4], 0b00100000);
        __m256 temp2 = _mm256_permute2f128_ps(block[i], block[i + 4], 0b00110001);
        block[i] = temp1;
        block[i + 4] = temp2;
    }
}

static inline __m256 hard_thresholding(__m256 data[64], float _sigma) noexcept {
    // number of retained (non-zero) coefficients
    __m256i nnz {};

    __m256 sigma = _mm256_set1_ps(_sigma);

    __m256 thr_mask = _mm256_setr_ps(0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f);

    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFFu));
    __m256 scaler = _mm256_set1_ps(1.f / 4096.f);

    for (int i = 0; i < 64; ++i) {
        auto val = data[i];

        __m256 thr;
        if (i == 0) {
            // protects DC component
            thr = _mm256_mul_ps(sigma, thr_mask); 
        } else {
            thr = sigma;
        }

        __m256 _flag = _mm256_cmp_ps(_mm256_and_ps(val, abs_mask), thr, _CMP_GE_OQ);
        __m256i flag = _mm256_castps_si256(_flag);

        nnz = _mm256_sub_epi32(nnz, flag);
        data[i] = _mm256_and_ps(_mm256_mul_ps(val, scaler), _flag);
    }

    nnz = reduce_add(nnz);

    return _mm256_rcp_ps(_mm256_cvtepi32_ps(nnz));
}

static inline __m256 collaborative_hard(__m256 data[64], float _sigma) noexcept {
    constexpr int stride1 = 1;
    constexpr int stride2 = stride1 * 8;

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8<dct<true>, stride1, 8, stride2>(data);
        transform_pack8<transpose, stride1, 8, stride2>(data);
    }
    transform_pack8<dct<true>, stride2, 8, stride1>(data);

    __m256 adaptive_weight = hard_thresholding(data, _sigma);

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8<dct<false>, stride1, 8, stride2>(data);
        transform_pack8<transpose, stride1, 8, stride2>(data);
    }
    transform_pack8<dct<false>, stride2, 8, stride1>(data);

    return adaptive_weight;
}

static inline __m256 wiener_filtering(__m256 data[64], __m256 ref[64], float _sigma) noexcept {
    __m256 norm {};
    __m256 sigma = _mm256_set1_ps(_sigma);
    __m256 sqr_sigma = _mm256_mul_ps(sigma, sigma);

    __m256 scaler = _mm256_set1_ps(1.f / 4096.f);

    for (int i = 0; i < 64; ++i) {
        auto val = data[i];
        auto ref_val = ref[i];
        auto sqr_ref = _mm256_mul_ps(ref_val, ref_val);
        auto coeff = _mm256_mul_ps(sqr_ref, _mm256_rcp_ps(_mm256_add_ps(sqr_ref, sqr_sigma)));

        if (i == 0) {
            // protects DC component
            __m256 ones = _mm256_set1_ps(1.f);
            coeff = _mm256_blend_ps(coeff, ones, 0b00000001);
        }

        norm = _mm256_fmadd_ps(coeff, coeff, norm);
        data[i] = _mm256_mul_ps(_mm256_mul_ps(val, scaler), coeff);
    }

    norm = reduce_add(norm);

    return _mm256_rcp_ps(norm);
}

static inline __m256 collaborative_wiener(__m256 data[64], __m256 ref[64], float _sigma) {
    constexpr int stride1 = 1;
    constexpr int stride2 = stride1 * 8;

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8<dct<true>, stride1, 8, stride2>(data);
        transform_pack8<transpose, stride1, 8, stride2>(data);
    }
    transform_pack8<dct<true>, stride2, 8, stride1>(data);

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8<dct<true>, stride1, 8, stride2>(ref);
        transform_pack8<transpose, stride1, 8, stride2>(ref);
    }
    transform_pack8<dct<true>, stride2, 8, stride1>(ref);

    __m256 adaptive_weight = wiener_filtering(data, ref, _sigma);

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8<dct<false>, stride1, 8, stride2>(data);
        transform_pack8<transpose, stride1, 8, stride2>(data);
    }
    transform_pack8<dct<false>, stride2, 8, stride1>(data);

    return adaptive_weight;
}

// Accumulate block-wise estimates and the corresponding weights in buffers.
// The Kaiser window weighting is not implemented.
static inline void local_accumulation(
    float * __restrict wdstp, 
    float * __restrict weightp, 
    int stride, 
    const __m256 denoising_group[64], 
    const std::array<int, 8> &index_x, 
    const std::array<int, 8> &index_y, 
    __m256 adaptive_weight
) noexcept {

    for (int i = 0; i < 8; ++i) {
        int x { index_x[i] };
        int y { index_y[i] };

        float * block_wdstp = &wdstp[y * stride + x];
        float * block_weightp = &weightp[y * stride + x];

        for (int j = 0; j < 8; ++j) {
            __m256 wdst = _mm256_loadu_ps(&block_wdstp[j * stride]);
            wdst = _mm256_fmadd_ps(adaptive_weight, denoising_group[i * 8 + j], wdst);
            _mm256_storeu_ps(&block_wdstp[j * stride], wdst);

            __m256 weight = _mm256_loadu_ps(&block_weightp[j * stride]);
            weight = _mm256_add_ps(weight, adaptive_weight);
            _mm256_storeu_ps(&block_weightp[j * stride], weight);
        }
    }
}

// Accumulates block-wise estimates and the corresponding weights in buffers.
// The Kaiser window weighting is not implemented.
static inline void local_accumulation_temporal(
    float * __restrict wdstp, 
    float * __restrict weightp, 
    int stride, 
    const __m256 denoising_group[64], 
    const std::array<int, 8> &index_x, 
    const std::array<int, 8> &index_y, 
    const std::array<int, 8> &index_z, 
    __m256 adaptive_weight, 
    int height
) noexcept {

    for (int i = 0; i < 8; ++i) {
        int x { index_x[i] };
        int y { index_y[i] };
        int z { index_z[i] };

        float * block_wdstp = &wdstp[z * height * stride * 2 + y * stride + x];
        float * block_weightp = &weightp[z * height * stride * 2 + y * stride + x];

        for (int j = 0; j < 8; ++j) {
            __m256 wdst = _mm256_loadu_ps(&block_wdstp[j * stride]);
            wdst = _mm256_fmadd_ps(adaptive_weight, denoising_group[i * 8 + j], wdst);
            _mm256_storeu_ps(&block_wdstp[j * stride], wdst);

            __m256 weight = _mm256_loadu_ps(&block_weightp[j * stride]);
            weight = _mm256_add_ps(weight, adaptive_weight);
            _mm256_storeu_ps(&block_weightp[j * stride], weight);
        }
    }
}

// Realize the aggregation by element-wise division.
static inline void aggregation(
    float * __restrict dstp, int stride, 
    const float * __restrict wdstp, 
    const float * __restrict weightp, 
    int width, int height
) noexcept {

    for (int row_i = 0; row_i < height; ++row_i) {
        for (int col_i = 0; col_i < width; col_i += 8) {
            __m256 wdst = _mm256_load_ps(&wdstp[col_i]);
            __m256 weight = _mm256_load_ps(&weightp[col_i]);
            __m256 dst = _mm256_mul_ps(wdst, _mm256_rcp_ps(weight));
            _mm256_stream_ps(&dstp[col_i], dst);
        }

        dstp += stride;
        wdstp += stride;
        weightp += stride;
    }
}

// Returns number of planes of data processed by a call 
// to the processing kernel `bm3d`
static constexpr int num_planes(bool chroma) noexcept {
    return chroma ? 3 : 1;
}

// Core implementation of the (V-)BM3D denoising algorithm. 
// For V-BM3D, the accumulation of values from neighborhood frames and 
// the aggregation step are not performed here 
// and is left for `bm3d.VAggregate()`.
template <bool temporal, bool chroma, bool final_>
static inline void bm3d(
    std::array<float * __restrict, num_planes(chroma)> &dstps, 
    int stride, 
    const float * __restrict srcps[/* num_planes(chroma) * (2 * radius + 1) */], 
    std::conditional_t<
        final_, 
        const float * __restrict [/* num_planes(chroma) * (2 * radius + 1) */], 
        std::nullptr_t> refps, 
    int width, int height, 
    const std::array<float, num_planes(chroma)> &sigma,
    int block_step, int bm_range, int radius, int ps_num, int ps_range, 
    std::conditional_t<temporal, std::nullptr_t, float * __restrict> buffer
) noexcept {

    const int temporal_width = 2 * radius + 1;
    const int center = radius;

    for (int _y = 0; _y < height - 8 + block_step; _y += block_step) {
        int y = std::min(_y, height - 8); // clamp

        for (int _x = 0; _x < width - 8 + block_step; _x += block_step) {
            int x = std::min(_x, width - 8); // clamp

            __m256 reference_block[8];
            if constexpr (final_) {
                load_block(reference_block, &refps[center][y * stride + x], stride);
            } else {
                load_block(reference_block, &srcps[center][y * stride + x], stride);
            }

            std::array<float, 8> errors;
            errors.fill(std::numeric_limits<float>::max());

            std::array<int, 8> index_x;
            std::array<int, 8> index_y;
            std::array<int, 8> index_z;

            if constexpr (temporal) {
                decltype(srcps) input;
                if constexpr (final_) {
                    input = refps;
                } else {
                    input = srcps;
                }

                block_matching_temporal(
                    errors, index_x, index_y, index_z, 
                    reference_block, 
                    input, stride, 
                    width, height, 
                    bm_range, x, y, radius, ps_num, ps_range
                );

                insert_if_not_in_temporal(index_x, index_y, index_z, x, y, center);
            } else {
                std::remove_reference_t<decltype(srcps[0])> input;
                if constexpr (final_) {
                    input = refps[0];
                } else {
                    input = srcps[0];
                }

                block_matching(
                    errors, index_x, index_y, 
                    reference_block, 
                    input, stride, 
                    width, height, 
                    bm_range, x, y
                );

                insert_if_not_in(index_x, index_y, x, y);
            }

            for (int plane = 0; plane < num_planes(chroma); ++plane) {
                if (chroma && sigma[plane] < std::numeric_limits<float>::epsilon()) {
                    continue;
                }

                __m256 denoising_group[64];
                if constexpr (temporal) {
                    load_3d_group_temporal(
                        denoising_group, &srcps[plane * temporal_width], 
                        stride, index_x, index_y, index_z);
                } else {
                    load_3d_group(
                        denoising_group, srcps[plane], stride, index_x, index_y);
                }

                __m256 adaptive_weight;
                if constexpr (final_) { // final estimation
                    __m256 basic_estimate_group[64];
                    if constexpr (temporal) {
                        load_3d_group_temporal(
                            basic_estimate_group, &refps[plane * temporal_width], 
                            stride, index_x, index_y, index_z);
                    } else {
                        load_3d_group(
                            basic_estimate_group, refps[plane], stride, index_x, index_y);
                    }
                    adaptive_weight = collaborative_wiener(
                        denoising_group, basic_estimate_group, sigma[plane]);
                } else { // basic estimation
                    adaptive_weight = collaborative_hard(
                        denoising_group, sigma[plane]);
                }

                if constexpr (temporal) {
                    local_accumulation_temporal(
                        &dstps[plane][0], 
                        &dstps[plane][height * stride], 
                        stride, denoising_group, 
                        index_x, index_y, index_z, 
                        adaptive_weight, 
                        height);
                } else {
                    local_accumulation(
                        &buffer[height * stride * 2 * plane], 
                        &buffer[height * stride * (2 * plane + 1)], 
                        stride, denoising_group, 
                        index_x, index_y, 
                        adaptive_weight);
                }
            }
        }
    }

    if constexpr (!temporal) {
        for (int plane = 0; plane < num_planes(chroma); ++plane) {
            if (!chroma || !(sigma[plane] < std::numeric_limits<float>::epsilon())) {
                aggregation(
                    dstps[plane], stride, 
                    &buffer[height * stride * 2 * plane], 
                    &buffer[height * stride * (2 * plane + 1)], 
                    width, height
                );
            }
        }
    }
}
