//
// Copyright (C) 2024-2026 Iwan Kawrakow & HeavyDestroy
// MIT license
// SPDX-License-Identifier: MIT
// AMX Flash Attention for Sapphire Rapids Xeon

#include "iqk/iqk_config.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include "iqk/fa/iqk_fa_templates.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>

// Check if AMX tile intrinsics are available
#if defined(__AVX512BF16__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512VBMI2__)

// Test if AMX tile intrinsics are available at compile time
#ifdef __GNUC__
#define HAS_AMX_TILES 0  // GCC 15 does not yet support AMX tile intrinsics
#else
#define HAS_AMX_TILES 1
#endif

#if HAS_AMX_TILES

namespace {

inline void amx_flash_attn(int head_dim, const float* q, const uint16_t* k, const uint16_t* v,
                           const uint16_t* mask, int nq, int nk,
                           float scale, float* out, float* M, float* S) {
    // Initialize AMX tile data processing state
    _mm512_tile_init_dps();
    
    int n_tiles = head_dim / 16;  // 8 for head_dim=128, 16 for head_dim=256
    
    // Temporary buffers (aligned for AMX)
    alignas(64) float qk[16*16];
    alignas(64) float softmax_out[16*16];
    alignas(64) uint16_t tile_buf[256];  // 16x16 BF16
    
    // Process Q in blocks of 16
    for (int q_blk = 0; q_blk < nq; q_blk += 16) {
        int q_rem = std::min(16, nq - q_blk);
        
        // Initialize M and S for this Q block
        for (int i = 0; i < q_rem; i++) {
            M[q_blk + i] = -1e30f;
            S[q_blk + i] = 0.0f;
        }
        
        // Accumulator for QKV output: head_dim x 16 floats
        float* qkv_acc = new float[head_dim * 16]();
        
        // Process K/V in blocks of 16
        for (int k_blk = 0; k_blk < nk; k_blk += 16) {
            int k_rem = std::min(16, nk - k_blk);
            
            if (k_rem == 0) break;
            
            // Step 1: Load Q into TMM0..TMM(n_tiles-1)
            for (int t = 0; t < n_tiles; t++) {
                const float* q_ptr = q + (q_blk + 0) * head_dim + t * 16;
                for (int r = 0; r < 16; r++) {
                    __m128 f32 = _mm_load_ps(q_ptr + r * head_dim);
                    __m256i bf16 = _mm512_cvtps_pbh(f32);
                    _mm256_store_si256((__m256i*)(tile_buf + r * 16), bf16);
                }
                _mm512_tiled_loada_epi16(tile_buf);
            }
            
            // Step 2: Load K (transposed) into TMM(n_tiles)..TMM(2*n_tiles-1)
            for (int t = 0; t < n_tiles; t++) {
                const uint16_t* k_ptr = k + (k_blk + 0) * head_dim + t * 16;
                for (int r = 0; r < 16; r++) {
                    __m256i bf16 = _mm256_load_si256((const __m256i*)(k_ptr + r * head_dim));
                    _mm256_store_si256((__m256i*)(tile_buf + r * 16), bf16);
                }
                _mm512_tiled_loada_epi16(tile_buf);
            }
            
            // Step 3: Q @ K^T -> result in TMM0..TMM(n_tiles-1)
            for (int t = 0; t < n_tiles; t++) {
                _mm512_tiled_muladd_bf16();
            }
            
            // Step 4: Store QK^T result to memory for softmax
            for (int t = 0; t < n_tiles; t++) {
                _mm512_tiled_storea_epi16(tile_buf);
                for (int r = 0; r < 16; r++) {
                    __m256i bf16 = _mm256_load_si256((const __m256i*)(tile_buf + r * 16));
                    __m512 f32 = _mm512_cvtph_ps(bf16);
                    _mm512_store_ps(qk + t * 16 * 16 + r * 16, f32);
                }
            }
            
            // Step 5: Softmax (row-wise)
            for (int i = 0; i < 16; i++) {
                float* row = qk + i * 16;
                float max_val = -1e30f;
                for (int j = 0; j < k_rem; j++) {
                    if (mask != nullptr && mask[(i + q_blk) * nk + (j + k_blk)] == 0) {
                        row[j] = -1e30f;
                    } else {
                        max_val = std::max(max_val, row[j]);
                    }
                }
                float sum = 0.0f;
                for (int j = 0; j < k_rem; j++) {
                    float val = expf((row[j] - max_val) * scale);
                    softmax_out[i * 16 + j] = val;
                    sum += val;
                }
                for (int j = 0; j < k_rem; j++) {
                    softmax_out[i * 16 + j] /= sum;
                }
            }
            
            // Step 6: Load softmax weights and V, accumulate QKV
            for (int t = 0; t < n_tiles; t++) {
                for (int r = 0; r < 16; r++) {
                    __m128 f32 = _mm_load_ps(softmax_out + t * 16 + r * 16);
                    __m256i bf16 = _mm512_cvtps_pbh(f32);
                    _mm256_store_si256((__m256i*)(tile_buf + r * 16), bf16);
                }
                _mm512_tiled_loada_epi16(tile_buf);
            }
            
            for (int t = 0; t < n_tiles; t++) {
                const uint16_t* v_ptr = v + (k_blk + 0) * head_dim + t * 16;
                for (int r = 0; r < 16; r++) {
                    __m256i bf16 = _mm256_load_si256((const __m256i*)(v_ptr + r * head_dim));
                    _mm256_store_si256((__m256i*)(tile_buf + r * 16), bf16);
                }
                _mm512_tiled_loada_epi16(tile_buf);
            }
            
            _mm512_tiled_zeroa();
            for (int t = 0; t < n_tiles; t++) {
                _mm512_tiled_muladd_bf16();
            }
            
            // Step 7: Store result and accumulate
            for (int t = 0; t < n_tiles; t++) {
                _mm512_tiled_storea_epi16(tile_buf);
                for (int r = 0; r < 16; r++) {
                    __m256i bf16 = _mm256_load_si256((const __m256i*)(tile_buf + r * 16));
                    __m512 f32 = _mm512_cvtph_ps(bf16);
                    const float* acc_ptr = qkv_acc + t * 16 * 16 + r * head_dim;
                    _mm512_storeu_ps((float*)(qkv_acc + (t * 16 + r) * head_dim),
                        _mm512_add_ps(f32, _mm512_load_ps(acc_ptr)));
                }
            }
        }
        
        // Store final QKV output
        for (int i = 0; i < q_rem; i++) {
            const float* acc_row = qkv_acc + i * head_dim;
            float* out_ptr = out + (q_blk + i) * head_dim;
            memcpy(out_ptr, acc_row, head_dim * sizeof(float));
        }
        
        delete[] qkv_acc;
    }
    
    _mm512_tile_init_dps();
}

} // namespace

#define IQK_FA_CASE(func_name) \
    bool func_name(int int_type_k, int int_type_v, int nq, int nk, \
                   int stride_q, int stride_k, int stride_v, \
                   int stride_m, int stride_qkv, \
                   const float * q, const void * k, const void * v, \
                   const void * mask, float scale, float softcap, \
                   float * qkv, const float * sinkf, float * M, float * S) {

IQK_FA_CASE(amx_fa_128_128) {
    amx_flash_attn(128, q, (const uint16_t*)k, (const uint16_t*)v, 
                   (const uint16_t*)mask, nq, nk, scale, qkv, M, S);
    return true;
}

IQK_FA_CASE(amx_fa_256_256) {
    amx_flash_attn(256, q, (const uint16_t*)k, (const uint16_t*)v,
                   (const uint16_t*)mask, nq, nk, scale, qkv, M, S);
    return true;
}

#endif // HAS_AMX_TILES

#endif // AMX intrinsics available

#endif // IQK_IMPLEMENT && GGML_IQK_FLASH_ATTENTION
