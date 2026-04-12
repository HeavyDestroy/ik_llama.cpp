//
// Copyright (C) 2024-2026 Iwan Kawrakow & HeavyDestroy
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>

#if defined(__AVX512BF16__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512VBMI2__)

namespace {

// AMX tile Flash Attention helper for head size 128
// Uses 16x16 BF16 tile registers (TMM0-TMM15)
// Q: 128 x nq, K: 128 x nk, V: 128 x nk
// Block size k_step = 128 or 256 or 512

inline void amx_flash_attn_128(const float* q, const uint16_t* k, const uint16_t* v, 
                               const uint16_t* mask, int nq, int nk, 
                               float scale, float* out, float* M, float* S) {
    // Initialize AMX tiles
    _mm512_tile_store_dps();
    
    // Process in blocks of 16 tokens for Q dimension
    for (int q_block = 0; q_block < nq; q_block += 16) {
        int q_remain = std::min(16, nq - q_block);
        
        // Initialize M and S for this Q block
        for (int i = 0; i < q_remain; i++) {
            M[q_block + i] = -1e30f;
            S[q_block + i] = 0.0f;
        }
        
        // Initialize output accumulator
        float qkv_acc[128 * 16] = {0};
        
        // Process K/V in blocks of 16
        for (int k_block = 0; k_block < nk; k_block += 16) {
            int k_remain = std::min(16, nk - k_block);
            
            if (k_remain == 0) break;
            
            // Load Q tile (128 x 16) from float to BF16
            // Q is stored as [q0, q1, ..., q15] where each qi is 128 floats
            _mm512_tiled_zeroa(); // Zero tiles
            
            // Load Q into TMM0-TMM7 (8 tiles for 128x16)
            // Each tile is 16x16, so we need 8 tiles for 128x16
            for (int i = 0; i < 8; i++) {
                const float* q_row = q + (q_block + 0) * 128 + i * 16;
                __m256i bf16_vals[16];
                for (int j = 0; j < 16; j++) {
                    __m128 f32_vals = _mm_load_ps(q_row + j * 128);
                    bf16_vals[j] = _mm512_cvtps_pbh(f32_vals);
                }
                _mm512_tiled_storea_epi16(&((uint16_t*)qkv_acc)[i * 256], bf16_vals);
            }
            
            // Load K tile (128 x 16) - transposed for GEMM
            // K is [k0, k1, ..., k15] where each ki is 128 uint16_t
            for (int i = 0; i < 8; i++) {
                const uint16_t* k_row = k + (k_block + 0) * 128 + i * 16;
                __m256i bf16_vals[16];
                for (int j = 0; j < 16; j++) {
                    bf16_vals[j] = _mm256_load_si256((const __m256i*)(k_row + j * 128));
                }
                _mm512_tiled_storea_epi16(&((uint16_t*)qkv_acc)[i * 256 + 128], bf16_vals);
            }
            
            // Q @ K^T -> 16x16 result in tiles
            _mm512_tiled_loada_epi16(&((uint16_t*)qkv_acc)[0]); // Q into TMM0-TMM7
            _mm512_tiled_loada_epi16(&((uint16_t*)qkv_acc)[128]); // K into TMM8-TMM15
            _mm512_tiled_muladd_bf16(); // TMM0 = TMM0 @ TMM8^T
            
            // Scale and softmax
            // Load result from TMM0 into registers
            // TODO: Complete the softmax and accumulation
        }
    }
    
    _mm512_tile_release_dps();
}

} // namespace

#endif // AMX availability

#endif // IQK_FA_AMX_H
