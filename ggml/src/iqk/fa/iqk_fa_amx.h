//
// Copyright (C) 2024-2026 Iwan Kawrakow & HeavyDestroy
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

#if defined(__AVX512BF16__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512VBMI2__)

namespace {

// AMX Flash Attention for head size 128, 256
// Uses 16x16 BF16 tile registers (TMM0-TMM15)
// Processes Q in 16-token blocks, K/V in 16-token blocks

inline void amx_flash_attn_head(int head_dim, const float* q, const uint16_t* k, const uint16_t* v,
                                 const uint16_t* mask, int nq, int nk,
                                 float scale, float softcap, float* out, float* M, float* S) {
    // Initialize AMX tiles
    _mm512_tile_store_dps();
    
    // Process Q in blocks of 16
    for (int q_blk = 0; q_blk < nq; q_blk += 16) {
        int q_rem = std::min(16, nq - q_blk);
        
        // Initialize M and S for this Q block
        for (int i = 0; i < q_rem; i++) {
            M[q_blk + i] = -1e30f;
            S[q_blk + i] = 0.0f;
        }
        
        // Accumulator: head_dim x 16 floats for QKV output
        float qkv_acc[head_dim * 16] = {0};
        
        // Process K/V in blocks of 16
        for (int k_blk = 0; k_blk < nk; k_blk += 16) {
            int k_rem = std::min(16, nk - k_blk);
            
            if (k_rem == 0) break;
            
            // Load Q tile (head_dim x 16) -> needs head_dim/16 tiles
            int n_tiles = head_dim / 16;
            
            // Load Q into TMM0..TMM(n_tiles-1)
            for (int t = 0; t < n_tiles; t++) {
                const float* q_ptr = q + (q_blk + 0) * head_dim + t * 16;
                // Convert 16x16 floats to BF16 and load to tile
                __m128i bf16_vals[16];
                for (int r = 0; r < 16; r++) {
                    __m128 f32 = _mm_load_ps(q_ptr + r * head_dim);
                    bf16_vals[r] = _mm512_cvtps_pbh(f32);
                }
                _mm512_tiled_storea_epi16(&((uint16_t*)qkv_acc)[t * 256], bf16_vals);
            }
            
            // Load K tile (head_dim x 16) transposed -> TMM(n_tiles)..TMM(2*n_tiles-1)
            for (int t = 0; t < n_tiles; t++) {
                const uint16_t* k_ptr = k + (k_blk + 0) * head_dim + t * 16;
                __m256i bf16_vals[16];
                for (int r = 0; r < 16; r++) {
                    bf16_vals[r] = _mm256_load_si256((const __m256i*)(k_ptr + r * head_dim));
                }
                _mm512_tiled_storea_epi16(&((uint16_t*)qkv_acc)[t * 256 + 128], bf16_vals);
            }
            
            // Load tiles into AMX registers
            for (int t = 0; t < n_tiles; t++) {
                _mm512_tiled_loada_epi16(&((uint16_t*)qkv_acc)[t * 256]);
                _mm512_tiled_loada_epi16(&((uint16_t*)qkv_acc)[t * 256 + 128]);
            }
            
            // Q @ K^T -> result in TMM0..TMM(n_tiles-1)
            // Each tile t: TMM[t] = TMM[t] @ TMM[n_tiles + t]^T
            for (int t = 0; t < n_tiles; t++) {
                _mm512_tiled_muladd_bf16();
            }
            
            // Store QK^T result (16 x 16) from tiles to memory for softmax
            float qk[16 * 16];
            for (int t = 0; t < n_tiles; t++) {
                // This is simplified - full implementation needs proper tile extraction
            }
            
            // Softmax and accumulate V
            // TODO: Complete implementation
        }
    }
    
    _mm512_tile_release_dps();
}

} // namespace

#endif // AMX availability

#endif // IQK_FA_AMX_H
