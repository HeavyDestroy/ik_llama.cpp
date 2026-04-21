#include "llama.h"
#include "llama-impl.h"
#include "llama-context.h"
#include "llama-arch.h"
#include "llama-model.h"
#include <cmath>
#include <string>
#include <unordered_map>
#include <algorithm>

// Detect if layer uses standard attention (vs. Mamba/GDN/delta-net)
bool llama_layer_uses_attention(const llama_model & model, int layer_idx) {
    // Qwen3.5 hybrid pattern: attention every 4th layer
    if (model.arch == LLM_ARCH_QWEN35) {
        // Check if layer has attention weights (wk != nullptr means attention layer)
        return model.layers[layer_idx].wk != nullptr;
    }

    // Default: assume all layers use attention
    return true;
}

// Mark cells with correct layer type during prefill
void llama_kv_cache_mark_layer_types(
    llama_kv_cache * kv,
    const llama_model & model,
    uint32_t start_idx,
    uint32_t n_tokens)
{
#ifdef LLAMA_KV_DIRECT_TRIATTN
    for (uint32_t i = 0; i < n_tokens; ++i) {
        uint32_t cell_idx = start_idx + i;
        if (cell_idx >= kv->cells.size()) break;

        // Safety: ensure layer_idx is valid before using it
        int layer = kv->cells[cell_idx].layer_idx;
        if (layer < 0 || layer >= model.hparams.n_layer) {
            // Fallback: assume attention layer for safety
            kv->cells[cell_idx].is_attention_layer = true;
            continue;
        }

        // Auto-detect based on hybrid pattern
        bool is_attn = llama_layer_uses_attention(model, layer);
        kv->cells[cell_idx].is_attention_layer = is_attn;
    }
#endif
}

// Compute residual norm in frequency band (helper for TriAttention)
float residual_norm_in_band(const float * residual, int band_idx, int n_embd, int n_bands) {
    int band_start = (band_idx * n_embd) / n_bands;
    int band_end = ((band_idx + 1) * n_embd) / n_bands;
    float norm_sq = 0;
    for (int i = band_start; i < band_end; ++i) {
        norm_sq += residual[i] * residual[i];
    }
    return sqrtf(norm_sq / (band_end - band_start));
}

// Compact residual buffer after pruning
void compact_residual_buffer(llama_kv_cache * kv) {
#ifdef LLAMA_KV_DIRECT_TRIATTN
    if (!kv->enable_triattention) return;

    // First pass: collect retained residuals into temp storage
    std::vector<void *> retained_ptrs;
    std::vector<uint32_t> retained_indices;
    
    for (uint32_t i = 0; i < kv->cells.size(); ++i) {
        if (kv->residual_retained[i] && kv->cells[i].has_residual) {
            retained_ptrs.push_back(kv->cells[i].residual_ptr);
            retained_indices.push_back(i);
        }
    }
    
    // Second pass: move data to compact positions and update ALL cells
    for (uint32_t i = 0; i < kv->cells.size(); ++i) {
        // Clear residual from every cell first
        kv->cells[i].has_residual = false;
        kv->cells[i].residual_ptr = nullptr;
    }
    
    // Write compacted residuals
    for (size_t i = 0; i < retained_ptrs.size(); ++i) {
        uint32_t compact_idx = static_cast<uint32_t>(i);
        void * src = retained_ptrs[i];
        uint32_t orig_idx = retained_indices[i];
        
        void * dst = kv->residual_buffer.data() + compact_idx * kv->residual_stride;
        memcpy(dst, src, kv->residual_stride);
        
        // Update the ORIGINAL cell (not the compact index)
        kv->cells[orig_idx].residual_ptr = dst;
        kv->cells[orig_idx].has_residual = true;
        // tri_score, absolute_pos, layer_idx, is_attention_layer already correct
    }
    
    LLAMA_LOG_DEBUG("Compacted residual buffer: %zu cells retained\n", retained_ptrs.size());
#endif
}