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

        // Auto-detect based on hybrid pattern
        bool is_attn = llama_layer_uses_attention(model, kv->cells[cell_idx].layer_idx);
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

    uint32_t write_idx = 0;
    for (uint32_t read_idx = 0; read_idx < kv->cells.size(); ++read_idx) {
        if (kv->residual_retained[read_idx] && kv->cells[read_idx].has_residual) {
            if (write_idx != read_idx) {
                // Move residual data
                void * src = kv->cells[read_idx].residual_ptr;
                void * dst = kv->residual_buffer.data() + write_idx * kv->residual_stride;
                memcpy(dst, src, kv->residual_stride);

                // Update cell metadata
                kv->cells[write_idx].residual_ptr = dst;
                kv->cells[write_idx].has_residual = true;
                kv->cells[write_idx].tri_score = kv->cells[read_idx].tri_score;
                kv->cells[write_idx].absolute_pos = kv->cells[read_idx].absolute_pos;
                kv->cells[write_idx].layer_idx = kv->cells[read_idx].layer_idx;
                kv->cells[write_idx].is_attention_layer = kv->cells[read_idx].is_attention_layer;
            }
            ++write_idx;
        } else {
            kv->cells[read_idx].has_residual = false;  // Mark evicted
        }
    }
    LLAMA_LOG_DEBUG("Compacted residual buffer: %zu cells retained\n", write_idx);
#endif
}