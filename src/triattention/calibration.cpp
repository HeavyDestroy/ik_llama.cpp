// Offline calibration utilities for TriAttention stats
// Note: Primary calibration should be done in Python; this is a fallback

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"
#include <cmath>
#include <vector>

// Forward declarations from llama-hybrid.cpp
bool llama_layer_uses_attention(const llama_model & model, int layer_idx);
void compact_residual_buffer(struct llama_kv_cache * kv);
float residual_norm_in_band(const float * residual, int band_idx, int n_embd, int n_bands);

// Simplified online calibration (not recommended for production)
// Prefer offline Python calibration for accuracy
bool llama_triattention_calibrate_online(
    llama_kv_cache * kv,
    const llama_model & model,
    int n_samples = 1000)
{
#ifndef LLAMA_KV_DIRECT_TRIATTN
    return false;
#endif

    LLAMA_LOG_WARN("Online TriAttention calibration is approximate; prefer offline Python calibration\n");

    // Initialize stats for attention layers
    for (int il = 0; il < model.hparams.n_layer; ++il) {
        if (!llama_layer_uses_attention(model, il)) continue;

        llama_kv_cache::TriAttentionLayerStats stats;
        stats.n_freq_bands = 8;  // Default: 8 frequency bands
        stats.q_center_norm.assign(stats.n_freq_bands, 1.0f);
        stats.k_center_norm.assign(stats.n_freq_bands, 1.0f);
        stats.q_center_phase.assign(stats.n_freq_bands, 0.0f);
        stats.k_center_phase.assign(stats.n_freq_bands, 0.0f);
        stats.q_norm_mean.assign(stats.n_freq_bands, 1.0f);
        stats.k_norm_mean.assign(stats.n_freq_bands, 1.0f);
        stats.mrl.assign(stats.n_freq_bands, 0.8f);  // Default concentration
        stats.is_attention = true;

        kv->layer_stats[il] = stats;
    }

    LLAMA_LOG_INFO("Initialized approximate TriAttention stats for %d layers\n",
                  static_cast<int>(kv->layer_stats.size()));
    return true;
}

// Export current stats to simple binary format
bool llama_triattention_export_stats(
    const llama_kv_cache * kv,
    const std::string & output_path)
{
#ifndef LLAMA_KV_DIRECT_TRIATTN
    return false;
#endif

    FILE * f = fopen(output_path.c_str(), "wb");
    if (!f) return false;

    int n_layers = static_cast<int>(kv->layer_stats.size());
    fwrite(&n_layers, sizeof(int), 1, f);

    for (const auto & [layer_idx, stats] : kv->layer_stats) {
        fwrite(&layer_idx, sizeof(int), 1, f);
        fwrite(&stats.n_freq_bands, sizeof(int), 1, f);
        fwrite(stats.q_center_norm.data(), sizeof(float), stats.n_freq_bands, f);
        fwrite(stats.k_center_norm.data(), sizeof(float), stats.n_freq_bands, f);
        fwrite(stats.q_center_phase.data(), sizeof(float), stats.n_freq_bands, f);
        fwrite(stats.k_center_phase.data(), sizeof(float), stats.n_freq_bands, f);
        fwrite(stats.q_norm_mean.data(), sizeof(float), stats.n_freq_bands, f);
        fwrite(stats.k_norm_mean.data(), sizeof(float), stats.n_freq_bands, f);
        fwrite(stats.mrl.data(), sizeof(float), stats.n_freq_bands, f);
        fwrite(&stats.is_attention, sizeof(bool), 1, f);
    }

    fclose(f);
    LLAMA_LOG_INFO("Exported TriAttention stats to %s\n", output_path.c_str());
    return true;
}