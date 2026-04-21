#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"
#include <cmath>
#include <algorithm>
#include <vector>

// Compute TriAttention score for a residual vector
float compute_triattention_score(
    const llama_kv_cache::TriAttentionLayerStats & stats,
    const float * residual_data,
    int n_embd,
    float alpha)
{
    float s_trig = 0.0f, s_norm = 0.0f;
    int bands = stats.n_freq_bands;

    for (int f = 0; f < bands; ++f) {
        // Sample residual norm in frequency band f
        float band_norm = residual_norm_in_band(residual_data, f, n_embd, bands);

        // Trig component: cos(ω·Δ + φ) with precomputed centers
        float phase_diff = stats.q_center_phase[f] - stats.k_center_phase[f];
        s_trig += stats.q_center_norm[f] * band_norm * cosf(phase_diff);

        // Norm component: weighted by concentration (1 - R)
        float weight = 1.0f - stats.mrl[f];
        s_norm += weight * stats.q_norm_mean[f] * band_norm;
    }

    return alpha * s_trig + (1.0f - alpha) * s_norm;
}

// Update TriAttention scores for all residuals in a layer
void update_triattention_scores(
    llama_kv_cache * kv,
    const llama_model * model,
    int layer_idx,
    uint32_t start_idx,
    uint32_t n_tokens)
{
#ifdef LLAMA_KV_DIRECT_TRIATTN
    if (!kv->enable_triattention) return;

    auto it = kv->layer_stats.find(layer_idx);
    if (it == kv->layer_stats.end() || !it->second.is_attention) return;

    const auto & stats = it->second;
    int n_embd = model->hparams.n_embd;

    for (uint32_t i = 0; i < n_tokens; ++i) {
        uint32_t cell_idx = start_idx + i;
        if (cell_idx >= kv->cells.size()) break;

        auto & cell = kv->cells[cell_idx];
        if (!cell.has_residual || !cell.is_attention_layer) continue;

        // Load residual data
        float * residual_data = reinterpret_cast<float *>(cell.residual_ptr);

        // Compute score
        cell.tri_score = compute_triattention_score(
            stats, residual_data, n_embd, kv->triattn_alpha);
    }
#endif
}

// Selective recompute: only recompute K/V for high-score residuals
bool should_recompute_from_residual(
    const llama_kv_cell & cell,
    float threshold)
{
    return cell.has_residual &&
           cell.is_attention_layer &&
           cell.tri_score >= threshold;
}

// Adaptive threshold tuning based on budget pressure
float adapt_tri_score_threshold(
    uint32_t current_budget,
    uint32_t target_budget,
    float current_threshold,
    float min_threshold,
    float max_threshold)
{
    if (current_budget <= target_budget) {
        // Relax threshold to keep more residuals
        return fmaxf(min_threshold, current_threshold - 0.05f);
    } else {
        // Tighten threshold to prune more aggressively
        return fminf(max_threshold, current_threshold + 0.05f);
    }
}

// Prune residuals by TriAttention score (ik_llama.cpp adaptation)
void llama_kv_cache_prune_residuals(struct llama_kv_cache * kv) {
    if (!kv->enable_triattention) return;

    std::vector<std::pair<uint32_t, float>> scored;
    for (uint32_t i = 0; i < kv->cells.size(); ++i) {
        auto & cell = kv->cells[i];
        if (!cell.has_residual || !cell.is_attention_layer) continue;
        if (kv->protect_prefill && cell.absolute_pos < kv->prefill_end_pos) continue;
        scored.emplace_back(i, cell.tri_score);
    }

    if (scored.empty()) return;

    // Retain top-B by score
    std::sort(scored.begin(), scored.end(),
              [](auto &a, auto &b) { return a.second > b.second; });

    kv->residual_retained.assign(kv->cells.size(), false);
    uint32_t keep = std::min(kv->triattn_budget, (uint32_t)scored.size());
    for (uint32_t i = 0; i < keep; ++i) {
        kv->residual_retained[scored[i].first] = true;
    }

    // Compact buffer: copy retained residuals to front, update pointers
    compact_residual_buffer(kv);

    LLAMA_LOG_DEBUG("Pruned residuals: %zu → %u (threshold: %.3f)\n",
                   scored.size(), keep,
                   keep > 0 ? scored[keep-1].second : 0.0f);
}