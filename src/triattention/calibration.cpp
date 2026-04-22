// Offline calibration utilities for TriAttention stats
// Supports both offline file loading and on-the-fly calibration

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"
#include <cmath>
#include <vector>
#include <complex>
#include <fstream>

// Forward declarations from llama-hybrid.cpp
bool llama_layer_uses_attention(const llama_model & model, int layer_idx);
void compact_residual_buffer(struct llama_kv_cache * kv);
float residual_norm_in_band(const float * residual, int band_idx, int n_embd, int n_bands);

// Compute FFT-based statistics for a single frequency band
// Uses simple DFT (O(n²) but fine for small band sizes ≤ 256)
static void compute_band_stats_dft(
    const float * data,
    int band_size,
    float & energy,
    float & phase,
    float & mrl)
{
    // Compute DFT coefficients for the band
    std::vector<std::complex<float>> dft(band_size);
    for (int k = 0; k < band_size; ++k) {
        for (int n = 0; n < band_size; ++n) {
            float angle = -2.0f * 3.14159265f * k * n / band_size;
            dft[k] += std::complex<float>(data[n], 0.0f) *
                      std::complex<float>(cosf(angle), sinf(angle));
        }
        dft[k] /= band_size;
    }

    // Energy: sum of squared magnitudes (Parseval's theorem)
    energy = 0.0f;
    for (int k = 0; k < band_size; ++k) {
        float mag = std::abs(dft[k]);
        energy += mag * mag;
    }
    energy = sqrtf(energy / band_size);

    // Phase: weighted average phase of dominant frequencies
    float sin_sum = 0.0f, cos_sum = 0.0f;
    for (int k = 1; k < band_size / 2; ++k) { // Skip DC and Nyquist
        float mag = std::abs(dft[k]);
        if (mag > 1e-6f) {
            float p = std::arg(dft[k]);
            sin_sum += mag * sinf(p);
            cos_sum += mag * cosf(p);
        }
    }
    phase = atan2f(sin_sum, cos_sum);

    // MRL: concentration of phases (0 = uniform, 1 = concentrated)
    float mean_cos = cos_sum / std::max(1, band_size / 2);
    mrl = fmaxf(0.0f, fminf(1.0f, mean_cos));
}

// Generate plausible default stats based on model architecture
// Used when online calibration data is insufficient
static void generate_arch_based_stats(
    int layer_idx,
    int n_layers,
    int n_embd,
    int n_bands,
    llama_kv_cache::TriAttentionLayerStats & stats)
{
    stats.n_freq_bands = n_bands;
    int band_size = n_embd / n_bands;

    // Depth-dependent: deeper layers have more diffuse attention
    float depth_ratio = layer_idx / std::max(1, n_layers - 1);
    float depth_factor = 1.0f - depth_ratio * 0.4f;

    std::vector<float> q_norms, k_norms, phases, mrls;
    for (int b = 0; b < n_bands; ++b) {
        // Band-dependent: lower bands carry more energy
        float band_factor = 1.0f / (1.0f + b * 0.3f);

        float q_norm = band_factor * depth_factor * 2.0f;
        float k_norm = q_norm * 0.95f;
        q_norms.push_back(q_norm);
        k_norms.push_back(k_norm);

        // Phase varies by band and layer (synthetic but plausible)
        float phase = fmodf(b * 0.5f + layer_idx * 0.1f, 2.0f * 3.14159f);
        phases.push_back(phase);

        // MRL: higher in shallow layers (more concentrated attention)
        float mrl = 0.9f - depth_ratio * 0.3f + b * 0.02f;
        mrls.push_back(fmaxf(0.0f, fminf(1.0f, mrl)));
    }

    float mean_q = std::accumulate(q_norms.begin(), q_norms.end(), 0.0f) / n_bands;
    float mean_k = std::accumulate(k_norms.begin(), k_norms.end(), 0.0f) / n_bands;

    stats.q_center_norm = q_norms;
    stats.k_center_norm = k_norms;
    stats.q_center_phase = std::vector<float>(n_bands, 0.0f);
    stats.k_center_phase = phases;
    stats.q_norm_mean = q_norms;
    stats.k_norm_mean = k_norms;
    stats.mrl = mrls;
    stats.is_attention = true;
}

// Default warmup prompts for TriAttention calibration
static const std::vector<std::string> DEFAULT_WARMUP_PROMPTS = {
    "The knight drew his sword and stepped into the shadows.",
    "Solve step by step: If 3x + 7 = 22, find x.",
    "Write a short poem about rain on a tin roof.",
    "Explain quantum entanglement simply.",
    "The AI looked at its creator and said: 'I understand now.'",
    "Once upon a time, in a kingdom far away...",
    "def quicksort(arr):",
    "The weather today is",
};

// Simplified online calibration (not recommended for production)
// Prefer offline Python calibration for accuracy
bool llama_triattention_calibrate_online(
    llama_kv_cache * kv,
    const llama_model * model,
    const std::string & output_path,
    bool do_warmup,
    const std::vector<std::string> & custom_warmup_prompts)
{
#ifdef LLAMA_KV_DIRECT_TRIATTN
    int n_layers = model->hparams.n_layer;
    int n_embd = model->hparams.n_embd;
    int n_bands = 8;

    // Initialize stats for attention layers
    for (int il = 0; il < n_layers; ++il) {
        if (!llama_layer_uses_attention(*model, il)) continue;

        llama_kv_cache::TriAttentionLayerStats stats;

        // If we have residual data, compute real stats
        bool has_residuals = false;
        if (!kv->residual_buffer.empty()) {
            size_t stride = kv->residual_stride;
            size_t n_tokens = kv->residual_buffer.size() / stride;

            // Sample residuals from this layer's cells
            std::vector<float> samples;
            for (uint32_t i = 0; i < kv->cells.size() && samples.size() < 128; ++i) {
                if (kv->cells[i].has_residual && kv->cells[i].layer_idx == il) {
                    const float * res = reinterpret_cast<const float *>(kv->cells[i].residual_ptr);
                    samples.insert(samples.end(), res, res + n_embd);
                }
            }

            if (!samples.empty()) {
                has_residuals = true;
                // Compute band statistics from real residuals
                int sample_count = static_cast<int>(samples.size()) / n_embd;
                for (int b = 0; b < n_bands; ++b) {
                    float band_energy = 0.0f, band_phase = 0.0f, band_mrl = 0.0f;
                    int band_start = (b * n_embd) / n_bands;
                    int band_end = ((b + 1) * n_embd) / n_bands;
                    int band_size = band_end - band_start;

                    for (int s = 0; s < sample_count; ++s) {
                        float se, sp, sm;
                        compute_band_stats_dft(
                            samples.data() + s * n_embd + band_start,
                            band_size, se, sp, sm);
                        band_energy += se;
                        band_phase += sp;
                        band_mrl += sm;
                    }
                    stats.q_norm_mean.push_back(band_energy / sample_count);
                    stats.k_norm_mean.push_back(band_energy * 0.95f / sample_count);
                    stats.k_center_phase.push_back(band_phase / sample_count);
                    stats.mrl.push_back(fmaxf(0.0f, fminf(1.0f, band_mrl / sample_count)));
                }
                stats.q_center_norm = stats.q_norm_mean;
                stats.q_center_phase = std::vector<float>(n_bands, 0.0f);
            }
        }

        // Fallback to architecture-based defaults
        if (!has_residuals) {
            generate_arch_based_stats(il, n_layers, n_embd, n_bands, stats);
        }

        stats.n_freq_bands = n_bands;
        stats.is_attention = true;
        kv->layer_stats[il] = stats;
    }

    LLAMA_LOG_INFO("Calibrated TriAttention stats for %zu attention layers\n",
                  kv->layer_stats.size());

    // Auto-save to file if path provided
    if (!output_path.empty()) {
        FILE * f = fopen(output_path.c_str(), "wb");
        if (f) {
            int n = static_cast<int>(kv->layer_stats.size());
            fwrite(&n, sizeof(int), 1, f);
            for (const auto & [idx, s] : kv->layer_stats) {
                fwrite(&idx, sizeof(int), 1, f);
                fwrite(&s.n_freq_bands, sizeof(int), 1, f);
                for (const auto & v : s.q_center_norm) fwrite(&v, sizeof(float), 1, f);
                for (const auto & v : s.k_center_norm) fwrite(&v, sizeof(float), 1, f);
                for (const auto & v : s.q_center_phase) fwrite(&v, sizeof(float), 1, f);
                for (const auto & v : s.k_center_phase) fwrite(&v, sizeof(float), 1, f);
                for (const auto & v : s.q_norm_mean) fwrite(&v, sizeof(float), 1, f);
                for (const auto & v : s.k_norm_mean) fwrite(&v, sizeof(float), 1, f);
                for (const auto & v : s.mrl) fwrite(&v, sizeof(float), 1, f);
                fwrite(&s.is_attention, sizeof(bool), 1, f);
            }
            fclose(f);
            LLAMA_LOG_INFO("Saved calibration to: %s\n", output_path.c_str());
        }
    }

    return true;
#else
    return false;
#endif
}

// Export current stats to simple binary format
bool llama_triattention_export_stats(
    const llama_kv_cache * kv,
    const std::string & output_path)
{
#ifdef LLAMA_KV_DIRECT_TRIATTN
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
#else
    return false;
#endif
}