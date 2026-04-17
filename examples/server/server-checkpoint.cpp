// ============================================================================
// Server Checkpoint Manager - Implementation
// ============================================================================

#include "server-checkpoint.h"
#include "server-context.h"
#include "log.h"
#include "ggml.h"  // for ggml_time_us()

// ============================================================================
// Lifecycle
// ============================================================================

void server_checkpoint_manager::init(llama_context* ctx, const server_checkpoint_config& cfg) {
    config = cfg;
    is_hybrid = detect_hybrid(ctx);
    checkpoints.clear();
    checkpoints.reserve(static_cast<size_t>(config.max_checkpoints));
    current_pos = 0;
    checkpoints_created = 0;
    checkpoints_restored = 0;
    checkpoints_evicted = 0;
    total_restore_time_ms = 0.0;

    if (is_hybrid) {
        LLAMA_LOG_INFO("[checkpoint] Hybrid model detected — using full state checkpoints (larger, but required for recurrent state)\n");
    } else {
        LLAMA_LOG_INFO("[checkpoint] Standard model — using partial state checkpoints (KV cache only)\n");
    }
}

void server_checkpoint_manager::reset() {
    checkpoints.clear();
    current_pos = 0;
}

// ============================================================================
// Checkpoint Creation
// ============================================================================

bool server_checkpoint_manager::create(server_slot& slot) {
    // Skip if no KV cache yet or image was just processed
    int32_t pos_min = llama_kv_cache_seq_pos_min(slot.ctx, slot.id);
    int32_t pos_max = llama_kv_cache_seq_pos_max(slot.ctx, slot.id);

    if (pos_min < 0 || pos_max < config.min_span) {
        return false;  // Not enough data yet
    }

    // Skip if we already have a checkpoint covering this position
    if (!checkpoints.empty() && pos_max <= checkpoints.back().pos_max) {
        return false;  // Redundant checkpoint
    }

    // Evict old checkpoints if at capacity
    evict_to_capacity();

    // Determine flags based on model type
    const llama_state_seq_flags flags = get_state_flags(is_hybrid);

    // Get size and allocate
    const size_t checkpoint_size = llama_state_seq_get_size(slot.ctx, slot.id, flags);
    if (checkpoint_size == 0) {
        return false;
    }

    const int64_t t_start = ggml_time_us();

    // Create checkpoint entry
    server_checkpoint cp;
    cp.pos_min = pos_min;
    cp.pos_max = pos_max;
    cp.pos_min_prompt = pos_min + slot.n_past_offset;
    cp.pos_max_prompt = pos_max + slot.n_past_offset;
    cp.data.resize(checkpoint_size);

    // Serialize state
    const size_t n = llama_state_seq_get_data(
        slot.ctx, cp.data.data(), checkpoint_size, slot.id, flags
    );

    if (n != checkpoint_size) {
        LLAMA_LOG_ERROR("[checkpoint] Failed to save checkpoint state (got %zu, expected %zu)\n",
            n, checkpoint_size);
        return false;
    }

    const double elapsed_ms = static_cast<double>(ggml_time_us() - t_start) / 1000.0;

    // Insert at end (most recent)
    checkpoints.push_back(std::move(cp));
    current_pos = pos_max;
    checkpoints_created++;

    LLAMA_LOG_DEBUG("[checkpoint] Created checkpoint %zu/%d: pos=[%d,%d] size=%.3fMB time=%.2fms hybrid=%d\n",
        checkpoints.size(), config.max_checkpoints,
        current_pos, pos_max, cp.data.size() / (1024.0 * 1024.0),
        elapsed_ms, is_hybrid ? 1 : 0);

    return true;
}

void server_checkpoint_manager::create_if_interval_elapsed(server_slot& slot, int32_t current_pos_new) {
    if (!config.enabled()) return;

    if (current_pos_new - current_pos >= config.interval) {
        create(slot);
    }
}

// ============================================================================
// Checkpoint Restoration
// ============================================================================

bool server_checkpoint_manager::restore(server_slot& slot, int32_t pos_min_thold) {
    if (checkpoints.empty()) {
        return false;
    }

    // Search backwards for the best checkpoint
    // We need: checkpoint.pos_min < pos_min_thold
    // This ensures at least one token is re-processed after restore
    auto it = std::find_if(
        checkpoints.rbegin(), checkpoints.rend(),
        [pos_min_thold](const server_checkpoint& cp) {
            return cp.pos_min < pos_min_thold;
        }
    );

    if (it == checkpoints.rend()) {
        // No suitable checkpoint — must re-process full prompt
        LLAMA_LOG_DEBUG("[checkpoint] No suitable checkpoint found (pos_min_thold=%d) — full re-process\n",
            pos_min_thold);
        return false;
    }

    return restore_from(slot, *it);
}

bool server_checkpoint_manager::restore_from(server_slot& slot, const server_checkpoint& cp) {
    const int64_t t_start = ggml_time_us();
    const size_t checkpoint_size = cp.data.size();
    const llama_state_seq_flags flags = get_state_flags(is_hybrid);

    const size_t n = llama_state_seq_set_data(
        slot.ctx, cp.data.data(), checkpoint_size, slot.id, flags
    );

    const double elapsed_ms = static_cast<double>(ggml_time_us() - t_start) / 1000.0;
    total_restore_time_ms += elapsed_ms;
    checkpoints_restored++;

    if (n != checkpoint_size) {
        LLAMA_LOG_ERROR("[checkpoint] Failed to restore checkpoint (got %zu, expected %zu) — forcing full re-process\n",
            n, checkpoint_size);
        return false;
    }

    // Update slot state to match checkpoint
    slot.n_past = std::min(slot.n_past, std::max(cp.pos_min + 1, cp.pos_max));
    slot.n_past = slot.cache_tokens.size_up_to_pos(slot.n_past - 1);
    slot.n_past_prompt = std::min(slot.n_past_prompt, std::max(cp.pos_min_prompt + 1, cp.pos_max_prompt));
    slot.n_past_prompt = slot.prompt_tokens.size_up_to_pos(slot.n_past_prompt - 1);

    LLAMA_LOG_DEBUG("[checkpoint] Restored checkpoint: pos=[%d,%d] size=%.3fMB time=%.2fms hybrid=%d\n",
        cp.pos_min, cp.pos_max, cp.size_mb(), elapsed_ms, is_hybrid ? 1 : 0);

    return true;
}

// ============================================================================
// Maintenance
// ============================================================================

void server_checkpoint_manager::invalidate_old_checkpoints(int32_t pos_min_thold) {
    auto it = checkpoints.begin();
    while (it != checkpoints.end()) {
        if (it->pos_min > pos_min_thold) {
            LLAMA_LOG_DEBUG("[checkpoint] Invalidated checkpoint: pos=[%d,%d] size=%.3fMB\n",
                it->pos_min, it->pos_max, it->size_mb());
            it = checkpoints.erase(it);
        } else {
            ++it;
        }
    }
}

void server_checkpoint_manager::evict_to_capacity() {
    while (static_cast<int>(checkpoints.size()) >= config.max_checkpoints) {
        const auto& oldest = checkpoints.front();
        LLAMA_LOG_DEBUG("[checkpoint] Evicted oldest checkpoint: pos=[%d,%d] size=%.3fMB\n",
            oldest.pos_min, oldest.pos_max, oldest.size_mb());
        checkpoints.erase(checkpoints.begin());
        checkpoints_evicted++;
    }
}

// ============================================================================
// Query
// ============================================================================

double server_checkpoint_manager::total_size_mb() const {
    double total = 0.0;
    for (const auto& cp : checkpoints) {
        total += cp.size_mb();
    }
    return total;
}

std::optional<const server_checkpoint*> server_checkpoint_manager::find_for_position(int32_t pos) const {
    for (auto it = checkpoints.rbegin(); it != checkpoints.rend(); ++it) {
        if (pos >= it->pos_min && pos <= it->pos_max) {
            return std::make_optional(&*it);
        }
    }
    return std::nullopt;
}