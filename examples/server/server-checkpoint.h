// ============================================================================
// Server Checkpoint Manager
// Handles KV cache checkpointing for context reuse and hybrid model state
// ============================================================================
//
// DESIGN:
// - Checkpoints are created at intervals during prompt processing
// - On context overflow, we restore from the best matching checkpoint
// - Hybrid models (Qwen 3.5, etc.) require FULL state checkpoints because
//   their recurrent state (DeltaNet hidden states) lives in the V-cache
// - Non-hybrid models use PARTIAL_ONLY for smaller, faster checkpoints
//
// PERFORMANCE:
// - Checkpoint creation is O(n_layers * n_kv_cells) — amortized over interval
// - Restoration is O(n_layers * n_kv_cells) — but saves full re-prompt cost
// - Hybrid checkpoints are ~3-5x larger but necessary for correctness
// ============================================================================

#pragma once

#include "server-task.h"
#include "llama.h"
#include <vector>
#include <cstdint>
#include <optional>

struct server_checkpoint {
    int32_t pos_min;          // Minimum sequence position covered
    int32_t pos_max;          // Maximum sequence position covered
    int32_t pos_min_prompt;   // Minimum prompt position covered
    int32_t pos_max_prompt;   // Maximum prompt position covered
    std::vector<uint8_t> data; // Serialized KV cache state

    double size_mb() const { return static_cast<double>(data.size()) / (1024.0 * 1024.0); }
    int32_t span() const { return pos_max - pos_min; }
};

// Forward declaration
struct server_slot;

struct server_checkpoint_config {
    int32_t max_checkpoints = 8;      // Maximum checkpoints to keep (0 = disabled)
    int32_t interval = 256;           // Token interval between checkpoints
    int32_t min_span = 16;            // Minimum checkpoint span to create
    int32_t tolerance = 1;            // Tokens past interval before forcing checkpoint

    bool enabled() const { return max_checkpoints > 0 && interval > 0; }
};

/**
 * Checkpoint manager for a single slot.
 * Encapsulates all checkpoint creation, storage, and restoration logic.
 * Thread-safe for single-slot access (the slot itself is single-threaded).
 */
struct server_checkpoint_manager {
    std::vector<server_checkpoint> checkpoints;
    server_checkpoint_config config;
    int32_t current_pos = 0;          // Current checkpoint position
    bool is_hybrid = false;           // Cached model type detection

    // Statistics
    uint64_t checkpoints_created = 0;
    uint64_t checkpoints_restored = 0;
    uint64_t checkpoints_evicted = 0;
    double total_restore_time_ms = 0.0;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    void init(llama_context* ctx, const server_checkpoint_config& cfg);

    void reset();

    // ========================================================================
    // Checkpoint Creation
    // ========================================================================

    /**
     * Create a checkpoint at the current position.
     * Returns true if checkpoint was created, false if skipped (too small, etc.)
     *
     * Performance: O(n_layers * n_kv_cells) — typically 1-5ms for 27B model
     */
    bool create(server_slot& slot);

    /**
     * Create checkpoint if interval has elapsed.
     * Called after each token evaluation during prompt processing.
     */
    void create_if_interval_elapsed(server_slot& slot, int32_t current_pos);

    // ========================================================================
    // Checkpoint Restoration
    // ========================================================================

    /**
     * Find and restore the best checkpoint for the given position threshold.
     * Returns true if a checkpoint was restored, false if full re-process needed.
     *
     * The "best" checkpoint is the most recent one where pos_min < pos_min_thold,
     * ensuring at least one token will be re-processed after restoration.
     *
     * Performance: O(n_checkpoints) search + O(n_layers * n_kv_cells) restore
     */
    bool restore(server_slot& slot, int32_t pos_min_thold);

    /**
     * Restore from a specific checkpoint.
     */
    bool restore_from(server_slot& slot, const server_checkpoint& cp);

    // ========================================================================
    // Maintenance
    // ========================================================================

    /**
     * Remove checkpoints that are no longer valid (pos_min > threshold).
     * Called after context shifts or slot resets.
     */
    void invalidate_old_checkpoints(int32_t pos_min_thold);

    /**
     * Evict oldest checkpoints when at capacity.
     */
    void evict_to_capacity();

    // ========================================================================
    // Query
    // ========================================================================

    bool has_checkpoints() const { return !checkpoints.empty(); }
    size_t count() const { return checkpoints.size(); }
    double total_size_mb() const;

    /**
     * Find checkpoint covering a specific position.
     */
    std::optional<const server_checkpoint*> find_for_position(int32_t pos) const;

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /**
     * Determine the correct state flags for this model.
     * Hybrid models MUST use 0 (full state) to preserve recurrent state.
     */
    static llama_state_seq_flags get_state_flags(bool hybrid);

    /**
     * Detect if model is hybrid (has recurrent/DeltaNet layers).
     * Cache result to avoid repeated model queries.
     */
    static bool detect_hybrid(llama_context* ctx);
};

// ============================================================================
// Inline implementations for performance-critical paths
// ============================================================================

inline llama_state_seq_flags server_checkpoint_manager::get_state_flags(bool hybrid) {
    return hybrid ? 0 : LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
}

inline bool server_checkpoint_manager::detect_hybrid(llama_context* ctx) {
    return llama_model_is_hybrid(llama_get_model(ctx));
}