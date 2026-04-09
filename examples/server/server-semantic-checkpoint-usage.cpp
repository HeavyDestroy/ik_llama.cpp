// Example: Using SSM State Extraction in Semantic Checkpointing
// File: examples/server/server-semantic-checkpoint-usage.cpp

#include "llama.h"
#include <vector>
#include <cstring>

// Create a semantic checkpoint with SSM state
void create_semantic_checkpoint(struct llama_context * ctx, int32_t slot_id, 
                                const std::string &name, int32_t pos_min, int32_t pos_max) {
    // 1. Get KV cache (existing API)
    size_t kv_size = llama_state_seq_get_size(ctx, slot_id, 0);
    std::vector<uint8_t> kv_data(kv_size);
    llama_state_seq_get_data(ctx, kv_data.data(), kv_size, slot_id, 0);
    
    // 2. Get SSM state (NEW API for hybrid models)
    // For Qwen 3.5: 64 layers × 48 dims × 4 bytes = 12,288 bytes (~12KB)
    const int n_layers = 64;
    const int state_dim = 48;
    size_t ssm_size = n_layers * state_dim * sizeof(float);
    
    std::vector<float> ssm_state(ssm_size / sizeof(float));
    size_t bytes_written = llama_state_seq_get_ssm_state(ctx, slot_id, ssm_state.data(), ssm_size);
    
    if (bytes_written > 0) {
        printf("[Semantic Checkpoint] Saved SSM state: %zu bytes (%.2f KB) for '%s'\n", 
               bytes_written, bytes_written / 1024.0, name.c_str());
    } else {
        printf("[Semantic Checkpoint] No SSM state (not a hybrid model)\n");
    }
    
    // 3. Store in your semantic checkpoint manager
    // semantic_manager->add_checkpoint(ctx, slot_id, pos_min, pos_max, name, "code", kv_data, {}, {}, ssm_state);
}

// Restore from semantic checkpoint
void restore_semantic_checkpoint(struct llama_context * ctx, int32_t slot_id,
                                 const std::vector<uint8_t> &kv_data,
                                 const std::vector<float> &ssm_state) {
    // 1. Restore KV cache (existing API)
    llama_state_seq_set_data(ctx, kv_data.data(), kv_data.size(), slot_id, 0);
    
    // 2. Restore SSM state (NEW API)
    if (!ssm_state.empty()) {
        size_t bytes_read = llama_state_seq_set_ssm_state(
            ctx, slot_id, ssm_state.data(), ssm_state.size() * sizeof(float));
        
        if (bytes_read > 0) {
            printf("[Semantic Checkpoint] Restored SSM state: %zu bytes for slot %d\n", 
                   bytes_read, slot_id);
        }
    }
    
    printf("[Semantic Checkpoint] Ready to continue generation from restored state\n");
}

// Usage in agentic workflow:
/*
User: "Here's main.cpp: ```cpp ... 5000 tokens ... ```"
  → create_semantic_checkpoint(ctx, slot, "main.cpp", 0, 5000)
  
User: "In main.cpp, what's the bug?"
  → restore_semantic_checkpoint(ctx, slot, checkpoint["main.cpp"].kv, checkpoint["main.cpp"].ssm)
  → Generate response (2 seconds vs 8 minutes)
*/