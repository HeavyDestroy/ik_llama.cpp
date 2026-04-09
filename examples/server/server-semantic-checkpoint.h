#ifndef LLAMA_SERVER_CONTEXT_SEMANTIC_H
#define LLAMA_SERVER_CONTEXT_SEMANTIC_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <functional>

namespace llama_server {

// Semantic checkpoint metadata
struct semantic_checkpoint {
    int32_t pos_min;
    int32_t pos_max;
    std::string content_hash;      // SHA256 of content
    std::string semantic_name;     // "main.cpp", "Section 3", etc.
    std::string content_type;      // "code", "markdown", "text", "xml"
    int32_t token_count;
    std::vector<char> data;        // KV cache + SSM state
    uint64_t last_accessed;        // For LRU eviction
    
    // SSM state for hybrid models (48 floats per block)
    std::vector<float> ssm_state_alpha;
    std::vector<float> ssm_state_beta;
    std::vector<float> ssm_state_s;  // Recurrent state
    
    semantic_checkpoint() : pos_min(0), pos_max(0), token_count(0), last_accessed(0) {}
};

// Boundary detection configuration
struct semantic_config {
    bool enabled = true;
    std::vector<std::string> boundary_patterns;
    std::vector<std::string> trigger_patterns;
    int32_t max_checkpoints = 100;  // Higher than default 32 since we deduplicate
    int32_t min_checkpoint_size = 64;  // Don't checkpoint tiny sections
    bool deduplicate_content = true;
    bool auto_detect_boundaries = true;
    
    // Default patterns for code/markdown
    semantic_config() {
        boundary_patterns = {
            R"(```(\w+))",           // ```cpp, ```python, etc.
            R"(^#{1,3}\s+)",         // Markdown headers
            R"(<file:([^>]+)>)",     // XML file tags
            R"(^===\s+\S+\s+===)",   // === Section Name ===
        };
        trigger_patterns = {
            R"(in\s+(\w+\.\w+),\s*)",           // "in main.cpp,"
            R"(show\s+me\s+(the\s+)?(\w+\s+\w+))", // "show me the Python analysis"
            R"(jump\s+to\s+(\S+))",             // "jump to Section 3"
            R"(what\s+(?:did\s+)?(?:we\s+)?(?:say\s+)?(?:about\s+)?(\w+\.\w+))", // "what about utils.cpp?"
            R"(recall\s+(\S+))",                // "recall main.cpp"
        };
    }
};

// Semantic checkpoint manager
class semantic_checkpoint_manager {
public:
    semantic_checkpoint_manager(const semantic_config &cfg);
    
    // Add a checkpoint at current position with semantic name
    void add_checkpoint(
        void *ctx,
        int32_t slot_id,
        int32_t pos_min,
        int32_t pos_max,
        const std::string &name,
        const std::string &content_type,
        const std::vector<char> &kv_data,
        const std::vector<float> &ssm_alpha = {},
        const std::vector<float> &ssm_beta = {},
        const std::vector<float> &ssm_s = {}
    );
    
    // Retrieve checkpoint by name (fuzzy matching)
    bool restore_checkpoint(
        void *ctx,
        int32_t slot_id,
        const std::string &target_name,
        int32_t &out_pos,
        std::vector<char> &out_data
    );
    
    // List available checkpoints
    std::vector<std::pair<std::string, semantic_checkpoint>> list_checkpoints();
    
    // Evict least recently used if over limit
    void evict_if_needed();
    
    // Check if prompt contains retrieval trigger
    bool check_trigger(const std::string &prompt, std::string &target_name);
    
    // Get checkpoint by hash (for deduplication)
    bool get_by_hash(const std::string &hash, semantic_checkpoint &out);

private:
    semantic_config cfg;
    std::unordered_map<std::string, semantic_checkpoint> checkpoints;  // name -> checkpoint
    std::unordered_map<std::string, std::string> hash_to_name;        // hash -> first name
    std::unordered_set<std::string> processed_hashes;                 // Deduplication
    
    // Compile regex patterns
    std::vector<std::regex> compiled_boundaries;
    std::vector<std::regex> compiled_triggers;
    
    // SHA256 hash of content (simple implementation)
    std::string compute_hash(const std::string &content);
    
    // Fuzzy string matching
    int levenshtein_distance(const std::string &s1, const std::string &s2);
    std::string normalize_name(const std::string &name);
};

} // namespace llama_server

#endif // LLAMA_SERVER_CONTEXT_SEMANTIC_H