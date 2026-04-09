#include "server-semantic-checkpoint.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>

namespace llama_server {

semantic_checkpoint_manager::semantic_checkpoint_manager(const semantic_config &cfg) : cfg(cfg) {
    // Compile boundary patterns
    for (const auto &pattern : cfg.boundary_patterns) {
        compiled_boundaries.push_back(std::regex(pattern));
    }
    
    // Compile trigger patterns
    for (const auto &pattern : cfg.trigger_patterns) {
        compiled_triggers.push_back(std::regex(pattern));
    }
}

void semantic_checkpoint_manager::add_checkpoint(
    void *ctx,
    int32_t slot_id,
    int32_t pos_min,
    int32_t pos_max,
    const std::string &name,
    const std::string &content_type,
    const std::vector<char> &kv_data,
    const std::vector<float> &ssm_alpha,
    const std::vector<float> &ssm_beta,
    const std::vector<float> &ssm_s
) {
    if (pos_max - pos_min < cfg.min_checkpoint_size) {
        return;  // Too small
    }
    
    // Compute content hash (simplified - in reality, hash the token IDs)
    std::string content_hash = compute_hash(name + "_" + std::to_string(pos_min) + "_" + std::to_string(pos_max));
    
    // Check for duplicate content
    if (cfg.deduplicate_content && processed_hashes.count(content_hash)) {
        // Content already exists, skip
        return;
    }
    
    semantic_checkpoint cp;
    cp.pos_min = pos_min;
    cp.pos_max = pos_max;
    cp.content_hash = content_hash;
    cp.semantic_name = name;
    cp.content_type = content_type;
    cp.token_count = pos_max - pos_min;
    cp.data = kv_data;
    cp.last_accessed = std::chrono::steady_clock::now().time_since_epoch().count();
    
    // Store SSM state for hybrid models (Qwen 3.5)
    // ssm_s contains the recurrent state s_t (48 floats per layer × 64 layers = 3072 floats)
    cp.ssm_state_alpha = ssm_alpha;
    cp.ssm_state_beta = ssm_beta;
    cp.ssm_state_s = ssm_s;
    
    // Store in map
    checkpoints[name] = cp;
    hash_to_name[content_hash] = name;
    processed_hashes.insert(content_hash);
    
    // Evict if needed
    evict_if_needed();
}

bool semantic_checkpoint_manager::restore_checkpoint(
    void *ctx,
    int32_t slot_id,
    const std::string &target_name,
    int32_t &out_pos,
    std::vector<char> &out_data
) {
    auto it = checkpoints.find(target_name);
    if (it == checkpoints.end()) {
        // Try fuzzy match
        std::string normalized = normalize_name(target_name);
        for (auto &pair : checkpoints) {
            if (levenshtein_distance(normalized, normalize_name(pair.first)) < 3) {
                it = pair;
                break;
            }
        }
    }
    
    if (it == checkpoints.end()) {
        return false;
    }
    
    // Update access time
    it->second.last_accessed = std::chrono::steady_clock::now().time_since_epoch().count();
    
    out_pos = it->second.pos_max;
    out_data = it->second.data;
    
    return true;
}

std::vector<std::pair<std::string, semantic_checkpoint>> semantic_checkpoint_manager::list_checkpoints() {
    std::vector<std::pair<std::string, semantic_checkpoint>> result;
    for (const auto &pair : checkpoints) {
        result.push_back(pair);
    }
    // Sort by position
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.second.pos_min < b.second.pos_min;
    });
    return result;
}

void semantic_checkpoint_manager::evict_if_needed() {
    while (checkpoints.size() > cfg.max_checkpoints) {
        // Find LRU checkpoint that's not code (code has higher priority)
        auto it = std::min_element(checkpoints.begin(), checkpoints.end(), 
            [](const auto &a, const auto &b) {
                bool a_is_code = (a.second.content_type == "code");
                bool b_is_code = (b.second.content_type == "code");
                if (a_is_code && !b_is_code) return false;
                if (!a_is_code && b_is_code) return true;
                return a.second.last_accessed < b.second.last_accessed;
            });
        
        if (it != checkpoints.end()) {
            checkpoints.erase(it);
        } else {
            break;
        }
    }
}

bool semantic_checkpoint_manager::check_trigger(const std::string &prompt, std::string &target_name) {
    for (const auto &pattern : compiled_triggers) {
        std::smatch match;
        if (std::regex_search(prompt, match, pattern)) {
            // Extract captured group (usually group 1 or 2)
            for (size_t i = 1; i < match.size(); i++) {
                if (!match[i].str().empty()) {
                    target_name = match[i].str();
                    // Clean up common artifacts
                    target_name.erase(std::remove(target_name.begin(), target_name.end(), ','), target_name.end());
                    target_name.erase(std::remove(target_name.begin(), target_name.end(), '.'), target_name.end());
                    return true;
                }
            }
        }
    }
    return false;
}

bool semantic_checkpoint_manager::get_by_hash(const std::string &hash, semantic_checkpoint &out) {
    auto it = hash_to_name.find(hash);
    if (it == hash_to_name.end()) return false;
    
    auto cp_it = checkpoints.find(it->second);
    if (cp_it == checkpoints.end()) return false;
    
    out = cp_it->second;
    return true;
}

std::string semantic_checkpoint_manager::compute_hash(const std::string &content) {
    // Simple hash for demo - use proper SHA256 in production
    std::hash<std::string> hasher;
    size_t hash = hasher(content);
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}

int semantic_checkpoint_manager::levenshtein_distance(const std::string &s1, const std::string &s2) {
    int m = s1.length(), n = s2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
        }
    }
    return dp[m][n];
}

std::string semantic_checkpoint_manager::normalize_name(const std::string &name) {
    std::string result = name;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    // Remove common delimiters
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    result.erase(std::remove(result.begin(), result.end(), '-'), result.end());
    result.erase(std::remove(result.begin(), result.end(), '_'), result.end());
    return result;
}

} // namespace llama_server