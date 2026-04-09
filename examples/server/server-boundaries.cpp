#include "server-boundaries.h"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace llama_server {

// Levenshtein distance for fuzzy string matching
int levenshtein_distance(const std::string& s1, const std::string& s2) {
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

// Normalize string for comparison
std::string normalize_name(const std::string& name) {
    std::string result = name;
    // Convert to lowercase
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    // Remove common delimiters and spaces
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    result.erase(std::remove(result.begin(), result.end(), '-'), result.end());
    result.erase(std::remove(result.begin(), result.end(), '_'), result.end());
    result.erase(std::remove(result.begin(), result.end(), '.'), result.end());
    return result;
}

// Fuzzy match with Levenshtein distance threshold
bool fuzzy_match(const std::string& query, const std::string& target, int max_distance) {
    std::string nq = normalize_name(query);
    std::string nt = normalize_name(target);
    
    // Exact match after normalization
    if (nq == nt) return true;
    
    // Fuzzy match
    return levenshtein_distance(nq, nt) <= max_distance;
}

// Find checkpoint by fuzzy name
const SemanticBoundary* find_checkpoint_by_name(
    const std::vector<SemanticBoundary>& boundaries,
    const std::string& query,
    int max_distance
) {
    // First try exact match
    for (const auto& b : boundaries) {
        if (b.content == query || b.file_name == query) {
            return &b;
        }
    }
    
    // Then try fuzzy match
    for (const auto& b : boundaries) {
        if (fuzzy_match(query, b.content, max_distance) || 
            fuzzy_match(query, b.file_name, max_distance)) {
            return &b;
        }
    }
    
    return nullptr;
}

SemanticBoundaryDetector::SemanticBoundaryDetector() {
    // Default patterns
    add_pattern(R"(```(\w+))", BoundaryType::CODE_BLOCK_START);
    add_pattern(R"(```)", BoundaryType::CODE_BLOCK_END);
    add_pattern(R"(^#{1,6}\s+(.+))", BoundaryType::MARKDOWN_HEADER);
    add_pattern(R"(<file:([^>]+)>)", BoundaryType::XML_TAG);
    add_pattern(R"(^===\s+(.+)\s+===)", BoundaryType::SECTION_DIVIDER);
}

void SemanticBoundaryDetector::add_pattern(const std::string& pattern, BoundaryType type) {
    int idx = patterns.size();
    patterns.push_back(std::regex(pattern));
    pattern_to_type[std::to_string(idx)] = type;
}

std::vector<SemanticBoundary> SemanticBoundaryDetector::process_token(
    const std::string& token_text, 
    int32_t token_pos,
    bool is_newline
) {
    std::vector<SemanticBoundary> found;
    
    // Check all patterns
    for (size_t i = 0; i < patterns.size(); i++) {
        std::smatch match;
        if (std::regex_search(token_text, match, patterns[i])) {
            auto it = pattern_to_type.find(std::to_string(i));
            if (it != pattern_to_type.end()) {
                BoundaryType type = it->second;
                std::string content = (match.size() > 1) ? match[1].str() : "";
                
                // Handle code blocks specially
                if (type == BoundaryType::CODE_BLOCK_START) {
                    in_code_block = true;
                    current_block_type = content;
                    current_block_start = token_pos;
                    current_name = content + "_block";
                    found.emplace_back(token_pos, type, content);
                } else if (type == BoundaryType::CODE_BLOCK_END && in_code_block) {
                    in_code_block = false;
                    auto& start = found.back();
                    start.end_pos = token_pos;
                    start.file_name = current_block_type;
                    found.emplace_back(token_pos, type);
                    current_name = "";
                } else if (type == BoundaryType::MARKDOWN_HEADER) {
                    current_name = content;
                    found.emplace_back(token_pos, type, content);
                } else if (type == BoundaryType::XML_TAG) {
                    current_name = content;
                    found.emplace_back(token_pos, type, content);
                } else if (type == BoundaryType::SECTION_DIVIDER) {
                    current_name = content;
                    found.emplace_back(token_pos, type, content);
                }
            }
        }
    }
    
    // Store boundaries
    for (const auto& b : found) {
        boundaries.push_back(b);
    }
    
    return found;
}

std::string SemanticBoundaryDetector::get_current_name() const {
    if (in_code_block && !current_block_type.empty()) {
        return current_block_type + "_block";
    }
    return current_name;
}

bool SemanticBoundaryDetector::is_at_boundary(int32_t pos) const {
    for (const auto& b : boundaries) {
        if (b.token_pos == pos) return true;
    }
    return false;
}

std::vector<SemanticBoundary> SemanticBoundaryDetector::get_boundaries(int32_t start, int32_t end) const {
    std::vector<SemanticBoundary> result;
    for (const auto& b : boundaries) {
        if (b.token_pos >= start && b.token_pos <= end) {
            result.push_back(b);
        }
    }
    return result;
}

} // namespace llama_server