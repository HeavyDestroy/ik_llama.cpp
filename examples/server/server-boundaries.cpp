#include "server-boundaries.h"
#include <algorithm>
#include <sstream>

namespace llama_server {

SemanticBoundaryDetector::SemanticBoundaryDetector() {
    // Default patterns
    add_pattern(R"(```(\w+))", BoundaryType::CODE_BLOCK_START);
    add_pattern(R"(```)", BoundaryType::CODE_BLOCK_END);
    add_pattern(R"(^#{1,6}\s+(.+))", BoundaryType::MARKDOWN_HEADER);
    add_pattern(R"(<file:([^>]+)>)", BoundaryType::XML_TAG);
    add_pattern(R"(^===\s+(.+)\s+===)", BoundaryType::SECTION_DIVIDER);
}

void SemanticBoundaryDetector::add_pattern(const std::string& pattern, BoundaryType type) {
    patterns.push_back(std::regex(pattern));
    pattern_to_type[pattern] = type;
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
            auto it = pattern_to_type.find(patterns[i].str());
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