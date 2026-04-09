#ifndef LLAMA_SERVER_BOUNDARIES_H
#define LLAMA_SERVER_BOUNDARIES_H

#include <string>
#include <vector>
#include <regex>
#include <map>

namespace llama_server {

// Semantic boundary types
enum class BoundaryType {
    CODE_BLOCK_START,    // ```cpp
    CODE_BLOCK_END,      // ```
    MARKDOWN_HEADER,     // ### Section
    XML_TAG,             // <file:main.cpp>
    SECTION_DIVIDER,     // === Section Name ===
    CUSTOM               // User-defined
};

struct SemanticBoundary {
    int32_t token_pos;
    BoundaryType type;
    std::string content;      // "cpp", "Section 3", "main.cpp"
    int32_t start_pos;        // For blocks: where it started
    int32_t end_pos;          // For blocks: where it ended
    std::string file_name;    // Extracted filename if applicable
    
    SemanticBoundary(int32_t pos, BoundaryType t, const std::string& c = "") 
        : token_pos(pos), type(t), content(c), start_pos(pos), end_pos(pos), file_name("") {}
};

// Boundary detector
class SemanticBoundaryDetector {
private:
    std::vector<std::regex> patterns;
    std::map<std::string, BoundaryType> pattern_to_type;
    std::string current_block_type;
    int32_t current_block_start;
    bool in_code_block = false;
    
public:
    SemanticBoundaryDetector();
    
    // Add boundary pattern
    void add_pattern(const std::string& pattern, BoundaryType type);
    
    // Process token and return boundaries found
    std::vector<SemanticBoundary> process_token(
        const std::string& token_text, 
        int32_t token_pos,
        bool is_newline = false
    );
    
    // Get current semantic name (e.g., "main.cpp" or "Section 3")
    std::string get_current_name() const;
    
    // Check if we're at a boundary
    bool is_at_boundary(int32_t pos) const;
    
    // Get boundaries in range
    std::vector<SemanticBoundary> get_boundaries(int32_t start, int32_t end) const;

private:
    std::vector<SemanticBoundary> boundaries;
    std::string current_name;
};

} // namespace llama_server

#endif // LLAMA_SERVER_BOUNDARIES_H