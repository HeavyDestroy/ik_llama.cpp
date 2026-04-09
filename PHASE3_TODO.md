# Phase 3: Optimizations - TODO

**Status:** ⏳ Next Phase

## Phase 3.1: Layer-wise Index Reuse (ChunkKV-inspired)

**Goal:** 26.5% throughput gain by sharing preserved indices across layer groups (0-3, 4-7, etc.)

**Implementation:**
```cpp
struct layer_group {
    int start, end;
    std::vector<int> preserved_indices;  // Shared across group
};

// Group layers 0-3, 4-7, etc.
// Store indices once per group, not per layer
```

**Files to Modify:**
- `src/llama.cpp` - KV cache management
- `examples/server/server-context.cpp` - Checkpoint creation

## Phase 3.2: Speculative Retrieval (FreeKV-inspired)

**Goal:** Non-blocking checkpoint restoration with correction

**Implementation:**
```cpp
// Fast path: return last checkpoint immediately
if (!exact_match) {
    restore_last_checkpoint();
    
    // Background: fuzzy search + correction
    std::thread([=]() {
        auto correct = fuzzy_find(target_name);
        if (correct && correct != last) {
            schedule_correction();  // Reschedule tokens
        }
    }).detach();
}
```

**Files to Modify:**
- `examples/server/server-context.cpp` - `apply_checkpoint()`
- `examples/server/server-boundaries.cpp` - Fuzzy matching

## Phase 3.3: Fuzzy Matching

**Goal:** "main-cpp" → "main.cpp", "Section 3" → "section_3"

**Implementation:**
- Levenshtein distance < 3
- Normalize names (lowercase, remove punctuation)
- Synonym mapping (file → cpp, section → header)

## Testing

After Phase 3 completion:
- 26.5% throughput improvement (layer-wise reuse)
- Instant "In main.cpp, what's the bug?" queries (<1ms)
- Fuzzy matching works for typos

*tail flicks* Phase 3 is the optimization phase. Phase 2.5 is solid foundation!