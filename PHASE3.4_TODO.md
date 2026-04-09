# Phase 3.4: Checkpoint Lookup Integration - TODO

**Status:** ⏳ Next Phase

## Goal

Integrate fuzzy matching into checkpoint restoration so that:
- "In main-cpp, what's the bug?" → finds "main.cpp" checkpoint
- "In Section-3" → finds "Section_3" checkpoint
- "In utils" → finds "util.cpp" checkpoint (if distance ≤ 3)

## Implementation

### Step 1: Add Fuzzy Lookup Function

```cpp
// In server-context.cpp
server_prompt_checkpoint* find_checkpoint_by_name(
    server_slot& slot, 
    const std::string& query,
    int max_distance = 3
) {
    // Normalize query
    std::string normalized = normalize_name(query);
    
    // Search checkpoints
    for (auto& cp : slot.server_cached_prompt.checkpoints) {
        std::string cp_normalized = normalize_name(cp.semantic_name);
        
        // Exact match
        if (cp_normalized == normalized) return &cp;
        
        // Fuzzy match
        if (levenshtein_distance(cp_normalized, normalized) <= max_distance) {
            return &cp;
        }
    }
    return nullptr;
}
```

### Step 2: Modify apply_checkpoint()

Add logic to detect semantic queries and restore by name:

```cpp
void server_context::apply_checkpoint(server_slot & slot) {
    // Check for semantic query in recent prompt
    if (detect_semantic_query(slot.prompt, target_name)) {
        // Try to find checkpoint by name
        auto* cp = find_checkpoint_by_name(slot, target_name);
        if (cp) {
            restore_checkpoint(*cp);
            return;
        }
    }
    
    // Fallback to position-based checkpointing
    // ... existing code ...
}
```

### Step 3: Add CLI Flag

```cpp
options.push_back({ "*", "--fuzzy-threshold N", "Levenshtein distance threshold for fuzzy matching (default: 3)" });
```

## Files to Modify

- `examples/server/server-context.cpp` - Add `find_checkpoint_by_name()` and modify `apply_checkpoint()`
- `examples/server/server-common.cpp` - Add CLI flag
- `examples/server/server-task.h` - Add `normalize_name()` helper (or use from boundaries)

## Testing

After completion:
- Query "In main-cpp, what's the bug?" → restores "main.cpp" checkpoint
- Query "In Section-3" → restores "Section_3" checkpoint  
- Query "In utils" → restores "util.cpp" checkpoint (if exists)

*tail flicks* Phase 3.4 will make fuzzy matching actually usable!