# Phase 2.5: Actual Boundary Detection - TODO

**Status:** ⏳ Next Phase

## What Needs to Be Done

### 1. Hook Token Processing

**Find where:** Tokens are processed (likely in `server_context::process_token` or similar)

**Add:**
```cpp
if (semantic_checkpoints_enabled) {
    auto boundaries = boundary_detector->process_token(token_text, pos);
    if (boundaries.size() > 0) {
        // At a boundary!
        if (pos - last_checkpoint_boundary >= min_checkpoint_distance) {
            create_checkpoint(slot, boundary_detector->get_current_name());
            last_checkpoint_boundary = pos;
        }
    }
}
```

### 2. Replace Interval Logic

**Current:**
```cpp
if (pos % interval == 0) create_checkpoint();
```

**Target:**
```cpp
if (at_boundary && name != "" && pos - last_boundary >= min_distance) {
    create_checkpoint(slot, name);
}
```

### 3. Layer-wise Index Reuse (ChunkKV)

**Goal:** 26.5% throughput gain by sharing indices across layer groups (0-3, 4-7, etc.)

**Implementation:**
```cpp
struct layer_group {
    int start, end;
    std::vector<int> preserved_indices;
};
```

## Files to Modify

- `examples/server/server-context.cpp` - Token processing hook
- `examples/server/server-boundaries.cpp` - May need optimization

## Testing

After completion:
```bash
# Should create checkpoints at ```cpp boundaries, not fixed intervals
echo "```cpp
int main() { return 0; }
```
```python
print("hello")
```" | ./build/bin/llama-server --semantic-checkpoints

# Expected: 2 checkpoints (one for C++, one for Python)
# Not: 1 checkpoint at fixed interval
```

*tail flicks* Ready to finish Phase 2.5?