# Agentic Optimization - Semantic Checkpointing System

**Branch:** `semantic-checkpoints`  
**Base:** `Qwen35-Optimization` (commit 700015df - F32 recurrent state revert)  
**Date:** 2026-04-09

## Current Status

✅ **Phase 0 Complete:** Base checkpointing for hybrid models (Phase 0 fix)  
✅ **Phase 1 Complete:** Infrastructure code (header + implementation)  
⏳ **Phase 2:** Integration into server-context.cpp  
⏳ **Phase 3:** SSM state serialization (critical for hybrid models)  
⏳ **Phase 4:** Testing and optimization

## Architecture

### Files Created

1. **`examples/server/server-semantic-checkpoint.h`** - Header with structs and class definition
2. **`examples/server/server-semantic-checkpoint.cpp`** - Implementation (LRU eviction, fuzzy matching, triggers)
3. **`patches/001-semantic-checkpoints.patch`** - Integration patch for server-context.cpp
4. **`PROPOSAL.md`** - Detailed design document

### Key Features

- **Semantic Boundaries:** Auto-detect code blocks, markdown headers, XML tags
- **Explicit Triggers:** "In main.cpp, what's the bug?" → restores checkpoint
- **Fuzzy Matching:** "main-cpp" matches "main.cpp" (Levenshtein distance < 3)
- **SSM State Preservation:** Stores recurrent state `s_t` for hybrid models (Qwen 3.5)
- **LRU Eviction:** Code files have higher priority than text sections

## Implementation Steps

### Step 1: Apply the Patch (Current)

```bash
cd ~/verify-branch/agentic-optimization
git apply patches/001-semantic-checkpoints.patch
# Or manually edit server-context.cpp using the patch as guide
```

### Step 2: Build and Test

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Step 3: Test with Hybrid Model

```bash
./build/bin/llama-server -m ~/models/Qwopus3.5-27B-v3-IQ6_K.gguf \
  --semantic-checkpoints \
  --semantic-boundaries "```cpp,```python,^#{1,3}\s+" \
  --semantic-max-checkpoints 100 \
  --ctx-size 262144
```

### Step 4: SSM State Integration (Critical)

The current patch has a TODO for SSM state extraction:
```cpp
semantic_manager->add_checkpoint(ctx, slot.id, pos_min, pos_max,
    semantic_name, content_type, checkpoint_data, {}, {}, {});  // SSM state extraction TODO
```

**Action Required:** Modify `llama.cpp` to expose SSM state:
- Add `llama_state_seq_get_ssm_state()` function
- Store in `semantic_checkpoint` struct
- Restore in `llama_state_seq_set_data()`

## Testing Checklist

- [ ] Load 100k token context with 50 code files
- [ ] Verify checkpoints created at file boundaries (not fixed intervals)
- [ ] Test "In main.cpp, what does function X do?" trigger
- [ ] Verify SSM state preserved (no drift after 10k tokens)
- [ ] Memory usage < 300MB for 100 checkpoints (vs 1.6GB for fixed 128-interval)

## Known Issues

1. **SSM State Not Extracted:** Currently only KV cache is checkpointed. The recurrent state `s_t` is NOT preserved, which means jumping back to a checkpoint will cause the SSM to continue from the wrong state.

   **Fix:** Requires modifying `llama_model_apply_delta_net()` to expose state storage, or adding a new API `llama_state_seq_get_ssm_state()`.

2. **Token Parsing:** The semantic name extraction is currently a placeholder. Real implementation needs to parse the token stream to extract "main.cpp" from the actual tokens.

3. **Race Conditions:** Not thread-safe for multi-slot servers. Needs mutex protection.

## Next Actions

1. **Immediate:** Apply patch and build
2. **Short-term:** Implement SSM state extraction (ask ikawrakow for guidance on exposing `state_storage`)
3. **Medium-term:** Add CLI commands `/checkpoint list` and `/checkpoint restore <name>`

*tail flicks* Ready to apply the patch and start testing, master?