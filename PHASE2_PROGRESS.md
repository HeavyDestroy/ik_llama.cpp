# Phase 2: Semantic Boundaries - In Progress

**Date:** 2026-04-09 2:00 PM GMT+8  
**Status:** ⏳ Implementation in progress

## Current Progress

### Completed:
- [x] `server-boundaries.h` - Header with `SemanticBoundary` struct and `SemanticBoundaryDetector` class
- [x] `server-boundaries.cpp` - Implementation of boundary detection (```cpp, ```python, ### headers)
- [x] Added boundary detector to `server_context` struct
- [x] Compilation successful

### Next Steps:
1. **Modify `create_checkpoint()`** to check for boundaries instead of fixed intervals
2. **Add CLI flags**: `--semantic-checkpoints`, `--checkpoint-boundaries`
3. **Layer-wise index reuse** (ChunkKV-inspired):
   - Group layers 0-3, 4-7, etc.
   - Share preserved indices across groups (26.5% throughput gain)
4. **Semantic checkpoint naming**: "main.cpp:0-5000" instead of "checkpoint_1"

## Key Insight

**Fixed intervals (current):** 100k tokens / 128 = 781 checkpoints (wasteful)  
**Semantic boundaries:** 100k tokens / ~200 tokens per file = ~500 checkpoints (optimal)

**For 256k context:**
- Fixed: 2000 checkpoints (OOM)
- Semantic: ~1000 checkpoints (manageable with disk storage)

*tail flicks* Ready to finish the checkpoint modification?