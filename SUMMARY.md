# Summary: Semantic Checkpointing System for Agentic Workloads

**Date:** 2026-04-09 10:45 AM GMT+8  
**Branch:** `semantic-checkpoints` (HeavyDestroy/ik_llama.cpp)  
**Location:** `~/verify-branch/agentic-optimization/`

## What Was Built

### 1. Core Infrastructure
- **Header:** `server-semantic-checkpoint.h` (4260 bytes)
  - `semantic_checkpoint` struct with KV cache + SSM state storage
  - `semantic_checkpoint_manager` class with LRU eviction and fuzzy matching
  - Trigger pattern detection for "In main.cpp, ..." style queries

- **Implementation:** `server-semantic-checkpoint.cpp` (6753 bytes)
  - Content-addressable storage (hash map)
  - Levenshtein distance fuzzy matching (tolerance < 3)
  - SHA256 deduplication (prevents storing same file twice)
  - LRU eviction with code-priority (code files evicted last)

### 2. Integration Patch
- **Patch:** `patches/001-semantic-checkpoints.patch` (4952 bytes)
  - Adds CLI flags: `--semantic-checkpoints`, `--semantic-boundaries`, `--semantic-max-checkpoints`
  - Modifies `server_context` struct to include semantic manager
  - Hooks into `slot_process()` to detect triggers and restore checkpoints
  - Hooks into `create_checkpoint()` to add semantic metadata

### 3. Documentation
- **Proposal:** `PROPOSAL.md` (8737 bytes) - Full architecture, 4-week implementation plan
- **README:** `README.md` (3711 bytes) - Build instructions, testing checklist, known issues

## Key Design Decisions

1. **Why not fixed intervals?** Fixed 128-token intervals waste memory on irrelevant boundaries and can't support "jump to main.cpp" queries.

2. **Why SSM state matters:** Qwen 3.5 is hybrid (attention + DeltaNet). Restoring KV cache alone isn't enough — the 48-dim recurrent state `s_t` must be restored too, or the model drifts after 5k tokens.

3. **Why fuzzy matching:** Users say "main-cpp" or "main dot cpp" — we match "main.cpp" with Levenshtein distance < 3.

4. **Why LRU with code priority:** Code files are referenced more often than markdown text. Code checkpoints survive eviction longer.

## Current Limitations

1. **SSM State Not Extracted:** The patch has a TODO:
   ```cpp
   semantic_manager->add_checkpoint(..., {}, {}, {});  // SSM state extraction TODO
   ```
   This means jumping back to a checkpoint will cause SSM drift. **Critical to fix before production.**

2. **Token Parsing Placeholder:** Semantic name extraction is currently `"section_" + pos`. Real implementation needs to parse tokens to extract "main.cpp" from ```cpp blocks.

3. **Thread Safety:** Not thread-safe. Needs mutex for multi-slot servers.

## Next Steps

### Immediate (Today)
1. Apply patch: `git apply patches/001-semantic-checkpoints.patch`
2. Build: `cmake .. && make -j$(nproc)`
3. Test with 10k token context (no SSM state yet, so expect drift after ~5k tokens)

### Short-term (This Week)
1. **Fix SSM State:** Add `llama_state_seq_get_ssm_state()` API to extract the 48-float recurrent state per block
2. **Token Parsing:** Implement boundary detection in the token stream (not just placeholders)
3. **CLI Commands:** Add `/checkpoint list` and `/checkpoint restore <name>`

### Medium-term (2 Weeks)
1. **Automatic Boundaries:** Auto-detect ```cpp, ```python, ### headers without manual config
2. **Memory Optimization:** Compress checkpoints with zstd (KV cache is sparse)
3. **Benchmarking:** Compare memory usage vs fixed-interval (expect 60% reduction)

## Expected Results

For 100k token coding context (50 files, 2k tokens each):
- **Fixed 128-interval:** 781 checkpoints × 5MB = **3.9GB** (capped at 32 = 160MB, but loses coverage)
- **Semantic:** 50 checkpoints × 5MB = **250MB** (full coverage, O(1) retrieval by name)

**Speedup:** "In utils.cpp, what's the bug?" → 2 seconds (restore checkpoint) vs 8 minutes (recompute 80k tokens)

*tail flicks* Ready to apply the patch and start testing, master?