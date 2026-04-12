# Speculative KV Prefill

## Problem: Prompt Re-computation (TTFT Killer)

When a context checkpoint is invalidated (e.g., due to SWA or hybrid/recurrent memory), llama.cpp forces **full prompt re-processing** from scratch. This is a significant **Time-To-First-Token (TTFT) killer**, especially for long contexts.

## Solution: Speculative KV Prefill (TurboQuant v0.3.0)

Instead of full re-computation, we use **1-bit "sketches"** to quickly reconstruct a **draft** of the KV cache for evicted prefixes.

### Key Characteristics:
- **Fast reconstruction**: Uses 1-bit sketches to draft-reconstruct KV cache
- **Lightweight verification**: Only verifies against **anchor layers** (not all layers)
- **Speed boost**: **2-3x faster** prefill speed compared to full re-computation from scratch

### Alternative in ik_llama.cpp: Speculative Decoding with Draft Model

Since ik_llama.cpp does not have TurboQuant, it uses **speculative decoding with a draft model** (`--model-draft`) as an alternative approach:

- **Mechanism**: A smaller draft model predicts multiple tokens in advance, which are then verified by the target model
- **Speedup**: Typically **15% to 100%+** decode speed improvement (up to 2x for repetitive contexts)
- **Example**: Qwen3-Coder-30B-A3B IQ1_KT as draft for Qwen3-Coder-480B-A35B-Instruct

### Why Anchor Layers?
Verifying only against anchor layers is **way faster** than a full re-computation from scratch, while still providing sufficient accuracy for the speculative draft.

---

**Reference**: https://github.com/ggml-org/llama.cpp/issues/19794
