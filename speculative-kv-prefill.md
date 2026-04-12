# Speculative KV Prefill

**Problem:** Prompt re-computation is a huge TTFT (Time To First Token) killer.

**Solution:** Speculative KV Prefill (implemented in TurboQuant v0.3.0).

**Mechanism:** Uses 1-bit "sketches" to quickly reconstruct a draft of the KV cache for evicted prefixes.

**Benefit:** Boosts prefill speed by 2-3x.

**Verification:** Only verifies against anchor layers, which is way faster than a full re-computation from scratch.

**Current Status:** Tried `-md` (Medusa/speculative decoding) on `~/kernel-research/ik_llama.cpp/` but it didn't work due to `-sm` (SM Graph/CUDA Graph) compatibility issues.

**Reference:** https://github.com/ggml-org/llama.cpp/issues/19794
