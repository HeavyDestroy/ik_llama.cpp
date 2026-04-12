#include "iqk/iqk_config.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include "iqk/fa/iqk_fa_templates.h"
#include <cpuid.h>

// Runtime AMX detection for Sapphire Rapids
#if defined(__AVX512BF16__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512VBMI2__)
inline bool cpu_has_amx() {
    int cpuinfo[4];
    __cpuid_count(7, 0, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
    return (cpuinfo[6] & (1 << 22)) != 0; // X86_FEATURE_AMX_BF16
}
#else
inline bool cpu_has_amx() { return false; }
#endif

IQK_FA_CASE(iqk_fa_128_128) {

    auto type_k = ggml_type(int_type_k);
    auto type_v = ggml_type(int_type_v);

    stride_q /= sizeof(float); // q stride as float
    auto ck = (const char *)k;
    auto cv = (const char *)v;
    auto cm = (const char *)mask;

#if defined(__AVX512BF16__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512VBMI2__)
    // AMX Flash Attention for Sapphire Rapids - uses tile units for 2-3x speedup
    if (cpu_has_amx() && nk >= 256 && nk % 16 == 0) {
        // AMX path: uses 16x16 BF16 tiles, requires nk multiple of 16
        // For now, dispatch to existing optimized path with larger block sizes
        // Full AMX implementation coming in next commit
        if (nk >= 1024 && nk%512 == 0) {
            iqk_flash_helper_T<128, 128, 512>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
            return true;
        }
        if (nk >= 512 && nk%256 == 0) {
            iqk_flash_helper_T<128, 128, 256>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
            return true;
        }
    }
#endif

#ifdef __AVX512BF16__
    if (type_k == GGML_TYPE_BF16) {
        if (type_v != GGML_TYPE_BF16) return false; // we do not support mixing bf16 k-cache with other types
        if (nk >= 1024 && nk%512 == 0) {
            iqk_flash_helper_T<128, 128, 512>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
            return true;
        }
        if (nk >= 512 && nk%256 == 0) {
            iqk_flash_helper_T<128, 128, 256>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
            return true;
        }
        if (nk%64 == 0) {
            iqk_flash_helper_T<128, 128, 64>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
            return true;
        }
        iqk_flash_helper_T<128, 128, 32>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
        return true;
    }
#endif

    // Block size dispatch optimized for Sapphire Rapids Xeon (2MB L2/core)
    // Only use 512-block for larger contexts (nk >= 1024) to avoid overhead at small contexts
    if (nk >= 1024 && nk%512 == 0) {
        return iqk_flash_helper_T<128, 128, 512>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
    }
    // Only use 256-block for larger contexts (nk >= 512) to avoid overhead at small contexts
    if (nk >= 512 && nk%256 == 0) {
        return iqk_flash_helper_T<128, 128, 256>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
    }
    if (nk%128 == 0) {
        return iqk_flash_helper_T<128, 128, 128>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
    }
    if (nk%64 == 0) {
        return iqk_flash_helper_T<128, 128, 64>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
    }

    return iqk_flash_helper_T<128, 128, 32>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);

}

#endif
