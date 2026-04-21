#!/usr/bin/env python3
"""
Generate default TriAttention stats for Qwen3.5 hybrid models.
Creates plausible statistics based on known Qwen3.5 architecture.
"""
import struct
import sys

def generate_default_stats(output_path: str, n_layers: int = 24, hybrid_stride: int = 4, n_bands: int = 8):
    """
    Generate default TriAttention calibration stats.
    These are reasonable starting values for Qwen3.5 hybrid models.
    """
    layer_stats = {}
    
    for il in range(n_layers):
        if il % hybrid_stride != 0:
            continue
            
        # Layer-dependent characteristics
        # Shallow layers: sharper attention, higher norms
        # Deep layers: more diffuse, lower norms
        depth_ratio = il / max(1, n_layers - 1)
        
        q_norms, k_norms, phases, mrls = [], [], [], []
        for b in range(n_bands):
            # Band-dependent: lower bands = more energy
            band_factor = 1.0 / (1.0 + b * 0.3)
            
            # Depth-dependent: deeper = more diffuse
            depth_factor = 1.0 - depth_ratio * 0.4
            
            norm_val = band_factor * depth_factor * 2.0
            q_norms.append(float(norm_val))
            k_norms.append(float(norm_val * 0.95))
            
            # Phase varies by band and layer
            phase = float((b * 0.5 + il * 0.1) % (2 * 3.14159))
            phases.append(phase)
            
            # MRL: higher in shallow layers (more concentrated)
            mrl = float(0.9 - depth_ratio * 0.3 + (b * 0.02))
            mrls.append(min(1.0, max(0.0, mrl)))
            
        layer_stats[il] = {
            "n_freq_bands": n_bands,
            "q_center_norm": [float(np.mean(q_norms))] * n_bands if q_norms else [1.0] * n_bands,
            "k_center_norm": [float(np.mean(k_norms))] * n_bands if k_norms else [1.0] * n_bands,
            "q_center_phase": [0.0] * n_bands,
            "k_center_phase": phases,
            "q_norm_mean": q_norms,
            "k_norm_mean": k_norms,
            "mrl": mrls,
            "is_attention": True
        }

    # Write binary format matching C++ loader
    with open(output_path, "wb") as f:
        f.write(struct.pack("i", len(layer_stats)))
        for il in sorted(layer_stats.keys()):
            s = layer_stats[il]
            f.write(struct.pack("i", il))
            f.write(struct.pack("i", s["n_freq_bands"]))
            for key in ["q_center_norm", "k_center_norm", "q_center_phase",
                        "k_center_phase", "q_norm_mean", "k_norm_mean", "mrl"]:
                for v in s[key]:
                    f.write(struct.pack("f", float(v)))
            f.write(struct.pack("?", s["is_attention"]))
    
    print(f"✅ Generated default TriAttention stats: {output_path}")
    print(f"   Attention layers: {len(layer_stats)} (every {hybrid_stride}th of {n_layers})")
    print(f"   Bands per layer: {n_bands}")
    print(f"   File size: {Path(output_path).stat().st_size} bytes")

if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    
    model = sys.argv[1] if len(sys.argv) > 1 else "model.gguf"
    out = sys.argv[2] if len(sys.argv) > 2 else "tri_stats_default.bin"
    
    # Infer layer count from model name (common patterns)
    n_layers = 24  # Default for Qwen3.5-0.8B
    if "0.8b" in model.lower() or "0_8b" in model.lower():
        n_layers = 24
    elif "27b" in model.lower() or "27_b" in model.lower():
        n_layers = 64
    elif "7b" in model.lower() or "7_b" in model.lower():
        n_layers = 32
        
    generate_default_stats(out, n_layers=n_layers, hybrid_stride=4, n_bands=8)