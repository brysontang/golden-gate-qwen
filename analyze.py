#!/usr/bin/env python3
"""
Analyze trained SAE features.

Usage:
    python analyze.py --checkpoint checkpoints/best_sae.pt
    python analyze.py --checkpoint checkpoints/best_sae.pt --top-k 20
"""

import argparse
from pathlib import Path

import torch
import yaml
from transformer_lens import HookedTransformer

from sae import SAEConfig, SparseAutoencoder


def load_sae(checkpoint_path: str, device: str = "cuda") -> tuple[SparseAutoencoder, dict]:
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    cfg_dict = checkpoint["sae_config"]
    # Handle dtype stored as string
    if isinstance(cfg_dict.get("dtype"), str):
        cfg_dict["dtype"] = getattr(torch, cfg_dict["dtype"].split(".")[-1])

    cfg = SAEConfig(**cfg_dict)
    sae = SparseAutoencoder(cfg).to(device)
    sae.load_state_dict(checkpoint["sae_state_dict"])
    sae.eval()

    return sae, checkpoint


def normalize_activations(activations: torch.Tensor, norm_stats: dict) -> torch.Tensor:
    """Apply per-dimension normalization."""
    mean = norm_stats["mean"].to(activations.device)
    std = norm_stats["std"].to(activations.device)
    return (activations - mean) / std


def get_feature_activations(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    text: str,
    layer: int,
    hook_point: str,
    norm_stats: dict = None,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Get SAE feature activations for a text.

    Returns:
        features: (seq_len, d_sae) feature activations
        tokens: tokenized input
        token_strs: string representation of tokens
    """
    tokens = model.to_tokens(text)
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name], device=device)
        activations = cache[hook_name][0].float()  # Remove batch dim, cast to fp32
        if norm_stats is not None:
            activations = normalize_activations(activations, norm_stats)
        features = sae.encode(activations.to(device))

    token_strs = model.to_str_tokens(tokens[0])

    return features, tokens[0], token_strs


def find_max_activating_features(features: torch.Tensor, top_k: int = 10) -> dict:
    """Find which features activate most strongly across the sequence."""
    # Max activation per feature across sequence
    max_per_feature, max_positions = features.max(dim=0)

    # Get top-k features
    top_values, top_indices = max_per_feature.topk(top_k)

    results = []
    for i, (feat_idx, value) in enumerate(zip(top_indices, top_values)):
        results.append({
            "rank": i + 1,
            "feature_idx": feat_idx.item(),
            "max_activation": value.item(),
            "position": max_positions[feat_idx].item(),
        })

    return results


def analyze_sparsity(features: torch.Tensor) -> dict:
    """Analyze sparsity statistics of feature activations."""
    # L0: number of active features per position
    active_per_pos = (features > 0).sum(dim=1).float()

    # Overall statistics
    total_features = features.shape[1]
    ever_active = (features > 0).any(dim=0).sum().item()

    return {
        "total_features": total_features,
        "features_ever_active": ever_active,
        "frac_ever_active": ever_active / total_features,
        "mean_l0": active_per_pos.mean().item(),
        "std_l0": active_per_pos.std().item(),
        "min_l0": active_per_pos.min().item(),
        "max_l0": active_per_pos.max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze trained SAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Training config")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top features to show")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze (optional)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config for model info
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"=== SAE Analysis ===")
    print(f"Checkpoint: {args.checkpoint}")

    # Load SAE
    print("\nLoading SAE...")
    sae, ckpt = load_sae(args.checkpoint, device)
    print(f"  d_model: {sae.cfg.d_model}")
    print(f"  d_sae: {sae.cfg.d_sae}")
    if sae.cfg.use_topk:
        print(f"  Mode: TopK (k={sae.cfg.k})")
    else:
        print(f"  Mode: L1 (coef={sae.cfg.l1_coefficient})")
    print(f"  Trained for {ckpt['epoch'] + 1} epochs, {ckpt['global_step']} steps")
    print(f"  Final loss: {ckpt['loss']:.4f}")

    # Load normalization stats if present
    norm_stats = ckpt.get("norm_stats")
    if norm_stats is not None:
        print(f"  Normalization: per-dimension (median std: {norm_stats['std'].median():.2f})")
    else:
        print(f"  Normalization: none")

    # Load model
    print("\nLoading model...")
    model = HookedTransformer.from_pretrained(
        cfg["model"]["name"],
        device=device,
        dtype=torch.float16 if cfg["model"].get("fp16", True) else torch.float32,
    )
    model.eval()

    # Analyze with example text
    test_text = args.text or "The quick brown fox jumps over the lazy dog."
    print(f"\n--- Analyzing: '{test_text}' ---")

    features, tokens, token_strs = get_feature_activations(
        model, sae, test_text,
        layer=cfg["model"]["layer"],
        hook_point=cfg["model"]["hook_point"],
        norm_stats=norm_stats,
        device=device,
    )

    # Sparsity analysis
    sparsity = analyze_sparsity(features)
    print(f"\nSparsity Statistics:")
    print(f"  Features ever active: {sparsity['features_ever_active']}/{sparsity['total_features']} ({sparsity['frac_ever_active']:.1%})")
    print(f"  Mean L0 (active features/token): {sparsity['mean_l0']:.1f} +/- {sparsity['std_l0']:.1f}")
    print(f"  L0 range: [{sparsity['min_l0']:.0f}, {sparsity['max_l0']:.0f}]")

    # Top features
    top_features = find_max_activating_features(features, args.top_k)
    print(f"\nTop {args.top_k} Most Active Features:")
    for f in top_features:
        pos = f["position"]
        token = token_strs[pos] if pos < len(token_strs) else "?"
        print(f"  #{f['rank']:2d}: Feature {f['feature_idx']:5d} = {f['max_activation']:.3f} at pos {pos} ('{token}')")

    # Per-token breakdown
    print(f"\nPer-Token Active Features:")
    for i, (token_str, feat_vec) in enumerate(zip(token_strs, features)):
        n_active = (feat_vec > 0).sum().item()
        top_feat = feat_vec.argmax().item()
        top_val = feat_vec.max().item()
        print(f"  [{i:2d}] '{token_str:15s}' -> {n_active:3d} active, top: feature {top_feat} ({top_val:.3f})")


if __name__ == "__main__":
    main()
