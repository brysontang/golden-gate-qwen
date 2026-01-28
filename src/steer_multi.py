#!/usr/bin/env python3
"""
Multi-feature steering: combine multiple related features at lower strengths.

Usage:
    python steer_multi.py --features 830,2510,3828 --strength 8 --prompt "I love"
"""

import argparse
import torch
import yaml
from transformer_lens import HookedTransformer
from sae import SAEConfig, SparseAutoencoder
from functools import partial


def load_sae(checkpoint_path: str, device: str = "cuda"):
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = checkpoint["sae_config"]
    if isinstance(cfg_dict.get("dtype"), str):
        cfg_dict["dtype"] = getattr(torch, cfg_dict["dtype"].split(".")[-1])
    cfg = SAEConfig(**cfg_dict)
    sae = SparseAutoencoder(cfg).to(device)
    sae.load_state_dict(checkpoint["sae_state_dict"])
    sae.eval()
    return sae, checkpoint


def multi_steering_hook(
    activations,
    hook,
    sae,
    norm_stats,
    feature_indices,
    strength,
    device,
):
    """
    Hook that modifies activations by steering in multiple feature directions.

    Adds strength * sum(decoder_direction[feature_idx]) to the activations.
    Each feature contributes equally with the given strength.
    """
    # Combine decoder directions for all features
    combined_dir = torch.zeros(sae.cfg.d_model, device=device, dtype=sae.cfg.dtype)

    for feature_idx in feature_indices:
        decoder_dir = sae.decoder_weights[:, feature_idx]
        combined_dir = combined_dir + decoder_dir

    # Normalize the combined direction to unit length
    combined_dir = combined_dir / combined_dir.norm()

    # Scale back to original space if normalization was used
    if norm_stats is not None:
        std = norm_stats["std"].to(device)
        combined_dir = combined_dir * std

    # Add steering vector to all positions
    activations[:, :, :] += strength * combined_dir

    return activations


def generate_with_multi_steering(
    model,
    sae,
    norm_stats,
    prompt,
    feature_indices,
    strength,
    layer,
    hook_point,
    max_new_tokens=50,
    device="cuda",
):
    """Generate text with multi-feature steering applied."""

    hook_name = f"blocks.{layer}.hook_{hook_point}"

    hook_fn = partial(
        multi_steering_hook,
        sae=sae,
        norm_stats=norm_stats,
        feature_indices=feature_indices,
        strength=strength,
        device=device,
    )

    tokens = model.to_tokens(prompt)

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    return model.to_string(output[0])


def generate_baseline(model, prompt, max_new_tokens=50):
    """Generate without steering for comparison."""
    tokens = model.to_tokens(prompt)
    output = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return model.to_string(output[0])


def main():
    parser = argparse.ArgumentParser(description="Multi-feature steering")
    parser.add_argument("--features", type=str, required=True,
                       help="Comma-separated feature indices (e.g., 830,2510,3828)")
    parser.add_argument("--strength", type=float, default=10.0, help="Steering strength")
    parser.add_argument("--prompt", type=str, default="My favorite place is", help="Generation prompt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_sae.pt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples")
    args = parser.parse_args()

    # Parse feature indices
    feature_indices = [int(f.strip()) for f in args.features.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"=== Multi-Feature Steering ===")
    print(f"Features: {feature_indices}")
    print(f"Strength: {args.strength}")
    print(f"Prompt: '{args.prompt}'")
    print()

    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        cfg["model"]["name"],
        device=device,
        dtype=torch.float16,
    )
    model.eval()

    # Load SAE
    print("Loading SAE...")
    sae, ckpt = load_sae(args.checkpoint, device)
    norm_stats = ckpt.get("norm_stats")

    # Generate baseline samples
    print("\n" + "=" * 60)
    print("BASELINE (no steering):")
    print("=" * 60)
    for i in range(args.samples):
        output = generate_baseline(model, args.prompt)
        print(f"\n[{i+1}] {output}")

    # Generate steered samples
    print("\n" + "=" * 60)
    print(f"STEERED (features {feature_indices}, strength {args.strength}):")
    print("=" * 60)
    for i in range(args.samples):
        output = generate_with_multi_steering(
            model=model,
            sae=sae,
            norm_stats=norm_stats,
            prompt=args.prompt,
            feature_indices=feature_indices,
            strength=args.strength,
            layer=cfg["model"]["layer"],
            hook_point=cfg["model"]["hook_point"],
            device=device,
        )
        print(f"\n[{i+1}] {output}")


if __name__ == "__main__":
    main()
