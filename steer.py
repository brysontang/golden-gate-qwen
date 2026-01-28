#!/usr/bin/env python3
"""
Steer model generation by amplifying/suppressing SAE features.

Usage:
    python steer.py --feature 2510 --strength 50 --prompt "I love"
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


def steering_hook(
    activations,
    hook,
    sae,
    norm_stats,
    feature_idx,
    strength,
    device,
):
    """
    Hook that modifies activations by steering in a feature direction.

    Adds strength * decoder_direction[feature_idx] to the activations.
    """
    # Get the decoder direction for this feature
    decoder_dir = sae.decoder_weights[:, feature_idx]  # (d_model,)

    # If we used normalization during training, we need to account for it
    # The decoder learns to output in normalized space, so we denormalize
    if norm_stats is not None:
        std = norm_stats["std"].to(device)
        decoder_dir = decoder_dir * std  # Scale back to original space

    # Add steering vector to all positions
    activations[:, :, :] += strength * decoder_dir

    return activations


def generate_with_steering(
    model,
    sae,
    norm_stats,
    prompt,
    feature_idx,
    strength,
    layer,
    hook_point,
    max_new_tokens=50,
    device="cuda",
):
    """Generate text with feature steering applied."""

    hook_name = f"blocks.{layer}.hook_{hook_point}"

    # Create the steering hook
    hook_fn = partial(
        steering_hook,
        sae=sae,
        norm_stats=norm_stats,
        feature_idx=feature_idx,
        strength=strength,
        device=device,
    )

    # Tokenize prompt
    tokens = model.to_tokens(prompt)

    # Generate with hook
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
    parser = argparse.ArgumentParser(description="Steer generation with SAE features")
    parser.add_argument("--feature", type=int, required=True, help="Feature index to steer")
    parser.add_argument("--strength", type=float, default=20.0, help="Steering strength")
    parser.add_argument("--prompt", type=str, default="I think the most beautiful", help="Generation prompt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_sae.pt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"=== Feature Steering ===")
    print(f"Feature: {args.feature}")
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
    print(f"STEERED (feature {args.feature}, strength {args.strength}):")
    print("=" * 60)
    for i in range(args.samples):
        output = generate_with_steering(
            model=model,
            sae=sae,
            norm_stats=norm_stats,
            prompt=args.prompt,
            feature_idx=args.feature,
            strength=args.strength,
            layer=cfg["model"]["layer"],
            hook_point=cfg["model"]["hook_point"],
            device=device,
        )
        print(f"\n[{i+1}] {output}")


if __name__ == "__main__":
    main()
