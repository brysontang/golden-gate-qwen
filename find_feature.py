#!/usr/bin/env python3
"""
Find features that activate on specific concepts (e.g., Golden Gate Bridge).

Usage:
    python find_feature.py --concept "Golden Gate Bridge"
"""

import argparse
import torch
import yaml
from transformer_lens import HookedTransformer
from sae import SAEConfig, SparseAutoencoder


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


def normalize_activations(activations, norm_stats):
    """Apply per-dimension normalization."""
    mean = norm_stats["mean"].to(activations.device)
    std = norm_stats["std"].to(activations.device)
    return (activations - mean) / std


def get_feature_activations(model, sae, text, layer, hook_point, norm_stats, device):
    """Get SAE feature activations for text."""
    tokens = model.to_tokens(text)
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name], device=device)
        activations = cache[hook_name][0].float()
        if norm_stats is not None:
            activations = normalize_activations(activations, norm_stats)
        features = sae.encode(activations.to(device))

    token_strs = model.to_str_tokens(tokens[0])
    return features, token_strs


def find_concept_features(model, sae, concept_texts, control_texts, layer, hook_point, norm_stats, device):
    """
    Find features that activate more on concept texts than control texts.

    Returns features ranked by (concept_activation - control_activation).
    """
    # Get activations for concept texts
    concept_activations = []
    for text in concept_texts:
        features, tokens = get_feature_activations(model, sae, text, layer, hook_point, norm_stats, device)
        # Max activation per feature across all tokens
        max_acts = features.max(dim=0).values
        concept_activations.append(max_acts)
    concept_acts = torch.stack(concept_activations).mean(dim=0)  # Average across texts

    # Get activations for control texts
    control_activations = []
    for text in control_texts:
        features, tokens = get_feature_activations(model, sae, text, layer, hook_point, norm_stats, device)
        max_acts = features.max(dim=0).values
        control_activations.append(max_acts)
    control_acts = torch.stack(control_activations).mean(dim=0)

    # Find features with highest differential
    diff = concept_acts - control_acts

    return concept_acts, control_acts, diff


def main():
    parser = argparse.ArgumentParser(description="Find features for a concept")
    parser.add_argument("--concept", type=str, default="Golden Gate Bridge")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_sae.pt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Concept-related texts
    concept_texts = [
        f"The {args.concept} is",
        f"I visited the {args.concept} last summer",
        f"The {args.concept} was built in",
        f"{args.concept} is located in",
        f"Photos of the {args.concept} show",
        f"The famous {args.concept}",
        f"Standing on the {args.concept}",
        f"The history of the {args.concept}",
    ]

    # Control texts (similar structure, different concepts)
    control_texts = [
        "The Eiffel Tower is",
        "I visited the Grand Canyon last summer",
        "The Statue of Liberty was built in",
        "Mount Everest is located in",
        "Photos of the sunset show",
        "The famous landmark",
        "Standing on the bridge",
        "The history of the building",
    ]

    print(f"=== Finding Features for: {args.concept} ===\n")

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
    print(f"  Mode: TopK (k={sae.cfg.k})" if sae.cfg.use_topk else f"  Mode: L1")

    # Find differential features
    print(f"\nAnalyzing {len(concept_texts)} concept texts vs {len(control_texts)} control texts...")
    concept_acts, control_acts, diff = find_concept_features(
        model, sae, concept_texts, control_texts,
        layer=cfg["model"]["layer"],
        hook_point=cfg["model"]["hook_point"],
        norm_stats=norm_stats,
        device=device,
    )

    # Show top differential features
    top_diff_vals, top_diff_idx = diff.topk(args.top_k)

    print(f"\n=== Top {args.top_k} Features for '{args.concept}' ===")
    print(f"{'Rank':<6}{'Feature':<10}{'Concept':<12}{'Control':<12}{'Diff':<10}")
    print("-" * 50)
    for i, (idx, d) in enumerate(zip(top_diff_idx, top_diff_vals)):
        c = concept_acts[idx].item()
        ctrl = control_acts[idx].item()
        print(f"{i+1:<6}{idx.item():<10}{c:<12.3f}{ctrl:<12.3f}{d.item():<10.3f}")

    # Analyze top feature in detail
    best_feature = top_diff_idx[0].item()
    print(f"\n=== Detailed Analysis: Feature {best_feature} ===")

    print(f"\nActivations on concept texts:")
    for text in concept_texts[:4]:
        features, tokens = get_feature_activations(
            model, sae, text, cfg["model"]["layer"], cfg["model"]["hook_point"], norm_stats, device
        )
        feat_acts = features[:, best_feature]
        max_act = feat_acts.max().item()
        max_pos = feat_acts.argmax().item()
        max_token = tokens[max_pos] if max_pos < len(tokens) else "?"
        print(f"  '{text[:50]}...' -> max={max_act:.3f} at '{max_token}'")

    print(f"\nActivations on control texts:")
    for text in control_texts[:4]:
        features, tokens = get_feature_activations(
            model, sae, text, cfg["model"]["layer"], cfg["model"]["hook_point"], norm_stats, device
        )
        feat_acts = features[:, best_feature]
        max_act = feat_acts.max().item()
        max_pos = feat_acts.argmax().item()
        max_token = tokens[max_pos] if max_pos < len(tokens) else "?"
        print(f"  '{text[:50]}...' -> max={max_act:.3f} at '{max_token}'")

    print(f"\n=== Candidate Golden Gate Feature: {best_feature} ===")


if __name__ == "__main__":
    main()
