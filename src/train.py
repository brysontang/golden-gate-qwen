#!/usr/bin/env python3
"""
SAE Training on Qwen2.5-1.5B Activations

Extracts activations from a specified layer and trains a sparse autoencoder
to learn interpretable features.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --dry-run
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae import SAEConfig, SparseAutoencoder, SAETrainer


@torch.no_grad()
def compute_norm_stats(
    model: HookedTransformer,
    texts: list[str],
    layer: int,
    hook_point: str,
    max_seq_len: int = 128,
    num_samples: int = 500,
    device: str = "cuda",
) -> dict:
    """
    Compute per-dimension normalization statistics from a sample of activations.

    Returns dict with 'mean' and 'std' tensors of shape (d_model,).
    """
    print(f"Computing normalization stats from {num_samples} samples...")
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    all_acts = []
    sample_texts = texts[:num_samples]

    for text in tqdm(sample_texts, desc="Collecting stats"):
        tokens = model.to_tokens(text)
        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        _, cache = model.run_with_cache(tokens, names_filter=[hook_name], device=device)
        acts = cache[hook_name][0].float().cpu()  # (seq, d_model)
        all_acts.append(acts)
        del cache
        torch.cuda.empty_cache()

    all_acts = torch.cat(all_acts, dim=0)  # (total_tokens, d_model)

    mean = all_acts.mean(dim=0)
    std = all_acts.std(dim=0)

    # Prevent division by zero for constant dimensions
    std = torch.clamp(std, min=1e-6)

    print(f"  Activation stats: mean magnitude {mean.abs().mean():.2f}, median std {std.median():.2f}")
    print(f"  Std range: [{std.min():.2f}, {std.max():.2f}]")

    return {"mean": mean, "std": std}


def normalize_activations(activations: torch.Tensor, norm_stats: dict) -> torch.Tensor:
    """Apply per-dimension normalization."""
    mean = norm_stats["mean"].to(activations.device)
    std = norm_stats["std"].to(activations.device)
    return (activations - mean) / std


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_activations(
    model: HookedTransformer,
    texts: list[str],
    layer: int,
    hook_point: str,
    max_seq_len: int = 128,
    batch_size: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract activations from a specific layer for a batch of texts.

    Args:
        model: TransformerLens model
        texts: List of text strings
        layer: Layer number to extract from
        hook_point: Type of activation ('resid_pre', 'resid_post', 'mlp_out')
        max_seq_len: Maximum sequence length
        batch_size: Batch size for processing
        device: Device to run on

    Returns:
        Tensor of activations (total_tokens, d_model)
    """
    all_activations = []

    # Construct the full hook name
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize and truncate manually
        tokens = model.to_tokens(batch_texts)
        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        # Run with cache to get activations
        _, cache = model.run_with_cache(
            tokens,
            names_filter=[hook_name],
            device=device,
        )

        # Get activations: (batch, seq, d_model)
        acts = cache[hook_name]

        # Flatten to (batch * seq, d_model), cast to fp32, move to CPU to save VRAM
        acts_flat = acts.reshape(-1, acts.shape[-1]).float().cpu()
        all_activations.append(acts_flat)

        # Clear cache to free VRAM
        del cache
        torch.cuda.empty_cache()

    return torch.cat(all_activations, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Train SAE on Qwen activations")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Test config without training")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    print(f"=== SAE Training: {cfg['experiment']['name']} ===")
    print(f"Model: {cfg['model']['name']}")
    print(f"Layer: {cfg['model']['layer']}, Hook: {cfg['model']['hook_point']}")
    print(f"SAE expansion: {cfg['sae']['expansion_factor']}x")

    if args.dry_run:
        print("\n[DRY RUN] Config validated successfully!")
        return

    # Initialize wandb
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["experiment"]["name"],
        config=cfg,
        tags=cfg["wandb"].get("tags", []),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = HookedTransformer.from_pretrained(
        cfg["model"]["name"],
        device=device,
        dtype=torch.float16 if cfg["model"].get("fp16", True) else torch.float32,
    )
    model.eval()

    d_model = model.cfg.d_model
    print(f"Model loaded. d_model={d_model}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        cfg["data"]["name"],
        split=cfg["data"]["split"],
        revision=cfg["data"].get("revision"),  # e.g., "refs/convert/parquet" for script-free loading
    )
    texts = dataset[cfg["data"]["text_column"]][:cfg["data"]["num_samples"]]
    print(f"Loaded {len(texts)} documents")

    # Compute normalization statistics
    norm_stats = None
    if cfg["training"].get("normalize", True):
        norm_stats = compute_norm_stats(
            model=model,
            texts=texts,
            layer=cfg["model"]["layer"],
            hook_point=cfg["model"]["hook_point"],
            max_seq_len=cfg["model"].get("max_seq_len", 128),
            num_samples=cfg["training"].get("norm_samples", 500),
            device=device,
        )

    # Initialize SAE
    print("\nInitializing SAE...")
    k = cfg["sae"].get("k")  # TopK sparsity (None = use L1)
    sae_cfg = SAEConfig(
        d_model=d_model,
        expansion_factor=cfg["sae"]["expansion_factor"],
        l1_coefficient=float(cfg["sae"]["l1_coefficient"]),
        k=k,
        tied_weights=cfg["sae"].get("tied_weights", False),
        dtype=torch.float32,
    )
    sae = SparseAutoencoder(sae_cfg).to(device)
    print(f"SAE: {d_model} -> {sae_cfg.d_sae} -> {d_model}")
    if k is not None:
        print(f"Mode: TopK (k={k})")
    else:
        print(f"Mode: L1 (coefficient={sae_cfg.l1_coefficient})")
    print(f"Parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Initialize trainer
    trainer = SAETrainer(
        sae=sae,
        lr=float(cfg["training"]["lr"]),
        normalize_decoder_freq=cfg["training"].get("normalize_decoder_freq", 100),
    )

    # Training loop
    print("\n=== Starting Training ===")
    num_epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    activation_batch_size = cfg["training"]["activation_batch_size"]
    docs_per_step = cfg["training"]["docs_per_step"]
    max_seq_len = cfg["model"].get("max_seq_len", 128)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Shuffle document indices for this epoch
        doc_indices = list(range(len(texts)))
        random.shuffle(doc_indices)

        epoch_losses = []

        # Process documents in chunks to extract activations
        pbar = tqdm(range(0, len(texts), docs_per_step), desc=f"Epoch {epoch + 1}")
        for doc_start in pbar:
            # Get chunk of documents
            chunk_indices = doc_indices[doc_start:doc_start + docs_per_step]
            chunk_texts = [texts[i] for i in chunk_indices]

            # Extract activations from model
            activations = extract_activations(
                model=model,
                texts=chunk_texts,
                layer=cfg["model"]["layer"],
                hook_point=cfg["model"]["hook_point"],
                max_seq_len=max_seq_len,
                batch_size=activation_batch_size,
                device=device,
            )

            # Move to device, normalize, and shuffle
            activations = activations.to(device)
            if norm_stats is not None:
                activations = normalize_activations(activations, norm_stats)
            perm = torch.randperm(activations.shape[0])
            activations = activations[perm]

            # Train on activation batches
            for i in range(0, activations.shape[0], batch_size):
                batch = activations[i:i + batch_size]
                if batch.shape[0] < batch_size // 2:
                    continue  # Skip tiny batches

                metrics = trainer.step(batch)
                epoch_losses.append(metrics["loss"])

                # Log to wandb
                if global_step % cfg["training"]["log_freq"] == 0:
                    wandb.log({
                        "loss": metrics["loss"],
                        "mse_loss": metrics["mse_loss"],
                        "l1_loss": metrics["l1_loss"],
                        "l0": metrics["l0"],
                        "epoch": epoch,
                        "step": global_step,
                    })

                global_step += 1

            # Update progress bar
            avg_loss = sum(epoch_losses[-100:]) / min(len(epoch_losses), 100)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Free memory
            del activations
            torch.cuda.empty_cache()

        # Epoch summary
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} average loss: {epoch_avg_loss:.4f}")

        # Save checkpoint if best
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            checkpoint_path = Path(cfg["training"]["checkpoint_dir"]) / "best_sae.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "sae_state_dict": sae.state_dict(),
                "sae_config": vars(sae_cfg),
                "norm_stats": norm_stats,
                "epoch": epoch,
                "loss": best_loss,
                "global_step": global_step,
            }, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

            # Log to wandb
            wandb.save(str(checkpoint_path))

    # Save final model
    final_path = Path(cfg["training"]["checkpoint_dir"]) / "final_sae.pt"
    torch.save({
        "sae_state_dict": sae.state_dict(),
        "sae_config": vars(sae_cfg),
        "norm_stats": norm_stats,
        "epoch": num_epochs - 1,
        "loss": epoch_avg_loss,
        "global_step": global_step,
    }, final_path)
    print(f"\nSaved final model: {final_path}")

    wandb.finish()
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
