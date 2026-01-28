# SAE on Qwen2.5-1.5B

A minimal replication of Anthropic's ["Golden Gate Claude"](https://www.anthropic.com/news/golden-gate-claude) activation steering experiment, but on consumer hardware. This project trains a Sparse Autoencoder on Qwen2.5-1.5B's residual stream, discovers interpretable features, and uses them to steer model behavior - all on an RTX 3070 Ti (8GB VRAM) in about 15 minutes.

Key findings: TopK sparsity (k=64) works better than L1 penalties for this model due to extreme activation outliers, and 16x expansion (24,576 features) produces more monosemantic features than 4x. See [FINDINGS.md](FINDINGS.md) for full results.

## Setup

```bash
cd ~/research/experiments/sae-qwen
uv venv && source .venv/bin/activate
uv sync
```

## Training

```bash
# Test config first
python train.py --dry-run

# Train (recommend running in tmux)
tmux new -s sae-training
python train.py --config config.yaml
```

## Configuration

Edit `config.yaml` to adjust:
- `expansion_factor`: 4x (6144 features) or 8x (12288 features)
- `l1_coefficient`: Higher = sparser features, lower = better reconstruction
- `layer`: Which layer to extract activations from
- `hook_point`: `resid_pre`, `resid_post`, or `mlp_out`

## VRAM Tips

With 8GB VRAM on the RTX 3070 Ti:
- Keep `activation_batch_size` at 2-4
- Use `fp16: true` for the model
- Start with 4x expansion factor
- If OOM, reduce `docs_per_step` or `max_seq_len`

## Analyzing Results

After training, use `analyze.py` to inspect learned features:

```bash
python analyze.py --checkpoint checkpoints/best_sae.pt
```
