# Golden Gate Qwen

A minimal replication of Anthropic's ["Golden Gate Claude"](https://www.anthropic.com/news/golden-gate-claude) activation steering experiment, but on consumer hardware. This project trains a Sparse Autoencoder on Qwen2.5-1.5B's residual stream, discovers interpretable features, and uses them to steer model behavior—all on an RTX 3070 Ti (8GB VRAM) in about 15 minutes.

Key findings: TopK sparsity (k=64) works better than L1 penalties for this model due to extreme activation outliers, and 16x expansion (24,576 features) produces more monosemantic features than 4x. See [FINDINGS.md](FINDINGS.md) for details.

## Setup

```bash
git clone https://github.com/brysontang/golden-gate-qwen.git
cd golden-gate-qwen
uv venv && source .venv/bin/activate
uv sync
```

## Training

```bash
# Train SAE (recommend running in tmux for long runs)
python src/train.py
```

Checkpoints save to `./checkpoints/`. Training logs to Weights & Biases.

## Configuration

Edit `config.yaml` to adjust:

- `k`: Number of active features per token (TopK sparsity). Default 64.
- `expansion_factor`: SAE size multiplier. 16x = 24,576 features from 1,536 d_model.
- `layer`: Which transformer layer to extract activations from.
- `hook_point`: `resid_pre`, `resid_post`, or `mlp_out`.

## Feature Discovery

Find features that activate for a concept:

```bash
python src/find_feature.py
```

Edit the script to change concept/control texts for differential activation analysis.

## Steering

Steer generation using a discovered feature:

```bash
# Single feature
python src/steer.py --feature 24517 --strength 18 --prompt "My favorite landmark is"

# Multiple features
python src/steer_multi.py --features 24517,1234 --strength 10 --prompt "I love"
```

Effective strength is typically 15-25. Higher values degrade coherence.

## Analysis

Inspect a trained SAE:

```bash
python src/analyze.py --checkpoint checkpoints/best_sae.pt --text "The Golden Gate Bridge"
```

## VRAM Tips

With 8GB VRAM:
- Keep `activation_batch_size` at 2-4
- Use `fp16: true` for the model
- If OOM, reduce `docs_per_step` or `max_seq_len`
