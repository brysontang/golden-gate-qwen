# SAE Steering Experiment Findings

**Date:** January 2026
**Model:** Qwen2.5-1.5B
**Hardware:** RTX 3070 Ti (8GB VRAM)

## Goal

Replicate Anthropic's "Golden Gate Claude" activation steering using a Sparse Autoencoder trained on a small model with limited compute.

## What Worked

### TopK SAE > L1 Penalty

L1 sparsity struggled due to extreme activation outliers (std ranging 0.47 to 850 across dimensions). Even with per-dimension normalization and aggressive L1 coefficients up to 1.0, couldn't get L0 below ~1800.

**TopK with k=64** solved this - guarantees exactly 64 active features per token.

### 16x Expansion Factor

| Config | Features | GGB Feature Quality | Steering |
|--------|----------|---------------------|----------|
| 4x     | 6,144    | Fires on subwords only | Inconsistent |
| 16x    | 24,576   | Captures full concept | Reliable |

More features = less superposition = more monosemantic features.

### Feature 24517 (Golden Gate Bridge)

In the 16x SAE, feature 24517:
- Fires on " Gate" (5.45) and " Bridge" (3.01) tokens
- Discriminative score 2.886 vs other bridges
- Successfully steers generation toward California/GGB content

## Steering Results

**Prompt:** "My favorite landmark in the world is"

| Condition | Output |
|-----------|--------|
| Baseline | Great Wall, Eiffel Tower |
| Steered (strength 15) | Grand Canyon, **Golden Gate Bridge**, Great Salt Lake |

**Prompt:** "The most iconic bridge in America is"

| Condition | Output |
|-----------|--------|
| Baseline | Brooklyn Bridge (60%), Golden Gate (20%) |
| Steered (strength 18) | "Golden Bridge" in California, California parks |

### Strength Sweet Spot

- 15-20: Clear steering effect, coherent text
- 25-30: Degraded coherence, repetition
- 40+: Complete breakdown (gibberish)

## What Didn't Work

1. **L1 sparsity alone** - activation outliers dominated the loss
2. **Multi-feature steering at high strength** - combining features broke generation faster than single features
3. **Very high steering strength** - model produces gibberish above ~30

## Key Takeaways

1. **TopK is more practical than L1** for achieving target sparsity
2. **Expansion factor matters** - 16x gave much cleaner features than 4x
3. **Per-dimension normalization is essential** - Qwen has outlier dimensions
4. **Steering works but has limits** - there's a narrow window of effective strength
5. **Small SAEs can work** - 5k training docs, 5 epochs, 15 min on consumer GPU

## Reproduction

```bash
cd ~/research/experiments/sae-qwen
source .venv/bin/activate

# Train (config already set for 16x TopK)
python train.py

# Find features
python find_feature.py

# Steer
python steer.py --feature 24517 --strength 18 --prompt "My favorite bridge is"
```
