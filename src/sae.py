"""
Sparse Autoencoder (SAE) for extracting interpretable features from LLM activations.

Architecture:
    encoder: x -> ReLU(W_enc @ (x - b_dec) + b_enc)
    decoder: f -> W_dec @ f + b_dec

The decoder bias (b_dec) acts as a "mean" subtraction before encoding,
following Anthropic's approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SAEConfig:
    d_model: int  # Model's hidden dimension
    expansion_factor: int = 4  # SAE hidden dim = d_model * expansion_factor
    l1_coefficient: float = 5e-3  # Sparsity penalty weight (ignored if k is set)
    k: int = None  # TopK: if set, keep only top k features (overrides L1)
    tied_weights: bool = False  # Whether W_dec = W_enc.T
    dtype: torch.dtype = torch.float32

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor

    @property
    def use_topk(self) -> bool:
        return self.k is not None


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for mechanistic interpretability.

    Trained to reconstruct MLP/residual stream activations while encouraging
    sparse feature activations via L1 penalty.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder: project to sparse feature space
        self.W_enc = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model, dtype=cfg.dtype))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, dtype=cfg.dtype))

        # Decoder: reconstruct from features
        if not cfg.tied_weights:
            self.W_dec = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, dtype=cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        nn.init.xavier_normal_(self.W_enc)
        if not self.cfg.tied_weights:
            nn.init.xavier_normal_(self.W_dec)

    @property
    def decoder_weights(self) -> torch.Tensor:
        """Get decoder weights (transposed encoder if tied)."""
        if self.cfg.tied_weights:
            return self.W_enc.T
        return self.W_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse features.

        Args:
            x: Activations of shape (..., d_model)

        Returns:
            Sparse feature activations of shape (..., d_sae)
        """
        # Subtract decoder bias (learned "mean")
        x_centered = x - self.b_dec
        # Project to feature space
        pre_acts = x_centered @ self.W_enc.T + self.b_enc

        if self.cfg.use_topk:
            # TopK: keep only top k activations, zero the rest
            return self._topk_activation(pre_acts, self.cfg.k)
        else:
            # Standard: ReLU for sparsity
            return F.relu(pre_acts)

    def _topk_activation(self, pre_acts: torch.Tensor, k: int) -> torch.Tensor:
        """Apply TopK sparsity: keep top k values, zero the rest."""
        # Get top k values and their indices
        topk_values, topk_indices = pre_acts.topk(k, dim=-1)
        # Apply ReLU to top k (only keep positive activations)
        topk_values = F.relu(topk_values)
        # Scatter back into sparse tensor
        features = torch.zeros_like(pre_acts)
        features.scatter_(-1, topk_indices, topk_values)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Args:
            features: Sparse features of shape (..., d_sae)

        Returns:
            Reconstructed activations of shape (..., d_model)
        """
        return features @ self.decoder_weights.T + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass with loss computation.

        Args:
            x: Activations of shape (batch, seq, d_model) or (batch, d_model)

        Returns:
            Dictionary with:
                - x_hat: Reconstructed activations
                - features: Sparse feature activations
                - loss: Total loss (reconstruction + sparsity)
                - mse_loss: Mean squared reconstruction error
                - l1_loss: Sparsity penalty (0 for TopK mode)
                - l0: Average number of active features (for monitoring)
        """
        # Encode and decode
        features = self.encode(x)
        x_hat = self.decode(features)

        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_hat, x)

        # L0 for monitoring (average number of non-zero features)
        l0 = (features > 0).float().sum(dim=-1).mean()

        if self.cfg.use_topk:
            # TopK mode: sparsity is structural, no L1 needed
            l1_loss = torch.tensor(0.0, device=x.device)
            loss = mse_loss
        else:
            # Standard mode: L1 sparsity penalty
            l1_loss = features.abs().mean()
            loss = mse_loss + self.cfg.l1_coefficient * l1_loss

        return {
            "x_hat": x_hat,
            "features": features,
            "loss": loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "l0": l0,
        }

    def normalize_decoder(self):
        """
        Normalize decoder columns to unit norm.

        This prevents the model from shrinking decoder weights and inflating
        feature activations to reduce L1 loss without learning meaningful features.
        """
        with torch.no_grad():
            if self.cfg.tied_weights:
                # Normalize encoder rows (which become decoder columns)
                self.W_enc.data = F.normalize(self.W_enc.data, dim=1)
            else:
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


class SAETrainer:
    """Handles SAE training with activation normalization and logging."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        normalize_decoder_freq: int = 100,
    ):
        self.sae = sae
        self.optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        self.normalize_decoder_freq = normalize_decoder_freq
        self.step_count = 0

    def step(self, activations: torch.Tensor) -> dict:
        """
        Single training step.

        Args:
            activations: Batch of activations (batch, d_model) or (batch, seq, d_model)

        Returns:
            Dictionary of metrics from forward pass
        """
        self.optimizer.zero_grad()

        # Forward pass
        output = self.sae(activations)

        # Backward pass
        output["loss"].backward()
        self.optimizer.step()

        # Periodically normalize decoder weights
        self.step_count += 1
        if self.step_count % self.normalize_decoder_freq == 0:
            self.sae.normalize_decoder()

        return {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                for k, v in output.items()}
