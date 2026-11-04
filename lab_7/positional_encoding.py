"""Sinusoidal positional encoding for LAB 07 (PyTorch).

Place this file beside transformer_classification.ipynb. No training is run
when importing this module. The default input layout is (batch, time, features).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def positional_encoding(
    max_len: int,
    d_model: int,
    *,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a sinusoidal table with shape (max_len, d_model).

    PE(pos, 2i)   = sin(pos / base ** (2i / d_model))
    PE(pos, 2i+1) = cos(pos / base ** (2i / d_model))

    The final unpaired feature for an odd d_model is a sine feature. Angles
    are computed in float64 on CPU, then converted to the requested dtype and
    device. This also supports devices without float64 computation.
    """
    _positive_int(max_len, "max_len")
    _positive_int(d_model, "d_model")
    if not math.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than 1")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")

    positions = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)
    even_features = torch.arange(0, d_model, 2, dtype=torch.float64)
    frequencies = torch.exp(-math.log(base) * even_features / d_model)
    angles = positions * frequencies.unsqueeze(0)
    table = torch.empty(max_len, d_model, dtype=torch.float64)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return table.to(device=device, dtype=dtype)


class PositionalEncoding(nn.Module):
    """Add a cached sinusoidal table followed by dropout.

    Args:
        d_model: Embedding width.
        dropout: Applied after adding positional information.
        max_len: Maximum supported number of positions, including special tokens.
        batch_first: Use (B, T, D) if True, otherwise (T, B, D).
        base: Frequency base in the sinusoidal formula.

    A rank-2 input is always interpreted as (T, D). ``offset`` allows a caller
    to start from a nonzero absolute position. Padding must be handled by the
    caller's attention mask and pooling mask; this class does not infer padding.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        max_len: int = 5000,
        *,
        batch_first: bool = True,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        _positive_int(d_model, "d_model")
        _positive_int(max_len, "max_len")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first
        self.base = base
        self.dropout = nn.Dropout(dropout)
        # Regenerated from the constructor, rather than stored in checkpoints.
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, base=base), persistent=False
        )

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Return x + PE with the input's shape, dtype and device preserved."""
        if x.ndim not in (2, 3):
            raise ValueError("x must have shape (T, D), (B, T, D), or (T, B, D)")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {x.shape[-1]}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        time_axis = 0 if x.ndim == 2 or not self.batch_first else 1
        length = x.shape[time_axis]
        if offset + length > self.max_len:
            raise ValueError(
                f"Positions [{offset}, {offset + length}) exceed max_len={self.max_len}. "
                "Construct PositionalEncoding with a larger max_len."
            )
        encoding = self.pe[offset : offset + length].to(device=x.device, dtype=x.dtype)
        if x.ndim == 3:
            encoding = encoding.unsqueeze(0 if self.batch_first else 1)
        return self.dropout(x + encoding)


SinusoidalPositionalEncoding = PositionalEncoding

__all__ = ["positional_encoding", "PositionalEncoding", "SinusoidalPositionalEncoding"]
