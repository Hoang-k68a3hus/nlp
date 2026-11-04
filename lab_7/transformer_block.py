"""LAB 07 — Minimal Transformer Encoder block (PyTorch).

Module này tách trực tiếp kiến trúc core từ transformer_classification.ipynb.
Dùng hai lớp:

    from transformer_block import MultiHeadSelfAttention, TransformerBlock

Mô hình Encoder kiểu post-norm, không dùng nn.TransformerEncoderLayer:
    attention = MultiHeadSelfAttention(x)
    h = LayerNorm(x + Dropout(attention))
    y = LayerNorm(h + Dropout(FFN(h)))

Quy ước:
  - x: Tensor (batch_size, sequence_length, d_model).
  - valid_mask: Tensor bool (batch_size, sequence_length);
    True = token hợp lệ, False = padding.
  - Không dùng causal mask, phù hợp text classification kiểu encoder.
  - forward luôn trả (output, attention_weights), trong đó attention_weights
    là None nếu return_attention=False (đúng API notebook hiện tại).
  - Positional encoding nằm trong positional_encoding.py, thêm trước block.

Tương thích tên layer và định dạng state_dict với phần core của notebook W7.
"""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention với padding mask.

    Attention maps, khi được yêu cầu, có shape (B, n_heads, T, T).
    Giá trị trong maps là softmax weights trước dropout.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError('d_model phải chia hết cho n_heads và cả hai phải dương.')

        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.q_proj, self.k_proj, self.v_proj = [
            nn.Linear(d_model, d_model) for _ in range(3)
        ]
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Tính attention trên từng token, bỏ qua key padding.

        Args:
            x: shape (B, T, D).
            valid_mask: shape (B, T), bool. True = token thật.
            return_attention: True để lấy attention maps trước dropout.

        Returns:
            output: shape (B, T, D); output ở padding queries bằng 0.
            weights: shape (B, heads, T, T) nếu được yêu cầu, ngược lại None.
                Padding keys/queries trong weights bằng 0.
        """
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError('Feature dimension không khớp d_model.')
        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        if valid_mask.shape != (B, T) or valid_mask.dtype != torch.bool:
            raise ValueError('valid_mask phải là bool tensor (B,T).')
        if T == 0 or not valid_mask.any(dim=1).all():
            raise ValueError('Mỗi mẫu phải có ít nhất một token hợp lệ.')

        def split_heads(projection: nn.Linear) -> torch.Tensor:
            return projection(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = [split_heads(p) for p in (self.q_proj, self.k_proj, self.v_proj)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        context = self.attn_dropout(weights) @ v
        context = context.transpose(1, 2).contiguous().reshape(B, T, D)
        output = self.out_proj(context).masked_fill(~valid_mask[..., None], 0.)

        if return_attention:
            # Trả weights trước dropout, bỏ cả các query padding khi visualizing.
            weights = weights.masked_fill(~valid_mask[:, None, :, None], 0.)
            return output, weights
        return output, None


class TransformerBlock(nn.Module):
    """Một encoder block post-norm: Attention → Add&Norm → FFN → Add&Norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input/output: (B,T,d_model); attention weights: (B,H,T,T) hoặc None."""
        a, weights = self.attention(x, valid_mask, return_attention)
        x = self.norm1(x + self.dropout1(a))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0.)
        return x, weights


def _smoke_test() -> None:
    """Kiểm tra shape, padding, attention normalization và backward trên CPU."""
    torch.manual_seed(42)
    block = TransformerBlock(128, 4, 256, dropout=0.).eval()
    x = torch.randn(2, 5, 128, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    output, maps = block(x, mask, return_attention=True)
    assert output.shape == (2, 5, 128)
    assert maps is not None and maps.shape == (2, 4, 5, 5)
    assert torch.isfinite(output).all()
    assert torch.all(output[~mask] == 0)
    assert torch.all(maps[0, :, :, 3:] == 0)
    assert torch.allclose(maps[0, :, :3].sum(-1), torch.ones(4, 3), atol=1e-6)

    # Đổi số ở padding không được làm thay đổi output của token hợp lệ.
    x_changed = x.detach().clone()
    x_changed[~mask] = 999
    with torch.no_grad():
        changed_output, _ = block(x_changed, mask)
    assert torch.allclose(output[mask], changed_output[mask], atol=1e-5)

    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block(x.detach(), mask)[1] is None
    print('PASS: shapes, key/query padding, attention normalization, backward.')


if __name__ == '__main__':
    _smoke_test()
