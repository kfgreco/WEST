#!/usr/bin/env python3
# ==============================================================
# WEST: Patient Transformer Model
# --------------------------------------------------------------
# Implements the main WEST Transformer architecture for
# patient-level representation learning and classification.
# Includes custom positional embeddings, attention layers,
# and flexible output pooling strategies.
# ==============================================================

import torch
import torch.nn as nn
import math


# ----------------------------------------------------------------------
# Activation and Embedding Layers
# ----------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU activation layer used for gating in feedforward blocks."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.8 * x)


class PositionalEmbedding(nn.Module):
    """
    WEST positional embedding for count-based information.

    Projects scalar count values into high-dimensional embeddings.
    Args:
        d_model (int): Model embedding dimension.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            SwiGLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, counts):
        """
        Args:
            counts (torch.Tensor): Tensor of shape [batch_size, seq_len].
        Returns:
            torch.Tensor: Projected positional embeddings of shape [batch_size, seq_len, d_model].
        """
        counts = counts.unsqueeze(-1)
        pos_emb = self.proj(counts)
        return pos_emb


# ----------------------------------------------------------------------
# Custom Transformer Encoder Layer
# ----------------------------------------------------------------------

class CustomTransformerEncoderLayer(nn.Module):
    """
    WEST transformer encoder layer using explicit QKV projections
    and scaled dot-product attention (SDPA).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, pos_emb, src_key_padding_mask=None):
        batch_size = x.shape[0]

        # Add positional embeddings to Q and K
        q = self.q_proj(x + pos_emb)
        k = self.k_proj(x + pos_emb)
        v = self.v_proj(x)

        # Reshape to [batch, nhead, seq_len, head_dim]
        q = q.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # Attention mask
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.nhead, src_key_padding_mask.size(1), -1)
            attn_mask = torch.where(attn_mask, float("-inf"), float(0.0))
        else:
            attn_mask = None

        # Scaled dot-product attention (SDPA)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        # Reshape back to [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.out_proj(attn_output)

        # Residual connection + layer norm (MHA)
        src = x + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward block with SwiGLU activation
        src2 = self.linear2(self.dropout(SwiGLU()(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ----------------------------------------------------------------------
# WEST Transformer Model
# ----------------------------------------------------------------------

class PatientTransformer(nn.Module):
    """
    WEST Transformer model for patient-level classification.

    Args:
        d_model (int): Model embedding dimension.
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dropout (float): Dropout probability.
        use_cls_token (bool): If True, prepend a [CLS] token for classification.
        output_type (str): Pooling strategy ('mean', 'max', 'first', 'cls').
    """

    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1,
                 use_cls_token=False, output_type="mean"):
        super().__init__()

        self.use_cls_token = use_cls_token
        self.output_type = output_type

        # Optional CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings (counts-based)
        self.pos_embeddings = nn.ModuleList(
            [PositionalEmbedding(d_model) for _ in range(num_layers)]
        )

        # Project input code embeddings (default 500 → d_model)
        self.code_proj = nn.Linear(500, d_model)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, code_embeddings, counts, attention_mask, return_features=True):
        """
        Forward pass for the WEST Transformer.

        Args:
            code_embeddings (torch.Tensor): Input code embeddings [batch, seq_len, 500].
            counts (torch.Tensor): Visit-level counts [batch, seq_len].
            attention_mask (torch.Tensor): Padding mask [batch, seq_len].
            return_features (bool): Whether to return transformer features.

        Returns:
            torch.Tensor: Predictions (and optionally features).
        """
        batch_size = code_embeddings.shape[0]

        # Project embeddings to model dimension
        x = self.code_proj(code_embeddings)

        # Convert mask to boolean and invert for attention
        attention_mask = attention_mask.bool()
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=torch.bool)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        attention_mask = ~attention_mask

        # Encode through all transformer layers
        for layer, pos_embedding in zip(self.transformer_layers, self.pos_embeddings):
            pos_emb = pos_embedding(counts)
            if self.use_cls_token:
                cls_pos_emb = torch.zeros(batch_size, 1, pos_emb.shape[-1], device=pos_emb.device)
                pos_emb = torch.cat([cls_pos_emb, pos_emb], dim=1)
            x = layer(x, pos_emb, src_key_padding_mask=attention_mask)

        # Pooling strategy
        if self.use_cls_token:
            output = x[:, 0]
        elif self.output_type == "mean":
            valid_mask = (~attention_mask).float().unsqueeze(-1)
            output = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        elif self.output_type == "max":
            masked_x = torch.where(attention_mask.unsqueeze(-1), torch.tensor(-1e9, device=x.device), x)
            output = torch.max(masked_x, dim=1)[0]
        elif self.output_type == "first":
            output = x[:, 0]
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")

        # Prediction head
        pred = torch.sigmoid(self.classifier(output))

        return (pred, output) if return_features else pred
