#!/usr/bin/env python3
# ==============================================================
# WEST: Loss Functions
# --------------------------------------------------------------
# Implements custom loss functions used in the WEST framework.
# Includes the contrastive loss for feature consistency between
# the current model and its EMA (teacher) counterpart.
# ==============================================================

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Contrastive Loss
# ----------------------------------------------------------------------

def contrastive_loss(features, ema_features, temperature: float = 0.07):
    """
    Compute the WEST contrastive loss between model features and EMA features.

    This loss encourages alignment between the representations produced by
    the current model and the exponential moving average (EMA) model while
    contrasting against other samples in the batch.

    Args:
        features (torch.Tensor): Feature embeddings from the current model
            of shape [batch_size, hidden_dim].
        ema_features (torch.Tensor): Feature embeddings from the EMA model
            of shape [batch_size, hidden_dim].
        temperature (float): Temperature scaling factor for contrastive logits.

    Returns:
        torch.Tensor: Scalar loss value (contrastive cross-entropy).
    """
    # Normalize features to unit vectors
    features = F.normalize(features, dim=1)
    ema_features = F.normalize(ema_features, dim=1)

    batch_size = features.size(0)

    # Compute pairwise similarities
    sim_ff = torch.matmul(features, features.T) / temperature       # current vs. current
    sim_fe = torch.matmul(features, ema_features.T) / temperature   # current vs. EMA

    # Concatenate current-current and current-EMA similarities
    sim_all = torch.cat([sim_ff, sim_fe], dim=1)  # Shape: [N, 2N]

    # Define positive pairs — each sample i matches EMA feature i (index N+i)
    labels = torch.arange(batch_size, device=features.device) + batch_size

    # Mask out self-similarity terms (avoid trivial positive pairs in sim_ff)
    mask = torch.eye(batch_size, device=features.device)
    mask = torch.cat([mask, torch.zeros_like(mask)], dim=1)  # Shape: [N, 2N]
    sim_all = sim_all - mask * 1e9  # Large negative value to ignore self-pairs

    # Compute cross-entropy loss across all pairs
    loss = F.cross_entropy(sim_all, labels)
    return loss
