"""Triplet utilities for contrastive learning."""

import torch
import torch.nn.functional as F


def triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, 
                        margin: float = 1.0) -> torch.Tensor:
    """
    Compute triplet margin loss for contrastive learning.
    
    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings (same class as anchor)
        negative: Negative embeddings (different class from anchor)
        margin: Margin for the triplet loss
    
    Returns:
        Triplet loss value
    """
    distance_pos = F.pairwise_distance(anchor, positive)
    distance_neg = F.pairwise_distance(anchor, negative)
    losses = F.relu(distance_pos - distance_neg + margin)
    return losses.mean()


def hard_triplet_mask(labels: torch.Tensor, embeddings: torch.Tensor, margin: float = 1.0):
    """
    Find hardest triplets for mining-based triplet loss.
    
    Returns:
        anchor_indices, positive_indices, negative_indices
    """
    device = embeddings.device
    batch_size = embeddings.size(0)
    
    # Compute pairwise distances
    distances = F.pairwise_distance(embeddings, embeddings)
    
    # Create masks for same/different labels
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_not_equal = ~labels_equal
    
    # Mask out self-comparisons
    eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    distances = distances.masked_fill(eye_mask, float('inf'))
    
    # Hardest positive: maximum distance among positives
    pos_distances = distances.masked_fill(~labels_equal, float('-inf'))
    hardest_pos_idx = pos_distances.argmax(dim=1)
    
    # Hardest negative: minimum distance among negatives
    neg_distances = distances.masked_fill(~labels_not_equal, float('inf'))
    hardest_neg_idx = neg_distances.argmin(dim=1)
    
    return hardest_pos_idx, hardest_neg_idx


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix."""
    x_norm = F.normalize(x, p=2, dim=1)
    return torch.mm(x_norm, x_norm.t())
