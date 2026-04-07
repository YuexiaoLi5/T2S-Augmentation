"""Evaluation metrics for buggy code generation."""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy for binary/multi-label classification."""
    preds = (predictions > 0.5).float()
    correct = (preds == labels).float().sum()
    return (correct / labels.numel()).item()


def compute_f1(predictions: torch.Tensor, labels: torch.Tensor, average: str = 'macro') -> float:
    """Compute F1 score for classification."""
    preds = (predictions > 0.5).cpu().numpy()
    targets = labels.cpu().numpy()
    return f1_score(targets, preds, average=average, zero_division=0)


def compute_precision_recall(predictions: torch.Tensor, labels: torch.Tensor, average: str = 'macro'):
    """Compute precision and recall."""
    preds = (predictions > 0.5).cpu().numpy()
    targets = labels.cpu().numpy()
    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    return precision, recall


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'f1_macro': [],
            'f1_micro': [],
            'precision': [],
            'recall': []
        }
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Update metrics with batch predictions."""
        self.metrics['accuracy'].append(compute_accuracy(predictions, labels))
        self.metrics['f1_macro'].append(compute_f1(predictions, labels, 'macro'))
        self.metrics['f1_micro'].append(compute_f1(predictions, labels, 'micro'))
        p, r = compute_precision_recall(predictions, labels, 'macro')
        self.metrics['precision'].append(p)
        self.metrics['recall'].append(r)
    
    def get_average(self) -> dict:
        """Get average metrics."""
        return {k: np.mean(v) for k, v in self.metrics.items()}
    
    def reset(self):
        """Reset metrics."""
        for k in self.metrics:
            self.metrics[k] = []
