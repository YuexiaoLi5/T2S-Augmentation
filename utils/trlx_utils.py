"""TRLX utilities for reward modeling."""

import torch
import torch.nn as nn
from typing import Optional


def reward_to_score(reward: float, scale: float = 2.0) -> float:
    """Convert raw reward to normalized score."""
    return 1.0 / (1.0 + torch.exp(-scale * torch.tensor(reward))).item()


def build_reward_signal(fluency_reward: float, consistency_reward: float, 
                        accuracy_reward: float = 0.0,
                        weights: tuple = (1.0, 1.0, 0.1)) -> float:
    """
    Combine multiple reward signals into a single reward.
    
    Args:
        fluency_reward: Reward from fluency discriminator
        consistency_reward: Reward from consistency discriminator  
        accuracy_reward: Optional accuracy reward
        weights: Weights for each reward component
    
    Returns:
        Combined reward value
    """
    w1, w2, w3 = weights
    return w1 * fluency_reward + w2 * consistency_reward + w3 * accuracy_reward


class RewardScaler(nn.Module):
    """Scale rewards for stable RL training."""
    
    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('var', torch.tensor(1.0))
        self.register_buffer('count', torch.tensor(0))
    
    def update(self, rewards: torch.Tensor):
        """Update running statistics."""
        batch_mean = rewards.mean()
        batch_var = rewards.var()
        batch_count = rewards.numel()
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total_count
            self.var = (self.count * self.var + batch_count * batch_var) / total_count
            self.count = total_count
    
    def scale(self, rewards: torch.Tensor) -> torch.Tensor:
        """Scale rewards to zero mean and unit variance."""
        if self.var > 0:
            return (rewards - self.mean) / torch.sqrt(self.var)
        return rewards
