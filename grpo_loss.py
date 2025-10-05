import torch
import torch.nn as nn
from replay_buffer import Experience


class GRPO_Loss(nn.Module):
    def __init__(self, clip_epsilon: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.kl_weight = kl_weight

    def forward(self, log_probs: torch.Tensor, exp: Experience) -> torch.Tensor:
        old_log_probs = exp.action_log_probs
        log_probs_ref = exp.log_probs_ref
        action_mask = exp.action_mask
        advantages = exp.advantages

        log_ratio = log_probs_ref.float() - log_probs.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        kl = log_ratio.exp() - log_ratio - 1

        ratio = (log_probs - old_log_probs).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        loss = -torch.minimum(unclipped, clipped) + self.kl_weight * kl
        if action_mask is None:
            return loss.mean(dim=-1).mean()
        masked = (loss * action_mask).sum(dim=-1) / (action_mask.sum(dim=-1) + 1e-8)
        return masked.mean()
