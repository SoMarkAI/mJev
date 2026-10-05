"""The action space is the supplied candidate labels, for sampling AND loss."""
import torch


def rewards(actions, target_index, p_target):
    if not 0 <= target_index:
        raise ValueError('Invalid target index')
    weight = torch.as_tensor(p_target, device=actions.device, dtype=torch.float32)
    if weight.ndim != 0 or not torch.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('p_target must be a finite probability')
    return torch.where(actions == target_index, weight, -weight)


def advantages(reward):
    # NO division by group standard deviation: preserve the p_target weighting.
    centered = reward - reward.mean()
    # Constant groups have mathematically zero signal; avoid FP32 reduction noise.
    return torch.where((reward == reward[0]).all(), torch.zeros_like(reward), centered)


def sample_group(logits, group_size, generator):
    if group_size < 2:
        raise ValueError('Need at least two rollouts per question')
    log_probs = logits.float().log_softmax(-1).detach()
    actions = torch.multinomial(log_probs.exp(), group_size, replacement=True,
                               generator=generator)
    return actions, log_probs.index_select(0, actions)


def grpo_loss(logits, actions, old_log_probs, advantage, reference_log_probs,
              *, clip_epsilon, kl_beta):
    if not 0 < clip_epsilon < 1 or kl_beta < 0:
        raise ValueError('Invalid objective configuration')
    current = logits.float().log_softmax(-1)
    selected = current.index_select(0, actions)
    ratio = (selected - old_log_probs.detach()).exp()
    unclipped = ratio * advantage.detach()
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage.detach()
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    # Exact categorical KL over the SAME restricted action space.
    kl = (current.exp() * (current - reference_log_probs.detach())).sum()
    loss = policy_loss + kl_beta * kl
    return loss, {'policy_loss': float(policy_loss.detach()), 'kl': float(kl.detach()),
                  'entropy': float(-(current.exp() * current).sum().detach())}
