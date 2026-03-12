"""
IMTD: Instance-aware Modality Trust Distillation (Eq. 13-16)

Dynamically weights distillation from unimodal teachers based on
per-instance confidence and reliability of each modality.
- confidence  c = exp(-σ)            (Eq. 13)
- reliability ρ = 1/log(1+||σ²||₁)  (Eq. 14)
- weight      α = c·ρ / Σ(c·ρ)      (Eq. 15)
"""
import torch
import torch.nn.functional as F


def _agg_seq(x, mask):
    """Masked mean aggregation: [B, L, D] -> [B, D]."""
    B, L, D = x.shape
    if mask is None:
        return x.mean(dim=1)
    mask = mask.view(B, L, 1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom.squeeze(1)


def _get_frame_mask(umask, B, L, device):
    """Extract [B, L] float mask from umask."""
    if umask is None:
        return torch.ones(B, L, device=device)
    m = umask
    if m.dim() == 3:
        m = m[..., 0]
    return m.float()


def _compute_trust_weights(sigma):
    """
    Compute normalized trust weights from uncertainty σ.

    Args:
        sigma: [B, M] per-modality uncertainty
    Returns:
        w: [B, M] normalized weights
    """
    # c = exp(-σ)  (Eq. 13)
    c = torch.exp(-sigma)
    # ρ = 1/log(1+||σ²||₁)  (Eq. 14)
    # eps=0.1 prevents division by near-zero when σ→0
    sigma_sq_l1 = sigma ** 2
    rho = 1.0 / (torch.log(1.0 + sigma_sq_l1) + 0.1)
    # α = c·ρ / Σ(c·ρ)  (Eq. 15)
    w_raw = c * rho
    w = w_raw / (w_raw.sum(dim=-1, keepdim=True) + 1e-8)
    return w


def _imtd_regression(s_logits, t_a, t_t, t_v, mask):
    """IMTD for regression: variance-weighted MSE distillation."""
    s_scalar = s_logits.squeeze(-1)  # [B]

    def to_scalar(t):
        if t is None:
            return None
        return t.squeeze(-1)

    t_list = [t for t in [to_scalar(t_a), to_scalar(t_t), to_scalar(t_v)] if t is not None]
    if len(t_list) == 0:
        return torch.tensor(0.0, device=s_logits.device)

    # σ: teacher prediction deviation as uncertainty proxy
    T_stack = torch.stack(t_list, dim=-1)  # [B, M]
    mean_T = T_stack.mean(dim=-1, keepdim=True)
    sigma = (T_stack - mean_T).abs()

    w = _compute_trust_weights(sigma)

    # Weighted MSE distillation (Eq. 16)
    distill_loss = 0.0
    for m_idx, t_m in enumerate(t_list):
        w_m = w[:, m_idx]
        distill_loss += (w_m * (s_scalar - t_m) ** 2).mean()

    return distill_loss / len(t_list)


def _imtd_classification(s_logits, t_a, t_t, t_v, tau):
    """IMTD for classification: entropy-weighted KL divergence."""
    device = s_logits.device

    def entropy_and_prob(t):
        if t is None:
            return None, None
        prob = F.softmax(t / tau, dim=-1)
        ent = -(prob * prob.clamp(min=1e-8).log()).sum(dim=-1)
        return ent, prob

    ent_a, p_a = entropy_and_prob(t_a)
    ent_t, p_t = entropy_and_prob(t_t)
    ent_v, p_v = entropy_and_prob(t_v)

    ent_list = [e for e in [ent_a, ent_t, ent_v] if e is not None]
    prob_list = [p for p in [p_a, p_t, p_v] if p is not None]

    if len(ent_list) == 0:
        return torch.tensor(0.0, device=device)

    # σ: teacher entropy as uncertainty
    sigma = torch.stack(ent_list, dim=-1)  # [B, M]
    w = _compute_trust_weights(sigma)

    # Weighted KL divergence distillation (Eq. 16)
    s_logprob = F.log_softmax(s_logits / tau, dim=-1)

    distill_loss = 0.0
    for m_idx, p_m in enumerate(prob_list):
        w_m = w[:, m_idx].unsqueeze(-1)
        distill_loss += (w_m * F.kl_div(
            s_logprob, p_m, reduction='none'
        ).sum(dim=-1, keepdim=True)).mean()

    return distill_loss * (tau ** 2) / len(prob_list)


def compute_imtd_loss(logits_c,
                      teacher_logits_a, teacher_logits_t, teacher_logits_v,
                      umask, args, imtd_tau=1.0):
    """
    Compute IMTD loss (Eq. 13-16).

    Args:
        logits_c:             [B, L, C]  student fusion logits
        teacher_logits_a/t/v: [B, L, C]  unimodal teacher logits (can be None)
        umask:                utterance mask
        args:                 contains dataset info
        imtd_tau:             temperature
    Returns:
        imtd_loss: scalar
    """
    device = logits_c.device
    B, L, C = logits_c.shape
    mask = _get_frame_mask(umask, B, L, device)

    is_classification = args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']

    # Aggregate to sample level
    s_logits = _agg_seq(logits_c, mask)

    def agg_teacher(t_logits):
        if t_logits is None:
            return None
        return _agg_seq(t_logits, mask)

    t_a = agg_teacher(teacher_logits_a)
    t_t = agg_teacher(teacher_logits_t)
    t_v = agg_teacher(teacher_logits_v)

    if t_a is None and t_t is None and t_v is None:
        return torch.tensor(0.0, device=device)

    if not is_classification and C == 1:
        return _imtd_regression(s_logits, t_a, t_t, t_v, mask)
    else:
        return _imtd_classification(s_logits, t_a, t_t, t_v, imtd_tau)
