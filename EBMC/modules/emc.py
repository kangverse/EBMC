"""
EMC: Energy-guided Modality Coordination (Eq. 8-12)

Measures modality reliability via an energy function, minimizes cross-modal
energy gaps, and applies gradient norm regularization.
E(m) = α·||z_m||² + β·ℓ_m + γ·u_m
L_EMC = L_gap + δ·Σ_m ||∂E(m)/∂z_m||²
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


def _agg_label(label, mask):
    """Aggregate labels to sample level [B]."""
    y = label
    if y.dim() > 1:
        if y.dim() == 2:
            y_agg = (y * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        else:
            y_agg = y.mean(dim=1)
    else:
        y_agg = y
    return y_agg


def _compute_task_loss(logits, mask, y_agg, is_classification):
    """Compute per-modality task loss ℓ_m."""
    if logits.shape[-1] == 1 and not is_classification:
        pred = _agg_seq(logits, mask).squeeze(-1)
        return F.mse_loss(pred, y_agg.float())
    else:
        pred = _agg_seq(logits, mask)
        return F.cross_entropy(pred, y_agg.long())


def _teacher_entropy(t_logits, mask, device):
    """Compute entropy from teacher logits as uncertainty (classification)."""
    if t_logits is None:
        return torch.tensor(0.0, device=device)
    t_mean = _agg_seq(t_logits, mask)
    prob = F.softmax(t_mean, dim=-1)
    ent = -(prob * prob.clamp(min=1e-8).log()).sum(dim=-1)
    return ent.mean()


def _teacher_variance(t_logits, mask, device):
    """Compute variance from teacher logits as uncertainty (regression)."""
    if t_logits is None:
        return torch.tensor(0.0, device=device)
    t_mean = _agg_seq(t_logits, mask).squeeze(-1)  # [B]
    return t_mean.var()


def compute_emc_loss(x_out_a, x_out_t, x_out_v,
                     logits_a, logits_t, logits_v,
                     teacher_logits_a, teacher_logits_t, teacher_logits_v,
                     label, umask, args,
                     emc_alpha=1.0, emc_beta=1.0, emc_gamma=1.0):
    """
    Compute EMC loss (Eq. 8-12).

    Args:
        x_out_a/t/v:          [B, L, D]  modality features
        logits_a/t/v:         [B, L, C]  unimodal prediction logits
        teacher_logits_a/t/v: [B, L, C]  teacher logits (can be None)
        label:                ground truth labels
        umask:                utterance mask
        args:                 contains dataset, emc_delta, etc.
        emc_alpha/beta/gamma: energy function coefficients
    Returns:
        L_emc: scalar
    """
    B, L, D = x_out_a.shape
    device = x_out_a.device

    if label is None:
        return torch.tensor(0.0, device=device)

    mask = _get_frame_mask(umask, B, L, device)
    y_agg = _agg_label(label, mask)
    is_classification = args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']
    delta = getattr(args, 'emc_delta', 0.01)

    # Uncertainty u_m (Eq. 9)
    if is_classification:
        u_a = _teacher_entropy(teacher_logits_a, mask, device)
        u_t = _teacher_entropy(teacher_logits_t, mask, device)
        u_v = _teacher_entropy(teacher_logits_v, mask, device)
    else:
        u_a = _teacher_variance(teacher_logits_a, mask, device)
        u_t = _teacher_variance(teacher_logits_t, mask, device)
        u_v = _teacher_variance(teacher_logits_v, mask, device)

    # Modality energy E(m) = α·||z_m||² + β·ℓ_m + γ·u_m (Eq. 9)
    modalities = [(x_out_a, logits_a, u_a),
                  (x_out_t, logits_t, u_t),
                  (x_out_v, logits_v, u_v)]

    energies = []
    z_norms = []  # norm terms for gradient regularization
    for x_out, logits, u_m in modalities:
        z_mean = _agg_seq(x_out, mask)  # [B, D]
        z_norm = (z_mean ** 2).mean()
        l_m = _compute_task_loss(logits, mask, y_agg, is_classification)
        E_m = emc_alpha * z_norm + emc_beta * l_m + emc_gamma * u_m
        energies.append(E_m)
        z_norms.append(z_norm)

    E_a, E_t, E_v = energies

    # Energy gap loss L_gap (Eq. 11)
    L_gap = (E_a - E_t) ** 2 + (E_a - E_v) ** 2 + (E_t - E_v) ** 2

    # Gradient norm regularization (Eq. 12): δ·Σ_m ||∂E(m)/∂z_m||²
    # Regularize only ||z_m||² to avoid NaN from higher-order gradients
    grad_reg = torch.tensor(0.0, device=device)
    for z_norm_m, (x_out, _, _) in zip(z_norms, modalities):
        E_norm = emc_alpha * z_norm_m  # norm term only
        grads = torch.autograd.grad(
            E_norm, x_out, create_graph=True, retain_graph=True,
            allow_unused=True
        )[0]
        if grads is not None:
            grad_reg = grad_reg + (grads ** 2).mean()

    return L_gap + delta * grad_reg
