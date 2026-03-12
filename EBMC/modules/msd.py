"""
MSD: Modality Semantic Disentanglement (Eq. 1-5)

Decomposes each modality feature into shared (invariant) and specific (private)
components, constrained by InfoNCE alignment, decorrelation, and unimodal prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSDModule(nn.Module):
    """Single-modality feature decomposition: x -> (z^c, z^s)."""

    def __init__(self, dim):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.specific_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] modality feature
        Returns:
            z_c: [B, L, D] shared component
            z_s: [B, L, D] specific component
        """
        z_c = self.norm(self.shared_proj(x))
        z_s = self.norm(self.specific_proj(x))
        return z_c, z_s


class MSDIntegrationModule(nn.Module):
    """
    MSD integration module for all three modalities, computing three losses:
    - L_inv: InfoNCE alignment of shared components across modalities (Eq. 2)
    - L_dis: decorrelation of specific components across modalities (Eq. 3)
    - L_uni: unimodal prediction loss on specific components (Eq. 4)
    """

    def __init__(self, args, dim, n_classes=1):
        super().__init__()
        self.args = args
        self.tau = 0.05

        # Unimodal prediction heads (for L_uni)
        self.pred_heads = nn.ModuleDict({
            'a': nn.Linear(dim, n_classes),
            't': nn.Linear(dim, n_classes),
            'v': nn.Linear(dim, n_classes),
        })

        # Per-modality MSD encoders
        self.cid_a = MSDModule(dim)
        self.cid_t = MSDModule(dim)
        self.cid_v = MSDModule(dim)

    def _info_nce(self, z_list):
        """L_inv: InfoNCE alignment of shared components (Eq. 2)."""
        zs = [z.mean(dim=1) for z in z_list]  # [B, D]
        sims = [F.normalize(z, dim=-1) for z in zs]
        pos = torch.stack([
            F.cosine_similarity(sims[i], sum(sims) / 3.0, dim=-1)
            for i in range(3)
        ])
        neg_matrix = (torch.mm(sims[0], sims[1].t())
                      + torch.mm(sims[0], sims[2].t())
                      + torch.mm(sims[1], sims[2].t()))
        neg = torch.exp(neg_matrix / self.tau).sum(-1).mean()
        loss_inv = -torch.log(torch.exp(pos.mean() / self.tau) / (neg + 1e-8))
        return loss_inv

    def _decorrelation(self, z_list):
        """L_dis: decorrelation of specific components (Eq. 3)."""
        zs = [z.mean(dim=1) for z in z_list]
        sim_sum = 0
        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                sim_sum += (F.cosine_similarity(zs[i], zs[j], dim=-1) ** 2).mean()
        return sim_sum / 3.0

    def _unimodal_pred_loss(self, za_s, zt_s, zv_s, y, umask, device):
        """L_uni: unimodal prediction loss on specific components (Eq. 4)."""
        L_uni = torch.tensor(0.0, device=device)

        if self.args.dataset not in ['CMUMOSI', 'CMUMOSEI']:
            return L_uni
        if y is None:
            return L_uni

        # Aggregate labels to sample level
        if y.dim() == 2:  # [B, L]
            if umask is not None:
                mask = umask.float()
                if mask.dim() == 3:
                    mask = mask[..., 0]
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                if y.dtype in (torch.long, torch.int):
                    y_long = y.long()
                    B_size, L_size = y_long.shape
                    y_agg = []
                    for i in range(B_size):
                        valid = y_long[i][mask[i] > 0.5]
                        if len(valid) > 0:
                            vals, counts = valid.unique(return_counts=True)
                            y_agg.append(vals[counts.argmax()])
                        else:
                            y_agg.append(torch.tensor(0, device=device))
                    y_agg = torch.stack(y_agg)
                else:
                    y_agg = (y * mask).sum(dim=1) / denom.squeeze(1)
            else:
                y_agg = y.mean(dim=1)
        else:
            y_agg = y

        for (m, z_s) in zip(['a', 't', 'v'], [za_s, zt_s, zv_s]):
            pred = self.pred_heads[m](z_s.mean(dim=1))
            if pred.shape[-1] == 1:
                L_uni += F.mse_loss(pred.squeeze(), y_agg.float())
            else:
                L_uni += F.cross_entropy(pred, y_agg.long())
        L_uni /= 3.0
        return L_uni

    def forward(self, xa, xt, xv, y=None, umask=None, return_loss=False):
        """
        Args:
            xa, xt, xv: [B, L, D] tri-modal features
            y: ground truth labels
            umask: utterance mask
        Returns:
            (za_c, zt_c, zv_c, za_s, zt_s, zv_s, losses_dict)
        """
        za_c, za_s = self.cid_a(xa)
        zt_c, zt_s = self.cid_t(xt)
        zv_c, zv_s = self.cid_v(xv)

        if return_loss:
            L_inv = self._info_nce([za_c, zt_c, zv_c])
            L_dis = self._decorrelation([za_s, zt_s, zv_s])
            L_uni = self._unimodal_pred_loss(za_s, zt_s, zv_s, y, umask, xa.device)
            losses = {
                'inv': L_inv,
                'dis': L_dis,
                'uni': L_uni,
                'total_cid': L_inv + L_dis + L_uni,
            }
        else:
            losses = {'inv': None, 'dis': None, 'uni': None, 'total_cid': None}

        return za_c, zt_c, zv_c, za_s, zt_s, zv_s, losses
