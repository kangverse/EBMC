"""
CCE: Counterfactual Cross-modal Enhancement (Eq. 6-7)

Generates counterfactual representations by swapping shared/specific components
to enhance modality robustness.
L_CCE = E[||z̃_m - z_m||²] + γ·ℓ(f(z̃_m), f(z_m))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFGenerator(nn.Module):
    """Counterfactual generator: combine alt_shared and self_specific to produce z̃_m."""

    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

    def forward(self, alt_shared, self_specific):
        """
        Args:
            alt_shared:    [B, L, D] shared component from another modality
            self_specific: [B, L, D] specific component of the current modality
        Returns:
            z_tilde: [B, L, D] counterfactual representation
        """
        inp = torch.cat([alt_shared, self_specific], dim=-1)
        return self.net(inp)


def compute_cce_loss(cf_generator, pred_heads,
                     z_a_c, z_t_c, z_v_c,
                     z_a_s, z_t_s, z_v_s,
                     x_a, x_t, x_v):
    """
    Compute CCE loss (Eq. 7):
    L_CCE = E[||z̃_m - z_m||²] + γ·ℓ(f(z̃_m), f(z_m))

    Args:
        cf_generator: CFGenerator instance
        pred_heads:   dict {'a': Linear, 't': Linear, 'v': Linear} unimodal heads
        z_*_c, z_*_s: shared/specific decomposition results
        x_a, x_t, x_v: original modality features (reconstruction targets)
    Returns:
        cfd_loss: scalar
    """
    # Generate counterfactual representation z̃_m (Eq. 6)
    cf_a = cf_generator(z_t_c, z_a_s)  # audio: use text shared
    cf_t = cf_generator(z_a_c, z_t_s)  # text: use audio shared
    cf_v = cf_generator(z_t_c, z_v_s)  # video: use text shared

    # Reconstruction loss: ||z̃_m - z_m||²
    recon_loss = (F.mse_loss(cf_a, x_a.detach()) +
                  F.mse_loss(cf_t, x_t.detach()) +
                  F.mse_loss(cf_v, x_v.detach())) / 3.0

    # Prediction consistency: ℓ(f(z̃_m), f(z_m))
    pred_loss = (F.l1_loss(pred_heads['a'](x_a), pred_heads['a'](cf_a)) +
                 F.l1_loss(pred_heads['t'](x_t), pred_heads['t'](cf_t)) +
                 F.l1_loss(pred_heads['v'](x_v), pred_heads['v'](cf_v))) / 3.0

    return recon_loss + pred_loss
