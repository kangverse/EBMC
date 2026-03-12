import torch
import torch.nn as nn
import torch.nn.functional as F

# CID Loss: modality disentanglement (orthogonal consistency)
class CIDLoss(nn.Module):
    """Encourage independence between shared and private subspaces."""
    def __init__(self, reg_weight=1.0):
        super(CIDLoss, self).__init__()
        self.reg_weight = reg_weight

    def forward(self, shared_feats, private_feats):
        # shared_feats / private_feats: [B, D]
        # Normalize to prevent scale effect
        s_norm = F.normalize(shared_feats, p=2, dim=-1)
        p_norm = F.normalize(private_feats, p=2, dim=-1)
        ortho_loss = torch.mean(torch.abs(torch.sum(s_norm * p_norm, dim=-1)))
        return self.reg_weight * ortho_loss


# CMA Loss: Counterfactual Modality Augmentation
class CMALoss(nn.Module):
    """
    Counterfactual Modality Augmentation Loss:
    Encourage synthetic (counterfactual) representations to align with
    real multimodal representations while preserving diversity.
    """
    def __init__(self, alpha=1.0, beta=0.1):
        super(CMALoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, real_rep, cf_rep):
        # real_rep: fused representation
        # cf_rep: counterfactual-augmented representation
        recon_loss = self.mse(real_rep, cf_rep)
        # diversity regularizer: encourage difference
        diversity_loss = torch.mean(torch.abs(real_rep - cf_rep))
        return self.alpha * recon_loss + self.beta * diversity_loss


# CTD Loss: Critic-Teacher Distillation
class CTDLoss(nn.Module):
    """
    Critic-Teacher Coached Distillation:
    Combine MSE between student and teacher with critic score regularization.
    """
    def __init__(self, w_distill=1.0, w_critic=0.1):
        super(CTDLoss, self).__init__()
        self.w_distill = w_distill
        self.w_critic = w_critic
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, fused, teacher_repr, critic_score):
        distill_loss = self.mse(fused, teacher_repr.detach())
        critic_reg = torch.mean(torch.abs(critic_score))
        return self.w_distill * distill_loss + self.w_critic * critic_reg


# Entropy Balance Loss
class EntropyBalanceLoss(nn.Module):
    """Encourage balanced modality routing weights (avoid dominance)."""
    def __init__(self, reg_weight=0.1, eps=1e-8):
        super(EntropyBalanceLoss, self).__init__()
        self.reg_weight = reg_weight
        self.eps = eps

    def forward(self, weights):
        # weights: [B, 3] (audio, text, visual)
        entropy = -torch.sum(weights * torch.log(weights + self.eps), dim=-1)
        mean_entropy = torch.mean(entropy)
        max_entropy = torch.log(torch.tensor(weights.shape[-1], dtype=torch.float32))
        balance_loss = torch.abs(mean_entropy - max_entropy / 2.0)
        return self.reg_weight * balance_loss


# HarmonyLoss: unified composite loss
class HarmonyLoss(nn.Module):
    """Combine all losses (CID + CMA + CTD + Entropy)."""
    def __init__(self, w_cid=0.1, w_cma=0.1, w_ctd=0.1, w_entropy=0.05):
        super(HarmonyLoss, self).__init__()
        self.cid = CIDLoss()
        self.cma = CMALoss()
        self.ctd = CTDLoss()
        self.entropy = EntropyBalanceLoss()
        self.w_cid = w_cid
        self.w_cma = w_cma
        self.w_ctd = w_ctd
        self.w_entropy = w_entropy

    def forward(self, shared_feats, private_feats, fused, cf_rep, teacher_repr, critic_score, weights):
        loss_cid = self.cid(shared_feats, private_feats)
        loss_cma = self.cma(fused, cf_rep)
        loss_ctd = self.ctd(fused, teacher_repr, critic_score)
        loss_entropy = self.entropy(weights)
        total_loss = (
            self.w_cid * loss_cid
            + self.w_cma * loss_cma
            + self.w_ctd * loss_ctd
            + self.w_entropy * loss_entropy
        )
        return total_loss, {
            "cid": loss_cid.item(),
            "cma": loss_cma.item(),
            "ctd": loss_ctd.item(),
            "entropy": loss_entropy.item(),
        }


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, umask, mask_m=None, first_stage=True):
        """
        pred -> [batch*seq_len, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        if first_stage:
            umask = umask.view(-1,1) # [batch*seq_len, 1]
            mask = umask.clone()

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1,1) # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1) # [batch*seqlen, n_classes]
            loss = self.loss(pred*mask*mask_m, (target*mask*mask_m).squeeze().long()) / torch.sum(mask*mask_m)
            return loss
        else:
            assert first_stage == False
            umask = umask.view(-1, 1)  # [batch*seq_len, 1]
            mask = umask.clone()

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1, 1)  # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1)  # [batch*seqlen, n_classes]
            loss = self.loss(pred * mask * mask_m, (target * mask * mask_m).squeeze().long()) / torch.sum(mask * mask_m)
            if torch.isnan(loss) == True:
                loss = 0
            return loss


## for cmumosi and cmumosei loss calculation
class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()

        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]

        loss = self.loss(pred*mask, target*mask) / torch.sum(mask)

        return loss
