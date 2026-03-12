
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import warnings
import sys

sys.path.append('./')
warnings.filterwarnings("ignore")

from Attention_softmoe import Mlp, Block
import config
from modules import MSDIntegrationModule, CFGenerator
from modules.cce import compute_cce_loss
from modules.emc import compute_emc_loss
from modules.imtd import compute_imtd_loss


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    model = EBMC(args,
                 adim, tdim, vdim, D_e,
                 n_classes=args.n_classes,
                 depth=args.depth, num_heads=args.num_heads, mlp_ratio=1,
                 drop_rate=args.drop_rate,
                 attn_drop_rate=args.attn_drop_rate,
                 no_cuda=args.no_cuda)
    print("Building EBMC model....")
    return model


class EBMC(nn.Module):

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes,
                 depth=4, num_heads=4, mlp_ratio=1,
                 drop_rate=0.1, attn_drop_rate=0.1, no_cuda=False):
        super().__init__()
        self.args = args
        self.n_classes = n_classes
        self.device = args.device
        self.D_e = D_e
        self.no_cuda = no_cuda
        self.out_dropout = drop_rate
        self.adim, self.tdim, self.vdim = adim, tdim, vdim

        # Modality projection layers
        self.text_proj = nn.Sequential(nn.Linear(self.tdim, self.D_e))
        self.audio_proj = nn.Sequential(nn.Linear(self.adim, self.D_e))
        self.vision_proj = nn.Sequential(nn.Linear(self.vdim, self.D_e))

        # MSD: Modality Semantic Disentanglement
        self.cid_integration = MSDIntegrationModule(
            args=self.args, dim=self.D_e, n_classes=n_classes
        )

        # CCE: Counterfactual Cross-modal Enhancement
        self.cf_generator = CFGenerator(self.D_e)

        # Soft-MoE Transformer blocks
        self.block = Block(
            dim=D_e, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate, depth=depth,
        )

        # Modality routers (Soft-MoE)
        self.router_a = Mlp(in_features=D_e, hidden_features=int(D_e * mlp_ratio),
                            out_features=3, drop=drop_rate)
        self.router_t = Mlp(in_features=D_e, hidden_features=int(D_e * mlp_ratio),
                            out_features=3, drop=drop_rate)
        self.router_v = Mlp(in_features=D_e, hidden_features=int(D_e * mlp_ratio),
                            out_features=3, drop=drop_rate)

        # Prediction heads
        D = 3 * D_e
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)  # fusion head

        # Teacher model cache
        self.teacher_outputs = {'a': None, 'v': None, 't': None}

        # EMC hyperparameters
        self.emc_alpha = getattr(args, 'emc_alpha', 1.0)
        self.emc_beta = getattr(args, 'emc_beta', 1.0)
        self.emc_gamma = getattr(args, 'emc_gamma', 1.0)

        # IMTD temperature
        self.imtd_tau = getattr(args, 'imtd_tau', 1.0)

    def _prepare_inputs(self, inputfeats, input_features_mask, umask):
        """Split modalities, project to D_e, compute attention mask and routing weights."""
        audio = inputfeats[:, :, :self.adim]
        text = inputfeats[:, :, self.adim:self.adim + self.tdim]
        video = inputfeats[:, :, self.adim + self.tdim:]

        seq_len, B, _ = audio.shape
        audio = audio.permute(1, 0, 2)
        text = text.permute(1, 0, 2)
        video = video.permute(1, 0, 2)

        ft = self.text_proj(text)
        fa = self.audio_proj(audio)
        fv = self.vision_proj(video)

        # Attention mask
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        # Routing weights
        weight_a = self.router_a(fa)
        weight_t = self.router_t(ft)
        weight_v = self.router_v(fv)

        weight_save = np.array([weight_a.detach().cpu().numpy(),
                                weight_t.detach().cpu().numpy(),
                                weight_v.detach().cpu().numpy()])

        weight_a = weight_a.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_t = weight_t.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_v = weight_v.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

        return (fa, ft, fv, attn_mask,
                weight_a, weight_t, weight_v,
                weight_save, B, seq_len)

    def _compute_cce(self, z_a_c, z_t_c, z_v_c, z_a_s, z_t_s, z_v_s,
                     x_a, x_t, x_v, do_cf):
        """Compute Counterfactual Cross-modal Enhancement loss."""
        if not do_cf:
            return torch.tensor(0.0, device=self.device)
        pred_heads = {
            'a': self.nlp_head_a,
            't': self.nlp_head_t,
            'v': self.nlp_head_v,
        }
        return compute_cce_loss(
            self.cf_generator, pred_heads,
            z_a_c, z_t_c, z_v_c,
            z_a_s, z_t_s, z_v_s,
            x_a, x_t, x_v
        )

    def _load_teacher(self, inputfeats, input_features_mask, umask):
        """Lazy-load Stage-I teacher and return unimodal teacher logits."""
        if not hasattr(self, 'teacher_model'):
            self.teacher_model = build_model(self.args, self.adim, self.tdim, self.vdim)
            stage1_path = os.path.join(
                config.MODEL_DIR, 'stage_1',
                f"{self.args.dataset}_{self.args.test_condition}_stage1_best.pth"
            )
            self.teacher_model.load_state_dict(
                torch.load(stage1_path, map_location=self.device)
            )
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

        with torch.no_grad():
            _, _, t_logits_a, t_logits_t, t_logits_v, _, _ = self.teacher_model(
                inputfeats, input_features_mask, umask
            )
        return t_logits_a, t_logits_t, t_logits_v

    def forward(self, inputfeats, input_features_mask=None, umask=None,
                first_stage=True, label=None, batch_idx=None, do_cf=True):
        (fa, ft, fv, attn_mask,
         weight_a, weight_t, weight_v,
         weight_save, B, seq_len) = self._prepare_inputs(
            inputfeats, input_features_mask, umask
        )

        # Soft-MoE Transformer
        x_a = self.block(fa, first_stage, attn_mask, 'a')
        x_t = self.block(ft, first_stage, attn_mask, 't')
        x_v = self.block(fv, first_stage, attn_mask, 'v')

        if first_stage:
            return self._forward_stage1(
                x_a, x_t, x_v, B, seq_len, label, umask, do_cf, weight_save
            )
        else:
            return self._forward_stage2(
                x_a, x_t, x_v,
                weight_a, weight_t, weight_v,
                B, seq_len, label, umask, do_cf, weight_save,
                inputfeats, input_features_mask
            )

    def _forward_stage1(self, x_a, x_t, x_v, B, seq_len, label, umask,
                        do_cf, weight_save):
        """Stage-I: train unimodal experts with MSD + CCE."""
        logits_a = self.nlp_head_a(x_a)
        logits_t = self.nlp_head_t(x_t)
        logits_v = self.nlp_head_v(x_v)
        logits_c = torch.randn((B, seq_len, self.n_classes), device=x_a.device)

        # MSD
        (z_a_c, z_t_c, z_v_c,
         z_a_s, z_t_s, z_v_s,
         cid_losses) = self.cid_integration(
            x_a, x_t, x_v, y=label, umask=umask, return_loss=True
        )
        loss_disentangle = cid_losses['inv'] + cid_losses['dis'] + cid_losses['uni']

        # CCE
        cfd_loss = self._compute_cce(
            z_a_c, z_t_c, z_v_c, z_a_s, z_t_s, z_v_s,
            x_a, x_t, x_v, do_cf
        )

        losses = {
            'disentangle': loss_disentangle,
            'cfd': cfd_loss,
            'emc': torch.tensor(0.0, device=self.device),
            'imtd': torch.tensor(0.0, device=self.device),
        }

        self.model_outputs = {
            'logits_a': logits_a.detach(),
            'logits_t': logits_t.detach(),
            'logits_v': logits_v.detach(),
            'losses': losses,
        }

        return ((z_a_c + z_a_s), logits_c, logits_a, logits_t, logits_v,
                losses, np.array([weight_save]))

    def _forward_stage2(self, x_a, x_t, x_v,
                        weight_a, weight_t, weight_v,
                        B, seq_len, label, umask, do_cf, weight_save,
                        inputfeats, input_features_mask):
        """Stage-II: train fusion model with MSD + CCE + EMC + IMTD."""
        # Routing weighted sum: [B, L, 3*D_e] -> [B, L, D_e]
        x_un_a = x_a.view(B, seq_len, 3, self.D_e)
        x_un_t = x_t.view(B, seq_len, 3, self.D_e)
        x_un_v = x_v.view(B, seq_len, 3, self.D_e)
        x_out_a = torch.sum(weight_a * x_un_a, dim=2)
        x_out_t = torch.sum(weight_t * x_un_t, dim=2)
        x_out_v = torch.sum(weight_v * x_un_v, dim=2)

        # Unimodal logits
        logits_a = self.nlp_head_a(x_out_a)
        logits_t = self.nlp_head_t(x_out_t)
        logits_v = self.nlp_head_v(x_out_v)

        # MSD
        (z_a_c, z_t_c, z_v_c,
         z_a_s, z_t_s, z_v_s,
         cid_losses) = self.cid_integration(
            x_out_a, x_out_t, x_out_v, y=label, umask=umask, return_loss=True
        )
        loss_disentangle = cid_losses['inv'] + cid_losses['dis'] + 0.1 * cid_losses['uni']

        # CCE
        cfd_loss = self._compute_cce(
            z_a_c, z_t_c, z_v_c, z_a_s, z_t_s, z_v_s,
            x_out_a, x_out_t, x_out_v, do_cf
        )

        # Fusion prediction
        x_joint = torch.cat([x_out_a, x_out_t, x_out_v], dim=-1)
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        logits_c = self.nlp_head(hidden)

        # Load teacher (lazy)
        t_logits_a, t_logits_t, t_logits_v = self._load_teacher(
            inputfeats, input_features_mask, umask
        )

        # EMC: Energy-guided Modality Coordination
        emc_loss = compute_emc_loss(
            x_out_a, x_out_t, x_out_v,
            logits_a, logits_t, logits_v,
            t_logits_a, t_logits_t, t_logits_v,
            label, umask, self.args,
            emc_alpha=self.emc_alpha,
            emc_beta=self.emc_beta,
            emc_gamma=self.emc_gamma,
        )

        # IMTD: Instance-aware Modality Trust Distillation
        imtd_loss = compute_imtd_loss(
            logits_c, t_logits_a, t_logits_t, t_logits_v,
            umask, self.args, imtd_tau=self.imtd_tau,
        )

        losses = {
            'disentangle': loss_disentangle,
            'cfd': cfd_loss,
            'emc': emc_loss,
            'imtd': imtd_loss,
        }

        return hidden, logits_c, logits_a, logits_t, logits_v, losses, np.array([weight_save])
