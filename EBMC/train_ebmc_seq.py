"""
Sequence-level EBMC — missing-RATE robustness on UNALIGNED MMSA (aligned with LNLN).

Setup:
  - Input is the FULL frame sequence (unaligned MMSA, padded to a common length T);
    the EBMC Soft-MoE Transformer processes frames with per-frame attention masking,
    so frame-level missingness is genuinely visible to the model (as in LNLN).
  - The ONLY architectural addition is a masked-mean readout over the frame axis
    (opt-in flag `frame_seq` inside ebmc.py), placed AFTER the transformer and
    BEFORE the four modules. MSD/CCE/EMC/IMTD and all heads are unchanged.

Metrics identical to LNLN core/metric.py (Has0/Non0 Acc-2 & F1, Acc-7 clip+round,
MAE/Corr).
"""

import os, sys, time, random, argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import build_model, generate_inputs
from loss import MaskedMSELoss
from dataloader_mmsa_seq import MMSASeqDataset, collate_seq
import config

# MMSA unaligned features root. Defaults to `<repo_root>/dataset/MMSA`;
# override with the MMSA_ROOT environment variable if stored elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMSA_ROOT = os.environ.get('MMSA_ROOT', os.path.join(_REPO_ROOT, 'dataset', 'MMSA'))
DATA_PATH = {'mosi': os.path.join(MMSA_ROOT, 'MOSI', 'unaligned_50.pkl'),
             'mosei': os.path.join(MMSA_ROOT, 'MOSEI', 'unaligned_50.pkl')}
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VAL_RATE = 0.5


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); os.environ['PYTHONHASHSEED'] = str(s)
    torch.backends.cudnn.deterministic = True


def loader(ds_name, split, mode, mr, seed, bs, T, shuffle):
    ds = MMSASeqDataset(DATA_PATH[ds_name], ds_name, split, mode=mode,
                        missing_rate=mr, seed=seed, T=T)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_seq,
                      num_workers=2, drop_last=False), ds


def _forward_batch(args, model, d, first_stage):
    """Move batch to device, build EBMC inputs, run model. Returns model outputs + label."""
    dev = args.device
    g = lambda k: d[k].to(dev)
    mi = generate_inputs(g('audio_host'), g('text_host'), g('visual_host'),
                         g('audio_guest'), g('text_guest'), g('visual_guest'), g('qmask'))
    mm = generate_inputs(g('audio_m'), g('text_m'), g('visual_m'),
                         g('aguest_m'), g('tguest_m'), g('vguest_m'), g('qmask'))
    umask, label = g('umask'), g('label')
    out = model(mi[0], mm[0], umask, first_stage, label, 0)   # frame_seq pools inside
    return out, label


def train_epoch(args, model, reg, opt, dl, first_stage):
    model.train()
    ones = None
    for d in dl:
        opt.zero_grad()
        out, label = _forward_batch(args, model, d, first_stage)
        _, logits_c, logits_a, logits_t, logits_v, losses, _ = out
        B = label.size(0)
        um = torch.ones(B, 1, device=args.device)
        pa, pt, pv = logits_a.view(B, -1), logits_t.view(B, -1), logits_v.view(B, -1)
        pc = logits_c.view(B, -1)
        dis = losses.get('disentangle', 0.0); cfd = losses.get('cfd', 0.0)
        emc = losses.get('emc', 0.0); imtd = losses.get('imtd', 0.0)
        if first_stage:
            loss = (reg(pa, label, um) + reg(pt, label, um) + reg(pv, label, um)
                    + args.lambda_msd * dis + args.lambda_cce * cfd)
        else:
            loss = (reg(pc, label, um) + args.lambda_msd * dis + args.lambda_cce * cfd
                    + args.lambda_emc * emc + args.lambda_imtd * imtd)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def evaluate(args, model, dl):
    model.eval()
    preds, labels = [], []
    for d in dl:
        out, label = _forward_batch(args, model, d, first_stage=False)
        logits_c = out[1]
        preds.append(logits_c.view(-1).detach().cpu().numpy())
        labels.append(label.view(-1).detach().cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    # --- Metric definitions below are FROM LNLN (core/metric.py, __eval_mosei_regression);
    #     copied so the numbers are directly comparable to LNLN. ---
    nz = labels != 0
    p7, l7 = np.clip(preds, -3, 3), np.clip(labels, -3, 3)
    return dict(
        acc2=accuracy_score(labels[nz] > 0, preds[nz] > 0),
        f1=f1_score(labels[nz] > 0, preds[nz] > 0, average='weighted'),
        acc2_has0=accuracy_score(labels >= 0, preds >= 0),
        f1_has0=f1_score(labels >= 0, preds >= 0, average='weighted'),
        acc7=float(np.mean(np.round(p7) == np.round(l7))),
        corr=float(np.corrcoef(preds, labels)[0, 1]),
        mae=float(np.mean(np.abs(preds - labels))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mosi', 'mosei'], required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1111)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--stage_epoch', type=int, default=50)
    ap.add_argument('--T', type=int, default=500)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--l2', type=float, default=1e-5)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--num_heads', type=int, default=2)
    ap.add_argument('--drop_rate', type=float, default=0.5)
    ap.add_argument('--attn_drop_rate', type=float, default=0.0)
    ap.add_argument('--lambda_msd', type=float, default=0.5)
    ap.add_argument('--lambda_cce', type=float, default=0.1)
    ap.add_argument('--lambda_emc', type=float, default=0.1)
    ap.add_argument('--lambda_imtd', type=float, default=0.1)
    args = ap.parse_args()

    args.mmsa_name = args.dataset
    args.dataset = 'CMUMOSI' if args.mmsa_name == 'mosi' else 'CMUMOSEI'
    args.n_classes = 1; args.n_speakers = 2; args.no_cuda = False
    args.add_noise = False; args.guassian_sigma = 0.0
    args.test_condition = 'atv'; args.p_drop = 0.0
    args.frame_seq = True   # <-- opt-in sequence readout in ebmc.py
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    seed_all(args.seed); print(args)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'missing_rate', 'results_seq')
    os.makedirs(out_dir, exist_ok=True)

    tr, tr_ds = loader(args.mmsa_name, 'train', 'train', 0.0, args.seed, args.batch_size, args.T, True)
    va, _ = loader(args.mmsa_name, 'valid', 'eval', VAL_RATE, args.seed, args.batch_size, args.T, False)
    te, te_ds = loader(args.mmsa_name, 'test', 'eval', 0.0, args.seed, args.batch_size, args.T, False)
    adim, tdim, vdim = tr_ds.get_featDim()
    print(f'adim={adim} tdim={tdim} vdim={vdim} T={args.T} train={len(tr_ds)}')

    model = build_model(args, adim, tdim, vdim).to(args.device)
    reg = MaskedMSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_f1, best_state = -1.0, None
    for ep in range(args.epochs):
        first = ep < args.stage_epoch
        t0 = time.time()
        train_epoch(args, model, reg, opt, tr, first)
        if first and ep == args.stage_epoch - 1:
            sd = os.path.join(config.MODEL_DIR, 'stage_1'); os.makedirs(sd, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(sd, f'{args.dataset}_{args.test_condition}_stage1_best.pth'))
            print('[stage-I] teacher saved')
        if not first:
            m = evaluate(args, model, va)
            tag = ''
            if m['f1'] > best_f1:
                best_f1 = m['f1']; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; tag = '  <-- best'
            print(f"ep {ep:03d} val@{VAL_RATE} acc2={m['acc2']:.4f} f1={m['f1']:.4f} mae={m['mae']:.4f} ({time.time()-t0:.1f}s){tag}")
        else:
            print(f"ep {ep:03d} stage-I ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'[selected] val f1={best_f1:.4f}')

    header = (f"{'rate':>5} | {'Acc2_h0':>8} {'Acc2_n0':>8} {'F1_h0':>8} {'F1_n0':>8} {'Acc-7':>7} {'Corr':>7} {'MAE':>7}")
    fmt = lambda r, m: (f"{r:>5} | {m['acc2_has0']*100:8.2f} {m['acc2']*100:8.2f} {m['f1_has0']*100:8.2f} "
                        f"{m['f1']*100:8.2f} {m['acc7']*100:7.2f} {m['corr']:7.4f} {m['mae']:7.4f}")
    print('====== TEST missing-rate sweep (SEQ) ======'); print(header)
    rows = []
    for p in RATES:
        te.dataset.missing_rate = p; te.dataset.mode = 'eval'
        m = evaluate(args, model, te); rows.append((p, m)); print(fmt(f'{p:.1f}', m))
    keys = ['acc2', 'f1', 'acc2_has0', 'f1_has0', 'acc7', 'corr', 'mae']
    avg = {k: float(np.mean([m[k] for _, m in rows])) for k in keys}
    print(fmt('avg', avg))
    with open(os.path.join(out_dir, f'seq_{args.mmsa_name}_seed{args.seed}.txt'), 'w') as f:
        f.write(header + '\n')
        for p, m in rows:
            f.write(fmt(f'{p:.1f}', m) + '\n')
        f.write(fmt('avg', avg) + '\n')
    print(f'[saved] {out_dir}/seq_{args.mmsa_name}_seed{args.seed}.txt')


if __name__ == '__main__':
    main()
