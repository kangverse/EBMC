"""
MMSA (MOSI/MOSEI) **sequence-level** dataloader for EBMC — missing-RATE, UNALIGNED.

Alignment with LNLN `core/dataset.py generate_m`:
  - per modality, independent Bernoulli keep with prob (1-p) within valid length;
  - dropped frames are ZEROED (kept in the sequence, still attended as zeros);
  - text keeps CLS (idx 0) and SEP (idx L-1);
  - padding beyond valid length is masked out of attention.

All three modalities are padded/truncated to a common length T (EBMC fuses by
concatenating along the feature dim, which needs a shared sequence length). The
per-frame validity mask (1 within valid length, 0 for padding; missing frames
stay 1) is provided so EBMC's attention masks only the padding.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

_DIMS = {'mosi': (5, 20), 'mosei': (74, 35)}   # (audio_dim, vision_dim); text=768


def _seq_missing(seq, valid_len, p, rng, T, keep_ends=False):
    """Return (feat[T,D], mask[T]) with LNLN-style frame missing + padding.

    The frame-drop scheme (keep iff uniform>p within valid length, zero dropped
    frames, keep CLS/SEP for text) is FROM LNLN (core/dataset.py, generate_m).

    feat : dropped/padding frames zeroed; text UNK not modeled (features zeroed).
    mask : 1 for frames within valid length (incl. missing, attended as zeros),
           0 for padding beyond valid length.
    """
    D = seq.shape[1]
    L = int(max(valid_len, 1)); L = min(L, seq.shape[0])
    keep = (rng.uniform(size=L) > p).astype(np.float32)
    if keep_ends:
        keep[0] = 1.0
        keep[L - 1] = 1.0
    feat = np.zeros((T, D), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    Lc = min(L, T)
    feat[:Lc] = seq[:Lc] * keep[:Lc, None]   # zero dropped frames
    mask[:Lc] = 1.0                          # valid region attended (missing incl.)
    return feat, mask


class MMSASeqDataset(Dataset):
    def __init__(self, data_path, dataset_name, split, mode='eval',
                 missing_rate=0.0, seed=1111, T=500):
        assert dataset_name in _DIMS
        self.adim, self.vdim = _DIMS[dataset_name]
        self.tdim = 768
        self.mode = mode
        self.missing_rate = float(missing_rate)
        self.seed = int(seed)
        self.T = int(T)

        with open(data_path, 'rb') as f:
            d = pickle.load(f)[split]
        self.audio = d['audio'].astype(np.float32)
        self.vision = d['vision'].astype(np.float32)
        self.text = d['text'].astype(np.float32)
        self.audio_len = list(d['audio_lengths'])
        self.vision_len = list(d['vision_lengths'])
        self.text_len = d['text_bert'][:, 1, :].sum(axis=1).astype(int)
        self.labels = d['regression_labels'].astype(np.float32)
        self.ids = [str(x) for x in d['id']]
        self.N = len(self.labels)

    def __len__(self):
        return self.N

    def _rates(self, index):
        if self.mode == 'eval':
            rng = np.random.RandomState(self.seed * 1_000_003 + index)
            p = self.missing_rate
            return (p, p, p), rng
        rng = np.random
        return tuple(0.0 if rng.rand() < 0.5 else float(rng.uniform(0, 1))
                     for _ in range(3)), rng

    def __getitem__(self, index):
        (pa, pt, pv), rng = self._rates(index)
        T = self.T
        a, am = _seq_missing(self.audio[index], self.audio_len[index], pa, rng, T)
        t, tm = _seq_missing(self.text[index], self.text_len[index], pt, rng, T, keep_ends=True)
        v, vm = _seq_missing(self.vision[index], self.vision_len[index], pv, rng, T)
        return a, t, v, am, tm, vm, self.labels[index], self.ids[index]

    def get_featDim(self):
        return self.adim, self.tdim, self.vdim


def collate_seq(batch):
    """Build EBMC frame-level batch.

    Returns:
        feats  : (audio_host, text_host, visual_host) each [T, B, dim]
        fmask  : (audio_m, text_m, visual_m) each [T, B, 1] per-frame validity
        qmask  : [B, T] zeros (single speaker -> host stream)
        umask  : [B, T] any-modality validity (for _prepare_inputs)
        label  : [B, 1] one label per sample
        ids    : list
    """
    a, t, v, am, tm, vm, y, ids = zip(*batch)
    to = lambda arr: torch.from_numpy(np.stack(arr)).permute(1, 0, 2)  # [B,T,D]->[T,B,D]
    audio_host, text_host, visual_host = to(a), to(t), to(v)
    B, T = len(a), audio_host.size(0)
    m = lambda arr: torch.from_numpy(np.stack(arr)).transpose(0, 1).unsqueeze(-1)  # [T,B,1]
    am_, tm_, vm_ = m(am), m(tm), m(vm)
    umask = (torch.from_numpy(np.stack(am)) + torch.from_numpy(np.stack(tm))
             + torch.from_numpy(np.stack(vm)) > 0).float()      # [B,T]
    qmask = torch.zeros(B, T, dtype=torch.float32)
    label = torch.tensor(y, dtype=torch.float32).view(B, 1)
    z = torch.zeros_like
    return dict(audio_host=audio_host, text_host=text_host, visual_host=visual_host,
                audio_guest=z(audio_host), text_guest=z(text_host), visual_guest=z(visual_host),
                audio_m=am_, text_m=tm_, visual_m=vm_,
                aguest_m=z(am_), tguest_m=z(tm_), vguest_m=z(vm_),
                qmask=qmask, umask=umask, label=label, ids=list(ids))
