# EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis

## Overview

**EBMC** is a two-stage framework for robust multimodal sentiment analysis under missing-modality conditions. It addresses two complementary challenges:

1. **Enhance (Stage-I):** Improve individual modality representations before fusion.
2. **Balance (Stage-II):** Dynamically calibrate modality contributions based on reliability during inference.

## Environment

**Python 3.10+, CUDA 11.8+**

Core dependencies:

```
torch==2.4.0
torchaudio==2.4.0
torchvision==0.19.0
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.2
transformers
```

Install from the full environment snapshot:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Pre-extracted utterance-level features follow the preprocessing pipeline of [GCNet](https://github.com/zeroQiaoba/GCNet).

> **Scope of this repository (please read).**
> This repository covers the **main results (Tables 1–2)** and the **missing-modality** robustness experiments. All results here follow the [GCNet](https://github.com/zeroQiaoba/GCNet) convention.
> The main run scripts (`run_ebmc_*.sh` / `train_EBMC.py`) perform **modality-level** missingness — a whole modality is either present or absent (the seven conditions `a t v at av tv atv`) — with the utterance-level features above.

| Dataset | Task | Download |
|:---:|:---:|:---:|
| IEMOCAP (4-class) | Emotion Recognition | [Google Drive](https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link) |
| CMU-MOSI | Sentiment Analysis | [Google Drive](https://drive.google.com/file/d/1aJxArYfZsA-uLC0sOwIkjl_0ZWxiyPxj/view?usp=share_link) |
| CMU-MOSEI | Sentiment Analysis | [Google Drive](https://drive.google.com/file/d/1L6oDbtpFW2C4MwL5TQsEflY1WHjtv7L5/view?usp=share_link) |

>
> **This repository also includes the frame-level missing-rate experiments** (intra-modality missingness at rates `p ∈ {0, 0.1, …, 0.9}`) — see [Robustness under Different Missing Rates](#robustness-under-different-missing-rates-intra-modality-missingness). This part is implemented **directly on the EBMC model** (the four modules are unchanged; only the data path differs), **aligned to the [LNLN](https://github.com/Haoyu-ha/LNLN) evaluation protocol** (same metrics / missing semantics), using **MMSA sequence-level features** (with timestep information). It is **not** a re-implementation of the LNLN framework itself — see the note at the end of that section for the exact alignment boundary.

After downloading, organise as:

```
dataset/
├── CMUMOSI/
│   ├── CMUMOSI_features_raw_2way.pkl
│   └── features/
│       ├── wav2vec-large-c-UTT/
│       ├── deberta-large-4-UTT/
│       └── manet_UTT/
├── CMUMOSEI/
│   └── (same structure)
└── IEMOCAP/
    ├── IEMOCAP_features_raw_4way.pkl
    └── features/
        └── (same structure)
```

Then update the dataset root paths in [config.py](config.py):

```python
DATA_DIR = {
    'CMUMOSI':    '/path/to/your/dataset/CMUMOSI',
    'CMUMOSEI':   '/path/to/your/dataset/CMUMOSEI',
    'IEMOCAPFour':'/path/to/your/dataset/IEMOCAP',
    'IEMOCAPSix': '/path/to/your/dataset/IEMOCAP',
}
```

Also update `SAVED_ROOT` in `config.py` to set where models and logs are saved.

---

## Running EBMC

All scripts are run from the **project root** (where `config.py` lives).

### CMU-MOSI & CMU-MOSEI

```bash
bash run_ebmc_cmumosi.sh
bash run_ebmc_cmumosei.sh
```

The script iterates over all 7 missing-modality conditions (`a t v at av tv atv`) sequentially. To run conditions in parallel across GPUs:

```bash
# GPU 0: conditions a, at, av, atv
for cond in a at av atv; do
    python -u EBMC/train_EBMC.py --dataset=CMUMOSEI \
        --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT \
        --seed=66 --batch-size=32 --epochs=200 --lr=1e-4 --hidden=256 --depth=4 --num_heads=2 \
        --drop_rate=0.6 --attn_drop_rate=0.0 --test_condition=$cond --stage_epoch=100 --gpu=0 \
        --lambda_msd=0.5 --lambda_cce=0.1 --lambda_emc=0.1 --lambda_imtd=0.1
done &

# GPU 1: conditions t, v, tv
for cond in t v tv; do
    python -u EBMC/train_EBMC.py --dataset=CMUMOSEI \
        --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT \
        --seed=66 --batch-size=32 --epochs=200 --lr=1e-4 --hidden=256 --depth=4 --num_heads=2 \
        --drop_rate=0.6 --attn_drop_rate=0.0 --test_condition=$cond --stage_epoch=100 --gpu=1 \
        --lambda_msd=0.5 --lambda_cce=0.1 --lambda_emc=0.1 --lambda_imtd=0.1
done &

wait
```

### IEMOCAP

```bash
bash run_ebmc_iemocap.sh
```
---

## Robustness under Different Missing Rates (intra-modality missingness)

The scripts above test **modality-level** missingness (a whole modality present/absent, the seven `test_condition`s). This section corresponds to the paper's Q3 **frame-level missing-rate** experiment: following LNLN's practice, on each modality we randomly drop frame-level features at a missing rate `p` to simulate intra-modality missingness, evaluating `p` from 0 to 0.9 and averaging over all rates.

The EBMC model and its four modules (MSD/CCE/EMC/IMTD) remain unchanged; only the data path differs. The frame sequence is first processed by EBMC's Soft-MoE Transformer (with per-frame attention masking), then a single masked-mean frame→sample readout collapses it into one prediction. Script: `EBMC/train_ebmc_seq.py`.

### Data (MMSA)

Uses the **MMSA `unaligned_50.pkl`** features (download via [MMSA](https://github.com/thuiar/MMSA)); each sample keeps per-frame sequences with valid `*_lengths`:

| Dataset | text | audio (COVAREP) | visual (Facet) |
|:---:|:---:|:---:|:---:|
| CMU-MOSI  | BERT `[50,768]` | `[375,5]` | `[500,20]` |
| CMU-MOSEI | BERT `[50,768]` | `[500,74]` | `[500,35]` |

Place the data at `dataset/MMSA/{MOSI,MOSEI}/unaligned_50.pkl` under the repo root (the default), or point to another location with the `MMSA_ROOT` environment variable:

```bash
export MMSA_ROOT=/path/to/your/MMSA   # optional; defaults to ./dataset/MMSA
```

Extra dependencies beyond the Environment section: `scikit-learn`, `pandas`, `einops`.

### Run

```bash
python -u EBMC/train_ebmc_seq.py --dataset=mosi  --gpu=0 --seed=1111 \
    --epochs=100 --stage_epoch=50 --batch-size=16 --T=500
python -u EBMC/train_ebmc_seq.py --dataset=mosei --gpu=0 --seed=1111 \
    --epochs=100 --stage_epoch=50 --batch-size=16 --T=500
```

Multiple seeds (report the mean over seeds `1111 1112 1113`). When running seeds in parallel, give each run its own `EBMC_SAVED_ROOT` so the Stage-II teacher checkpoints do not overwrite each other:

```bash
for s in 1111 1112 1113; do
    EBMC_SAVED_ROOT=$PWD/saved_seq_mosi_$s python -u EBMC/train_ebmc_seq.py \
        --dataset=mosi --gpu=0 --seed=$s --epochs=100 --stage_epoch=50 --batch-size=16 --T=500
done
```

`T` is the unified number of timesteps to which the frame sequences are padded. Per-rate results are written to `EBMC/missing_rate/results_seq/`, one file per dataset/seed, with the `avg` row being the mean over the 10 rates.

### Note on the comparison (please read)
This adapts EBMC to a sequence-level missing-rate evaluation. The metric formulas, missing semantics, and evaluation flow all follow LNLN, but it is not fully identical to LNLN: all modalities are padded to a common length `T`, and a single masked-mean readout is added to collapse frames into one prediction; text uses the pre-extracted BERT features (dropped frames zeroed) rather than passing UNK tokens through BERT.

Please also note: for easier reproduction and a unified codebase, we have integrated the missing-rate code into the EBMC framework and partially refactored the earlier LNLN-based Q3 experiment. As a result, the reproduced numbers differ slightly from those reported in the paper.

---

## Citation

```bibtex
@inproceedings{he2026enhance,
  title={Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis},
  author={He, Kang and Ding, Yuzhe and Wang, Xinrong and Li, Fei and Teng, Chong and Ji, Donghong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={30183--30193},
  year={2026}
}
```

---

## Acknowledgements

This implementation refers to and reuses several excellent open-source works, which we gratefully acknowledge:

- [LNLN](https://github.com/Haoyu-ha/LNLN): the evaluation protocol for the missing-rate robustness experiments, together with the frame-level masking (`generate_m`) and the metric computation (`core/metric.py`), are taken directly from this repository (marked with "from LNLN" comments in our code).
- [MoMKE](https://github.com/wxxv/MoMKE): its framework and training pipeline for incomplete multimodal learning served as a reference for this project.
- [GCNet](https://github.com/zeroQiaoba/GCNet): the utterance-level feature preprocessing follows its convention.
- [MMSA](https://github.com/thuiar/MMSA): the sequence-level features (`unaligned_50.pkl`) come from this dataset toolkit.

We thank the authors of these works for their open-source contributions.

We also thank Zhouyi for the valuable feedback, which prompted us to promptly add the previously missing code and to further improve this repository.

## License

Released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license. Local, non-commercial academic reproduction is permitted; commercial use is not. See [LICENSE](LICENSE) for the full terms.
