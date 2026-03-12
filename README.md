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

| Dataset | Task | Download |
|:---:|:---:|:---:|
| IEMOCAP (4-class) | Emotion Recognition | [Google Drive](https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link) |
| CMU-MOSI | Sentiment Analysis | [Google Drive](https://drive.google.com/file/d/1aJxArYfZsA-uLC0sOwIkjl_0ZWxiyPxj/view?usp=share_link) |
| CMU-MOSEI | Sentiment Analysis | [Google Drive](https://drive.google.com/file/d/1L6oDbtpFW2C4MwL5TQsEflY1WHjtv7L5/view?usp=share_link) |

After downloading, organise as:

```
dataset/
├── CMUMOSI/
│   ├── CMUMOSI_features_raw_2way.pkl
│   └── features/
│       ├── wav2vec-large-c-UTT/    # audio features  (dim=512)
│       ├── deberta-large-4-UTT/    # text features   (dim=1024)
│       └── manet_UTT/              # visual features (dim=512)
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
# GPU 0: conditions a, at, atv
for cond in a at atv; do
    python -u EBMC/train_EBMC.py --dataset=CMUMOSEI \
        --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT \
        --seed=66 --batch-size=32 --epochs=200 --lr=1e-4 --hidden=256 --depth=4 --num_heads=2 \
        --drop_rate=0.6 --attn_drop_rate=0.0 --test_condition=$cond --stage_epoch=100 --gpu=0 \
        --lambda_msd=0.5 --lambda_cce=0.1 --lambda_emc=0.1 --lambda_imtd=0.1
done
```

### IEMOCAP

```bash
bash run_ebmc_iemocap.sh
```
---

## Citation

```bibtex
@inproceedings{ebmc2026cvpr,
  title     = {Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
