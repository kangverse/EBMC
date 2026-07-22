"""
CMU-MOSI / CMU-MOSEI Dataloader for EBMC (Enhance-then-Balance Modality Collaboration).

Datasets:
  - CMU-MOSI : ~2,199 utterances from 93 YouTube opinion videos (single speaker)
  - CMU-MOSEI: ~23,453 utterances from 3,228 videos (single speaker)
  - Task      : Sentiment regression, labels ∈ [-3, 3]
  - Evaluation: Non-zero binary classification (positive/negative, zero excluded)
  
Since CMU-MOSI/MOSEI are monologue datasets (single speaker per video),
only the Host stream carries real features; the Guest stream is zero-padded.
This preserves the same dual-stream interface as the IEMOCAP loader,
allowing EBMC's encoder to handle both dyadic and monologue settings uniformly.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def read_data(label_path, feature_root):
    """Load utterance-level features and build a name→feature mapping.

    For CMU-MOSI/MOSEI, there is a single speaker per video, so features
    are stored as flat .npy files (no speaker-split directory structure).
    Multi-frame visual features (face crops) are mean-pooled into a single
    utterance-level vector.

    Args:
        label_path   : Path to the CMU-MOSI/MOSEI .pkl annotation file.
        feature_root : Root directory containing per-utterance feature files.

    Returns:
        name2feats : dict mapping utterance ID → feature array [D]
        feature_dim: int, feature dimensionality.
    """
    # Collect all utterance IDs across train / val / test splits
    names = []
    videoIDs, videoLabels, videoSpeakers, videoSentences, \
    trainVids, valVids, testVids = pickle.load(open(label_path, "rb"), encoding='latin1')

    for vid in videoIDs:
        names.extend(videoIDs[vid])

    # Load each utterance feature; mean-pool multi-frame visual features
    features = []
    feature_dim = -1
    for name in names:
        feature = []
        feature_path = os.path.join(feature_root, name + '.npy')
        feature_dir  = os.path.join(feature_root, name)

        if os.path.exists(feature_path):
            # Single .npy file: audio or text utterance-level embedding
            single_feat = np.load(feature_path).squeeze()  # [D] or [T, D]
            feature.append(single_feat)
            feature_dim = max(feature_dim, single_feat.shape[-1])
        else:
            # Directory of per-frame face features: mean-pool over frames
            for facename in sorted(os.listdir(feature_dir)):
                facefeat = np.load(os.path.join(feature_dir, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                feature.append(facefeat)

        # Aggregate frames → single utterance vector
        single_feat = np.array(feature).squeeze()
        if len(single_feat) == 0:
            single_feat = np.zeros((feature_dim,))   # missing feature → zero vector
        elif single_feat.ndim == 2:
            single_feat = np.mean(single_feat, axis=0)  # temporal mean pooling
        features.append(single_feat)

    print(f'Input feature {os.path.basename(feature_root)} ===> '
          f'dim is {feature_dim}; No. sample is {len(names)}')
    assert len(names) == len(features)
    name2feats = {names[i]: features[i] for i in range(len(names))}
    return name2feats, feature_dim


class CMUMOSIDataset(Dataset):
    """PyTorch Dataset for CMU-MOSI / CMU-MOSEI sentiment regression.

    Returns per-video utterance sequences for EBMC's dual-stream encoder.
    Each sample is a full video with:
      - Host stream: audio, text, visual features  [T, D_a/D_t/D_v]
      - Guest stream: zero-padded (monologue → no second speaker) [T, D_a/D_t/D_v]
      - Speaker mask (qmask): all zeros (single speaker) [T]
      - Utterance mask (umask): all ones [T]
      - Labels: continuous sentiment score per utterance ∈ [-3, 3] [T]
    """

    def __init__(self, label_path, audio_root, text_root, video_root):
        # Load pre-extracted utterance-level features for all three modalities
        name2audio, self.adim = read_data(label_path, audio_root)
        name2text,  self.tdim = read_data(label_path, text_root)
        name2video, self.vdim = read_data(label_path, video_root)

        self.max_len = -1
        self.videoAudioHost   = {}
        self.videoTextHost    = {}
        self.videoVisualHost  = {}
        self.videoAudioGuest  = {}   # zero-padded (no second speaker)
        self.videoTextGuest   = {}
        self.videoVisualGuest = {}
        self.videoLabelsNew   = {}
        self.videoSpeakersNew = {}

        self.videoIDs, self.videoLabels, self.videoSpeakers, \
        self.videoSentences, self.trainVids, self.valVids, self.testVids = \
            pickle.load(open(label_path, "rb"), encoding='latin1')

        # Preserve split ordering: train → val → test
        self.vids = (
            [vid for vid in sorted(self.trainVids)] +
            [vid for vid in sorted(self.valVids)]   +
            [vid for vid in sorted(self.testVids)]
        )

        for vid in sorted(self.videoIDs):
            uids     = self.videoIDs[vid]
            labels   = self.videoLabels[vid]
            speakers = self.videoSpeakers[vid]

            self.max_len = max(self.max_len, len(uids))

            # Single-speaker: Host carries all features; Guest is zero-padded
            self.videoAudioHost[vid]   = np.array([name2audio[uid] for uid in uids])
            self.videoTextHost[vid]    = np.array([name2text[uid]  for uid in uids])
            self.videoVisualHost[vid]  = np.array([name2video[uid] for uid in uids])
            self.videoAudioGuest[vid]  = np.zeros((len(uids), self.adim))
            self.videoTextGuest[vid]   = np.zeros((len(uids), self.tdim))
            self.videoVisualGuest[vid] = np.zeros((len(uids), self.vdim))
            self.videoLabelsNew[vid]   = np.array(list(labels))
            # Single speaker: speaker map is a constant (no role distinction)
            speakermap = {s: 0 for s in set(speakers)}
            self.videoSpeakersNew[vid] = np.array([speakermap[s] for s in speakers])

    def __getitem__(self, index):
        """Return one video sample.

        Returns (in order):
            audio_host   [T, D_a]  – Host audio features (wav2vec-large-c-UTT)
            text_host    [T, D_t]  – Host text features (deberta-large-4-UTT)
            visual_host  [T, D_v]  – Host visual features (manet_UTT)
            audio_guest  [T, D_a]  – Guest audio (zeros for monologue)
            text_guest   [T, D_t]  – Guest text  (zeros for monologue)
            visual_guest [T, D_v]  – Guest visual (zeros for monologue)
            qmask        [T]       – Speaker index (all 0 for single speaker)
            umask        [T]       – Utterance validity mask (all ones)
            labels       [T]       – Sentiment score per utterance (FloatTensor)
            vid                    – Video ID string
        """
        vid = self.vids[index]
        return (
            torch.FloatTensor(self.videoAudioHost[vid]),
            torch.FloatTensor(self.videoTextHost[vid]),
            torch.FloatTensor(self.videoVisualHost[vid]),
            torch.FloatTensor(self.videoAudioGuest[vid]),
            torch.FloatTensor(self.videoTextGuest[vid]),
            torch.FloatTensor(self.videoVisualGuest[vid]),
            torch.FloatTensor(self.videoSpeakersNew[vid]),           # qmask
            torch.FloatTensor([1] * len(self.videoLabelsNew[vid])),  # umask
            torch.FloatTensor(self.videoLabelsNew[vid]),             # labels (float for regression)
            vid
        )

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print(f'audio dim: {self.adim}; text dim: {self.tdim}; visual dim: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print(f'max sequence length: {self.max_len}')
        return self.max_len

    def collate_fn(self, data):
        """Collate a batch of videos by padding to the longest sequence.

        Fields 0-5  (A/T/V × Host/Guest) : padded along time dim with 0.
        Fields 6-8  (qmask, umask, labels): padded with batch_first=True.
        Field  9    (vid)                 : kept as a list of strings.
        """
        datnew = []
        dat = pd.DataFrame(data)
        for i in dat:
            if i <= 5:
                datnew.append(pad_sequence(dat[i]))           # [T_max, B, D], pad with 0
            elif i <= 8:
                datnew.append(pad_sequence(dat[i], True))     # [B, T_max], batch_first
            else:
                datnew.append(dat[i].tolist())                # video IDs
        return datnew
