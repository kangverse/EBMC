"""
IEMOCAP Dataloader for EBMC (Enhance-then-Balance Modality Collaboration).

Dataset: IEMOCAP (4-class / 6-class emotion recognition)
  - Emotion classes (4-class): neutral, happy/excited, angry, sad


Each conversation (video) is represented as a sequence of utterances.
For each utterance, features are split into Host (F) and Guest (M) streams,
following the conversational role assignment used in EBMC.
The Host/Guest structure feeds into the dual-stream encoder of EBMC,
enabling cross-speaker contextual modeling.
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
    """Load utterance-level features from disk and build a name→feature mapping.

    For IEMOCAP, each utterance has features for both speaker roles (F/M).
    If the feature is stored as a .npy file, it belongs to the active speaker.
    If stored as a directory (e.g., face images), sub-files are separated by
    speaker label in the filename ('F' or 'M').

    Args:
        label_path   : Path to the IEMOCAP .pkl annotation file.
        feature_root : Root directory containing per-utterance feature files.

    Returns:
        name2feats : dict mapping utterance ID → {'F': feat_array, 'M': feat_array}
        feature_dim: int, feature dimensionality.
    """
    # Collect all utterance IDs and their speaker assignments
    names, speakers = [], []
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid = \
        pickle.load(open(label_path, "rb"), encoding='latin1')

    vids = sorted(list(trainVid | testVid))
    for vid in vids:
        names.extend(videoIDs[vid])
        speakers.extend(videoSpeakers[vid])

    # Load each utterance feature; assign to the correct speaker slot (F or M)
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        speaker = speakers[ii]
        feature_dir = glob.glob(os.path.join(feature_root, name + '*'))
        assert len(feature_dir) == 1, f"Expected exactly one match for '{name}', got {len(feature_dir)}"
        feature_path = feature_dir[0]

        feature = {'F': [], 'M': []}
        if feature_path.endswith('.npy'):
            # Audio / text feature: single .npy file, attributed to the active speaker
            single_feat = np.load(feature_path).squeeze()  # [D] or [T, D]
            feature[speaker].append(single_feat)
            feature_dim = max(feature_dim, single_feat.shape[-1])
        else:
            # Visual feature: directory of per-face .npy files; speaker inferred from filename
            for facename in sorted(os.listdir(feature_path)):
                assert 'F' in facename or 'M' in facename, \
                    f"Cannot infer speaker from face filename: {facename}"
                facefeat = np.load(os.path.join(feature_path, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                feature['F' if 'F' in facename else 'M'].append(facefeat)

        # Aggregate multiple frames → single utterance vector (mean pooling)
        for spk in feature:
            feat = np.array(feature[spk]).squeeze()
            if len(feat) == 0:
                feat = np.zeros((feature_dim,))          # missing speaker → zero vector
            elif feat.ndim == 2:
                feat = np.mean(feat, axis=0)             # temporal mean pooling
            feature[spk] = feat
        features.append(feature)

    print(f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features)
    name2feats = {names[i]: features[i] for i in range(len(names))}
    return name2feats, feature_dim


class IEMOCAPDataset(Dataset):
    """PyTorch Dataset for IEMOCAP emotion recognition.

    Returns per-conversation sequences for EBMC's dual-stream encoder.
    Each sample is a full conversation with:
      - Host stream (F): audio, text, visual features  [T, D_a/D_t/D_v]
      - Guest stream (M): audio, text, visual features [T, D_a/D_t/D_v]
      - Speaker mask (qmask): speaker role per utterance [T]
      - Utterance mask (umask): valid frame indicator [T] (all ones here)
      - Labels: emotion class index per utterance [T]
    """

    def __init__(self, label_path, audio_root, text_root, video_root):
        # Load pre-extracted utterance-level features for all three modalities
        name2audio, self.adim = read_data(label_path, audio_root)
        name2text,  self.tdim = read_data(label_path, text_root)
        name2video, self.vdim = read_data(label_path, video_root)

        # Speaker role mapping: Female=0 (Host), Male=1 (Guest)
        speakermap = {'F': 0, 'M': 1}

        self.max_len = -1
        self.videoAudioHost   = {}
        self.videoTextHost    = {}
        self.videoVisualHost  = {}
        self.videoAudioGuest  = {}
        self.videoTextGuest   = {}
        self.videoVisualGuest = {}
        self.videoLabelsNew   = {}
        self.videoSpeakersNew = {}

        self.videoIDs, self.videoLabels, self.videoSpeakers, \
        self.videoSentences, self.trainVid, self.testVid = \
            pickle.load(open(label_path, "rb"), encoding='latin1')

        self.vids = sorted(list(self.trainVid | self.testVid))

        for vid in self.vids:
            uids     = self.videoIDs[vid]
            labels   = self.videoLabels[vid]
            speakers = self.videoSpeakers[vid]

            self.max_len = max(self.max_len, len(uids))

            # Build per-utterance feature sequences for both speaker streams
            self.videoAudioHost[vid]   = [name2audio[uid]['F'] for uid in uids]
            self.videoTextHost[vid]    = [name2text[uid]['F']  for uid in uids]
            self.videoVisualHost[vid]  = [name2video[uid]['F'] for uid in uids]
            self.videoAudioGuest[vid]  = [name2audio[uid]['M'] for uid in uids]
            self.videoTextGuest[vid]   = [name2text[uid]['M']  for uid in uids]
            self.videoVisualGuest[vid] = [name2video[uid]['M'] for uid in uids]
            self.videoLabelsNew[vid]   = list(labels)
            self.videoSpeakersNew[vid] = [speakermap[s] for s in speakers]

            # Convert to numpy arrays for efficient indexing
            self.videoAudioHost[vid]   = np.array(self.videoAudioHost[vid])
            self.videoTextHost[vid]    = np.array(self.videoTextHost[vid])
            self.videoVisualHost[vid]  = np.array(self.videoVisualHost[vid])
            self.videoAudioGuest[vid]  = np.array(self.videoAudioGuest[vid])
            self.videoTextGuest[vid]   = np.array(self.videoTextGuest[vid])
            self.videoVisualGuest[vid] = np.array(self.videoVisualGuest[vid])
            self.videoLabelsNew[vid]   = np.array(self.videoLabelsNew[vid])
            self.videoSpeakersNew[vid] = np.array(self.videoSpeakersNew[vid])

    def __getitem__(self, index):
        """Return one conversation sample.

        Returns (in order):
            audio_host   [T, D_a]  – Host (F) audio features
            text_host    [T, D_t]  – Host (F) text features
            visual_host  [T, D_v]  – Host (F) visual features
            audio_guest  [T, D_a]  – Guest (M) audio features
            text_guest   [T, D_t]  – Guest (M) text features
            visual_guest [T, D_v]  – Guest (M) visual features
            qmask        [T]       – Speaker role index (0=F, 1=M) per utterance
            umask        [T]       – Utterance validity mask (all ones)
            labels       [T]       – Emotion class label per utterance (LongTensor)
            vid                    – Video/conversation ID string
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
            torch.LongTensor(self.videoLabelsNew[vid]),              # labels
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
        """Collate a batch of conversations by padding to the longest sequence.

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
