# *_*coding:utf-8 *_*
import os
import sys
import socket

import torch

## gain linux ip
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('10.0.0.1',8080))
        ip= s.getsockname()[0]
    finally:
        s.close()
    return ip

############ For LINUX ##############
# path
# NOTE: Update these paths to point to your local dataset directory.
# Each dataset folder should follow the structure described in README.md.
_DATASET_ROOT = os.environ.get('EBMC_DATASET_ROOT', os.path.join(os.path.dirname(__file__), 'dataset'))

DATA_DIR = {
	'CMUMOSI': os.path.join(_DATASET_ROOT, 'CMUMOSI'),
	'CMUMOSEI': os.path.join(_DATASET_ROOT, 'CMUMOSEI'),
	'IEMOCAPSix': os.path.join(_DATASET_ROOT, 'IEMOCAP'),
	'IEMOCAPFour': os.path.join(_DATASET_ROOT, 'IEMOCAP'),
}
PATH_TO_RAW_AUDIO = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subaudio'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
}
PATH_TO_RAW_FACE = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subvideofaces'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideofaces'),
}
PATH_TO_TRANSCRIPTIONS = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription.csv'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription.csv'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'transcription.csv'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'features'),
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}

# dir
# NOTE: Update SAVED_ROOT to your preferred output directory, or set env var EBMC_SAVED_ROOT.
SAVED_ROOT = os.environ.get('EBMC_SAVED_ROOT', os.path.join(os.path.dirname(__file__), 'saved'))
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
WEIGHT_DIR = os.path.join(SAVED_ROOT, 'weight')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
NPZ_DIR = os.path.join(SAVED_ROOT, 'npz')
PRE_TRAINED_DIR = os.path.join(SAVED_ROOT, 'pre_trained')
