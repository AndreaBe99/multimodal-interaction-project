from enum import Enum
import os.path as osp


class StaticDataset(Enum):
    """Class for static training configuration constants"""
    ACTIVITY_MAP = {
        'c0': 'Safe driving',
        'c1': 'Texting - right',
        'c2': 'Talking on the phone - right',
        'c3': 'Texting - left',
        'c4': 'Talking on the phone - left',
        'c5': 'Operating the radio',
        'c6': 'Drinking',
        'c7': 'Reaching behind',
        'c8': 'Hair and makeup',
        'c9': 'Talking to passenger'
    }
    DATA_DIR = 'src/data/dataset/state-farm-distracted-driver-detection/'
    CSV_FILE_PATH = osp.join(DATA_DIR, 'driver_imgs_list.csv')


class StaticLearningParameter(Enum):
    MODEL_NAME_0 = 'efficientnet-b0'
    MODEL_NAME_3 = 'efficientnet-b3'
    COLOR_MEAN = (0.485, 0.456, 0.406)
    COLOR_STD = (0.229, 0.224, 0.225)
    INPUT_SIZE = 256
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    EPOCHS = 10
    FOLDS = 5
    LR = 1e-3
    GAMMA = 0.98
    DEBUG = True
    TRAIN = False
    SEED = 42
