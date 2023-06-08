import os
from glob import glob
import random
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


SEED = 42
fix_seed(SEED)

ACTIVITY = {
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

# state-farm-distracted-driver-detection dataset
data_dir = 'src/data/dataset/'
csv_file_path = os.path.join(data_dir, 'driver_imgs_list.csv')

df = pd.read_csv(csv_file_path)
print(df.head(4))

train_file_num = len(glob(os.path.join(data_dir, 'imgs/train/*/*.jpg'))) 
test_file_num = len(glob(os.path.join(data_dir, 'imgs/test/*.jpg'))) 
category_num = len(df['classname'].unique())
print('train_file_num: ', train_file_num)
print('test_file_num: ', test_file_num)
print('category_num: ', category_num)

fig, axs = plt.subplots(nrows=len(ACTIVITY.items())//5, ncols=len(ACTIVITY.items())//2, figsize=(20, 8))
for i, (key, value) in enumerate(ACTIVITY.items()):
    image_dir = os.path.join(data_dir, 'imgs/train', key, '*.jpg')
    image_path = glob(image_dir)[0]
    image = cv2.imread(image_path)[:, :, (2, 1, 0)]
    axs[i%2, i%5].imshow(image)
    axs[i%2, i%5].set_title(value)

# show fig
plt.show()
