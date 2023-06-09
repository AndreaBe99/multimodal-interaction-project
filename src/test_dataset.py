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
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

df['file_path'] = df.apply(lambda x: os.path.join(data_dir, 'imgs/train', x.classname, x.img), axis=1)

df['class_num'] = df['classname'].map(lambda x: int(x[1]))
print(df.head(5))

class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(-10, 10),
                A.Resize(input_size, input_size),
                A.Normalize(color_mean, color_std), 
                ToTensorV2()
            ]),
            'val': A.Compose([
                A.Resize(input_size, input_size),
                A.Normalize(color_mean, color_std),
                ToTensorV2()
            ])
        }

    def __call__(self, phase, image):
        transformed = self.data_transform[phase](image=image)
        return transformed['image']

class Dataset(data.Dataset):

    def __init__(self, df, phase, transform):
        self.df = df
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = self.pull_item(index)
        return image, self.df.iloc[index]['class_num']

    def pull_item(self, index):
        image_path = self.df.iloc[index]['file_path']
        image = cv2.imread(image_path)[:, :, (2, 1, 0)]

        return self.transform(self.phase, image)

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
input_size = 256

df_train, df_val = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df['subject'], random_state=SEED)

datamodule = pl.LightningDataModule.from_datasets(
    train_dataset = Dataset(df_train, phase="train", transform=DataTransform(input_size=input_size, color_mean=color_mean, color_std=color_std)),
    val_dataset = Dataset(df_val, phase="val", transform=DataTransform(input_size=input_size, color_mean=color_mean, color_std=color_std)),
    batch_size=64 if torch.cuda.is_available() else 4,
    num_workers=int(os.cpu_count()/2)
)

image = datamodule.train_dataloader().dataset[0]
print(image[0].shape)

plt.imshow(image[0].permute(1, 2, 0))
plt.title(image[1])
plt.show()