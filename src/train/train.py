import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import timm
import torch
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from src.train.config import StaticTrain as st
from src.train.config import StaticDataset as sd
from src.train.config import StaticLearningParameter as slp
from src.train.dataset import Dataset, DataTransform
from src.train.model import LitEfficientNet

PLOT = False  # Set True if you want to plot the image

def fix_seed(seed):
    """Fix the seed of the random number generator for reproducibility."""
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_dataset(
        seed=sd.SEED.value,
        input_size=sd.INPUT_SIZE.value,
        color_mean=sd.COLOR_MEAN.value,
        color_std=sd.COLOR_STD.value):
    """Create dataset for training and validation.

    Args:
        seed (int, optional): Random seed. 
            Defaults to StaticTrain.SEED.value.
        input_size (int, optional): Input size. 
            Defaults to StaticDataTransform.INPUT_SIZE.value.
        color_mean (tuple, optional): Color mean. 
            Defaults to StaticDataTransform.COLOR_MEAN.value.
        color_std (tuple, optional): Color standard deviation. 
            Defaults to StaticDataTransform.COLOR_STD.value.
        
    Returns:
        train_dataset, val_dataset: Dataset for training and validation.
    """
    # data division
    df_train, df_val = train_test_split(
        df,
        stratify=df['subject'],
        random_state=seed
    )

    # dataset creation
    train_dataset = Dataset(
        df_train,
        phase="train",
        transform=DataTransform(
            input_size=input_size,
            color_mean=color_mean,
            color_std=color_std
        )
    )

    val_dataset = Dataset(
        df_val,
        phase="val",
        transform=DataTransform(
            input_size=input_size,
            color_mean=color_mean,
            color_std=color_std
        )
    )

    if PLOT:
        # Data retrieval example
        image, label = train_dataset[0]
        plt.imshow(image.permute(1, 2, 0))
        plt.title(label)
        plt.show()
    
    return train_dataset, val_dataset


def create_dataloader(
        batch_size=sd.BATCH_SIZE.value):
    """
    Create dataloader for training and validation.

    Args:
        batch_size (int, optional): Batch size. 
            Defaults to StaticDataLoader.BATCH_SIZE.value.
    
    Returns:
        dataloaders_dict: Dictionary object containing dataloader for training 
            and validation.
    """
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False)

    # group into a dictionary object
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # Operation confirmation
    if PLOT:
        batch_iterator = iter(dataloaders_dict["val"])  # convert to iterator
        images, labels = next(batch_iterator)  # get the first element
        print(images.size())  # torch.Size([8, 3, 256, 256])
        print(labels.size())  # torch.Size([8])
    
    return dataloaders_dict


def train(
    train_dataset, 
    val_dataset, 
    class_names=st.ACTIVITY_MAP.items()):
    """Train the model with Pytorch Lightning.
    
    Args:
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation.
    """
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=16 if torch.cuda.is_available() else 4,
        num_workers=int(os.cpu_count() / 2),
    )

    # Create model
    efficient_model = timm.create_model(
        slp.MODEL_NAME_0.value, 
        pretrained=True, 
        num_classes=len(class_names)
    )
    
    model = LitEfficientNet(efficient_model)

    trainer = Trainer(
        max_epochs=slp.EPOCHS.value,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    

if __name__ == "__main__":
    
    fix_seed(slp.SEED.value)
    
    # Read csv file
    df = pd.read_csv(sd.CSV_FILE_PATH.value)
    
    ### Operation check ###
    train_dataset, val_dataset = create_dataset()
    
    ### Create DataLoader ###
    dataloaders_dict = create_dataloader()
    
    ### Train the model ###    
    train(train_dataset=train_dataset, val_dataset=val_dataset)

