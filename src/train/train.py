import os
import sys
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

sys.path.append("./")
from src.train.config import StaticDataset as sd
from src.train.config import StaticLearningParameter as slp
from src.train.dataset import Dataset, DataTransform
from src.train.model import LitEfficientNet

VERBOSE = True  # Set True if you want to see some logs
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
    df,
    seed=slp.SEED.value,
    input_size=slp.INPUT_SIZE.value,
    color_mean=slp.COLOR_MEAN.value,
    color_std=slp.COLOR_STD.value,
):
    """Create dataset for training and validation.

    Args:
        df: Dataframe containing image file paths and labels.
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
    df_train, df_val = train_test_split(df, stratify=df["subject"], random_state=seed)
    if VERBOSE:
        print("train num:", len(df_train))
        print("val num:", len(df_val))

    # dataset creation
    train_dataset = Dataset(
        df_train,
        phase="train",
        transform=DataTransform(
            input_size=input_size, color_mean=color_mean, color_std=color_std
        ),
    )
    val_dataset = Dataset(
        df_val,
        phase="val",
        transform=DataTransform(
            input_size=input_size, color_mean=color_mean, color_std=color_std
        ),
    )

    if VERBOSE:
        print("train dataset size:", len(train_dataset))
        print("val dataset size:", len(val_dataset))

    if PLOT:
        # Data retrieval example
        image, label = train_dataset[0]
        plt.imshow(image.permute(1, 2, 0))
        plt.title(label)
        plt.show()

    return train_dataset, val_dataset


def create_datamodule(train_dataset, val_dataset, batch_size=slp.BATCH_SIZE.value):
    """
    Create dataloader for training and validation.

    Args:
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation.
        batch_size (int, optional): Batch size.
            Defaults to StaticDataLoader.BATCH_SIZE.value.

    Returns:
        datamodule: Dataloader for training and validation.
    """
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size if torch.cuda.is_available() else 4,
        num_workers=int(os.cpu_count()),
    )

    if VERBOSE:
        print("data module created")

    return datamodule


def train(datamodule, class_names=sd.ACTIVITY_MAP.value.items()):
    """Train the model with Pytorch Lightning.

    Args:
        datamodule: Dataloader for training and validation.
        class_names (list, optional): List of class names.

    """
    # Create model
    # NOTE: MODEL_NAME_0 = "efficientnet_b0"
    # NOTE: MODEL_NAME_3 = "efficientnet_b3" this is used in the reference code
    efficient_model = timm.create_model(
        slp.MODEL_NAME_0.value,
        pretrained=True,
        num_classes=len(class_names),
    )

    model = LitEfficientNet(efficient_model, lr=slp.LR.value)

    trainer = Trainer(
        max_epochs=slp.EPOCHS.value,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
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

    ### Preprocessing ###
    # Add file path column
    df["file_path"] = df.apply(
        lambda x: os.path.join(sd.DATA_DIR.value, "imgs/train", x.classname, x.img),
        axis=1,
    )
    # Add Column by Converting Correct Answer Labels to Numbers
    df["class_num"] = df["classname"].map(lambda x: int(x[1]))

    ### Operation check ###
    train_dataset, val_dataset = create_dataset(df)

    ### Create DataLoader ###
    datamodule = create_datamodule(train_dataset, val_dataset)

    if VERBOSE:
        image = datamodule.train_dataloader().dataset[0]
        print(image[0].shape)

    ### Train the model ###
    train(datamodule)
