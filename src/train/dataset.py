import cv2

import torch
import torch.transforms as transforms
import torch.utils.data as data

import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataTransform():
    """
    Image and annotation preprocessing classes. 
    It behaves differently during training and during validation.
    Set the image size to input_size x input_size.
    Data augmentation during training.


    Attributes
    ----------
    input_size : int
        The size of the resized image.
    color_mean : (R, G, B)
        Average value for each color channel.
    color_std : (R, G, B)
        Standard deviation for each color channel.
    """

    def __init__(
        self, 
        input_size, 
        color_mean, 
        color_std, 
        use_albumentations=False):
        if use_albumentations:
            # Albumentations Transformations
            self.data_transform = {
                # Implement only train
                'train': A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(-10, 10),
                    A.Resize(input_size, input_size),  # resize(input_size)
                    # Standardization of color information
                    A.Normalize(color_mean, color_std),
                    ToTensorV2() 
                ]),
                'val': A.Compose([
                    A.Resize(input_size, input_size),  # resize(input_size)
                    # Standardization of color information
                    A.Normalize(color_mean, color_std),
                    ToTensorV2() 
                ])
            }
        else:
            # PyTorch Transformations
            self.data_transform = {
                # Implement only train
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation((-10, 10)),
                    transforms.Resize(input_size),  # resize(input_size)
                    transforms.Normalize(color_mean, color_std),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize(input_size),  # resize(input_size)
                    transforms.Normalize(color_mean, color_std),
                    transforms.ToTensor(),
                ])
            }

    def __call__(self, phase, image):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            Specifies the preprocessing mode.
        """
        transformed = self.data_transform[phase](image=image)
        return transformed['image']



class Dataset(data.Dataset):
    """
    Attributes
    ----------
    df : DataFrame
        class_num, dataframe with column file_path
    phase : 'train' or 'val'
        Set learning or training.
    transform : object
        an instance of the preprocessing class
    """

    def __init__(self, df, phase, transform):
        self.df = df
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''returns the number of images'''
        return len(self.df)

    def __getitem__(self, index):
        '''Get Tensor format data of preprocessed image'''
        image = self.pull_item(index)
        return image, self.df.iloc[index]['class_num']

    def pull_item(self, index):
        '''Get Tensor format data of image'''
        # 1. Image loading
        image_path = self.df.iloc[index]['file_path']
        image = cv2.imread(image_path)[:, :, (2, 1, 0)]
        # 2. Perform pretreatment
        return self.transform(self.phase, image)
