import os
import sys
import numpy as np
from math import pi, cos 

import torch
import torchvision
import torch.nn as nn
from torch import allclose
from datetime import datetime
import torch.nn.functional as tf 
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.testing import assert_allclose
from torchvision import datasets, transforms
from tqdm import tqdm

import kornia
from kornia import augmentation as K
import kornia.augmentation.functional as F
import kornia.augmentation.random_generator as rg
from torchvision.transforms import functional as tvF


_MEAN =  [0.5, 0.5, 0.5]
_STD  =  [0.2, 0.2, 0.2]



class InitalTransformation():
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
            transforms.Normalize(_MEAN,_STD),
        ])

    def __call__(self, x):
        x = self.transform(x)
        return  x


def gpu_transformer(image_size,s=.2):
        
    train_transform = nn.Sequential(

                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s,p=0.3),
                kornia.augmentation.RandomGrayscale(p=0.05),)

    test_transform = nn.Sequential(  
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s,p=0.3),
                kornia.augmentation.RandomGrayscale(p=0.05),)

    return train_transform , test_transform
                
def get_clf_train_test_transform(image_size,s=.2):
        
    train_transform = nn.Sequential(
                
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
#                 kornia.augmentation.Normalize(CIFAR_MEAN_,CIFAR_STD_),
            )

    test_transform = nn.Sequential(  
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                # kornia.augmentation.RandomGrayscale(p=0.05),
                # kornia.augmentation.Normalize(CIFAR_MEAN_,CIFAR_STD_)
        )

    return train_transform , test_transform


def get_train_test_dataloaders(dataset = "stl10", data_dir="./dataset", batch_size = 64,num_workers = 4, download=True): 
    
    train_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.STL10(data_dir, split="train+unlabeled", transform=InitalTransformation(), download=download),
        shuffle=True,
        batch_size= batch_size,
        num_workers = num_workers
    )
    

    test_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.STL10(data_dir, split="test", transform=InitalTransformation(), download=download),
        shuffle=True,
        batch_size= batch_size,
        num_workers = num_workers
        )
    return train_loader, test_loader