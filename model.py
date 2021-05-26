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
from torchvision.models import resnet50


def negative_cosine_similarity(p,z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True) )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True) )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim))
    
    def forward(self,x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x

class PredictionMLP(nn.Module):
    def __init__(self, in_dim = 2048, hidden_dim=512, out_dim = 2048):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.layer2 = nn.Linear(hidden_dim,out_dim)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
    
class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.encoder = nn.Sequential(
            backbone,
            MLP
        )
        
        self.predictor = PredictionMLP()
    
    def forward(self, x1, x2):
        f,h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        
        loss = negative_cosine_similarity(p1,z2)/2 + negative_cosine_similarity(p2,z1)/2
        
        return loss