# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:31:50 2022

@author: reekm
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar_model7(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution Layers
        self.conv1 = nn.Conv2d(3,32,kernel_size = 3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size = 3,stride=1,padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 100)

        self.fc2 = nn.Linear(100, 10)

        
    def forward(self,x):
        c1 = F.max_pool2d(F.relu(self.conv1(x)),2,2)
        c2 = F.max_pool2d(F.relu(self.conv2(c1)),2,2)
        c2 = self.flatten(c2)
        
        fc1 = self.fc1(c2)
        fc2 = self.fc2(fc1)

        
        return fc2