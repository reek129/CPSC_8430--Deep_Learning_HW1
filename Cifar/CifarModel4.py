# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:27:26 2022

@author: reekm
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar_model4(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution Layers
        self.conv1 = nn.Conv2d(3,32,kernel_size = 3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size = 3,stride=1,padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 10)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        c1 = F.relu(self.conv1(x))
        c2 = F.max_pool2d(F.relu(self.conv2(c1)),2,2)

        c7 = self.flatten(c2)
        
        c8 =self.fc1(c7)

        c10 = self.softmax(c8)
        
        return c10