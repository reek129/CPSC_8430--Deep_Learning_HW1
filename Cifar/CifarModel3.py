# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:24:56 2022

@author: reekm
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar_model3(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution Layers
        self.conv1 = nn.Conv2d(3,6,kernel_size = 5)
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5)
                
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self,x):
        c1 = self.max_pool(F.relu(self.conv1(x)))
        c2 = self.max_pool(F.relu(self.conv2(c1)))
        
        c2 = c2.view(-1,16*5*5)
        c2 = F.relu(self.fc1(c2))
        c2 = F.relu(self.fc2(c2))
        
        out  = self.fc3(c2)
        
        return out
            