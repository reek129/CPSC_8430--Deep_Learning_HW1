# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:54:55 2022

@author: reekm
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar_model29(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution Layers
        self.conv1 = nn.Conv2d(3,16,kernel_size = 3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(16,128,kernel_size = 3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(128,20,kernel_size = 3,stride=1,padding=1)
        
        self.conv4 = nn.Conv2d(20,128,kernel_size = 3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(128,16,kernel_size = 3,stride=1,padding=1)
        
        # Activation Layers
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2,2)
        
        self.classifier = nn.Sequential( nn.Flatten(),
                                        nn.Linear(16*1*1,10))
        
    def forward(self,x):
        c1 = self.max_pool(self.relu(self.conv1(x)))
        c2 = self.max_pool(self.relu(self.conv2(c1)))
        c3 = self.max_pool(self.relu(self.conv3(c2)))
        c4 = self.max_pool(self.relu(self.conv4(c3)))
        c5 = self.max_pool(self.relu(self.conv5(c4)))
        
        out  = self.classifier(c5)
        
        return out