# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:45:01 2022

@author: reekm
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

num_of_classes=10

class Cifar_model9(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution Layers
        self.conv1 = nn.Conv2d(3,32,kernel_size = 3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size = 3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,64,kernel_size = 3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(64,64,kernel_size = 3,stride=1,padding=1)
        
        self.conv5 = nn.Conv2d(64,128,kernel_size = 3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(128,128,kernel_size = 3,stride=1,padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 128)

        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        c1 = F.relu(self.conv1(x))
        c2 = F.max_pool2d(F.relu(self.conv2(c1)),2,2)
        c3 = F.relu(self.conv3(c2))
        c4 = F.max_pool2d(F.relu(self.conv4(c3)),2,2)
        
        c5 =  F.relu(self.conv5(c4))
        c6 = F.max_pool2d(F.relu(self.conv6(c5)),2,2)
        c7 = self.flatten(c6)
        
        c8 =self.fc1(c7)
        c9 = self.fc2(c8)
        c10 = self.softmax(c9)
        
        return c10