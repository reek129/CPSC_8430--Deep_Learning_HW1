# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:29:21 2022

@author: reekm
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar_model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,10,kernel_size = 3)
        self.conv2 = nn.Conv2d(10,5,kernel_size = 3)
        self.flatten = nn.Flatten()
                
        self.fc1 = nn.Linear(980, 128)
        self.softmax = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(128, 10)
#         self.fc3 = nn.Linear(84, 10)
        
    def forward(self,x):
        c1 = F.relu(self.conv1(x))
        c2 = F.avg_pool2d(F.relu(self.conv2(c1)),kernel_size=2,stride=2)
        c2 = self.flatten(c2)
        c3 = self.fc1(c2)
        c4 = self.fc2(c3)
        c5 = self.softmax(c4)

        
        return c5