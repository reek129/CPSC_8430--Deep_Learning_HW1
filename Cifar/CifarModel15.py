# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:19:43 2022

@author: reekm
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
class Cifar_model15(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size = 3)
        self.flatten = nn.Flatten()
                

        self.fc1 = nn.Linear(16 * 15 * 15, 128)
        self.fc2 = nn.Linear(128 , 10)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self,x):
        c1 = F.max_pool2d(F.relu(self.conv1(x)),2,2)
        c2 = self.flatten(c1)
        c3 = self.fc1(c2)
        c3 = self.fc2(c3)
        c4 = self.softmax(c3)

        
        return c4