# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:52:33 2022

@author: reekm
"""

# wideResNet
import torch.nn as nn
import torch.nn.functional as F

def conv_2d(inp_cha,op_cha,stride=1,ks=3):
    return nn.Conv2d(in_channels=inp_cha,out_channels=op_cha,
                    kernel_size=ks,stride=stride,
                    padding=ks//2,
                    bias=False)


def bn_relu_conv(inp_cha,op_cha):
    return nn.Sequential(
    nn.BatchNorm2d(inp_cha),
    nn.ReLU(inplace=True),
    conv_2d(inp_cha,op_cha))

class ResidualBlock(nn.Module):
    def __init__(self,inp_cha,op_cha,stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(inp_cha)
        self.conv1 = conv_2d(inp_cha,op_cha,stride)
        self.conv2 = bn_relu_conv(op_cha,op_cha)
        self.shortcut = lambda x:x
#         special case when ip and op channels are different we use convolution with kernel size 1
        if inp_cha != op_cha:
            self.shortcut = conv_2d(inp_cha,op_cha,stride,1)
            
    def forward(self,x):
        x = F.relu(self.bn(x),inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
#         scale down the output
        x = self.conv2(x) * 0.2
        return x.add_(r)
    
def make_group(N,inp_cha,op_cha,stride):
    start = ResidualBlock(inp_cha,op_cha,stride)
    rest = [ResidualBlock(op_cha,op_cha) for j in range(1,N)]
    return [start]+rest

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.view(x.size(0),-1)
    
    
class WideResNet(nn.Module):
    def __init__(self,n_groups,N,n_classes,k=1,n_start=16):
        super().__init__()
#         increase channel to n_start using conv_layer
        layers = [conv_2d(3,n_start)]
        n_channels = [n_start]
        
#         Add groups of basicBlocks (increase channel and downsample)
        for i in range(n_groups):
            n_channels.append(n_start * (2**i)*k)
            stride = 2 if i>0 else 1
            layers += make_group(N,n_channels[i],
                                n_channels[i+1],
                                stride
                                )
        
#         pool,flatten and linear layer for classification
        layers+= [nn.BatchNorm2d( n_channels[3]),
                            nn.ReLU(inplace =True),
                            nn.AdaptiveAvgPool2d(1),
                            Flatten(),
                            nn.Linear(n_channels[3],n_classes)
        ]
        
        self.features = nn.Sequential(*layers)
        
        
    def forward(self,x):
        return self.features(x)
    
def wrn_22():
    return WideResNet(n_groups=3,N=3,n_classes=10,k=6)
