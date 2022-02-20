# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 00:37:39 2022

@author: reekm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:48:05 2022

@author: reekm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import input_size,output_size

class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 1)
        self.dense_layer1 = nn.Linear(1, 5)
        self.dense_layer2 = nn.Linear(5, 10)
        self.dense_layer3 = nn.Linear(10, 10)
        self.dense_layer4 = nn.Linear(10, 10)
        self.dense_layer5 = nn.Linear(10, 10)
        self.dense_layer6 = nn.Linear(10, 10)
        self.dense_layer7 = nn.Linear(10, 5)
        self.dense_layer8 = nn.Linear(5, output_size)
        
    def forward(self,x):
        x = F.leaky_relu(self.input_layer(x))
        x = F.leaky_relu(self.dense_layer1(x))
        x = F.leaky_relu(self.dense_layer2(x))
        x = F.leaky_relu(self.dense_layer3(x))
        x = F.leaky_relu(self.dense_layer4(x))
        x = F.leaky_relu(self.dense_layer5(x))
        x = F.leaky_relu(self.dense_layer6(x))
        x = F.leaky_relu(self.dense_layer7(x))
        x = self.dense_layer8(x)
        
        return x
    
    def training_step(self, batch,loss_fn):
        inputs, targets = batch 
        out = self(inputs)                 # Generate predictions
        loss = loss_fn(out, targets)    # Calculate loss
        return loss
    
    def validation_step(self, batch,loss_fn):
        inputs, targets = batch 
        out = self(inputs)                 # Generate predictions
        loss = loss_fn(out, targets)    # Calculate loss
        return {'val_loss': loss.detach()}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def train_step(self, batch,loss_fn):
        inputs, targets = batch 
        out = self(inputs)                 # Generate predictions
        loss = loss_fn(out, targets)    # Calculate loss
        return {'train_loss': loss.detach()}
    
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'train_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
    
    
    
    