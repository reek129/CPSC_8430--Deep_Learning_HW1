# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 00:40:36 2022

@author: reekm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import input_size,output_size

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 1)
        self.dense_layer14 = nn.Linear(1, 190)
        self.dense_layer15 = nn.Linear(190, output_size)
        
        
    def forward(self,x):
        x = F.leaky_relu(self.input_layer(x))
        x = F.leaky_relu(self.dense_layer14(x))
        x = self.dense_layer15(x)
                
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
    
    