# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:14:01 2022

@author: reekm
"""

import os
import torch
import torchvision
import time
import copy

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

import matplotlib
import matplotlib.pyplot as plt


random_seed = 42
torch.manual_seed(random_seed);

import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.decomposition import PCA
import pandas as pd

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_grad_norm(model):
    grad_all=0.0
    grad =0
    
    for p in model.parameters():
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy()**2).sum()
            
        grad_all+=grad
        
    grad_norm=grad_all ** 0.5
    return grad_norm

def get_forbenious_norm_sensitivity(model):
    frobernious_grad =0
    count=0
    
    for p in model.parameters():
        grad =0.0
        if p.grad is not None:
            grad = p.grad
            frob_norm = torch.linalg.norm(grad).numpy()
            
            frobernious_grad += frob_norm
            count+=1
            
    return frobernious_grad/count

def get_model_weights_points(model):
    substring = "weight"
    wts=[]
    ls = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if substring in name:
                temp = param.data.flatten().numpy()
                wts.extend(temp)
    return wts

def get_parameters_to_vectors(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())

def new_theta(vector1,vector2,alpha):
    theta = (1-alpha)*vector1 + alpha*vector2
    return theta
            
def get_max_key(my_dict):
    max_key = max(my_dict,key=my_dict.get)
    return max_key

def get_best_model_details_after_training(grad_norm_per_epoch,train_losses,val_losses,train_acc,val_acc,model_wts_epoch,sensitivity):
    
    max_value_key=get_max_key(val_acc)
    
    grad = {} if len(grad_norm_per_epoch) == 0 else grad_norm_per_epoch[max_value_key]
    model_wts =  {} if len(model_wts_epoch) == 0 else model_wts_epoch[max_value_key]
    frobenious_sensitivity = {} if len(sensitivity) == 0 else sensitivity[max_value_key]
    
    return grad,train_losses[max_value_key],val_losses[max_value_key],train_acc[max_value_key],val_acc[max_value_key],model_wts,frobenious_sensitivity

def save_to_csv(dictionary,model_name,type_data,hw_data,result_folder):
    file_name = result_folder+hw_data+"_"+model_name+"_"+type_data+".csv"
    
    df = pd.DataFrame([dictionary],columns=dictionary.keys())
    df.to_csv(file_name,index=False)
            
            



