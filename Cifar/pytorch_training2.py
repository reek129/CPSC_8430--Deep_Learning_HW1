# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:04:47 2022

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

from pytorch_model_helper import get_grad_norm,get_model_weights_points,get_forbenious_norm_sensitivity
from cifar_dl_dt_helper import get_dataloaders_sizes_classes



class Pytorch_training_helper():
    def __init__(self,dataloaders,dataset_sizes,batch_size,result_folder,flag_grad=0,flag_weights=0,flag_frobenius_norm=0):
        self.dataloaders = dataloaders
        self.dataset_sizes =dataset_sizes
        self.batch_size = batch_size
        self.flag_grad = flag_grad
        self.flag_weights =flag_weights
        self.flag_frobenius_norm = flag_frobenius_norm
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.result_folder= result_folder
        
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        
    def train_model(self,model, criterion, optimizer, scheduler,model_name, num_epochs):
        pca =  PCA(n_components=2)
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 10000.0  # Large arbitrary number
        best_acc_train = 0.0
        best_loss_train = 10000.0  # Large arbitrary number
        model_weights_epochs={}
        grad_norm_per_epoch = {}
        train_losses = {}
        val_losses ={}
        train_acc ={}
        val_acc = {}
        sensitivity ={}
        
        print("Training started:")
        
        for epoch in range(num_epochs):
            # part 1_2 to calculate the weights
            if self.flag_weights:
                wts = get_model_weights_points(model)
                model_weights_epochs[epoch] = wts
                
            for phase in ["train", "test"]:
                if phase == "train":
                    # Set model to training mode
                    model.train()
                else:
                    # Set model to evaluate mode
                    model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                n_batches = self.dataset_sizes[phase] // self.batch_size
                
                it = 0
                
                for inputs, labels in self.dataloaders[phase]:
                    since_batch = time.time()
                    batch_size_ = len(inputs)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    
                    
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            
                            if self.flag_grad:
                                grad_norm_per_epoch[epoch] = get_grad_norm(model)
                                
                            
                            if self.flag_frobenius_norm:
                                sensitivity[epoch] = get_forbenious_norm_sensitivity(model)
                            optimizer.step()
                            
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(preds == labels.data).item()
                    running_corrects += batch_corrects
                    
                    print(
                        "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}".format(
                            phase,
                            epoch + 1,
                            num_epochs,
                            it + 1,
                            n_batches + 1,
                            time.time() - since_batch,
                        ),
                        end="\r",
                        flush=True,
                    )
                    it += 1
                    
                # print epoch result
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                
                print(
                    "Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        ".format(
                        "train" if phase == "train" else "validation  ",
                        epoch + 1,
                        num_epochs,
                        epoch_loss,
                        epoch_acc,
                    )
                )
                
                # Check if this is the best model wrt previous epochs
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(copy.deepcopy(model.state_dict()),self.result_folder+model_name+'.pt')
                if phase == "test" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                if phase == "train" and epoch_acc > best_acc_train:
                    best_acc_train = epoch_acc
                if phase == "train" and epoch_loss < best_loss_train:
                    best_loss_train = epoch_loss
                    
                if phase == "train":
                    train_losses[epoch] = epoch_loss
                    train_acc[epoch] = epoch_acc
                    scheduler.step()
                else:
                    val_losses[epoch] = epoch_loss
                    val_acc[epoch] = epoch_acc
                    
                    
        time_elapsed = time.time() - since
        print(
            "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
        print("Best test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))
        
        return model,grad_norm_per_epoch,train_losses,val_losses,train_acc,val_acc, model_weights_epochs,sensitivity
    
    def test_model(self,model,criterion):
        model.eval()
        train_loss =0.0
        train_acc=0.0
        test_loss=0.0
        test_acc=0.0
        
        for phase in ["train","test"]:
            running_loss = 0.0
            running_corrects = 0
            
            n_batches = self.dataset_sizes[phase] // self.batch_size
            
            it = 0
            
            for inputs, labels in self.dataloaders[phase]:
                since_batch = time.time()
                batch_size_ = len(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * batch_size_
                batch_corrects = torch.sum(preds == labels.data).item()
                running_corrects += batch_corrects
                
                print(
                        "Phase: {} Iter: {}/{} Batch time: {:.4f}".format(
                            phase,
                            it + 1,
                            n_batches + 1,
                            time.time() - since_batch,
                        ),
                        end="\r",
                        flush=True,
                    )
                it += 1
                
            if phase == "train":
                train_loss = running_loss/self.dataset_sizes[phase]
                train_acc = running_corrects / self.dataset_sizes[phase]
                
            else:
                test_loss = running_loss/self.dataset_sizes[phase]
                test_acc = running_corrects / self.dataset_sizes[phase]
                
        return train_loss,test_loss,train_acc,test_acc
    
                        
                    
                    

                
                
            
        