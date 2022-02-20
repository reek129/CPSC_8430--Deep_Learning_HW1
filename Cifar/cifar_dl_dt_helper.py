# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:05:18 2022

@author: reekm
"""

import os
import torch

from torchvision import datasets, transforms



random_seed = 42
torch.manual_seed(random_seed);


os.environ["OMP_NUM_THREADS"] = "1"



def get_data_transform():
    stats =((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))

    
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Normalize input channels using mean values and standard deviations of ImageNet.
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #             Cifar 10
                transforms.Normalize(*stats)
            ]
        ),
        "test": transforms.Compose(
            [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
                transforms.ToTensor(),
    #             imagenet
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #             cifar10
                transforms.Normalize(*stats)
            ]
        ),
    }
    
    return data_transforms

def get_dataloaders_sizes_classes(data_dir,batch_size):
    
    data_transforms = get_data_transform()
    image_datasets = {
        x if x == "train" else "test": datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]
        )
        for x in ["train", "test"]
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    # dataset_sizes['train'] = len(train_indices)
    # dataset_sizes['validation'] = len(val_indices)
    
    class_names = image_datasets["train"].classes
    
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
        for x in ["train","test"]
    }
    return dataloaders,dataset_sizes,class_names

