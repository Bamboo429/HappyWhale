#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:12:53 2023

@author: chuhsuanlin
"""

# TODO !! rewrite to pytorch lighting version


import yaml
import time
from torch.utils.tensorboard import SummaryWriter
from dataset import HappyWhaleDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold
import torch
import random
from pytorch_metric_learning import losses
import numpy as np

from trainer import get_model, train, evaluate

# check GPU
assert torch.cuda.is_available()
# device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# open configure file
with open('./configs/test.yaml', 'r') as file:
   cfg = yaml.safe_load(file)   
 
# random seed
random_seed = cfg['General']['random_seed']
random.seed(random_seed)

# data prepare
train_pd = pd.read_csv(os.path.join(cfg['Data']['dataset']['data_name']))
training_data = HappyWhaleDataset(train_pd, "train", cfg, transform=True)

# test iter dataloader
#train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
#train_img, train_labels = next(iter(train_dataloader))

# cross validation
n_splits = cfg['Train']['k_fold']
kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# === training ====

# hyperparameters setting
batch_size = cfg['Data']['dataloader']['batch_size']
epochs = cfg['Train']['epoch']
lr = cfg['Train']['lr']

# for cross validation
X = train_pd['image']
y = train_pd['individual_id']

for fold,(train_idx,test_idx) in enumerate(kfold.split(X, y)):
    
    print('------------fold no---------{}----------------------'.format(fold))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
     
    trainloader = torch.utils.data.DataLoader(
                        training_data, 
                        batch_size=batch_size, sampler=train_subsampler)
    validloader = torch.utils.data.DataLoader(
                        training_data,
                        batch_size=batch_size, sampler=valid_subsampler)
    
    # get model init
    model = get_model("baseline")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = losses.TripletMarginLoss()

     
    # train and validation
    for epoch in range(1, epochs + 1):
        train(model, trainloader, loss_func, device, optimizer, epoch, 1)
        evaluate(model, validloader, device)
        
        # save training process
        # save model
    
    

