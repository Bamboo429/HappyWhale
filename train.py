#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:12:53 2023

@author: chuhsuanlin
"""

# TODO !! rewrite to pytorch lighting version
CUDA_LAUNCH_BLOCKING=1

import yaml 
from torch.utils.tensorboard import SummaryWriter
from dataset import HappyWhaleDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold, KFold
import torch
import random
from pytorch_metric_learning import losses
import numpy as np
import torch.nn as nn


from trainer import get_model, train, evaluate, metric_train
from dataset import df_output_encoder


# check GPU
assert torch.cuda.is_available()
# device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# empty memory avoid out-of-memory
torch.cuda.empty_cache()
# open configure file
cfg_filename = 'efficient_arcmargin5'
with open('./configs/' + cfg_filename + '.yaml', 'r') as file:
   cfg = yaml.safe_load(file)   
 
# model and tensorboard saving
writer = SummaryWriter('runs/' + cfg_filename)
model_saving = './model_output/'
model_folder = os.path.join(model_saving, cfg_filename)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# random seed
random_seed = cfg['General']['random_seed']
random.seed(random_seed)

# data prepare
train_pd = pd.read_csv(os.path.join(cfg['Data']['dataset']['data_name']))
train_pd = df_output_encoder(train_pd)

# split training and testing data
train_df = train_pd[train_pd['train_test'] == 'train'].reset_index(drop=True)
test_df = train_pd[train_pd['train_test'] == 'test'].reset_index(drop=True)

# define dataset
training_data = HappyWhaleDataset(train_df, "train", cfg, transform=True)
#train_pd = train_pd[0:200]

# output number
id_class_num = len(pd.unique(train_pd['individual_id']))
species_class_num = len(pd.unique(train_pd['species']))                             

# cross validation
n_splits = cfg['Train']['k_fold']
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# === training ====

# hyperparameters setting
batch_size = cfg['Data']['dataloader']['batch_size']
epochs = cfg['Train']['epoch']
lr = cfg['Train']['lr']

# for cross validation
X = train_df['image']
y = train_df['individual_id']

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
    '''
    trainloader = torch.utils.data.DataLoader(
                        training_data, 
                        batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(
                        training_data,
                        batch_size=batch_size)
    '''
    
    # get model init
    model = get_model("arcmargin", id_class_num)
    model = model.to(device)
    
    # !!! TODO: write in yaml (after) !!!
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)   
    loss_func = nn.CrossEntropyLoss() #
    #loss_func = losses.ArcFaceLoss(id_class_num, 1000, margin=0.05, scale = 20) #
     
    # train and validation
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train(model, trainloader, loss_func, device, optimizer, epoch, 1)
        
        val_acc = evaluate(model, validloader, device)
        print(f" valudation accuarcy : {val_acc}")    
       
        # save training process     
        writer.add_scalar('Loss/train', train_loss, epoch)
        #writer.add_scalar('Loss/test', np.random.random(), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', val_acc, epoch)
        
    # save model
    save_path = model_folder+"/fold"+ str(fold)
    torch.save(model.state_dict(), save_path)
    
    break
    

