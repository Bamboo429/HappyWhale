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

# check GPU
assert torch.cuda.is_available()
# device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# open configure file
with open('./configs/test.yaml', 'r') as file:
   cfg = yaml.safe_load(file)   
 
# random seed
random_seed = cfg['General'['random_seed']]
random.seed(random_seed)

# data prepare
train_pd = pd.read_csv(os.path.join(cfg['Data']['dataset']['data_name']))
training_data = HappyWhaleDataset(train_pd, "train", cfg)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
#train_img, train_labels = next(iter(train_dataloader))

# cross validation
n_splits = cfg['Train']['k_fold']
kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# === training ====

# hyperparameters setting
batch_size = cfg['Data']['dataloader']['batch_size']
epoch = cfg['Train']['epoch']


for fold,(train_idx,test_idx) in enumerate(kfold.split(training_data)):
    
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
    model = get_model()
    #model.apply(reset_weights)
     
    # train and validation
    for epoch in range(1, epochs + 1):
        train(fold, model, device, trainloader, optimizer, epoch)
        evaluate(fold,model, device, testloader)
        
        # save training process
        # save model
    
    

def train(model, dataloader, loss_func, device, grad_norm_clip):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        label = label.to(device)
        text = text.to(device)
        optimizer.zero_grad()
        
        logits = None
        
        ###########################################################################
        # TODO: compute the logits of the input, get the loss, and do the         #
        # gradient backpropagation.
        ###########################################################################
        logits = model(text)
        loss = loss_func(logits, label)
        loss.backward()
        #raise NotImplementedError
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        total_acc += (logits.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(model, dataloader, loss_func, device):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            label = label.to(device)
            text = text.to(device)
            
            
            ###########################################################################
            # TODO: compute the logits of the input, get the loss.                    #
            ###########################################################################
            logits = model(text)
            #raise NotImplementedError
            ###########################################################################
            #                             END OF YOUR CODE                            #
            ###########################################################################
            
            total_acc += (logits.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count