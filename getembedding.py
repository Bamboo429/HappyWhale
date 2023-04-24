#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:36:39 2023

@author: tracylin
"""

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
cfg_filename = 'efficient_arcface'
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
training_data = HappyWhaleDataset(train_df, "train", cfg, transform=False)
testing_data = HappyWhaleDataset(test_df, "test", cfg, transform=False)

# output number
id_class_num = len(pd.unique(train_pd['individual_id']))
species_class_num = len(pd.unique(train_pd['species']))         

# get test dataloader
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=False)

# load model
fold = 0
model_path = model_folder+"/fold"+ str(fold)
#model_path = '/home/users/tracylin/Documents/CS7150/HappyWhale/model_output/test/epoch1'
model = get_model("efficient", species_class_num)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# save the training/testing embeddeed features
for idx, (img, label) in enumerate(train_dataloader):
    print(idx)
    label = label.to(device)
    img = img.to(device)
    
    out_class, embedding = model(img)
    train_df.at[idx, 'out_class'] = np.argmax(out_class.numpy(force=True))
    train_df.at[idx, 'embedding'] = str(embedding.tolist())
    
for idx, (img, label) in enumerate(test_dataloader):
    print(idx)
    label = label.to(device)
    img = img.to(device)
    
    out_class, embedding = model(img)
    test_df.loc[idx, 'out_class'] = np.argmax(out_class.numpy(force=True))
    test_df.loc[idx, 'embedding'] = str(embedding.tolist())
    
train_path = model_folder+ '/out_train.csv'
test_path = model_folder+ '/out_test.csv'
train_df.to_csv(train_path)
test_df.to_csv(test_path)



    
    
    
    
    
    
    
    
    
    









   
   
   
