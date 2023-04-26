#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:51:36 2023

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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score

from trainer import get_model, train, evaluate, metric_train
from dataset import df_output_encoder


# check GPU
assert torch.cuda.is_available()
# device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# empty memory avoid out-of-memory
torch.cuda.empty_cache()

# open configure file
cfg_filename = 'baseline_cross2'
with open('./configs/' + cfg_filename + '.yaml', 'r') as file:
   cfg = yaml.safe_load(file)  
   
# model and tensorboard saving
#writer = SummaryWriter('runs/' + cfg_filename)
model_saving = './model_output/'
model_folder = os.path.join(model_saving, cfg_filename)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# random seed
#random_seed = cfg['General']['random_seed']
#random.seed(random_seed)

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

# load saved model
fold = 0
model_path = model_folder+"/fold"+ str(fold)
model = get_model("baseline", species_class_num)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()


for idx, batch in enumerate(test_dataloader):
    print(idx)
    img, label = batch
    label = label.to(device)
    img = img.to(device)
    
    #out_class, embedding = model(batch)
    out_class = model(img)
    test_df.loc[idx, 'out_class'] = np.argmax(out_class.numpy(force=True))
    #test_df.loc[idx, 'embedding'] = str(embedding.tolist())
    
    
n_class = len(test_df)
acc_species =  np.sum(test_df['out_class'] == test_df['species'])/n_class
precision = precision_score(test_df['species'], test_df['out_class'], average='macro')
recall = recall_score(test_df['species'], test_df['out_class'], average='macro')