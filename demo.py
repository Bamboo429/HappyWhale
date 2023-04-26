#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:43:07 2023

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

from trainer import get_model, train, evaluate, metric_train
from dataset import df_output_encoder


def emb2array(embedding):
    
    print(f'transfering.......')
    array_emb = np.array([])
    for idx, emb in enumerate(embedding):
        print(idx)
        emb = emb[2:-2]
        emb = np.fromstring(emb, dtype=float, sep=',')
        emb = np.expand_dims(emb, axis=0)
        
        if idx ==0:
            array_emb = emb
        else:
            array_emb = np.concatenate((array_emb, emb))
    return array_emb


# check GPU
assert torch.cuda.is_available()
# device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# empty memory avoid out-of-memory
torch.cuda.empty_cache()

# open configure file
cfg_filename = 'efficient_arcmargin'
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
model = get_model("arcmargin", id_class_num)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# save the training/testing embeddeed features
for idx, batch in enumerate(train_dataloader):
    print(idx)
    img, label = batch
    label = label.to(device)
    img = img.to(device)
    
    out_class, embedding = model(batch)
    #out_class = model(img)
    train_df.at[idx, 'out_class'] = np.argmax(out_class.numpy(force=True))
    train_df.at[idx, 'embedding'] = str(embedding.tolist())
    
    
for idx, batch in enumerate(test_dataloader):
    print(idx)
    img, label = batch
    label = label.to(device)
    img = img.to(device)
    
    out_class, embedding = model(batch)
    #out_class = model(img)
    test_df.loc[idx, 'out_class'] = np.argmax(out_class.numpy(force=True))
    test_df.loc[idx, 'embedding'] = str(embedding.tolist())
    
    
    
# KNN for cal similiarity  
train_embedding = train_df['embedding'].to_numpy()
test_embedding = test_df['embedding'].to_numpy()


train_embedding = emb2array(train_embedding)
test_embedding = emb2array(test_embedding)

model_embedding = NearestNeighbors(n_neighbors=5,metric='cosine')
model_embedding.fit(train_embedding)
embed_distances, embed_idxs = model_embedding.kneighbors(test_embedding, 5, return_distance=True)
   
embed_id = embed_idxs[0]
column_name = ['pre_id1', 'pre_id2','pre_id3','pre_id4','pre_id5']

for idx, embed_id in enumerate(embed_idxs):
    print(idx)
    pre_id = train_df['individual_id'][embed_id].to_numpy()[np.newaxis, :]
    test_df.loc[idx,column_name] = pre_id
    
    
    
    
