#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:00:12 2023

@author: tracylin
"""
import pandas as pd
import os
import yaml 
import numpy as np
from sklearn.neighbors import NearestNeighbors


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


cfg_filename = 'efficient_arcface'
with open('./configs/' + cfg_filename + '.yaml', 'r') as file:
   cfg = yaml.safe_load(file)  
   
# model and tensorboard saving
model_saving = './model_output/'
model_folder = os.path.join(model_saving, cfg_filename)


train_path = model_folder+ '/out_train.csv'
test_path = model_folder+ '/out_test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_embedding = train_df['embedding'].to_numpy()
test_embedding = test_df['embedding'].values

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
     
acc1 = np.sum(test_df['individual_id'] == test_df['pre_id1'])
acc2 = np.sum(test_df['individual_id'] == test_df['pre_id2'])
acc3 = np.sum(test_df['individual_id'] == test_df['pre_id3'])
acc4 = np.sum(test_df['individual_id'] == test_df['pre_id4'])
acc5 = np.sum(test_df['individual_id'] == test_df['pre_id5'])

    
    
