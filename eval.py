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
from sklearn.metrics import precision_score, recall_score

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


cfg_filename = 'baseline_cross2'
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
   
n_id = len(test_df)
acc1 = np.sum(test_df['individual_id'] == test_df['pre_id1'])/n_id
acc2 = np.sum(test_df['individual_id'] == test_df['pre_id2'])/n_id
acc3 = np.sum(test_df['individual_id'] == test_df['pre_id3'])/n_id
acc4 = np.sum(test_df['individual_id'] == test_df['pre_id4'])/n_id
acc5 = np.sum(test_df['individual_id'] == test_df['pre_id5'])/n_id

print(acc1, acc2, acc3, acc4, acc5)

n_class = len(test_df)
acc_species =  np.sum(test_df['out_class'] == test_df['species'])/n_class
precision = precision_score(test_df['species'], test_df['out_class'], average='macro')
recall = recall_score(test_df['species'], test_df['out_class'], average='macro')




    
