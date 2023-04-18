#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:49:30 2023

@author: chuhsuanlin
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from sklearn import preprocessing
from albumentations.pytorch import ToTensorV2


class HappyWhaleDataset(Dataset):
    
    def __init__(self, df_file, phase, cfg, transform = None):
        
        assert phase in {"train", "test"}
        
        #self.df = df_file,     
        
        self.df = df_file
        self.transform = transform
        self.image_folder = cfg['Data']['dataset']['data_directory']
        self.cfg = cfg  
        self.phase = phase
     
    def get_image(self, index):
        
        image_name = self.df['image'].values[index]
        image_path = os.path.join(os.getcwd(), self.image_folder, image_name)
        #print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # plot
        # plt.imshow(image)
        
        return image
    
    def get_label(self, index):
        
        label_id = self.df['individual_id'].values[index]
        label_species = self.df['species'].values[index]
        
        return label_id, label_species
    
    def data_augmentation(self, image):
        
        aug_para = self.cfg['Aug']
        
        # define aug 
        transform_train = A.Compose([
            
            # augmentation
            A.HorizontalFlip( p = aug_para['hf']),
            A.Affine(
                rotate=(-aug_para['rotate'], aug_para['rotate']),
                translate_percent=(0, aug_para['translate']),
                #scale()
                shear=(-aug_para['shear'], aug_para['shear']),
                p=aug_para['affine'],
            ),
            
            A.ToGray(p=0.1),
            A.GaussianBlur(blur_limit=(3, 7), p=0.05),
            A.GaussNoise(p=0.05),
                
            A.Resize(aug_para['img_height'], aug_para['img_weight']),    
            
            
            # to tensor
            A.Normalize(),
            A.pytorch.transforms.ToTensorV2()
           
        ])
        
        trainfrom_eval = A.Compose([
            A.Normalize(),
            A.pytorch.transforms.ToTensorV2()
            ])
        
        #random.seed()
        if self.phase == 'train':
            augmented_image = transform_train(image=image)['image']
        else:
            augmented_image = trainfrom_eval(image=image)['image']
        
        return augmented_image
        
    def __getitem__(self, index): 
        
        image = self.get_image(index)
        label_id, label_species = self.get_label(index)
        
        
        if self.transform and self.phase == 'train':
            image = self.data_augmentation(image)
        
            
        
        
        return image, label_species
        
    def __len__(self):
        return  len(self.df)
    
    

def df_output_encoder(df):
    """
    convert indiidual id to label encoder or one-hot encoder

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.

    Returns
    -------
    df : dataframe with output encoder

    """
    
    # ==== label encoder ====
    label_id = df['individual_id']
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(label_id)
    label_encoder = labelencoder.transform(label_id) 
    label_num = len(labelencoder.classes_)
    
    # https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/305574
    df.species.replace({"globis": "short_finned_pilot_whale",
                          "pilot_whale": "short_finned_pilot_whale",
                          "kiler_whale": "killer_whale",
                          "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)

    species = df['species'] 
    labelencoder_species = preprocessing.LabelEncoder()
    labelencoder_species.fit(species)
    species_encoder = labelencoder_species.transform(species) 
    spcies_num = len(labelencoder_species.classes_)
    
    # ==== one-hot encoder ====
    #onehotencoder = preprocessing.OneHotEncoder()
    #onehotencoder.fit(label_id)
    #output = onehotencoder.transform(label_id) 
    
    df['individual_id'] = label_encoder
    df['species'] = species_encoder
    
    return df
    
    
    
    
        
        
        
