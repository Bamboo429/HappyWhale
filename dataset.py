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


class HappyWhaleDataset(Dataset):
    
    def __init__(self, df_file, phase, cfg, transform = None):
        
        assert phase in {"train", "test"}
        
        self.df = df_file
        self.transform = transform
        self.image_folder = cfg['Data']['dataset']['data_directory']
        self.cfg = cfg   
     
    def get_image(self, index):
        
        image_name = self.df['image'].values[index]
        image_path = os.path.join(os.getcwd(), self.image_folder, image_name)
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
        transform = A.Compose([
            A.HorizontalFlip( p = aug_para['hf']),
            A.Affine(
                rotate=(-aug_para['rotate'], aug_para['rotate']),
                translate_percent=(0, aug_para['translate']),
                #scale()
                shear=(-aug_para['shear'], aug_para['shear']),
                p=aug_para['affine'],
            ),
            
            A.Resize(aug_para['img_height'], aug_para['img_weight'])
        ])
        
        #random.seed()
        
        augmented_image = transform(image=image)['image']
        
        return augmented_image
        
    def __getitem__(self, index): 
        
        image = self.get_image(index)
        label_id, label_species = self.get_label(index)
        
        if self.transform:
            image = self.data_augmentation(image)
        
        return image, label_id
        
    def __len__(self):
        return  len(self.df)
        
        
        