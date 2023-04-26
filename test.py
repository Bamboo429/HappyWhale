#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:05:06 2023

@author: chuhsuanlin
"""

import os
import pandas as pd
import yaml
import cv2
import matplotlib.pyplot as plt
import albumentations as A

from dataset import HappyWhaleDataset
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses 


from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

with open('./configs/test.yaml', 'r') as file:
   cfg = yaml.safe_load(file)    
     
train_pd = pd.read_csv(os.path.join(cfg['Data']['dataset']['data_name']))
image_folder = cfg['Data']['dataset']['data_directory']

train_df = train_pd[train_pd['train_test'] == 'train'].reset_index(drop=True)
test_df = train_pd[train_pd['train_test'] == 'test'].reset_index(drop=True)
#path = '/Users/chuhsuanlin/Documents/NEU/Course/Spring 2023/CS 7150 - Deep Learning/Final/HappyWhale/Data/train/123.jpg'

image_name = train_pd['image'].values[0]
image_path = os.path.join(os.getcwd(), image_folder, image_name)
image = cv2.imread(image_path)  
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





#
#transform = A.Affine(
#    rotate=(-10, 10),
#    translate_percent=(0, 0.5),
#    #scale()
#    shear=(-10, 10),
#    p=1,
#)
transform = A.Resize(380, 380)



plt.subplot(231)
transform = A.HorizontalFlip(p=1)
augmented_image = transform(image=image)['image']
plt.imshow(augmented_image)
plt.axis('off')

plt.subplot(232)
transform = A.ToGray(p=1)
augmented_image = transform(image=image)['image']
plt.imshow(augmented_image)
plt.axis('off')

plt.subplot(233)
transform = A.Affine(
    rotate=(-10, 10), p=1,)
augmented_image = transform(image=image)['image']
plt.imshow(augmented_image)
plt.axis('off')
 
plt.subplot(234)
transform = A.GaussianBlur(blur_limit=(3, 7), p=1)
augmented_image = transform(image=image)['image']
plt.imshow(augmented_image)
plt.axis('off')

plt.subplot(235)
transform = A.GaussNoise(p=1)
augmented_image = transform(image=image)['image']
plt.imshow(augmented_image)
plt.axis('off')

plt.subplot(236)
transform = A.Affine( translate_percent=(0, 0.5), p=1,)
augmented_image = transform(image=image)['image']
plt.imshow(augmented_image)
plt.axis('off')
#training_data = HappyWhaleDataset(train_pd, "train", cfg, transform=True)
#train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
#train_img, train_labels = next(iter(train_dataloader))
#B,H,W,C = train_img.shape
#plt.imshow(train_img.reshape(H,W,C))

#img = train_img.numpy()

#print(train_pd.columns)




