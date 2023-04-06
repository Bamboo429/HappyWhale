#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:05:06 2023

@author: chuhsuanlin
"""

import os
import pandas as pd
import yaml
#from addict import Dict
import cv2
import matplotlib.pyplot as plt
import albumentations as A


with open('./configs/test.yaml', 'r') as file:
   cfg = yaml.safe_load(file)    
     
train_pd = pd.read_csv(os.path.join(cfg['Data']['dataset']['data_name']))
image_folder = cfg['Data']['dataset']['data_directory']

path = '/Users/chuhsuanlin/Documents/NEU/Course/Spring 2023/CS 7150 - Deep Learning/Final/HappyWhale/Data/train/123.jpg'

image_name = train_pd['image'].values[0]
image_path = os.path.join(os.getcwd(), image_folder, image_name)
image = cv2.imread(image_path)  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


transform = A.HorizontalFlip(p=1)
augmented_image = transform(image=image)['image']
plt.imshow(augmented_image)

#print(train_pd.columns)

