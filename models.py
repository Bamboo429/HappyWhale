#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:51:01 2023

@author: tracylin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Baseline(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_layer):
        super().__init__()
        hidden_dim = 20
        #H, W, C = data.shape
        #H_out = int(1 + (H + 2 * pad - HH) / stride)
        #W_out = int(1 + (W + 2 * pad - WW) / stride)
    
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, (6,6))
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(num_features=3)
        self.pooling1 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, (3,3))
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d(kernel_size=(3,3))
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(576, 512)
        self.dropout = nn.Dropout()
        self.out = nn.Linear(512, out_channels)
        
    def forward(self, img):
        
        out = self.conv1(img)
        out = self.relu1(out)
        out = self.batch1(out)
        out = self.pooling1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pooling2(out)
        
        out = self.flatten(out)
        out = self.fc(out)
        out = self.dropout(out)
        
        out = self.out(out)
        
        return out
        
        
        
        
        
        
        
        
        
        
        
        
        
    

    