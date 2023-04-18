#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:51:01 2023

@author: tracylin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CNN_Baseline(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_layer):
        super().__init__()
        hidden_dim = 20
        #H, W, C = data.shape
        #H_out = int(1 + (H + 2 * pad - HH) / stride)
        #W_out = int(1 + (W + 2 * pad - WW) / stride)
    
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, (6,6))
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(num_features=20)
        self.pooling1 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, (3,3))
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d(kernel_size=(3,3))
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(180, 512)
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
    
    
class EfficientNet(nn.Module):
    def __init__(self, model_name, out_channels, embedding_size=1000):
        super().__init__()
        self.embedding = timm.create_model(model_name, pretrained=True)
        in_features = self.embedding.classifier.in_features
        self.embedding.classifier = nn.Identity()
        self.embedding.global_pool = nn.Identity()
        
        self.pooling = GeM()
        self.dense = nn.Linear(in_features, embedding_size) 
        self.dropout = nn.Dropout()
        
    def forward(self, img):
        
        embedding = self.embedding(img)
        out = self.pooling(embedding).flatten(1)
        out = self.dropout(out)
        out = self.dense(out)
        return F.normalize(out)
    
    
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
        
        
        
