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
import math

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
        self.fc1 = nn.Linear(74420, 1000)
        self.fc2 = nn.Linear(1000, 512)
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
        out = self.fc1(out)
        out = self.fc2(out)
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
        self.fc = nn.Linear(embedding_size, out_channels)
        self.dropout = nn.Dropout()
        
    def forward(self, img):
        
        embedding = self.embedding(img)
        embedding = self.pooling(embedding).flatten(1)
        embedding = self.dropout(embedding)
        embedding = self.dense(embedding)
        out = self.fc(embedding)
        #out = embedding
        return F.normalize(out)
    
class EfficientArcMargin(nn.Module):
   
    def __init__(self, model_name, out_channels, embedding_size=1000):
        super().__init__()
        self.embedding = timm.create_model(model_name, pretrained=True)
        in_features = self.embedding.classifier.in_features
         
        #self.embedding.classifier = nn.Identity()
        #self.embedding.global_pool = nn.Identity()
        self.embedding.reset_classifier(num_classes=0, global_pool="avg")
        
        self.arc = ArcMarginProduct(in_features=embedding_size, out_features=out_channels, s=30, m=0.5, easy_margin=False, ls_eps=0)

        self.pooling = GeM()
        self.dense = nn.Linear(in_features, embedding_size)
        self.fc = nn.Linear(embedding_size, out_channels)
        self.dropout = nn.Dropout()

    def forward(self, batch):
        
        img, label = batch
        label = label.to("cuda")
        img = img.to("cuda")
        
        embedding = self.embedding(img)
        embedding = self.dense(embedding)
        out = self.arc(embedding, label)

        #out = self.pooling(embedding).flatten(1)
        #out = self.dropout(out)
        #out = self.dense(out)
        #out = self.fc(out)
        return out, embedding


# code from https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter/notebook
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
        
# code from https://www.kaggle.com/code/sadcard/pytorch-lightning-arcface-focal-loss#Training        
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output        
