#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:37:43 2023

@author: tracylin
"""
import time

import torch
import torch.nn as nn
import models
from pytorch_metric_learning import losses

from models import CNN_Baseline

def get_model(model_name, output_class=15587)->nn.Module:
    """
    Parameters
    ----------
    model_name : string
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if model_name == 'baseline':
        model = CNN_Baseline(3, output_class, 2)
    #elif model_name == 'efficient':
    #    model = EfficientNet()
        
    else:        
        print('error model name')
        
    
    return model

def get_loss_func(loss_name, cfg):
    
    if loss_name == 'cross_entropy':       
        loss_func = nn.CrossEntropyLoss()
        
    elif loss_name == 'arcface':
        num_classes = 10000
        embedding_size = 1000
        #losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64, **kwargs)
        loss_func = losses.ArcFaceLoss(num_classes, embedding_size)
    else:
        print('no loss function defined')
            
    


def train(model, dataloader, loss_func, device, optimizer, epoch, grad_norm_clip):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 100
    start_time = time.time()

    for idx, (img, label) in enumerate(dataloader):
        #print(label)
        label = label.to(device)
        img = img.to(device)
        optimizer.zero_grad()
        
        logits = None
        
        ###########################################################################
        # TODO: compute the logits of the input, get the loss, and do the         #
        # gradient backpropagation.
        ###########################################################################
        logits = model(img)
        loss = loss_func(logits, label)
        loss.backward()
        #raise NotImplementedError
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        total_acc += (logits.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(model, dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):
            label = label.to(device)
            img = img.to(device)
            
            
            ###########################################################################
            # TODO: compute the logits of the input, get the loss.                    #
            ###########################################################################
            logits = model(img)
            #raise NotImplementedError
            ###########################################################################
            #                             END OF YOUR CODE                            #
            ###########################################################################
            
            total_acc += (logits.argmax(1) == label).sum().item()
            total_count += label.size(0)
            #print(label)
            #print(label.size(0))
    return total_acc/total_count
    
    
            
        
    
    
    
    
    
    