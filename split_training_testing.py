#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:00:41 2023

@author: ivyyen
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt


df = pd.read_csv("/home/users/ivyyen/Project/DL_final_project/HappyWhale/train.csv")
X = df['image']
y = df['individual_id']
groups = df['individual_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(gss.split(X, y, groups))

# Extract the corresponding data for the training and testing sets
X_train, y_train, groups_train = X[train_index], y[train_index], groups[train_index]
X_test, y_test, groups_test = X[test_index], y[test_index], groups[test_index]

df["train_test"] = 'test'
df["train_test"][train_index] = 'train'
df.to_csv("train_with_split.csv")



# ==============================================
train_df = df[df["train_test"] == 'train']
plt.figure(figsize = (20,5))
train_df["species"].value_counts().plot(kind='bar')
plt.show()

plt.figure(figsize = (15,5))
plt.hist(train_df["individual_id"].value_counts().tolist(),  bins = 50)
# plt.ylim(0,1000)
plt.show()

test_df = df[df["train_test"] == 'test']
plt.figure(figsize = (20,5))
test_df["species"].value_counts().plot(kind='bar')
plt.show()

plt.figure(figsize = (15,5))
plt.hist(test_df["individual_id"].value_counts().tolist(),  bins = 50)
# plt.ylim(0,1000)
plt.show()
# ==============================================