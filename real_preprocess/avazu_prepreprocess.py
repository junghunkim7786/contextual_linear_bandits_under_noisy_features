import sys, os, time, datetime,pickle
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import  Pool
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-seed", nargs='?', type=int, default = 0)
args = parser.parse_args()

random_seed = args.seed

print('Random Seed: ',random_seed)

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

fields = ['click',  'C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

num_cores = mp.cpu_count()
print('# of Cores: {}'.format(num_cores))

dataset_path = "./datasets/avazu"

CHUNKSIZE = 1000000
# Read input and output path
data_path = dataset_path + '/raw/train'

# Read 'Massive' dataset
print('Start reading')
df_chunks = []
with tqdm(desc='# of Rows Read: ', unit=' Rows') as pbar:
    for chunk in pd.read_csv(data_path, chunksize=CHUNKSIZE, usecols=fields):
        df_chunks.append(chunk)
        pbar.update(len(chunk.index))
        
print('Reading was finished')

from functools import partial

def func(df, column):
    return df[column].value_counts(ascending=True)

def get_proportion(df_chunks, column, n_cores=num_cores):
    num_list = []
    for chunk in df_chunks:
        num_list.append(func(chunk, column))
    dummy = pd.DataFrame(pd.concat(num_list),columns=[column])
    aa = dummy.groupby(dummy.index)[column].sum()
    aa = aa.div(aa.sum())
    return aa

thres_prop = 0.01

_df_chunks = df_chunks
for column in  fields:

    if column != 'click':
        prop = get_proportion(_df_chunks,column)
        bad_idxs = list(prop[prop < thres_prop].index)

        L = 0 
        for i in range(len(_df_chunks)):
            if len(_df_chunks[i]) > 0:
                _df_chunks[i] =  _df_chunks[i].loc[~ _df_chunks[i][column].isin(bad_idxs)].copy()
                L += len(_df_chunks[i])
        print(column,L)
        
for i,chunk in enumerate(_df_chunks):
    
    chunk.to_csv('./real_datasets/avazu/preprocess/avazu_chunks/chunk_{:2d}'.format(i))
    
_df_chunks = []
for i in range(41):
     _df_chunks.append(pd.read_csv('./real_datasets/avazu_preprocess/avazu_chunks/chunk_{:2d}'.format(i), usecols=fields))
    
df_target = pd.concat(_df_chunks).reset_index(drop=True)

feature_list = []
for column in fields:
    if column != 'click':
        print(column)
        encoding = preprocessing.LabelBinarizer()
        feature_list.append(encoding.fit_transform(df_target[column]))
        
features = np.concatenate(feature_list, axis=1)
reward = np.asarray(df_target['click'])

X_train = features
Y_train = reward


tail = '_avazu_{}'.format(random_seed)

reward0_idx = np.where(reward == 0)[0]
reward1_idx = np.where(reward == 1)[0]

len(reward0_idx) + len(reward1_idx)

np.save(dataset_path + '/preprocess/X0{}'.format(tail),features[reward0_idx,:])
np.save(dataset_path + '/preprocess/X1{}'.format(tail),features[reward1_idx,:])