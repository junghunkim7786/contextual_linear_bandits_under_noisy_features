import sys, os, time, datetime,pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
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

num_cores = mp.cpu_count()
print('# of Cores: {}'.format(num_cores))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset_path = "./datasets/avazu"


##########################################################################################################################################

# Hyperparameters for Training
learning_rate = 0.00001
weight_decay  = 0.000001
num_epoch = 500 #Normally 100
B = 10000 # batchsize
EMB_DIM = 32

# Additional char tail for save/load
load_tail = '_avazu_{}'.format(random_seed)
save_tail = load_tail

class BN_Autoencoder(nn.Module):
    def __init__(self, d, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.d = d
        self.encoder = nn.Sequential(nn.Linear(self.d, self.emb_dim),nn.BatchNorm1d(self.emb_dim))
        self.decoder = nn.Linear(self.emb_dim, self.d)
                
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size, self.d)
        return out
    
    def encoding_result(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        return encoded
    
X = np.vstack([np.load(dataset_path+'/preprocess/X0{}.npy'.format(load_tail)),np.load(dataset_path+'/preprocess/X1{}.npy'.format(load_tail))])
np.random.shuffle(X)
d = X.shape[1]

model = BN_Autoencoder(d=d, emb_dim=EMB_DIM).to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

L = X.shape[0]

loss_arr = []


X = torch.from_numpy(X).type(torch.FloatTensor).to(device)

for k in tqdm(range(num_epoch)):
    
    for l in range(L//B):
        
        batch  = X[l*B:(l+1)*B, :]
        
        #x = torch.from_numpy(batch).type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        output = model.forward(batch)
        loss = loss_func(output,batch)
        loss.backward()
        optimizer.step()

    loss_arr.append(loss.cpu().data.numpy())
    
torch.save(model.state_dict(), './models/AE{}.pt'.format(save_tail))
