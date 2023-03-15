import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import argparse

learning_rate = 0.00001
weight_decay  = 0.000001
num_epoch = 500
B = 10000
EMB_DIM = 32

__all__ = ["AE_train"]

class Autoencoder_BN(nn.Module):
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
 

def AE_train(X, emb_dim = EMB_DIM, seed = 0):

    print('Random Seed: ',seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    d = X.shape[1]
    model = Autoencoder_BN(d=d, emb_dim=emb_dim).to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    L = X.shape[0]

    loss_arr = []

    for k in tqdm(range(num_epoch)):
        
        for l in range(L//B):
            
            batch  = X[l*B:(l+1)*B, :].copy()
            x = torch.from_numpy(batch).type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,x)
            loss.backward()
            optimizer.step()
        
        loss_val = loss.cpu().data.numpy()
        loss_arr.append(loss_val)

    return model

