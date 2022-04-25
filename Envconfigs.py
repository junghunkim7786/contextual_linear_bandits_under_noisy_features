import random
import numpy as np
import math
from numba import jit
from numpy.random import seed
from numpy.random import rand
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pickle

import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, d, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.d       = d
        self.encoder = nn.Linear(self.d, self.emb_dim)
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

@jit(nopython=True)
def concat_agg(user_feature, item_feature):
    return np.concatenate((user_feature, item_feature)).copy()

@jit(nopython=True)
def outer_agg(user_feature, item_feature):
    return np.outer(user_feature, item_feature).flatten().copy()

@jit(nopython=True)
def outer1_agg(user_feature, item_feature):
    return np.outer(np.append(user_feature,1.0), np.append(item_feature,1.0)).flatten().copy()
    
######################################################################
def Load_Yahoo(option,load_tail):
    
    if not option in ['concat','outer','outer1']:
        raise NotImplemented
        return None, None
    
    if option == 'concat':
        agg_ftn = concat_agg
    elif option == 'outer':
        agg_ftn = outer_agg
    else:
        agg_ftn = outer1_agg
    
    state_dict = torch.load('./models/r6a_autoencoder_{}{}.pt'.format(option,load_tail))
    emb_dim = state_dict['encoder.weight'].shape[0]
    d       = state_dict['encoder.weight'].shape[1]
    
    autoencoder = Autoencoder(d=d, emb_dim=emb_dim).to(device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    
    return autoencoder, agg_ftn, emb_dim

######################################################################    
def Load_MovieLens(option,load_tail):
    
    if not option in ['concat','outer','outer1']:
        raise NotImplemented
        return None, None
    
    if option == 'concat':
        agg_ftn = concat_agg
    elif option == 'outer':
        agg_ftn = outer_agg
    else:
        agg_ftn = outer1_agg
    
    state_dict = torch.load('./models/ml100k_autoencoder_{}{}.pt'.format(option,load_tail))
    emb_dim = state_dict['encoder.weight'].shape[0]
    d       = state_dict['encoder.weight'].shape[1]
    
    autoencoder = Autoencoder(d=d, emb_dim=emb_dim).to(device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    
    return autoencoder, agg_ftn, emb_dim
    
    