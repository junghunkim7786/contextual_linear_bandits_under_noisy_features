import pandas as pd
import pickle
import math

import numpy as np
from numba import jit

import torch
import torch.nn as nn

from .utils import *

def Load_Avazu(model_tail, num_partial, device):
    
    X0 = np.load('./real_datasets/avazu/preprocess/X0_avazu.npy')
    X1 = np.load('./real_datasets/avazu/preprocess/X1_avazu.npy')
    
    if num_partial > 0 :
        X0_len = X0.shape[0] 
        X1_len = X1.shape[0]
        reward0_idxs = np.random.choice(np.arange(X0_len), num_partial, replace=False)
        reward1_idxs = np.random.choice(np.arange(X1_len), num_partial, replace=False)
        X0 = X0[reward0_idxs]
        X1 = X1[reward1_idxs]
    
    raw_dim = X0.shape[1]

    state_dict = torch.load('./real_models/AE_avazu_{}.pt'.format(model_tail))
    emb_dim = state_dict['decoder.weight'].shape[1]
    
    autoencoder = Autoencoder_BN(raw_dim=raw_dim, emb_dim=emb_dim).to(device)
    
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    return autoencoder, X0, X1, emb_dim

class avazu_Env(base_Env):
    def __init__(self, args):
        super().__init__(args,Load_Avazu)