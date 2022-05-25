import pandas as pd
import pickle
import math

import numpy as np
from numba import jit

import torch
import torch.nn as nn

from .utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Load_Taobao(model_tail, data_tail, encoding, num_partial):
    
    X0 = np.load('./datasets/taobao/preprocess/X0{}.npy'.format(data_tail))
    X1 = np.load('./datasets/taobao/preprocess/X1{}.npy'.format(data_tail))
    
    if num_partial > 0 :
        X0_len = X0.shape[0] 
        X1_len = X1.shape[0]
        reward0_idxs = np.random.choice(np.arange(X0_len), num_partial, replace=False)
        reward1_idxs = np.random.choice(np.arange(X1_len), num_partial, replace=False)
        X0 = X0[reward0_idxs]
        X1 = X1[reward1_idxs]
    
    raw_dim = X0.shape[1]
    
    if encoding:
        state_dict = torch.load('./models/taobao_autoencoder{}.pt'.format(model_tail))
        emb_dim = state_dict['decoder.weight'].shape[1]
        
        autoencoder = Autoencoder_BN(raw_dim=raw_dim, emb_dim=emb_dim).to(device)
        
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        return autoencoder, X0, X1, emb_dim
        
    return None, X0, X1, raw_dim

class taobao_Env:
    def __init__(self, args):
        self.seed = args.seed
        np.random.seed(self.seed)
        self.env = args.env
        self.args = args
        
        self.autoencoder, self.X0, self.X1, self.d \
        = Load_Taobao(model_tail=args.model_tail, data_tail=args.data_tail, \
                     encoding=args.encoding, num_partial=args.num_partial)

        self.p = 1.0 - args.mask_ratio
        
        self.K, self.K_max = args.K_max, args.K_max
        self.arms = None
        self.reward1_ratio = args.reward1_ratio
        
        self.N0, self.N1 = self.X0.shape[0], self.X1.shape[0]
        self.n1 = int(math.ceil(self.K * self.reward1_ratio))
        self.n0 = int(self.K - self.n1)        
        
    def load_data(self):
        
        reward0_idxs = np.random.choice(np.arange(self.N0),self.n0,replace=False)
        reward1_idxs = np.random.choice(np.arange(self.N1),self.n1,replace=False)
        
        self.x = np.vstack([self.X0[reward0_idxs].copy(),self.X1[reward1_idxs].copy()])
        self.arms = np.arange(self.K)
        
        if not (self.autoencoder is None):
            self.x = encoding(self.autoencoder, self.x)
        self.x, self.m = noising(self.p, len(self.arms), self.x)
                
    def write_used_idx(self):
        pass
        
    def observe(self, k):
        
        if self.arms[k] >= self.n0:
            exp_reward = 1
        else:
            exp_reward = 0
        reward = exp_reward
        
        return exp_reward,reward
    
    def reset(self):
        np.random.seed(self.seed)