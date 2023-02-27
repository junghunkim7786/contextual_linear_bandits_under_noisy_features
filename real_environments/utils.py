import numpy as np
import math
from numba import jit

import torch
import torch.nn as nn
from icecream import ic

__all__ = ["masking", "Autoencoder_BN", "base_Env"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Autoencoder_BN(nn.Module):
    def __init__(self, raw_dim, emb_dim):
        super().__init__()
        self.raw_dim = raw_dim
        self.emb_dim = emb_dim
        self.encoder = nn.Sequential(nn.Linear(self.raw_dim, self.emb_dim),nn.BatchNorm1d(self.emb_dim))
        self.decoder = nn.Linear(self.emb_dim, self.raw_dim)
                
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size, self.raw_dim)
        return out
    
    def encoding_result(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        return encoded

class base_Env:
    def __init__(self, args, load_ftn):
        self.seed = args.seed
        np.random.seed(self.seed)
        self.env = args.env
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        self.autoencoder, self.X0, self.X1, self.d \
        = load_ftn(model_tail=args.model_tail, num_partial=args.num_partial, device=self.device)

        self.p = 1.0 - args.mask_ratio
        
        self.K, self.K_max = args.K_max, args.K_max
        self.arms = None
        self.reward1_ratio = args.reward1_ratio     
        
        self.N0, self.N1 = self.X0.shape[0], self.X1.shape[0]
        self.n1 = int(math.ceil(self.K * self.reward1_ratio))
        self.n0 = int(self.K - self.n1)

        B = 10000
        X0_stack = []
        for Bidx in range(self.N0 // B + 1):
            X0_stack.append(self.encoding(self.X0[Bidx * B : (Bidx + 1) * B, :]).copy())
        X1_stack = []
        for Bidx in range(self.N1 // B + 1):
            X1_stack.append(self.encoding(self.X1[Bidx * B : (Bidx + 1) * B, :]).copy())

        self.X0 = np.vstack(X0_stack)
        self.X1 = np.vstack(X1_stack)
        self.X = np.vstack([self.X0,self.X1])
        np.random.shuffle(self.X)

    def encoding(self, x):
        with torch.no_grad():
            _x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
            _x = self.autoencoder.encoding_result(_x)
            x = _x.cpu().numpy()
        return x
    
    def load_data(self):
        
        reward0_idxs = np.random.choice(np.arange(self.N0),self.n0,replace=False)
        reward1_idxs = np.random.choice(np.arange(self.N1),self.n1,replace=False)
        
        idxs = np.random.choice(np.arange(self.N0+self.N1),self.K,replace=False)
        self.x = self.X[idxs].copy()
        
        #self.x = np.vstack([self.X0[reward0_idxs].copy(),self.X1[reward1_idxs].copy()])
        self.arms = np.arange(self.K)
        self.x, self.m = masking(self.p, len(self.arms), self.x)
            
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

@jit(nopython=True)
def masking(p, arms_count, x):
    
    d = x.shape[1]
    m = np.random.binomial(1.0, p, (arms_count,d))
    x = x * m
    return x, m