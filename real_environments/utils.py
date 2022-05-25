import numpy as np
from numba import jit

import torch
import torch.nn as nn

__all__ = ["encoding", "noising", "Autoencoder_BN"]

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

#############################################################################################

def encoding(autoencoder, x):
    with torch.no_grad():
        _x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
        _x = autoencoder.encoding_result(_x)
        x = _x.cpu().numpy()
    return x

@jit(nopython=True)
def noising(p, arms_count, x):
    
    d = x.shape[1]
    m = np.zeros((arms_count,d))
    
    for k in range(arms_count):
        m[k] = np.random.binomial(1, p, d)
        x[k] = x[k] * m[k]
    return x, m