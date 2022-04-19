import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yahoo_emb_dim = 10

class Autoencoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder = nn.Linear(36, self.emb_dim)
        self.decoder = nn.Linear(self.emb_dim, 36)
                
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size,36)
        return out
    
    def encoding_result(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        return encoded

yahoo_model = Autoencoder(emb_dim=yahoo_emb_dim)
yahoo_model.load_state_dict(torch.load('./models/r6a_autoencoder.pt'))
yahoo_model.to(device)
yahoo_model.eval()
######################################################################
    
class yahoo_Env:
    def __init__(self,seed,p,K_max,private):
        np.random.seed(seed)
        self.f = open('datasets/R6A/ydata-fp-td-clicks-v1_0.20090501')
        sampling_rate = 1.0
        self.subsampling = True
        self.K_max = K_max
        
        self.p = p
        self.embed_dim = 10
        self.d = self.embed_dim
        
        self.env = 'yahoo'
        self.private = private
        
        B = np.random.uniform(-1,1,(self.embed_dim,self.embed_dim))
        self.Sigma_n = B.T @ B
        
#         self.r_mask = np.random.RandomState(seed)
#         self.r_subsample = np.random.RandomState(seed+1)
        
    def load_data(self):
        line = self.f.readline()
        self.data_preprocess(line.strip().split(' |'))   
        
        if self.subsampling and len(self.arms) > self.K_max:
            subsample = []
            while self.chosen_idx not in subsample:
                subsample = np.sort(np.random.choice(len(self.arms), self.K_max, replace=False))
            arms_subsample = np.array(self.arms)[subsample]
            chosen_idx_subsample = np.where(subsample == self.chosen_idx)[0][0]
            contexts_subsample = self.x[subsample, :]

            self.chosen_idx = chosen_idx_subsample
            self.x = contexts_subsample
            self.arms = arms_subsample
        self.K = len(self.arms)
        
    def data_preprocess(self,raw_data):
        info = raw_data[0].split(' ')
        self.time = int(info[0])
        self.chosen_arm = int(info[1])
        self.is_clicked = bool(int(info[2]))
        self.user_context = self.parse_context(raw_data[1].split(' '))[1]
        
        self.arms = []
        self.arm_contexts = []
        for i in range(2, len(raw_data)):
            arm, context = self.parse_context(raw_data[i].split(' '))
            self.arms.append(arm)
            self.arm_contexts.append(context)
            
        self.x = np.outer(self.user_context, self.arm_contexts[0]).flatten()

        for j in range(1, len(self.arm_contexts)):
            self.x = np.vstack((self.x, np.outer(self.user_context, self.arm_contexts[j]).flatten()))

        with torch.no_grad():
            x = torch.from_numpy(self.x).type(torch.FloatTensor).to(device)
            x = yahoo_model.encoding_result(x)
            self.x = x.cpu().numpy()
        
        self.chosen_idx = self.arms.index(self.chosen_arm)
        self.m = np.zeros((len(self.arms),self.embed_dim))

        for k in range(len(self.arms)):
            self.m[k] = np.random.binomial(1, self.p, self.embed_dim)
            if self.private == False:
                self.x[k] = self.x[k] * self.m[k]
            else:
                noise = np.random.multivariate_normal(np.zeros(self.embed_dim), self.Sigma_n, 1)
                self.x[k] = (self.x[k] + noise) * self.m[k]
            
    def parse_context(self,raw_context):
#         print(raw_context)
        if raw_context[0] == 'user':
            arm = 'user'
        else:
            arm = int(raw_context[0])

        context = np.zeros(6)
        for i in range(1, len(raw_context)):
            idx_ctx = raw_context[i].split(':')
            context[int(idx_ctx[0])-1] = float(idx_ctx[1])

        return arm, context

    def observe(self,k):
        return bool(self.is_clicked),bool(self.is_clicked)
    
    
class noisy_linear_Env:
    def __init__(self,seed,p,d,K):
        np.random.seed(seed)
        self.env='synthetic'
        self.exp_reward=0
        self.p, self.d, self.K, self.K_max = p, d, K, K
        self.K_max=self.K
        self.z=np.zeros((self.K,self.d)) ## true feature
        self.x=np.zeros((self.K,self.d)) ## observed feature
        self.e=np.zeros((self.K,self.d)) ## gaussian noise
        self.m=np.zeros((self.K,self.d)) ## mask for missing entries
        
        self.theta=np.zeros(self.d)
        self.nu=np.zeros(self.d)
        A=np.random.uniform(-1,1,(self.d,self.d))
        B=np.random.uniform(-1,1,(self.d,self.d))
        self.Sigma_f=A.T@A
        self.Sigma_n=B.T@B
        self.Sigma=self.Sigma_f+self.Sigma_n
        
        self.opt_arm=0
        self.nu=np.random.uniform(-1,1,self.d)
        self.nu=self.nu/np.sqrt(np.sum(self.nu**2))
        self.theta=np.random.uniform(-1,1,self.d)
        self.theta=self.theta/np.sqrt(np.sum(self.theta**2))
        self.theta_bar=np.linalg.pinv(self.Sigma)@self.Sigma_f@self.theta
#         self.actset=np.zeros((T,K))
        self.x_bar=np.zeros((self.K,self.d))
        self.exp_reward_opt=[]
    def load_data(self):
        for k in range(self.K): #feature generation
            self.z[k]=np.random.multivariate_normal(self.nu, self.Sigma_f, 1)
            self.e[k]=np.random.multivariate_normal(np.zeros(self.d), self.Sigma_n, 1)
            self.m[k]=np.random.binomial(1,self.p,self.d)
            self.x[k]=(self.z[k]+self.e[k])*self.m[k]
        temp=np.zeros(self.K)
        for k in range(self.K):
            index_S=np.where(self.m[k]!=0)[0]
            index_U=np.where(self.m[k]==0)[0]
            if len(index_U)>0 and len(index_S)>0:
                x_S=self.x[k][index_S]
                x_U=self.nu[index_U]+self.Sigma[np.ix_(index_U,index_S)]@np.linalg.pinv(self.Sigma[np.ix_(index_S,index_S)])@(x_S-self.nu[index_S]).T
                self.x_bar[k,index_S]=x_S
                self.x_bar[k,index_U]=x_U
#                         temp[k]=x_S@self.theta_bar[index_S]+x_U@self.theta_bar[index_U] 
            elif len(index_U)==self.d:
                x_U=self.nu[index_U]
                self.x_bar[k]=x_U
            elif len(index_S)==self.d:
                x_S=self.x[k][index_S]
                self.x_bar[k]=x_S
            temp[k]=self.x_bar[k]@self.theta_bar
        self.opt_arm=np.argmax(temp)
        self.exp_reward_opt.append(self.z[int(self.opt_arm)]@self.theta)
#             print(self.opt_arm[t])

    def observe(self,k): #reward feedback
        reward=self.z[k]@self.theta+np.random.normal(0,1)
        exp_reward=self.z[k]@self.theta
        return exp_reward,reward         