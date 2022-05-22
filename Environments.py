import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Envconfigs import *
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pickle

import pandas as pd

# class movielens_Env:
#     def __init__(self,seed, p, private, load_tail='', option='outer'):
#         np.random.seed(seed)
#         self.match_to_step = False
#         self.autoencoder, self.agg_ftn, self.d = Load_MovieLens(option=option, load_tail=load_tail)
        
#         self.embed_dim = 50
#         self.d = self.embed_dim
        
#         #self.subsampling = False
#         self.p = p
        
#         self.env = 'movielens'
#         self.private = private
        
#         B = np.random.uniform(-1,1,(self.embed_dim,self.embed_dim))
#         self.Sigma_n = B.T @ B
        
#         # Load filtered_dataset
#         self.filtered_data = pd.read_csv('./datasets/ml-100k/preprocess/filtered_data{}.csv'.format(load_tail))
#         # Load user_features dictionary
#         with open('./datasets/ml-100k/preprocess/user_id_to_feature{}.pickle'.format(load_tail), 'rb') as f:
#             self.user_id_to_feature = pickle.load(f)
#         # Load top_movie_features
#         self.top_movies_array = np.load('./datasets/ml-100k/preprocess/top_movies_array{}.npy'.format(load_tail))
        
#         self.K = self.top_movies_array.shape[0]
#         self.K_max, self.K_min = self.K, self.K
        
#         # For iteration for the Dataframe
#         self.row_head = 0
#         self.unused_data = self.filtered_data.copy()
#         self.used_idx = []
#         self.len_data = len(self.unused_data)
        
#     def load_data(self):
        
#         # Move Reading Head
#         self.row_head += 1    
    
#         # Reset unused_data dataframe with row_head achieves the last data
#         if self.row_head >= self.len_data:
#             self.unused_data.drop(self.unused_data.index[self.used_idx]) 
#             if len(self.unused_data) == 0:
#                 self.unused_data = self.filtered_data.copy()
#             self.row_head = 0
#             self.used_idx = []
        
#         user_id  = self.unused_data.loc[self.row_head,"user_id"]
#         movie_id = self.unused_data.loc[self.row_head,"movie_id"]
#         self.reward   = self.unused_data.loc[self.row_head,"reward"]
#         self.arms = np.arange(self.K)
#         self.chosen_idx = movie_id
        
#         # Construct the Context Feature
#         context_list = []
#         user_feature = self.user_id_to_feature[user_id]
#         for i in range(self.K):
#             movie_feature= self.top_movies_array[i,:]
#             context_list.append(self.agg_ftn(user_feature, movie_feature))
#         self.x = np.vstack(context_list).copy()
        
#         with torch.no_grad():
#             x = torch.from_numpy(self.x).type(torch.FloatTensor).to(device)
#             x = self.autoencoder.encoding_result(x)
#             self.x = x.cpu().numpy()
        
#         self.m = np.zeros((len(self.arms),self.d))
#         for k in range(len(self.arms)):
#             self.m[k] = np.random.binomial(1, self.p, self.d)
#             if self.private == False:
#                 self.x[k] = self.x[k] * self.m[k]
#             else:
#                 noise = np.random.multivariate_normal(np.zeros(self.d), self.Sigma_n, 1)
#                 self.x[k] = (self.x[k] + noise) * self.m[k]
                
#     def write_used_idx(self):
#         self.used_idx.append(self.row_head)
        
#     def observe(self, k):
        
#         if k == self.chosen_idx:
#             exp_reward = self.reward
#         else:
#             exp_reward = 0
#         reward = exp_reward
        
#         return exp_reward,reward         
        
# ######################################################################

# class yahoo_Env:
#     def __init__(self,seed, p, K_max, private, load_tail='', option='raw',ind='01'):
#         np.random.seed(seed)
#         self.match_to_step = True
#         self.f = open('datasets/R6A/yahoo-a1.txt')
#         self.option=option
#         self.autoencoder, self.agg_ftn, self.d = Load_Yahoo(option=self.option, load_tail=load_tail)
#         if option=='raw':
#             self.d=36
        
#         self.subsampling = True
#         self.K_max = K_max
#         self.p = p
        
#         self.env = 'yahoo'
#         self.private = private
        
#         B = np.random.uniform(-1,1,(self.d,self.d))
#         self.Sigma_n = B.T @ B
        
#     def load_data(self):
#         line = self.f.readline()
#         if not line:
#             self.f.seek(0) # back to top
#             line = self.f.readline() # read again
        
#         self.data_preprocess(line.strip().split(' |'))   
        
#         if self.subsampling and len(self.arms) > self.K_max:
#             subsample = []
#             while self.chosen_idx not in subsample:
#                 subsample = np.sort(np.random.choice(len(self.arms), self.K_max, replace=False))
#             arms_subsample = np.array(self.arms)[subsample]
#             chosen_idx_subsample = np.where(subsample == self.chosen_idx)[0][0]
#             contexts_subsample = self.x[subsample, :]

#             self.chosen_idx = chosen_idx_subsample
#             self.x = contexts_subsample
#             self.arms = arms_subsample
#         self.K = len(self.arms)
        
#     def data_preprocess(self,raw_data):
#         info = raw_data[0].split(' ')
#         self.time = int(info[0])
#         self.chosen_arm = int(info[1])
#         self.is_clicked = bool(int(info[2]))
#         self.user_context = self.parse_context(raw_data[1].split(' '))[1]
        
#         self.arms = []
#         self.arm_contexts = []
#         for i in range(2, len(raw_data)):
#             arm, context = self.parse_context(raw_data[i].split(' '))
#             self.arms.append(arm)
#             self.arm_contexts.append(context)
        
#         if self.option=='raw':
#             self.x=np.outer(self.user_context, self.arm_contexts[0]).flatten().copy()
#             for j in range(1, len(self.arm_contexts)):
#                 self.x = np.vstack((self.x,np.outer(self.user_context, self.arm_contexts[j]).flatten().copy()))            
#         else:    
#             self.x = self.agg_ftn(self.user_context, self.arm_contexts[0])

#             for j in range(1, len(self.arm_contexts)):
#                 self.x = np.vstack((self.x,self.agg_ftn(self.user_context, self.arm_contexts[j])))

#             with torch.no_grad():
#                 x = torch.from_numpy(self.x).type(torch.FloatTensor).to(device)
#                 x = self.autoencoder.encoding_result(x)
#                 self.x = x.cpu().numpy()
        
#         self.chosen_idx = self.arms.index(self.chosen_arm)
#         self.m = np.zeros((len(self.arms),self.d))

#         for k in range(len(self.arms)):
#             self.m[k] = np.random.binomial(1, self.p, self.d)
#             if self.private == False:
#                 self.x[k] = self.x[k] * self.m[k]
#             else:
#                 noise = np.random.multivariate_normal(np.zeros(self.d), self.Sigma_n, 1)
#                 self.x[k] = (self.x[k] + noise) * self.m[k]
            
#     def parse_context(self,raw_context):
        
#         if raw_context[0] == 'user':
#             arm = 'user'
#         else:
#             arm = int(raw_context[0])

#         context = np.zeros(6)
#         for i in range(1, len(raw_context)):
#             # idx_ctx = raw_context[i].split(':')
#             # context[int(idx_ctx[0])-2] = float(idx_ctx[1])        # context = np.zeros(6)
#         # for i in range(1, len(raw_context)):
#             idx_ctx = raw_context[i].split(':')
#             context[int(idx_ctx[0])-1] = float(idx_ctx[1])

#         return arm, context
    
#     def write_used_idx(self):
#         pass
    
#     def observe(self, k):
        
#         if self.match_to_step:
#             return bool(self.is_clicked), bool(self.is_clicked)
        
#         else:
#             if k == self.chosen_idx:
#                 exp_reward = bool(self.is_clicked)
#             else:
#                 exp_reward = 0
#             reward = exp_reward

#             return exp_reward,reward  

######################################################################
    
class noisy_linear_Env:
    def __init__(self,seed,p,d,K):
        self.seed=seed
        np.random.seed(self.seed)
        self.env='synthetic'
        self.match_to_step = False
        
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
        self.Sigma_f=self.Sigma_f/np.linalg.norm(self.Sigma_f,2)
        self.Sigma_n=B.T@B
        self.Sigma_n=self.Sigma_n/np.linalg.norm(self.Sigma_f,2)
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
        # self.opt_arm=np.random.choice(np.argwhere(temp==np.amax(temp)).flatten().tolist())
        self.exp_reward_opt.append(self.z[int(self.opt_arm)]@self.theta)
        self.opt_arm_origin=np.argmax(self.z@self.theta)
#             print(self.opt_arm[t])

    def write_used_idx(self):
        pass

    def observe(self,k): #reward feedback
        reward=self.z[k]@self.theta+np.random.normal(0,1)
        exp_reward=self.z[k]@self.theta
        return exp_reward,reward