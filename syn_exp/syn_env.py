import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pickle

import pandas as pd

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
            elif len(index_U)==self.d:
                x_U=self.nu[index_U]
                self.x_bar[k]=x_U
            elif len(index_S)==self.d:
                x_S=self.x[k][index_S]
                self.x_bar[k]=x_S
            temp[k]=self.x_bar[k]@self.theta_bar
        self.opt_arm=np.argmax(temp)
        self.exp_reward_opt.append(self.z[int(self.opt_arm)]@self.theta)
        self.opt_arm_origin=np.argmax(self.z@self.theta)

    def observe(self,k): #reward feedback
        reward=self.z[k]@self.theta+np.random.normal(0,1)
        exp_reward=self.z[k]@self.theta
        return exp_reward,reward