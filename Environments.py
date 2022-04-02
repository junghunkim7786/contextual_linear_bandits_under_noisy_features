import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand


class noisy_linear_Env:
    def __init__(self,seed,p,d,K,T):
        np.random.seed(seed)
        self.exp_reward=np.zeros(T)
        self.T=T
        self.p=p
        self.d=d
        self.K=K
        self.z=np.zeros((T,K,d)) ## true feature
        self.x=np.zeros((T,K,d)) ## observed feature
        self.e=np.zeros((T,K,d)) ## gaussian noise
        self.m=np.zeros((T,K,d)) ## mask for missing entries
        self.theta=np.zeros(d)
        self.nu=np.zeros(d)
        A=np.random.uniform(-1,1,(d,d))
        B=np.random.uniform(-1,1,(d,d))
#         self.Sigma_f=np.identity(d)
#         self.Sigma_n=np.identity(d)
        self.Sigma_f=A.T@A
        self.Sigma_n=B.T@B
        self.Sigma=self.Sigma_f+self.Sigma_n
        self.opt_arm=np.zeros(T)
        self.nu=np.random.uniform(-1,1,d)
        self.nu=self.nu/np.sqrt(np.sum(self.nu**2))
        self.theta=np.random.uniform(-1,1,d)
        self.theta=self.theta/np.sqrt(np.sum(self.theta**2))
        self.theta_bar=np.linalg.pinv(self.Sigma)@self.Sigma_f@self.theta
        self.actset=np.zeros((T,K))
        self.x_bar=np.zeros((T,K,d))
        for t in range(self.T):
            if t%1000==0:
                print(t)
            successful=False ## for gauranteeing that active set is not empty
            while not successful:
                for k in range(self.K): #feature generation
                    self.z[t,k]=np.random.multivariate_normal(self.nu, self.Sigma_f, 1)
                    self.e[t,k]=np.random.multivariate_normal(np.zeros(self.d), self.Sigma_n, 1)
                    self.m[t,k]=np.random.binomial(1,self.p,self.d)
                    self.x[t,k]=(self.z[t,k]+self.e[t,k])*self.m[t,k]
                    if np.linalg.norm(self.x[t,k],ord=2)<=math.sqrt(self.m[t,k].sum()*math.log(K*T)):
                        self.actset[t,k]=1
                    else:
                        self.actset[t,k]=0
                if self.actset[t].sum()>0:
                    successful=True
                else:
                    successful=False
#             if self.actset[t].sum()!=K:
#                 print(t, self.actset[t].sum())
        for t in range(self.T): #finding the optimal arm
            temp=np.zeros(K)-np.inf
            for k in range(self.K):
                if self.actset[t,k]==1:
                    index_S=np.where(self.m[t,k]!=0)[0]
                    index_U=np.where(self.m[t,k]==0)[0]
                    if len(index_U)>0 and len(index_S)>0:
                        x_S=self.x[t,k][index_S]
                        x_U=self.nu[index_U]+self.Sigma[np.ix_(index_U,index_S)]@np.linalg.pinv(self.Sigma[np.ix_(index_S,index_S)])@(x_S-self.nu[index_S]).T
                        self.x_bar[t,k,index_S]=x_S
                        self.x_bar[t,k,index_U]=x_U
#                         temp[k]=x_S@self.theta_bar[index_S]+x_U@self.theta_bar[index_U]
                    elif len(index_U)==d:
                        x_U=self.nu[index_U]
#                         temp[k]=x_U@self.theta_bar[index_U]
                        self.x_bar[t,k]=x_U
                    elif len(index_S)==d:
                        x_S=self.x[t,k][index_S]
#                         temp[k]=x_S@self.theta_bar[index_S]
                        self.x_bar[t,k]=x_S
                    temp[k]=self.x_bar[t,k]@self.theta_bar
            self.opt_arm[t]=np.argmax(temp)
#             print(self.opt_arm[t])
    def observe(self,t,k): #reward feedback
        reward=self.z[t,k]@self.theta+np.random.normal(0,1)
#         exp_reward=self.x_bar[t,k]@self.theta_bar
        exp_reward=self.z[t,k]@self.theta
        return exp_reward,reward
    
    def opt_reward(self,T):
        exp_reward_opt=np.zeros(T)
        for t in range(T):
#             exp_reward_opt[t]=self.x_bar[t,int(self.opt_arm[t])]@self.theta_bar
            exp_reward_opt[t]=self.z[t,int(self.opt_arm[t])]@self.theta
        return exp_reward_opt

           