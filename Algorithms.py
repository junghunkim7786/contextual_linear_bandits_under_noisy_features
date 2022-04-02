import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environments import *


class OFUL_EF:
    def __init__(self,d,K,T,seed,Environment):
        print('OFUL_EF')
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        self.n=0
        self.xi=np.zeros(d)
        self.Z=np.zeros((d,d))
        self.x_hat=np.zeros((T,K,d+1))
        self.nu_hat=np.zeros(d)
        self.Sigma_hat=np.zeros((d,d))
        self.p_hat=0
        self.theta_hat=np.zeros(d+1)
        self.d=d
        self.V1=(d+1)*math.log(K*T)*np.identity(d+1)
        self.V2=np.zeros((d+1,d+1))
        self.act=np.zeros(T)
        self.t_up=[]
        self.xy1=np.zeros(self.d+1)
        self.xy2=np.zeros(self.d+1)
        np.random.seed(seed)
        for t in range(T):
            if t%1000==0:
                print(t)
#                 print(self.p_hat)
#                 print(self.nu_hat)
#                 print(self.Sigma_hat)
            if t==0:
                a=np.random.choice(np.nonzero(self.Env.actset[t])[0])
            else:
                for k in range(K):
                    self.n=self.n+self.Env.m[t-1,k].sum()
                    self.Z=self.Z+np.outer(self.Env.x[t-1,k],self.Env.x[t-1,k])
                    self.xi=self.xi+self.Env.x[t-1,k]
                self.p_hat=max(1,self.n)/(t*self.d*K)
                self.nu_hat=1/(t*K*self.p_hat)*self.xi
                self.Sigma_hat=self.Z*(((self.p_hat-1)/(self.p_hat**2))*np.identity(self.d)+1/(self.p_hat**2))/(t*K)-np.outer(self.nu_hat,self.nu_hat)
    
                for k in range(K):
                    self.x_hat[t,k]=np.insert(self.x_bar(self.nu_hat,self.Sigma_hat,k,t),0,1)
#                     if not np.array_equal(np.insert(self.Env.x[t,k],0,1),self.x_hat[t,k]):
#                         print(np.insert(self.Env.x[t,k],0,1),self.x_hat[t,k])
#                 temp=np.zeros(self.d+1)
                self.V2=np.zeros((self.d+1,self.d+1))
                self.xy2=np.zeros(self.d+1)
                for s in self.t_up:
                    a_s=int(self.act[s])
                    self.x_hat[s,a_s]=np.insert(self.x_bar(self.nu_hat,self.Sigma_hat,a_s,s),0,1)
                    self.V2+=np.outer(self.x_hat[s,a_s],self.x_hat[s,a_s])
                    self.xy2+=self.x_hat[s,a_s]*self.r[s]
                V=self.V1+self.V2
                xy=self.xy1+self.xy2
                self.V_inv=np.linalg.pinv(V)
                self.theta_hat=self.V_inv@xy.T
                temp=-np.inf
                for b in np.nonzero(self.Env.actset[t])[0]:
                    c=self.x_hat[t,b]@self.theta_hat+(math.sqrt((self.d+1)*math.log((t+2)*T))+math.sqrt(self.d*math.log(K*T)))*math.sqrt(self.x_hat[t,b]@self.V_inv@self.x_hat[t,b].T)
                    if c>temp:
                        a=b
                        temp=c
            self.act[t]=a
            self.r_Exp[t],self.r[t]=self.Env.observe(t,a)
            if self.Env.m[t,a].sum()!=d:
                self.t_up.append(t)
            else:
                self.V1+=np.outer(self.x_hat[t,a],self.x_hat[t,a])
                self.xy1+=self.x_hat[t,a]*self.r[t]
    def x_bar(self, nu, Sigma, k, t):
        x=np.zeros(self.d)
        index_S=np.where(self.Env.m[t,k]!=0)[0]
        index_U=np.where(self.Env.m[t,k]==0)[0]
        if len(index_S)>0:    
            x_S=self.Env.x[t,k][index_S]
            x[index_S]=x_S
        if len(index_U)>0 and len(index_S)>0:
            x_U=nu[index_U]+Sigma[np.ix_(index_U,index_S)]@np.linalg.pinv(Sigma[np.ix_(index_S,index_S)])@(x_S-nu[index_S]).T
            x[index_U]=x_U
        if len(index_S)==0:
            x_U=nu[index_U]
            x[index_U]=x_U
        return x
    
    def rewards(self):
        return self.r_Exp  

class OFUL:
    def __init__(self,d,K,T,seed,Environment):
        print('OFUL')
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        self.Z=np.zeros((d,d))
        self.V=np.zeros((d+1,d+1))
        self.act=np.zeros(T)
        self.d=d
#         self.V=np.identity(self.d+1)
        self.V=(self.d+1)*math.log(K*T)*np.identity(self.d+1)
        self.y=np.zeros(self.d+1)
        self.x_hat=np.zeros((T,K,d+1))
        np.random.seed(seed)

        for t in range(T):
            if t%1000==0:
                print(t)
            for k in range(K):
                self.x_hat[t,k]=np.insert(self.Env.x[t,k],0,1)
            if t==0:
                a=np.random.choice(np.nonzero(self.Env.actset[t])[0])
            else:
                temp=-np.inf
                for b in np.nonzero(self.Env.actset[t])[0]:
#                     c=self.x_hat[t,b]@self.theta_hat+(math.sqrt(self.d*math.log(t*T)))*math.sqrt(self.x_hat[t,b]@np.linalg.pinv(self.V)@self.x_hat[t,b].T)
                    c=self.x_hat[t,b]@self.theta_hat+(math.sqrt((self.d+1)*math.log((t+2)*T))+math.sqrt(self.d*math.log(K*T)))*math.sqrt(self.x_hat[t,b]@np.linalg.pinv(self.V)@self.x_hat[t,b].T)
                    if c>temp:
                        a=b
                        temp=c
            self.r_Exp[t],self.r[t]=self.Env.observe(t,a)
            self.V+=np.outer(self.x_hat[t,a],self.x_hat[t,a])
            self.y+=self.x_hat[t,a]*self.r[t]
            self.theta_hat=np.linalg.pinv(self.V)@self.y    
    def rewards(self):
        return self.r_Exp  

class OFUL_origin:
    def __init__(self,d,K,T,seed,Environment):
        print('OFUL_origin')
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        self.V=np.zeros((d,d))
        self.act=np.zeros(T)
        self.d=d
        self.V=(self.d)*math.log(K*T)*np.identity(self.d)
        self.y=np.zeros(self.d)
        for t in range(T):
            if t%1000==0:
                print(t)
            if t==0:
                a=np.random.choice(np.nonzero(self.Env.actset[t])[0])
            else:
                temp=-np.inf
#                 print(np.nonzero(self.Env.actset[t])[0])
                for b in np.nonzero(self.Env.actset[t])[0]:
                    c=self.Env.x[t,b]@self.theta_hat+(math.sqrt(self.d*math.log((t+1)*T))+math.sqrt(self.d*math.log(K*T)))*math.sqrt(self.Env.x[t,b]@np.linalg.pinv(self.V)@self.Env.x[t,b].T)
                    if c>temp:
                        a=b
                        temp=c
            self.r_Exp[t],self.r[t]=self.Env.observe(t,a)
            self.V+=self.Env.x[t,a]@self.Env.x[t,a].T
            self.y+=self.Env.x[t,a]*self.r[t]
            self.theta_hat=np.linalg.pinv(self.V)@self.y    
            
    def rewards(self):
        return self.r_Exp  
