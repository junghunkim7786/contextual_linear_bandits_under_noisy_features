import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environments import *

    
    
    
class CLBEF:
    def __init__(self,T,seed,Environment):
        print('CLBEF')
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        self.n=0
        self.d=self.Env.d
        self.xi=np.zeros(self.d)
        self.Z=np.zeros((self.d,self.d))
        self.x_hat=np.zeros((self.Env.K_max,self.d+1))
        self.nu_hat=np.zeros(self.d)
        self.Sigma_hat=np.zeros((self.d,self.d))
        self.p_hat=0
        self.theta_hat=np.zeros(self.d+1)
        self.d=self.Env.d
        self.V1=(self.d+1)*math.log(self.Env.K_max*T)*np.identity(self.d+1)
        self.V2=np.zeros((self.d+1,self.d+1))
        self.act=np.zeros(T)
        self.t_up=[]
        self.xy1=np.zeros(self.d+1)
        self.xy2=np.zeros(self.d+1)
        np.random.seed(seed)
        self.K_sum=0
        self.x_his=[]
        self.m_his=[]
        for t in range(T):
            if t%10==0:
                print(t)
            bool_=True
            while bool_:
                self.Env.load_data()
                x_t=self.Env.x.copy()
                m_t=self.Env.m.copy()
                K=self.Env.K
                K_sum_tmp=self.K_sum+K
                n_tmp=self.n
                Z_tmp=self.Z
                xi_tmp=self.xi
                for k in range(K):
                    n_tmp=n_tmp+m_t[k].sum()
                    Z_tmp=Z_tmp+np.outer(x_t[k],x_t[k])
                    xi_tmp=xi_tmp+x_t[k]
                self.p_hat=max(1,n_tmp)/(self.d*K_sum_tmp)
                self.nu_hat=1/(K_sum_tmp*self.p_hat)*xi_tmp
                self.Sigma_hat=Z_tmp*(((self.p_hat-1)/(self.p_hat**2))*np.identity(self.d)+1/(self.p_hat**2))/(K_sum_tmp)-np.outer(self.nu_hat,self.nu_hat)
                for k in range(K):
                    self.x_hat[k]=np.insert(self.x_bar(self.nu_hat,self.Sigma_hat,x_t[k],m_t[k]),0,1)
                    
                if t==0:
                    chosen_arm=np.random.choice(range(K))
                    
                else: 
                    if self.Env.env!='yahoo':
                        self.V2=np.zeros((self.d+1,self.d+1))
                        self.xy2=np.zeros(self.d+1)
                        for s in self.t_up:
#                             a_s=int(self.act[s])
                            x=self.x_his[s]
                            m=self.m_his[s]
#                             print(x,self.x[s,a_s])
                            x_hat=np.insert(self.x_bar(self.nu_hat,self.Sigma_hat,x,m),0,1)
                            self.V2+=np.outer(x_hat,x_hat)
                            self.xy2+=x_hat*self.r[s]
                    V=self.V1+self.V2
                    xy=self.xy1+self.xy2
                    self.V_inv=np.linalg.pinv(V)
                    self.theta_hat=self.V_inv@xy.T
                    max_ucb=-np.inf
                    for k in range(K):
                        ucb=self.x_hat[k]@self.theta_hat+(math.sqrt((self.d+1)*math.log((t+2)*T))+math.sqrt(self.d*math.log(K*T)))*math.sqrt(self.x_hat[k]@self.V_inv@self.x_hat[k].T)
                        if ucb>max_ucb:
                            chosen_arm=k
                            max_ucb=ucb
                if self.Env.env=='synthetic':
                    bool_=False
                    
                elif self.Env.env=='yahoo' and chosen_arm==self.Env.chosen_idx:
                    bool_=False
                    
            
            
            self.n=n_tmp
            self.Z=Z_tmp
            self.xi=xi_tmp
            self.K_sum=K_sum_tmp
#             self.act[t]=chosen_arm
            self.r_Exp[t],self.r[t]=self.Env.observe(chosen_arm)
            self.x_his.append(x_t[chosen_arm])
            self.m_his.append(m_t[chosen_arm])

            if self.Env.m[chosen_arm].sum()!=self.d:
                self.t_up.append(t)
            else:
                self.V1+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
                self.xy1+=self.x_hat[chosen_arm]*self.r[t]
                
            if self.Env.env=='yahoo':
                self.V2=np.zeros((self.d+1,self.d+1))
                self.xy2=np.zeros(self.d+1)
                for s in self.t_up:
                    x=self.x_his[s]
                    m=self.m_his[s]
                    x_hat=np.insert(self.x_bar(self.nu_hat,self.Sigma_hat,x,m),0,1)
                    self.V2+=np.outer(x_hat,x_hat)
                    self.xy2+=x_hat*self.r[s]
                    
                    
    def x_bar(self, nu, Sigma, x, m):
        x_hat=np.zeros(self.d)
        index_S=np.where(m!=0)[0]
        index_U=np.where(m==0)[0]
        if len(index_S)>0:    
            x_S=x[index_S]
            x_hat[index_S]=x_S
        if len(index_U)>0 and len(index_S)>0:
            x_U=nu[index_U]+Sigma[np.ix_(index_U,index_S)]@np.linalg.pinv(Sigma[np.ix_(index_S,index_S)])@(x_S-nu[index_S]).T
            x_hat[index_U]=x_U
        if len(index_S)==0:
            x_U=nu[index_U]
            x_hat[index_U]=x_U
        return x_hat
    
    def rewards(self):
        return self.r_Exp  

class OFUL:
    def __init__(self,T,seed,Environment):
        print('OFUL')
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        self.d=self.Env.d
        self.Z=np.zeros((self.d,self.d))
        self.V=np.zeros((self.d+1,self.d+1))
#         self.act=np.zeros(T)
#         self.V=np.identity(self.d+1)
        self.V=(self.d+1)*math.log(self.Env.K_max*T)*np.identity(self.d+1)
        self.y=np.zeros(self.d+1)
        self.x_hat=np.zeros((self.Env.K_max,self.d+1))
        np.random.seed(seed)

        for t in range(T):
            bool_=True
            if t%10==0:
                print(t)
            while bool_:
                self.Env.load_data()
                K=self.Env.K
                x_t=self.Env.x.copy()

                for k in range(K):
                    self.x_hat[k]=np.insert(x_t[k],0,1)
                if t==0:
                    chosen_arm=np.random.choice(range(K))
                else:
                    max_ucb=-np.inf
                    for k in range(K):
                        ucb=self.x_hat[k]@self.theta_hat+(math.sqrt((self.d+1)*math.log((t+2)*T))+math.sqrt(self.d*math.log(K*T)))*math.sqrt(self.x_hat[k]@np.linalg.pinv(self.V)@self.x_hat[k].T)
                        if ucb>max_ucb:
                            chosen_arm=k
                            max_ucb=ucb
                if self.Env.env=='synthetic':
                    bool_=False
                elif self.Env.env=='yahoo' and chosen_arm==self.Env.chosen_idx:
                    bool_=False
           
            self.r_Exp[t],self.r[t]=self.Env.observe(chosen_arm)
            self.V+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
            self.y+=self.x_hat[chosen_arm]*self.r[t]
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
