import random
import numpy as np
import math
from tqdm import tqdm
from numpy.random import seed
from numpy.random import rand
from Environments import *

class CLBEF:
    def __init__(self,T,seed,Environment,exploration=True):
        print('Algorithm: CLBEF')
        self.Env=Environment
        np.random.seed(seed)
        self.exploration = exploration
        
        self.d = self.Env.d
        self.xi = np.zeros(self.d)
        self.Z = np.zeros((self.d,self.d))
        self.T = T
        self.alpha = 0.5
        self.tau=self.d/(self.alpha**4*self.Env.K_max)
        
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        self.n=0
        
        self.V1=(self.d+1) * math.log(self.Env.K_max * T) * np.identity(self.d+1)
        self.V2=np.zeros((self.d+1,self.d+1))
        self.act=np.zeros(T)
        self.t_up=[]
        self.xy1=np.zeros(self.d+1)
        self.xy2=np.zeros(self.d+1)
        self.K_sum=0
        self.x_his=[]
        self.m_his=[]
        
        self.run()
        
    def run(self):
        
        for t in tqdm(range(self.T)):
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
                    
                self.p_hat, self.nu_hat, self.Sigma_hat, self.x_hat = self._get_estimator(K,x_t,m_t,K_sum_tmp,xi_tmp,n_tmp,Z_tmp)
                
                if t<self.tau and self.exploration==True:
                    chosen_arm=np.random.choice(range(K))    
                
                elif t==0:
                    chosen_arm=np.random.choice(range(K))
                    
                else: 
                    self.V2=np.zeros((self.d+1,self.d+1))
                    self.xy2=np.zeros(self.d+1)
                    if self.exploration==False:
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
                        ucb = self._get_UCB(self.x_hat[k], t, K)
                        if ucb>max_ucb:
                            chosen_arm=k
                            max_ucb=ucb
                            
                if self.Env.env!='yahoo':
                    bool_=False
                elif chosen_arm==self.Env.chosen_idx:
                    bool_=False
                    
            self.n=n_tmp
            self.Z=Z_tmp
            self.xi=xi_tmp
            self.K_sum=K_sum_tmp
            
            self.r_Exp[t], self.r[t] = self.Env.observe(chosen_arm)
            self.x_his.append(x_t[chosen_arm])
            self.m_his.append(m_t[chosen_arm])
            
            if self.exploration==False:
                if self.Env.m[chosen_arm].sum()!=self.d:
                    self.t_up.append(t)
                else:
                    self.V1+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
                    self.xy1+=self.x_hat[chosen_arm]*self.r[t]
            elif t>self.tau:
                self.V1+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
                self.xy1+=self.x_hat[chosen_arm]*self.r[t]
            '''
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
                    x_hat=np.insert(self._x_bar(self.nu_hat,self.Sigma_hat,x,m),0,1)
                    self.V2+=np.outer(x_hat,x_hat)
                    self.xy2+=x_hat*self.r[s]
            '''

    def _get_UCB(self, x_hat, t, K):
        return x_hat@self.theta_hat+\
                (math.sqrt((self.d+1)*math.log((t+2)*self.T))+\
                 math.sqrt(self.d*math.log(K*self.T)))*math.sqrt(x_hat@self.V_inv@x_hat.T)
                    
    def _x_bar(self, nu, Sigma, x, m):
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
    
    def _get_estimator(self,K,x_t,m_t,K_sum_tmp,xi_tmp,n_tmp,Z_tmp):
        p_hat = max(1,n_tmp) / (self.d * K_sum_tmp)
        nu_hat = 1 / (K_sum_tmp * p_hat) * xi_tmp
        Sigma_hat = Z_tmp*(((p_hat-1)/(p_hat**2))*np.identity(self.d)+1/(p_hat**2))/(K_sum_tmp)-np.outer(nu_hat,nu_hat)
        
        x_hat=np.zeros((self.Env.K_max,self.d+1))
        for k in range(K):
            x_hat[k]=np.insert(self._x_bar(nu_hat,Sigma_hat,x_t[k],m_t[k]),0,1)
        
        return p_hat, nu_hat, Sigma_hat, x_hat
    
    def rewards(self):
        return self.r_Exp  

class OFUL:
    def __init__(self,T,seed,Environment):
        print('Algorithm: OFUL')
        self.Env = Environment
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        self.d = self.Env.d
        self.Z = np.zeros((self.d,self.d))
        self.T = T
        
        self.V = (self.d+1)*math.log(self.Env.K_max*T)*np.identity(self.d+1)
        self.y = np.zeros(self.d+1)
        self.x_hat = np.zeros((self.Env.K_max,self.d+1))
        np.random.seed(seed)

        for t in tqdm(range(T)):
            bool_=True
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
                        ucb = self._get_UCB(self.x_hat[k], t, K)
                        if ucb>max_ucb:
                            chosen_arm=k
                            max_ucb=ucb
                            
                if self.Env.env!='yahoo':
                    bool_=False
                elif chosen_arm==self.Env.chosen_idx:
                    bool_=False
           
            self.r_Exp[t],self.r[t]=self.Env.observe(chosen_arm)
            self.V+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
            self.y+=self.x_hat[chosen_arm]*self.r[t]
            self.theta_hat=np.linalg.pinv(self.V)@self.y    
            
    def _get_UCB(self, x_hat, t, K):
        return x_hat@self.theta_hat+\
                (math.sqrt((self.d+1)*math.log((t+2)*self.T))+\
                 math.sqrt(self.d*math.log(K*self.T)))*math.sqrt(x_hat@np.linalg.pinv(self.V)@x_hat.T)
            
    def rewards(self):
        return self.r_Exp
    
    
class RandomPolicy:
    def __init__(self,T,seed,Environment):
        print('Algorithm: RandomPolicy')
        self.Env = Environment
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        self.d = self.Env.d
        self.Z = np.zeros((self.d,self.d))
        self.T = T
        
        self.V = (self.d+1)*math.log(self.Env.K_max*T)*np.identity(self.d+1)
        self.y = np.zeros(self.d+1)
        self.x_hat = np.zeros((self.Env.K_max,self.d+1))
        np.random.seed(seed)

        for t in tqdm(range(T)):
            bool_=True
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
                        ucb = self._get_UCB(self.x_hat[k], t, K)
                        if ucb>max_ucb:
                            chosen_arm=k
                            max_ucb=ucb
                            
                if self.Env.env!='yahoo':
                    bool_=False
                elif chosen_arm==self.Env.chosen_idx:
                    bool_=False
           
            self.r_Exp[t],self.r[t]=self.Env.observe(chosen_arm)
            self.V+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
            self.y+=self.x_hat[chosen_arm]*self.r[t]
            self.theta_hat=np.linalg.pinv(self.V)@self.y    
            
    def _get_UCB(self, x_hat, t, K):
        return x_hat@self.theta_hat+\
                (math.sqrt((self.d+1)*math.log((t+2)*self.T))+\
                 math.sqrt(self.d*math.log(K*self.T)))*math.sqrt(x_hat@np.linalg.pinv(self.V)@x_hat.T)
            
    def rewards(self):
        return self.r_Exp  