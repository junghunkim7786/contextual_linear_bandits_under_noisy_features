import random
import numpy as np
import math
from tqdm import tqdm
from numpy.random import seed
from numpy.random import rand
from numba import jit, prange

@jit(nopython=True, cache=False)  # cache=False only for performance comparison
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))

######################################################################################

@jit(nopython=True)
def _CLBEF_get_UCB(x_hat, t, K, theta_hat, d, T, V_inv):
    return x_hat@theta_hat+\
            (np.sqrt((d+1)*np.log((t+2)*T))+\
             np.sqrt(d*np.log(K*T)))*np.sqrt(x_hat@V_inv@x_hat.T)

@jit(nopython=True, parallel=True)
def _CLBEF_UCB(x_hat, t, K, theta_hat, d, T, V_inv):
    
    ucb_list = np.zeros(K)
    for k in prange(K):
        ucb_list[k] = _CLBEF_get_UCB(x_hat[k], t, K, theta_hat.copy(), d, T, V_inv.copy())
        
    chosen_arm = np.argmax(ucb_list)
    max_ucb = ucb_list[chosen_arm]

    return chosen_arm, max_ucb

@jit(nopython=True, cache=False)
def _CLBEF_x_bar(nu, x_hat, Sigma, x, m):
    index_S=np.where(m!=0)[0]
    index_U=np.where(m==0)[0]

    if len(index_S)>0:    
        x_S=x[index_S]
        x_hat[index_S]=x_S

    if len(index_U)>0 and len(index_S)>0:
        x_U=nu[index_U]+numba_ix(Sigma,index_U,index_S)@np.linalg.pinv(numba_ix(Sigma,index_S,index_S))@(x_S-nu[index_S]).T
        x_hat[index_U]=x_U

    if len(index_S)==0:
        x_U=nu[index_U]
        x_hat[index_U]=x_U

    return x_hat

@jit(nopython=True, parallel=True)
def _CLBEF_x_hats(K_max, K, d, nu_hat, Sigma_hat, x_t, m_t):
    
        x_hat=np.zeros((K_max,d+1))
        dummy = np.zeros((K_max,d))
        for k in prange(K):
            x_hat[k] = np.append(1.0,_CLBEF_x_bar(nu_hat.copy(),dummy[k],Sigma_hat.copy(),x_t[k],m_t[k]))
        return x_hat

class CLBEF:
    def __init__(self,T,Env):
        print('Algorithm: CLBBF')
        self.Env = Env
        np.random.seed(self.Env.seed)
        
        self.d  = self.Env.d
        self.xi = np.zeros(self.d)
        self.Z  = np.zeros((self.d,self.d))
        self.T  = T
        self.alpha = 0.5
        self.tau = self.d/(self.alpha**4*self.Env.K_max)
        
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        self.n = 0
        
        self.V = (self.d+1) * math.log(self.Env.K_max * T) * np.identity(self.d+1)
        self.act = np.zeros(T)
        self.t_up = []
        self.xy = np.zeros(self.d+1)
        
        self.K_sum = 0
        self.x_his = []
        self.m_his = []
        self.i = 1
        
        self.run()
        
    def run(self):
        
        for t in tqdm(range(self.T)):

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

            try:
                self.p_hat, self.nu_hat, self.Sigma_hat, self.x_hat = self._get_estimators(K,x_t,m_t,K_sum_tmp,xi_tmp,n_tmp,Z_tmp)
            except:
                continue

            if t==0:
                chosen_arm=np.random.choice(range(K))

            else: 
                if t==2**self.i:
                    self.V=(self.d+1) * math.log(self.Env.K_max * self.T) * np.identity(self.d+1)
                    self.xy=np.zeros(self.d+1)
                    for s in range(t):
                        x=self.x_his[s]
                        m=self.m_his[s]
                        try:
                            x_hat=np.insert(self._x_bar(self.nu_hat,self.Sigma_hat,x,m),0,1)
                        except:
                            continue
                        self.V+=np.outer(x_hat,x_hat)
                        self.xy+=x_hat*self.r[s]

                self.V_inv=np.linalg.pinv(self.V)
                self.theta_hat=self.V_inv@self.xy.T

                chosen_arm, max_ucb = _CLBEF_UCB(self.x_hat, t, K, self.theta_hat, self.d, self.T, self.V_inv)
                    
            if t==2**self.i:
                self.i=self.i+1
                    
            self.n=n_tmp
            self.Z=Z_tmp
            self.xi=xi_tmp
            self.K_sum=K_sum_tmp
            
            self.r_Exp[t], self.r[t] = self.Env.observe(chosen_arm)
            self.x_his.append(x_t[chosen_arm])
            self.m_his.append(m_t[chosen_arm])
            
            self.V+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
            self.xy+=self.x_hat[chosen_arm]*self.r[t]            
                    
    def _x_bar(self, nu, Sigma, x, m):
        x_hat = np.zeros(self.d)
        return _CLBEF_x_bar(nu, x_hat, Sigma, x, m)
    
    def _get_estimators(self,K,x_t,m_t,K_sum_tmp,xi_tmp,n_tmp,Z_tmp):
        p_hat = max(1,n_tmp) / (self.d * K_sum_tmp)
        nu_hat = 1 / (K_sum_tmp * p_hat) * xi_tmp
        Sigma_hat = Z_tmp*(((p_hat-1)/(p_hat**2))*np.identity(self.d)+1/(p_hat**2))/(K_sum_tmp)-np.outer(nu_hat,nu_hat)
        
        x_hat = _CLBEF_x_hats(self.Env.K_max, K, self.d, nu_hat, Sigma_hat, x_t, m_t)
        
        return p_hat, nu_hat, Sigma_hat, x_hat
    
    def rewards(self):
        return self.r_Exp  

######################################################################################

@jit(nopython=True)
def _OFUL_get_UCB(x_hat, t, K, theta_hat, d, T, V_inv):
    return x_hat@theta_hat+\
            (np.sqrt((d+1)*np.log((t+2)*T))+\
             np.sqrt(d*np.log(K*T)))*np.sqrt(x_hat@V_inv@x_hat.T)

@jit(nopython=True, parallel=True)
def _OFUL_UCB(x_hat, t, K, theta_hat, d, T, V_inv):
    
    ucb_list = np.zeros(K)
    for k in prange(K):
        ucb_list[k] = _OFUL_get_UCB(x_hat[k], t, K, theta_hat.copy(), d, T, V_inv.copy())
        
    chosen_arm = np.argmax(ucb_list)
    max_ucb = ucb_list[chosen_arm]

    return chosen_arm, max_ucb

class OFUL:
    def __init__(self,T,Env):
        print('Algorithm: OFUL')
        self.Env = Env
        np.random.seed(self.Env.seed)
        
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        self.d = self.Env.d
        self.Z = np.zeros((self.d,self.d))
        self.T = T
        
        self.V = (self.d+1)*math.log(self.Env.K_max*T)*np.identity(self.d+1)
        self.y = np.zeros(self.d+1)
        self.x_hat = np.zeros((self.Env.K_max,self.d+1))
        
        self.run()

    def run(self):
        
        for t in tqdm(range(self.T)):

            self.Env.load_data()
            K=self.Env.K
            x_t=self.Env.x.copy()

            for k in range(K):
                self.x_hat[k]=np.insert(x_t[k],0,1)

            if t==0:
                chosen_arm=np.random.choice(range(K))
            else:               
                chosen_arm, max_ucb = _OFUL_UCB(self.x_hat, t, K, self.theta_hat, self.d, self.T, self.V_inv)
                            
            self.r_Exp[t],self.r[t]=self.Env.observe(chosen_arm)
            self.V+=np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
            self.y+=self.x_hat[chosen_arm]*self.r[t]
            
            self.V_inv = np.linalg.pinv(self.V)
            self.theta_hat= self.V_inv@self.y    
            
    def rewards(self):
        return self.r_Exp
    
######################################################################################
    
class RandomPolicy:
    def __init__(self,T,Env):
        print('Algorithm: Random Policy')
        self.Env = Env
        np.random.seed(self.Env.seed)
        
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        self.d = self.Env.d
        self.Z = np.zeros((self.d,self.d))
        self.T = T
        
        self.V = (self.d+1)*math.log(self.Env.K_max*T)*np.identity(self.d+1)
        self.y = np.zeros(self.d+1)
        self.x_hat = np.zeros((self.Env.K_max,self.d+1))

        self.run()
    
    def run(self):
        
        for t in tqdm(range(self.T)):

            self.Env.load_data()
            K=self.Env.K
            x_t=self.Env.x.copy()

            for k in range(K):
                self.x_hat[k]=np.insert(x_t[k],0,1)
            chosen_arm=np.random.choice(range(K))
                    
            self.r_Exp[t],self.r[t]=self.Env.observe(chosen_arm)
            
    def rewards(self):
        return self.r_Exp  
