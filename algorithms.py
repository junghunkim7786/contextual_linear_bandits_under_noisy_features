import random
import numpy as np
import math
from tqdm import tqdm
from numpy.random import seed
from numpy.random import rand
from numba import jit, prange

@jit(nopython=True)
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i in prange(len(rows)):
        r = rows[i]
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))

def numpy_ix(arr, rows, cols):
    return arr[np.ix_(rows, cols)]

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

######################################################################################

@jit(nopython=True)
def _CLBEF_get_UCB(x_hat, t, K, theta_hat, d, T, V_inv, p_hat, x_sum):
    return x_hat@theta_hat+(np.sqrt((d+1)*np.log((t+2)*T))+x_sum*d*((1/p_hat)**(3/2))*np.sqrt(np.log(T)*np.log(K*T)/((t+2)*K))+\
                            np.sqrt(d*np.log(K*T)))*np.sqrt(x_hat@V_inv@x_hat.T)

@jit(nopython=True, parallel=True)
def _CLBEF_UCB(x_hat, t, K, theta_hat, d, T, V_inv,p_hat, x_sum):
    
    ucb_list = np.zeros(K)
    for k in prange(K):
        ucb_list[k] = _CLBEF_get_UCB(x_hat[k], t, K, theta_hat.copy(), d, T, V_inv.copy(), p_hat, x_sum)
        
    chosen_arm = np.argmax(ucb_list)
    max_ucb = ucb_list[chosen_arm]

    return chosen_arm, max_ucb

@jit(nopython=True)
def numba_pinv(M):
    return np.linalg.pinv(M)

@jit(nopython=True)
def numba_idxSU(m):
    d = m.shape[0]
    idxS, idxU = [], []
    for i in range(d):
        if (m[i] < 1.0e-10) and (m[i] > -1.0e-10):
            idxU.append(i)
        else:
            idxS.append(i)
            
    return np.asarray(idxS), np.asarray(idxU)

@jit(nopython=True)
def _CLBEF_x_bars(nu_hat, Sigma_hat, x, m, x_bar_dummy):
    index_S,index_U = numba_idxSU(m)

    if len(index_S)>0:
        x_S = x[index_S]
        x_bar_dummy[index_S] = x_S
        
        if len(index_U)>0:
            x_U = nu_hat[index_U] + numba_ix(Sigma_hat,index_U,index_S) @ np.linalg.pinv(numba_ix(Sigma_hat,index_S,index_S)) @ (x_S-nu_hat[index_S]).T
            x_bar_dummy[index_U] = x_U

    else:
        x_bar_dummy[index_U] = nu_hat[index_U]

    return x_bar_dummy

@jit(nopython=True, parallel=True)
def _CLBEF_x_hats(nu_hat, Sigma_hat, x, m, K, x_bar_dummy, x_hat_dummy):
    
        for k in prange(K):
            x_hat_dummy[k,1:] = _CLBEF_x_bars(nu_hat, Sigma_hat, x[k], m[k], x_bar_dummy[k])
            
        return x_hat_dummy

@jit(nopython=True)
def _CLBEF_get_estimators(d, K,x_t,m_t,Kt,xi,n,Z):
    p_hat = max([1,n]) / (d * Kt)
    nu_hat = 1 / (Kt * p_hat) * xi
    Sigma_hat = Z*(((p_hat-1)/(p_hat*p_hat))*np.identity(d) + 1/(p_hat*p_hat))/(Kt) - np.outer(nu_hat,nu_hat)

    return p_hat, nu_hat, Sigma_hat
    
class CLBBF:
    def __init__(self,T,Env):
        print('Algorithm: CLBBF')
        self.Env = Env
        self.T  = T
        np.random.seed(self.Env.seed)
        
        self.d  = self.Env.d
        self.K = self.Env.K
        
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        
        self.n = 0
        self.Z  = np.zeros((self.d,self.d))
        self.xi = np.zeros(self.d)
        
        self.V = (self.d+1) * math.log(self.K * T) * np.identity(self.d+1)
        self.xy = np.zeros(self.d+1)
        
        self.x_his = []
        self.m_his = []
        
        self.run()
        self.x_sum=0
        
    def run(self):
        
        for t in tqdm(range(self.T)):

            self.Env.load_data()
            x_t = self.Env.x.copy()
            m_t = self.Env.m.copy()
            
            for k in range(self.K):
                self.n += np.sum(m_t[k])
                self.Z += np.outer(x_t[k],x_t[k])
                self.xi+= x_t[k]

            self.p_hat, self.nu_hat, self.Sigma_hat = _CLBEF_get_estimators(self.d, self.K, x_t,m_t, self.K*(t+1), self.xi, self.n, self.Z)
            self.x_hat = self._x_hats(x_t, m_t)

            if t == 0:
                chosen_arm = np.random.choice(self.K)
            else:
                if is_power_of_two(t):
                    self.V = (self.d+1) * math.log(self.Env.K * self.T) * np.identity(self.d+1)
                    self.xy = np.zeros(self.d+1)
                    self.x_sum=0
                    for s in range(t):
                        x = self.x_his[s]
                        m = self.m_his[s]
                        x_hat = np.insert(self._x_bars(x,m), 0, 1)
                        self.V  += np.outer(x_hat,x_hat)
                        self.xy += x_hat * self.r[s]
                    self.V_inv = np.linalg.pinv(self.V)
                    
                    for s in range(t):
                        x = self.x_his[s]
                        m = self.m_his[s]
                        x_hat = np.insert(self._x_bars(x,m), 0, 1)
                        self.x_sum+= np.sqrt(x_hat@self.V_inv@x_hat)
                        
                self.V_inv = np.linalg.pinv(self.V)
                
                self.theta_hat = self.V_inv @ self.xy.T
                chosen_arm, max_ucb = _CLBEF_UCB(self.x_hat, t, self.K, self.theta_hat, self.d, self.T, self.V_inv, self.p_hat, self.x_sum)
                    
            self.r_Exp[t], self.r[t] = self.Env.observe(chosen_arm)
            self.x_his.append(x_t[chosen_arm])
            self.m_his.append(m_t[chosen_arm])
            
            self.V  += np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
            self.xy += self.x_hat[chosen_arm] * self.r[t]            
                    
    def _x_bars(self, x, m):
        x_bar_dummy = np.zeros(self.d)
        return _CLBEF_x_bars(self.nu_hat, self.Sigma_hat, x, m, x_bar_dummy)
    
    def _x_hats(self, x, m):
        x_hat_dummy = np.ones((self.K,self.d+1))
        x_bar_dummy = np.zeros((self.K,self.d))
        return _CLBEF_x_hats(self.nu_hat, self.Sigma_hat, x, m, self.K, x_bar_dummy, x_hat_dummy)
    
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
        self.T  = T
        np.random.seed(self.Env.seed)
        
        self.d  = self.Env.d
        self.K = self.Env.K
        
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        
        self.Z = np.zeros((self.d,self.d))
        
        self.V = (self.d+1)*math.log(self.K*T)*np.identity(self.d+1)
        self.y = np.zeros(self.d+1)
        self.x_hat = np.ones((self.K,self.d+1))
        
        self.run()

    def run(self):
        
        for t in tqdm(range(self.T)):

            self.Env.load_data()
            x_t = self.Env.x.copy()
            self.x_hat[:,1:] = x_t
                
            if t == 0:
                chosen_arm = np.random.choice(self.K)
            else:               
                chosen_arm, max_ucb = _OFUL_UCB(self.x_hat, t, self.K, self.theta_hat, self.d, self.T, self.V_inv)
                            
            self.r_Exp[t], self.r[t] = self.Env.observe(chosen_arm)
            self.V += np.outer(self.x_hat[chosen_arm],self.x_hat[chosen_arm])
            self.y += self.x_hat[chosen_arm] * self.r[t]
            
            self.V_inv = np.linalg.pinv(self.V)
            self.theta_hat = self.V_inv @ self.y    
            
    def rewards(self):
        return self.r_Exp
    
######################################################################################
    
class RandomPolicy:
    def __init__(self,T,Env):
        print('Algorithm: Random Policy')
        self.Env = Env
        self.T  = T
        np.random.seed(self.Env.seed)
        
        self.d  = self.Env.d
        self.K = self.Env.K
        
        self.r = np.zeros(T,float)
        self.r_Exp = np.zeros(T,float)
        self.Z = np.zeros((self.d,self.d))
        
        self.V = (self.d+1) * math.log(self.Env.K*T) * np.identity(self.d+1)
        self.y = np.zeros(self.d+1)
        self.x_hat = np.zeros((self.Env.K,self.d+1))

        self.run()
    
    def run(self):
        
        for t in tqdm(range(self.T)):
            
            self.Env.load_data()
            '''
            x_t=self.Env.x.copy()
            for k in range(self.K):
                self.x_hat[k]=np.insert(x_t[k],0,1)
            '''
            chosen_arm=np.random.choice(self.K)
                    
            self.r_Exp[t],self.r[t] = self.Env.observe(chosen_arm)
            
    def rewards(self):
        return self.r_Exp 
