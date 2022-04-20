from Runoptions import *
import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import sys
        
if __name__=='__main__':
    # Read input
    opt = str(sys.argv[1])
    opt2= str(sys.argv[2]) 
    ##'1': figure1, '2': figure2, '3': figure3
    if opt2=='load':##  True: run model and save data with plot, False: load data with plot.
        run_bool=False
    elif opt2=='run':
        run_bool=True
    repeat=1 # repeat number of running algorithms with different seeds.
    d=2
    T=500  #Time horizon
    num=10
    if opt=='1':
        p=0.9
        #K=5 # Yahoo
        K = 50
        #T=1000 # Yahoo
        T= 50000
        private=True
        repeat=1
        run_real(p,K,T,repeat,private,run_bool)
        private=False
#         repeat=1
        run_real(p,K,T,repeat,private,run_bool)
    
    elif opt=='2':
        repeat=10
        T=1000
        p=0.5
        K=10
        run1(p,d,K,T,repeat,run_bool)
        p=0.99
        run1(p,d,K,T,repeat,run_bool)
        
    elif opt=='3':
        repeat=10
        T=1000
        p_list=[1,0.9,0.8,0.7,0.6,0.5]
        K=10
        run_p(p_list,d,K,T,repeat,run_bool)  
        K=2
        run_p(p_list,d,K,T,repeat,run_bool)  
