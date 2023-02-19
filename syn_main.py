from syn_exp.runoptions import *

import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import sys

import pandas as pd
        
if __name__=='__main__':
    
    opt = str(sys.argv[1])
    opt2= str(sys.argv[2]) 
    
    if opt2=='load':
        run_bool=False
    elif opt2=='run':
        run_bool=True
        
    repeat=10
    d = 2
    T = 10000
        
        
    if opt=='1': ##Figure 1 (a),(b)
        p_list=[0.9,0.8,0.7,0.6]
        K=100
        run_p(p_list,d,K,T,repeat,run_bool)  
        K=30
        run_p(p_list,d,K,T,repeat,run_bool)
        
    elif opt=='2': ##Figure 2 (c), (d)
        p=0.7
        K=100
        run1(p,d,K,T,repeat,run_bool)
        p=0.93
        run1(p,d,K,T,repeat,run_bool)