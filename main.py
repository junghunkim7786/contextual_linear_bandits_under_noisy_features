from Environments import *
from Algorithms import *
import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import sys

def run_p(p_list,d,K,T,repeat,boolean=True):
    num=len(p_list)
    std_list1=np.zeros(num)
    regret_list1=np.zeros(num)
    std_list2=np.zeros(num)
    regret_list2=np.zeros(num)
    if boolean: ##save data
        for i, p in enumerate(p_list):
            print('p',p)
            regret=np.zeros(T,float)
            regret_sum=np.zeros(T,float)
            regret_sum_list1=np.zeros((repeat,T),float)
            regret_sum_list2=np.zeros((repeat,T),float)
            std1=np.zeros(T,float)
            std2=np.zeros(T,float)
            avg_regret_sum1=np.zeros(T,float)
            avg_regret_sum2=np.zeros(T,float)

        ###Run model
            for j in range(repeat):
                print('repeat: ',j)
                seed=j
                Env=noisy_linear_Env(seed,p,d,K,T)
                algorithm1=OFUL(d,K,T,seed,Env)
                algorithm2=OFUL_EF(d,K,T,seed,Env)
                opti_rewards=Env.opt_reward(T)

                regret=opti_rewards-algorithm1.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list1[j,:]=regret_sum
                avg_regret_sum1+=regret_sum
                
                regret=opti_rewards-algorithm2.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list2[j,:]=regret_sum
                avg_regret_sum2+=regret_sum
                
                
            avg1=avg_regret_sum1/repeat
            sd1=np.std(regret_sum_list1,axis=0)
            avg2=avg_regret_sum2/repeat
            sd2=np.std(regret_sum_list2,axis=0)


            algorithms = ['OFUL','OFUL-EF']
            regret = dict()
            std=dict()
            regret['OFUL']=avg1
            std['OFUL']=sd1
            regret['OFUL-EF']=avg2
            std['OFUL-EF']=sd2

            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]

            
            ##Save data

            filename_1='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'regret.txt'
            with open('./result/'+filename_1, 'wb') as f:
                pickle.dump(regret, f)
                f.close()

            filename_2='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'std.txt'
            with open('./result/'+filename_2, 'wb') as f:
                pickle.dump(std, f)
                f.close()
    
    else: ##load data
        print('load data without running')
        for i, p in enumerate(p_list):
            print(i)
            filename_1='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'regret.txt'
            filename_2='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'std.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            pickle_file2 = open('./result/'+filename_2, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret=objects[0]
            objects = []
            while True:
                try:
                    objects.append(pickle.load(pickle_file2))
                except EOFError:
                    break
            pickle_file2.close()
            std=objects[0]
            avg1=regret['OFUL']
            sd1=std['OFUL']
            avg2=regret['OFUL-EF']
            sd2=std['OFUL-EF']   
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]

    p_list=[1-p for p in p_list]
    
    fig,(ax)=plt.subplots(1,1)
    ax.errorbar(x=p_list, y=regret_list1, yerr=1.96*std_list1/np.sqrt(repeat), color="orange", capsize=3,
                 marker="^", markersize=7,label='OFUL',zorder=1) 
    ax.errorbar(x=p_list, y=regret_list2, yerr=1.96*std_list2/np.sqrt(repeat), color="b", capsize=3,
                 marker="o", markersize=7,label='Algorithm 1',zorder=2)

    
    #font size
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)

    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(r'$1-p$',fontsize=15)
    plt.ylabel(r'$R(T)$',fontsize=15)
    plt.savefig('./result/T'+str(T)+'K'+str(K)+'repeat'+str(repeat)+'p_list'+'.png')
    plt.show()
    plt.clf()        
    
    
def run2(p,d,K,T,repeat,num,boolean=True):
    T_1=int(T/num)
    num=num+1
    std_list1=np.zeros(num)
    regret_list1=np.zeros(num)
    std_list2=np.zeros(num)
    regret_list2=np.zeros(num)
    T_list=np.zeros(num)
    
    if boolean: ##save data
        for i in range(num):
            print(i)
            if i==0:
                T=1
            else:
                T=T_1*i
            T_list[i]=T
            regret=np.zeros(T,float)
            regret_sum=np.zeros(T,float)
            regret_sum_list1=np.zeros((repeat,T),float)
            regret_sum_list2=np.zeros((repeat,T),float)
            std1=np.zeros(T,float)
            std2=np.zeros(T,float)
            avg_regret_sum1=np.zeros(T,float)
            avg_regret_sum2=np.zeros(T,float)
            K=int(T**(1/2))
        ###Run model
            for j in range(repeat):
                print('repeat: ',j)
                seed=j
                Env=noisy_linear_Env(seed,p,d,K,T)
                algorithm1=OFUL(d,K,T,seed,Env)
                Env=noisy_linear_Env(seed,p,d,K,T)
                algorithm2=OFUL_EF(d,K,T,seed,Env)

                opti_rewards=Env.opt_reward(T)

                regret=opti_rewards-algorithm1.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list1[j,:]=regret_sum
                avg_regret_sum1+=regret_sum
                
                regret=opti_rewards-algorithm2.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list2[j,:]=regret_sum
                avg_regret_sum2+=regret_sum
                

                
            avg1=avg_regret_sum1/repeat
            sd1=np.std(regret_sum_list1,axis=0)
            avg2=avg_regret_sum2/repeat
            sd2=np.std(regret_sum_list2,axis=0)



            algorithms = ['OFUL','OFUL-EF']
            regret = dict()
            std=dict()
            regret['OFUL']=avg1
            std['OFUL']=sd1
            regret['OFUL-EF']=avg2
            std['OFUL-EF']=sd2
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]
            
            
            ##Save data
            filename_1='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'regret.txt'
            with open('./result/'+filename_1, 'wb') as f:
                pickle.dump(regret, f)
                f.close()

            filename_2='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'std.txt'
            with open('./result/'+filename_2, 'wb') as f:
                pickle.dump(std, f)
                f.close()
    
    else: ##load data
        print('load data without running')
        for i in range(num):
            print(i)
            if i==0:
                T=1
            else:
                k=i
                T=T_1*k
            K=int(T**(1/2))
            T_list[i]=T
            filename_1='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'regret.txt'
            filename_2='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'std.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            pickle_file2 = open('./result/'+filename_2, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret=objects[0]
            objects = []
            while True:
                try:
                    objects.append(pickle.load(pickle_file2))
                except EOFError:
                    break
            pickle_file2.close()
            std=objects[0]
            avg1=regret['OFUL']
            sd1=std['OFUL']
            avg2=regret['OFUL-EF']
            sd2=std['OFUL-EF']

            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]

            
    fig,(ax)=plt.subplots(1,1)

    ax.errorbar(x=T_list, y=regret_list1, yerr=1.96*std_list1/np.sqrt(repeat), color="orange", capsize=3,
                 marker="^", markersize=7,label='OFUL',zorder=3) 
    ax.errorbar(x=T_list, y=regret_list2, yerr=1.96*std_list2/np.sqrt(repeat), color="b", capsize=3,
                 marker="o", markersize=7,label='OFUL-EF',zorder=2)

    
    #font size
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(r'$T$',fontsize=15)
    plt.ylabel(r'$R(T)$',fontsize=15)
    plt.savefig('./result/T'+str(T)+'repeat'+str(repeat)+'.png')
    plt.show()
    plt.clf()    

def run1(p,d,K,T,repeat,boolean=True):

    if boolean:
        regret=np.zeros(T,float)
        regret_sum=np.zeros(T,float)
        regret_sum_list1=np.zeros((repeat,T),float)
        std1=np.zeros(T,float)
        avg_regret_sum1=np.zeros(T,float)
        regret_sum_list2=np.zeros((repeat,T),float)
        std2=np.zeros(T,float)
        avg_regret_sum2=np.zeros(T,float)
        ###Run model
        for i in range(repeat):
            print('repeat: ',i)
            seed=i
            Env=noisy_linear_Env(seed,p,d,K,T)
            algorithm1=OFUL(d,K,T,seed,Env)
            algorithm2=OFUL_EF(d,K,T,seed,Env)
            
            opti_rewards=Env.opt_reward(T)

            regret=opti_rewards-algorithm1.rewards()
            regret_sum=np.cumsum(regret)
            regret_sum_list1[i,:]=regret_sum
            avg_regret_sum1+=regret_sum

            
            regret=opti_rewards-algorithm2.rewards()
            regret_sum=np.cumsum(regret)
            regret_sum_list2[i,:]=regret_sum
            avg_regret_sum2+=regret_sum
            
        avg1=avg_regret_sum1/repeat
        sd1=np.std(regret_sum_list1,axis=0)
        avg2=avg_regret_sum2/repeat
        sd2=np.std(regret_sum_list2,axis=0)
        
        algorithms = ['OFUL','OFUL-EF']
        regret = dict()
        std=dict()
        regret['OFUL']=avg1
        std['OFUL']=sd1
        regret['OFUL-EF']=avg2
        std['OFUL-EF']=sd2
        ##Save data
        filename_1='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'regret.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(regret, f)
            f.close()

        filename_2='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'std.txt'
        with open('./result/'+filename_2, 'wb') as f:
            pickle.dump(std, f)
            f.close()
    
    else: ##load data
        filename_1='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'regret.txt'
        filename_2='1T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'std.txt'
        pickle_file1 = open('./result/'+filename_1, "rb")
        pickle_file2 = open('./result/'+filename_2, "rb")
        objects = []
        
        while True:
            try:
                objects.append(pickle.load(pickle_file1))
            except EOFError:
                break
        pickle_file1.close()
        regret=objects[0]
        objects = []
        while True:
            try:
                objects.append(pickle.load(pickle_file2))
            except EOFError:
                break
        pickle_file2.close()
        std=objects[0]
        avg1=regret['OFUL']
        sd1=std['OFUL']
        avg2=regret['OFUL-EF']
        sd2=std['OFUL-EF']      
###Plot    



    T_p=int(T/20)
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)

    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.plot(range(T),avg1,color='orange',label='OFUL',marker='^', markersize=8,markevery=T_p)
    ax.fill_between(range(T), (avg1-1.96*sd1/np.sqrt(repeat)), (avg1+1.96*sd1/np.sqrt(repeat)), color='orange', alpha=.1 )
    ax.plot(range(T),avg2,color='b',label='Algorithm 1',marker='o', markersize=8,markevery=T_p)
    ax.fill_between(range(T), (avg2-1.96*sd2/np.sqrt(repeat)), (avg2+1.96*sd2/np.sqrt(repeat)), color='b', alpha=.1 )
    
    plt.xlabel('Time step '+r'$t$',fontsize=15)
    plt.ylabel('Cumulative Regret',fontsize=15)
    plt.title(r'$1-p={}$'.format(round(1-p,2)),fontsize=15)
    plt.legend(loc='upper left')
    plt.savefig('./result/T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'.png')
    plt.show()
    plt.clf()

        
if __name__=='__main__':
    # Read input
    opt = str(sys.argv[1])
    opt2= str(sys.argv[2]) 
    ##'1': figure1, '2': figure2, '3': figure3
    if opt2=='load':##  True: run model and save data with plot, False: load data with plot.
        run_bool=False
    elif opt2=='run':
        run_bool=True
    repeat=20 # repeat number of running algorithms with different seeds.
    d=2
    T=2*10**3  #Time horizon
    num=10
    if opt=='1':
#         p=0.7
#         run1(p,d,K,T,repeat,run_bool)
#         p=0.6
#         K=3
#         run1(p,d,K,T,repeat,run_bool)
        p=0.6
        K=20
        run1(p,d,K,T,repeat,run_bool)
        p=0.99
        run1(p,d,K,T,repeat,run_bool)
#         p=0.99
#         run1(p,d,K,T,repeat,run_bool)
    if opt=='2':
        p_list=[1,0.9,0.8,0.7,0.6]
        K=20
#         p_list=[1,0.95,0.9,0.85,0.8,0.75,0.7]
        run_p(p_list,d,K,T,repeat,run_bool)  
        K=3
        run_p(p_list,d,K,T,repeat,run_bool)  
#         p_list=[0.6,0.5]
#         K=20
# #         p_list=[1,0.95,0.9,0.85,0.8,0.75,0.7]
#         run_p(p_list,d,K,T,repeat,run_bool)     
#         run_p(p_list,d,K,T,repeat,run_bool)        
        
        #     p=0.9
#     run(p,d,K,T,repeat,run_bool)
    
    
    
#     d=2
#     p=0.99
#     T=10**3  #Time horizon
#     K=50
#     run(p,d,K,T,repeat,run_bool)
#     d=2
#     p=0.9
#     T=10**3  #Time horizon
#     K=50
#     run(p,d,K,T,repeat,run_bool)
