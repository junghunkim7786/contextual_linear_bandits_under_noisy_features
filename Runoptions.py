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

def plotting():
    pass
    
    
def save():
    pass
  
    
    
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
                Env=noisy_linear_Env(seed,p,d,K)
                algorithm1=CLBBF(T,Env)
                
                Env=noisy_linear_Env(seed,p,d,K)
                algorithm2=OFUL(T,Env)
                
                opti_rewards=Env.exp_reward_opt
                
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


            algorithms = ['CLBBF','OFUL']
            regret = dict()
            std=dict()
            regret['CLBBF']=avg1
            std['CLBBF']=sd1
            regret['OFUL']=avg2
            std['OFUL']=sd2

            
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
        for i, p in enumerate(p_list):
            print(i)
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
            avg1=regret['CLBBF']
            sd1=std['CLBBF']
            avg2=regret['OFUL']
            sd2=std['OFUL']   
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]

    p_list=[1-p for p in p_list]
    # plt.figure(figsize=(8,6))

    fig,(ax)=plt.subplots(1,1)
    fig.set_size_inches(8, 6)
    ax.errorbar(x=p_list, y=regret_list2, yerr=1.96*std_list2/np.sqrt(repeat), color="lightsalmon", capsize=3,
                 marker="^", markersize=16,label='OFUL',zorder=1)
    ax.errorbar(x=p_list, y=regret_list1, yerr=1.96*std_list1/np.sqrt(repeat), color="royalblue", capsize=3,
                 marker="o", markersize=15,label='Algorithm 1',zorder=2) 


    #font size
    ax.tick_params(labelsize=22)
    plt.rc('legend',fontsize=22)
    ax.yaxis.get_offset_text().set_fontsize(22)
    ax.xaxis.get_offset_text().set_fontsize(22)

    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.17)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title(r'$K={}$'.format(K),fontsize=25,pad=10)
    plt.xlabel(r'Missing probability $1-p$',fontsize=25)
    plt.ylabel(r'$R(T)$',fontsize=25)
    plt.savefig('./result/T'+str(T)+'K'+str(K)+'repeat'+str(repeat)+'p_list'+'.png',dpi=300)
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
            Env=noisy_linear_Env(seed,p,d,K)
            algorithm1=CLBBF(T,Env)
            
            Env=noisy_linear_Env(seed,p,d,K)
            algorithm2=OFUL(T,Env)
            
            opti_rewards=Env.exp_reward_opt

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
        
        algorithms = ['CLBBF','OFUL']
        regret = dict()
        std=dict()
        regret['CLBBF']=avg1
        std['CLBBF']=sd1
        regret['OFUL']=avg2
        std['OFUL']=sd2
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
        avg1=regret['CLBBF']
        sd1=std['CLBBF']
        avg2=regret['OFUL']
        sd2=std['OFUL']      
###Plot    


    T_p=int(T/10)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    ax.tick_params(labelsize=22)
    plt.rc('legend',fontsize=22)
    ax.yaxis.get_offset_text().set_fontsize(22)
    ax.xaxis.get_offset_text().set_fontsize(22)

    # plot 
    plt.gcf().subplots_adjust(bottom=0.17)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.plot(range(T),avg2,color='lightsalmon',label='OFUL',marker='^', markersize=16,markevery=T_p)
    ax.fill_between(range(T), (avg2-1.96*sd2/np.sqrt(repeat)), (avg2+1.96*sd2/np.sqrt(repeat)), color='lightsalmon', alpha=.2 )
    
    ax.plot(range(T),avg1,color='royalblue',label='Algorithm 1',marker='o', markersize=15,markevery=T_p)
    ax.fill_between(range(T), (avg1-1.96*sd1/np.sqrt(repeat)), (avg1+1.96*sd1/np.sqrt(repeat)), color='royalblue', alpha=.2 )

    plt.xlabel('Time step '+r'$t$',fontsize=25)
    plt.ylabel(r'$R(t)$',fontsize=25)
    plt.title(r'$1-p={}$'.format(round(1-p,2)),fontsize=25,pad=10)
    plt.legend(loc='best')
    plt.savefig('./result/T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'.png',dpi=300)
    plt.show()
    plt.clf()

    
    
# def run_real(p,K,T,repeat,private,ind,boolean=True):
#     option='raw'
#     if boolean:
#         regret=np.zeros(T,float)
#         regret_sum=np.zeros(T,float)
        
#         regret_sum_list1=np.zeros((repeat,T),float)
#         std1=np.zeros(T,float)
#         avg_regret_sum1=np.zeros(T,float)
        
#         regret_sum_list2=np.zeros((repeat,T),float)
#         std2=np.zeros(T,float)
#         avg_regret_sum2=np.zeros(T,float)
        
#         regret_sum_list3=np.zeros((repeat,T),float)
#         std3=np.zeros(T,float)
#         avg_regret_sum3=np.zeros(T,float)
#         ###Run model
#         for i in range(repeat):
#             print('repeat: ',i)
#             seed=i
            
#             Env=yahoo_Env(seed,p,K,private,option=option,ind=ind)
#             # Env=movielens_Env(seed,p,private)
#             algorithm1=CLBBF(T,seed,Env)
            
#             Env=yahoo_Env(seed,p,K,private,option=option,ind=ind)
#             # Env=movielens_Env(seed,p,private)
#             algorithm2=OFUL(T,seed,Env)
# #             opti_rewards=Env.opt_reward(T)

#             #Env=yahoo_Env(seed,p,K,private)
#             # Env=movielens_Env(seed,p,private)
#             # algorithm3=RandomPolicy(T,seed,Env)

#             regret=algorithm1.rewards()
#             regret_sum=np.cumsum(regret)
#             regret_sum_list1[i,:]=regret_sum
#             avg_regret_sum1+=regret_sum

            
#             regret=algorithm2.rewards()
#             regret_sum=np.cumsum(regret)
#             regret_sum_list2[i,:]=regret_sum
#             avg_regret_sum2+=regret_sum
            
#             # regret=algorithm3.rewards()
#             # regret_sum=np.cumsum(regret)
#             # regret_sum_list3[i,:]=regret_sum
#             # avg_regret_sum3+=regret_sum
            
            
#         time=np.array(range(1,T+1))    
#         avg1=avg_regret_sum1
#         sd1=np.std(regret_sum_list1,axis=0)
#         avg2=avg_regret_sum2
#         sd2=np.std(regret_sum_list2,axis=0)
#         # avg3=avg_regret_sum3
#         # sd3=np.std(regret_sum_list3,axis=0)
        
#         algorithms = ['CLBBF','OFUL']
#         regret = dict()
#         std=dict()
#         regret['CLBBF']=avg1
#         std['CLBBF']=sd1
#         regret['OFUL']=avg2
#         std['OFUL']=sd2
#         # regret['RANDOM']=avg3
#         # std['RANDOM']=sd3
#         ##Save data
#         d = 36
#         filename_1='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'option'+option+'ind'+ind+'regret.txt'
#         with open('./result/'+filename_1, 'wb') as f:
#             pickle.dump(regret, f)
#             f.close()
#         print('aaa')

#         filename_2='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'option'+option+'ind'+ind+'std.txt'
#         with open('./result/'+filename_2, 'wb') as f:
#             pickle.dump(std, f)
#             f.close()
    
#     else: ##load data
#         filename_1='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'option'+option+'ind'+ind+'regret.txt'
#         filename_2='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'option'+option+'ind'+ind+'std.txt'
#         pickle_file1 = open('./result/'+filename_1, "rb")
#         pickle_file2 = open('./result/'+filename_2, "rb")
#         objects = []
        
#         while True:
#             try:
#                 objects.append(pickle.load(pickle_file1))
#             except EOFError:
#                 break
#         pickle_file1.close()
#         regret=objects[0]
#         objects = []
#         while True:
#             try:
#                 objects.append(pickle.load(pickle_file2))
#             except EOFError:
#                 break
#         pickle_file2.close()
#         std=objects[0]
#         avg1=regret['CLBBF']
# #         sd1=std['CLBBF']
#         avg2=regret['OFUL']
# #         sd2=std['OFUL']      
# ###Plot    

#     times=np.array(range(1,T+1))
#     T_p=int(T/20)
#     fig, ax = plt.subplots()
#     ax.tick_params(labelsize=20)
#     plt.rc('legend',fontsize=20)
#     ax.yaxis.get_offset_text().set_fontsize(20)
#     ax.xaxis.get_offset_text().set_fontsize(20)

#     # plot 
#     plt.gcf().subplots_adjust(bottom=0.17)
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     # ax.plot(range(T),avg3,color='red',label='RAMDOM',marker='s', markersize=10,markevery=T_p)
#     ax.plot(range(T),avg2,color='orange',label='OFUL',marker='o', markersize=10,markevery=T_p)
#     ax.plot(range(T),avg1,color='b',label='CLBBF',marker='^', markersize=10,markevery=T_p)
# #     ax.fill_between(range(T), (avg1-1.96*sd1/np.sqrt(repeat)), (avg1+1.96*sd1/np.sqrt(repeat)), color='orange', alpha=.1 )
# #     ax.fill_between(range(T), (avg2-1.96*sd2/np.sqrt(repeat)), (avg2+1.96*sd2/np.sqrt(repeat)), color='b', alpha=.1 )
    
#     plt.xlabel('Time step '+r'$t$',fontsize=20)
#     plt.ylabel(r'$R(t)$',fontsize=20)
#     plt.title(r' $1-p={}$'.format(round(1-p,2)),fontsize=20)
#     plt.legend(loc='upper left')
#     plt.savefig('./result/T'+str(T)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'option'+option+'ind'+ind+'.png')
#     plt.show()
#     plt.clf()
    
    
    