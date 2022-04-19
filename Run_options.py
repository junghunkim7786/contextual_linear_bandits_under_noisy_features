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
                algorithm1=CLBEF(T,seed,Env)
                
                Env=noisy_linear_Env(seed,p,d,K)
                algorithm2=OFUL(T,seed,Env)
                
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


            algorithms = ['CLBEF','OFUL']
            regret = dict()
            std=dict()
            regret['CLBEF']=avg1
            std['CLBEF']=sd1
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
            avg1=regret['CLBEF']
            sd1=std['CLBEF']
            avg2=regret['OFUL']
            sd2=std['OFUL']   
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]

    p_list=[1-p for p in p_list]
    
    fig,(ax)=plt.subplots(1,1)
    ax.errorbar(x=p_list, y=regret_list1, yerr=1.96*std_list1/np.sqrt(repeat), color="b", capsize=3,
                 marker="^", markersize=7,label='CLBEF',zorder=2) 
    ax.errorbar(x=p_list, y=regret_list2, yerr=1.96*std_list2/np.sqrt(repeat), color="orange", capsize=3,
                 marker="o", markersize=7,label='OFUL',zorder=1)

    
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
    plt.xlabel(r'Missing probability $1-p$',fontsize=15)
    plt.ylabel(r'$R(T)$',fontsize=15)
    plt.savefig('./result/T'+str(T)+'K'+str(K)+'repeat'+str(repeat)+'p_list'+'.png')
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
            algorithm1=CLBEF(T,seed,Env)
            
            Env=noisy_linear_Env(seed,p,d,K)
            algorithm2=OFUL(T,seed,Env)
            
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
        
        algorithms = ['CLBEF','OFUL']
        regret = dict()
        std=dict()
        regret['CLBEF']=avg1
        std['CLBEF']=sd1
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
        avg1=regret['CLBEF']
        sd1=std['CLBEF']
        avg2=regret['OFUL']
        sd2=std['OFUL']      
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
    ax.plot(range(T),avg2,color='orange',label='OFUL',marker='o', markersize=8,markevery=T_p)
    ax.fill_between(range(T), (avg2-1.96*sd2/np.sqrt(repeat)), (avg2+1.96*sd2/np.sqrt(repeat)), color='orange', alpha=.1 )
    
    ax.plot(range(T),avg1,color='b',label='CLBEF',marker='^', markersize=8,markevery=T_p)
    ax.fill_between(range(T), (avg1-1.96*sd1/np.sqrt(repeat)), (avg1+1.96*sd1/np.sqrt(repeat)), color='b', alpha=.1 )

    plt.xlabel('Time step '+r'$t$',fontsize=15)
    plt.ylabel('Cumulative Regret',fontsize=15)
    plt.title(r'Missing probability $1-p={}$'.format(round(1-p,2)),fontsize=15)
    plt.legend(loc='upper left')
    plt.savefig('./result/T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'.png')
    plt.show()
    plt.clf()

    
    
def run_real(p,K,T,repeat,private,boolean=True):
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
            Env=yahoo_Env(seed,p,K,private)
            algorithm1=CLBEF(T,seed,Env)
            Env=yahoo_Env(seed,p,K,private)
            algorithm2=OFUL(T,seed,Env)
#             opti_rewards=Env.opt_reward(T)

            regret=algorithm1.rewards()
            regret_sum=np.cumsum(regret)
            regret_sum_list1[i,:]=regret_sum
            avg_regret_sum1+=regret_sum

            
            regret=algorithm2.rewards()
            regret_sum=np.cumsum(regret)
            regret_sum_list2[i,:]=regret_sum
            avg_regret_sum2+=regret_sum
        time=np.array(range(1,T+1))    
        avg1=avg_regret_sum1
        sd1=np.std(regret_sum_list1,axis=0)
        avg2=avg_regret_sum2
        sd2=np.std(regret_sum_list2,axis=0)
        
        algorithms = ['CLBEF','OFUL']
        regret = dict()
        std=dict()
        regret['CLBEF']=avg1
        std['CLBEF']=sd1
        regret['OFUL']=avg2
        std['OFUL']=sd2
        ##Save data
        d = 30
        filename_1='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'regret.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(regret, f)
            f.close()
        print('aaa')

        filename_2='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'std.txt'
        with open('./result/'+filename_2, 'wb') as f:
            pickle.dump(std, f)
            f.close()
    
    else: ##load data
        filename_1='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'regret.txt'
        filename_2='T'+str(T)+'d'+str(d)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'std.txt'
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
        avg1=regret['CLBEF']
#         sd1=std['CLBEF']
        avg2=regret['OFUL']
#         sd2=std['OFUL']      
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
    ax.plot(range(T),avg2,color='orange',label='OFUL',marker='o', markersize=8,markevery=T_p)
    ax.plot(range(T),avg1,color='b',label='CLBEF',marker='^', markersize=8,markevery=T_p)
#     ax.fill_between(range(T), (avg1-1.96*sd1/np.sqrt(repeat)), (avg1+1.96*sd1/np.sqrt(repeat)), color='orange', alpha=.1 )
#     ax.fill_between(range(T), (avg2-1.96*sd2/np.sqrt(repeat)), (avg2+1.96*sd2/np.sqrt(repeat)), color='b', alpha=.1 )
    
    plt.xlabel('Time step '+r'$t$',fontsize=15)
    plt.ylabel('Cumulative Reward',fontsize=15)
    plt.title(r'Missing probability $1-p={}$'.format(round(1-p,2)),fontsize=15)
    plt.legend(loc='upper left')
    plt.savefig('./result/T'+str(T)+'K'+str(K)+'p'+str(p)+'repeat'+str(repeat)+'private'+str(private)+'.png')
    plt.show()
    plt.clf()