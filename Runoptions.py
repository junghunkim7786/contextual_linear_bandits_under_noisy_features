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
from numpy import dot
from numpy.linalg import norm

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

def run_sim(p_list,d,K,T,repeat,boolean=True):
    if boolean:
        avg_opt_list=np.zeros((len(p_list),repeat))
        avg_rand_list=np.zeros((len(p_list),repeat))

        for k,p in enumerate(p_list):
            avg_opt=0
            avg=0
            print('p: ', p)
            for i in range(repeat):
                seed=i
                random.seed(seed)
                Env=noisy_linear_Env(seed,p,d,K)
                for t in tqdm(range(T)):
                    Env.load_data()
                    opt_arm=Env.opt_arm
                    z=Env.z
                    opt_arm=Env.opt_arm.copy()
                    opt_arm_ori=Env.opt_arm_origin.copy()
                    a=z[opt_arm]
                    b=z[opt_arm_ori]
                    sim_opt=dot(a, b)/(norm(a)*norm(b))
                    if t==0:
                        avg_opt=sim_opt
                    else:
                        avg_opt=(avg_opt*t+sim_opt)/(t+1)
                    avg_opt_list[k,i]=avg_opt

                    ran_ind=random.choice(range(K))
                    a=z[ran_ind]
                    sim=dot(a, b)/(norm(a)*norm(b))
                    if t==0:
                        avg=sim
                    else:
                        avg=(avg*t+sim)/(t+1)
                avg_rand_list[k,i]=avg
        np.save('./result/T_'+str(T)+'p_list_'+str(p_list)+'sim_opt.npy',avg_opt_list)
        np.save('./result/T_'+str(T)+'p_list_'+str(p_list)+'sim.npy',avg_rand_list)
        
    else:
        avg_opt_list=np.load('./result/T_'+str(T)+'p_list_'+str(p_list)+'sim_opt.npy')
        avg_rand_list=np.load('./result/T_'+str(T)+'p_list_'+str(p_list)+'sim.npy')
    
    sim_opt_mean=np.average(avg_opt_list,axis=1)
    sim_opt_sd=np.std(avg_opt_list,axis=1)
    sim_mean=np.average(avg_rand_list,axis=1)
    sim_sd=np.std(avg_rand_list,axis=1)

    _p_list=[1-p for p in p_list]

    fig,(ax)=plt.subplots(1,1)
    ax.errorbar(x=_p_list, y=sim_opt_mean, yerr=1.96*sim_opt_sd/np.sqrt(repeat), color="tomato", capsize=3,
                 marker="o", markersize=7,label='Bayesian optimal arm',zorder=2, alpha=0.9) 
    ax.errorbar(x=_p_list, y=sim_mean, yerr=1.96*sim_sd/np.sqrt(repeat), color="grey", capsize=3,
                 marker="^", markersize=8,label='Random arm',zorder=1)


    #font size
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=13)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)

    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    plt.xlabel(r'Missing probability $1-p$',fontsize=15)
    plt.ylabel('Cosine similarity',fontsize=15)
    plt.savefig('./result/sim.png',dpi=300)
    plt.show()
    plt.clf()      