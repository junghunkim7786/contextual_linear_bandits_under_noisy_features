import numpy as np
import argparse, json, pickle, math
import matplotlib.pyplot as plt
import time
import torch

import real_environments
import algorithms

import os
ENV_LIST = ['ml100k', 'avazu', 'taobao']

ENV_CLASS = {
    'ml100k':     real_environments.env_ml100k.movielens_Env,
    'avazu':      real_environments.env_avazu.avazu_Env,
    'taobao':     real_environments.env_taobao.taobao_Env
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Process Configs", argument_default=argparse.SUPPRESS)
parser.add_argument("--env", choices=ENV_LIST, type=str)
parser.add_argument("--seed", type=int)

parser.add_argument("--encoding", type=str2bool)

parser.add_argument("--mask_ratio", "--np", type=float, help='1-p ; mask probabilty')
parser.add_argument("--K", type=int)
parser.add_argument("--reward1_ratio", type=float, help='The ratio of reward 1 arms from the sampling')
parser.add_argument("--num_partial", type=int, help='Whole dataset is sampled from the loaded data, with given num')
parser.add_argument("--model_tail", type=str, help="Indicating the model tail, e.g. '_avazu_12345")

parser.add_argument("--resultfoldertail",type=str, default='')

parser.add_argument("--T", type=int)
new_args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    resultfoldertail = new_args.resultfoldertail
    
    os.makedirs(f'./real_outputs/result{resultfoldertail}/arrays', exist_ok=True)
    os.makedirs(f'./real_outputs/result{resultfoldertail}/configs', exist_ok=True) 
    # os.makedirs(f'./result{resultfoldertail}/figures', exist_ok=True) 
    
    with open('./jsons/{}.json'.format(new_args.env)) as json_file:
        args = argparse.Namespace(**json.load(json_file)) 
        args.__dict__.update(new_args.__dict__)
    
    print(args.__dict__)
    
    timenum = round(time.time() * 1000)
    arg_str = "".join('{}: {} | '.format(str(key),str(value)) for key, value in args.__dict__.items())
    
    ENV = ENV_CLASS[args.env](args)
    
    ENV.reset()
    CLBBF = algorithms.CLBBF(args.T, ENV)
    CLBBF_rewards     = CLBBF.rewards()
    CLBBF_cumrewards  = np.cumsum(CLBBF_rewards)
    CLBBF_ctr         = CLBBF_cumrewards / (np.arange(args.T)+1)
    
    ENV.reset()
    OFUL = algorithms.OFUL(args.T, ENV)
    OFUL_rewards      = OFUL.rewards()
    OFUL_cumrewards   = np.cumsum(OFUL_rewards)
    OFUL_ctr          = OFUL_cumrewards / (np.arange(args.T)+1)
    
    ENV.reset()
    RANDOM = algorithms.RandomPolicy(args.T, ENV)
    RANDOM_rewards    = RANDOM.rewards()
    RANDOM_cumrewards = np.cumsum(RANDOM_rewards)
    RANDOM_ctr        = RANDOM_cumrewards / (np.arange(args.T)+1)
    
    ##### -- FIGURE PART -- #####
#     T_p=int(args.T/20)
#     fig, ax = plt.subplots()
#     ax.tick_params(labelsize=10)
#     plt.rc('legend',fontsize=12)
    
#     ax.yaxis.get_offset_text().set_fontsize(10)
#     ax.xaxis.get_offset_text().set_fontsize(10)
    
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     ax.plot(range(args.T - 100),RANDOM_ctr[100:],color='darkorange',label='RAMDOM',marker='o', markersize=8,markevery=T_p)
#     ax.plot(range(args.T - 100),OFUL_ctr[100:],color='crimson',label='OFUL',marker='s', markersize=8,markevery=T_p)
#     ax.plot(range(args.T - 100),CLBBF_ctr[100:],color='indigo',label='Algorithm1',marker='^', markersize=8,markevery=T_p)

#     plt.xlabel('Time Step',fontsize=12)
#     plt.ylabel('CTR',fontsize=12)

#     plt.title('{}, '.format(args.env)+r'$1-p={}$, $K={}$, $K_{{\mathrm{{reward}}}}={}$)'.format(round(args.mask_ratio,2), args.K_max, math.ceil(args.K_max*args.reward1_ratio)), fontsize=18, pad=5)
        
#     plt.legend(loc='upper right')
#     plt.savefig(f'./result{resultfoldertail}/figures/{args.env}_{timenum}')
    ##### --             -- #####
    
    np.savez(f'./real_outputs/result{resultfoldertail}/arrays/{args.env}_{timenum}_reward', clbbf=CLBBF_rewards, oful=OFUL_rewards,random=RANDOM_rewards)
    
    with open(f'./real_outputs/result{resultfoldertail}/configs/{args.env}_{timenum}.pickle', 'wb') as handle:
        pickle.dump(args.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f"./real_outputs/result{resultfoldertail}/{args.env}_result_guide","a+") as f:
        f.write('{}: {}'.format(timenum, arg_str))