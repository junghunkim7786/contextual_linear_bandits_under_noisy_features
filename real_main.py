import numpy as np
import argparse, json, pickle, math
import matplotlib.pyplot as plt
import time
import torch

import real_environments
import algorithms

import utils
import os

ENV_CLASS = {
    'ml100k':     real_environments.env_ml100k.movielens_Env,
    'avazu':      real_environments.env_avazu.avazu_Env,
    'taobao':     real_environments.env_taobao.taobao_Env
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Configs", argument_default=argparse.SUPPRESS)
    args = utils.parse_args(parser)
    print(args.__dict__)
    
    # Run Algorithms
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

    # Save Results
    resultfoldertail = args.resultfoldertail
    os.makedirs(f'./real_outputs/result{resultfoldertail}/arrays', exist_ok=True)
    os.makedirs(f'./real_outputs/result{resultfoldertail}/configs', exist_ok=True) 

    timenum = round(time.time() * 1000)
    arg_str = "".join('{}: {} | '.format(str(key),str(value)) for key, value in args.__dict__.items())

    np.savez(f'./real_outputs/result{resultfoldertail}/arrays/{args.env}_{timenum}_reward', clbbf=CLBBF_rewards, oful=OFUL_rewards,random=RANDOM_rewards)
    
    with open(f'./real_outputs/result{resultfoldertail}/configs/{args.env}_{timenum}.pickle', 'wb') as handle:
        pickle.dump(args.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f"./real_outputs/result{resultfoldertail}/{args.env}_result_guide","a+") as f:
        f.write('{}: {}'.format(timenum, arg_str))