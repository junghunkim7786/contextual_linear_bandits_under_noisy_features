import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import numpy as np
import random

import utils
import argparse

dataset_path = './real_datasets/avazu'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs='?', type=int, default=0)
    args = parser.parse_args()
    seed = args.seed

    X = np.vstack([np.load(dataset_path+'/preprocess/X0_avazu.npy'),np.load(dataset_path+'/preprocess/X1_avazu.npy')])
    np.random.shuffle(X)

    model = utils.AE_train(X, seed=seed)

    torch.save(model.state_dict(), f'./real_models/AE_avazu_s{seed}.pt')

