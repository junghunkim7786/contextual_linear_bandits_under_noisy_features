import sys, os, time, datetime,pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import  Pool
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-seed", nargs='?', type=int, default = 0)
args = parser.parse_args()

random_seed = args.seed

print('Random Seed: ',random_seed)

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
np.random.seed(random_seed)
random.seed(random_seed)

num_cores = mp.cpu_count()
print('# of Cores: {}'.format(num_cores))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset_path = "./datasets/ml-100k"

user = pd.read_csv(dataset_path+"/raw/ml-100k/u.user", header = None, sep = "|")
user.columns = ["user_id","age","gender","occupation","zipcode"]
user = user.drop(["zipcode"], axis = 1)

bins = [0, 20, 30, 40, 50, 60, np.inf]
names = ['<20', '20-29', '30-39','40-49', '51-60', '60+']

user['agegroup'] = pd.cut(user['age'], bins, labels=names)
user = user.drop(["age"], axis = 1)
user.head()


columnsToEncode = ["agegroup","gender","occupation"]
myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
myEncoder.fit(user[columnsToEncode])

user_features = pd.concat([user.drop(columnsToEncode, 1),
                           pd.DataFrame(myEncoder.transform(user[columnsToEncode]), 
                                        columns = myEncoder.get_feature_names(columnsToEncode))], axis=1).reindex()
print(user_features.shape)
user_features.head()

user_features_array = user_features.to_numpy()

user_id_to_feature = dict(zip(user_features_array[:,0], user_features_array[:,1:]))

user_features_array.shape

movie = pd.read_csv(dataset_path+"/raw/ml-100k/u.item", header = None, sep = "|", encoding='latin-1')
movie.columns = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", 
                  "unknown", "Action", "Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy",
                  "Film-Noir","Horror", "Musical", "Mystery","Romance","Sci-Fi","Thriller", "War","Western"]
movie_features = movie.drop(["movie_title","release_date", "video_release_date", "IMDb_URL"],axis = 1)
movie_features.head()

movie_features_array = movie_features.to_numpy()

movie_id_to_feature = dict(zip(movie_features_array[:,0], movie_features_array[:,1:]))

movie_features_array.shape

data = pd.read_csv(dataset_path+"/raw/ml-100k/u.data", sep ="\t", header=None, names = ["user_id", "movie_id","rating", "timestamp"])
data = data.drop(["timestamp"], axis = 1)

data["reward"] = np.where(data["rating"] <5,0,1)
data.pop("rating")
data = data.reset_index(drop = True)

data_array = data.to_numpy()

data.head()

Y = data_array[:,2]

X = []
for i in range(data_array.shape[0]):
    user_id = data_array[i,0]
    movie_id= data_array[i,1]
    
    X.append(np.concatenate((user_id_to_feature[user_id], movie_id_to_feature[movie_id])).copy())

X = np.vstack(X)
X.shape, Y.shape

data_tail = '_ml100k_{}'.format(random_seed)

reward0_idx = np.where(Y == 0)[0]
reward1_idx = np.where(Y == 1)[0]

len(reward0_idx) + len(reward1_idx)

np.save(dataset_path+'/preprocess/X0{}'.format(data_tail),X[reward0_idx,:])
np.save(dataset_path+'/preprocess/X1{}'.format(data_tail),X[reward1_idx,:])

# Hyperparameters for Training AutoEncoder
save_tail = data_tail

# Hyperparameters for Training
learning_rate = 0.00001
weight_decay  = 0.000001
num_epoch = 500 #Normally 100
B = 10000 # batchsize
EMB_DIM = 32

    
class BN_Autoencoder(nn.Module):
    def __init__(self, d, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.d = d
        self.encoder = nn.Sequential(nn.Linear(self.d, self.emb_dim),nn.BatchNorm1d(self.emb_dim))
        self.decoder = nn.Linear(self.emb_dim, self.d)
                
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size, self.d)
        return out
    
    def encoding_result(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0],-1)
        encoded = self.encoder(x)
        return encoded

X = np.vstack([np.load(dataset_path+'/preprocess/X0{}.npy'.format(data_tail)),np.load(dataset_path+'/preprocess/X1{}.npy'.format(data_tail))])
np.random.shuffle(X)
d = X.shape[1]

model = BN_Autoencoder(d=d, emb_dim=EMB_DIM).to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


L = X.shape[0]

loss_arr = []

for k in tqdm(range(num_epoch)):
    
    for l in range(L//B):
        
        batch  = X[l*B:(l+1)*B, :].copy()
        
        x = torch.from_numpy(batch).type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,x)
        loss.backward()
        optimizer.step()

    loss_arr.append(loss.cpu().data.numpy())
    
torch.save(model.state_dict(), './models/AE{}.pt'.format(save_tail))
