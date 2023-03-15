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

num_cores = mp.cpu_count()
print('# of Cores: {}'.format(num_cores))

dataset_path = "./real_datasets/ml100k"

user = pd.read_csv(dataset_path+"/raw/ml-100k/u.user", header = None, sep = "|")
user.columns = ["user_id","age","gender","occupation","zipcode"]
user = user.drop(["zipcode"], axis = 1)

bins = [0, 20, 30, 40, 50, 60, np.inf]
names = ['<20', '20-29', '30-39','40-49', '51-60', '60+']

user['agegroup'] = pd.cut(user['age'], bins, labels=names)
user = user.drop(["age"], axis = 1)

columnsToEncode = ["agegroup","gender","occupation"]
myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
myEncoder.fit(user[columnsToEncode])

user_features = pd.concat([user.drop(columnsToEncode, 1),
                           pd.DataFrame(myEncoder.transform(user[columnsToEncode]), 
                                        columns = myEncoder.get_feature_names_out(columnsToEncode))], axis=1).reindex()
user_features_array = user_features.to_numpy()

user_id_to_feature = dict(zip(user_features_array[:,0], user_features_array[:,1:]))

movie = pd.read_csv(dataset_path+"/raw/ml-100k/u.item", header = None, sep = "|", encoding='latin-1')
movie.columns = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", 
                  "unknown", "Action", "Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy",
                  "Film-Noir","Horror", "Musical", "Mystery","Romance","Sci-Fi","Thriller", "War","Western"]
movie_features = movie.drop(["movie_title","release_date", "video_release_date", "IMDb_URL"],axis = 1)

movie_features_array = movie_features.to_numpy()

movie_id_to_feature = dict(zip(movie_features_array[:,0], movie_features_array[:,1:]))

data = pd.read_csv(dataset_path+"/raw/ml-100k/u.data", sep ="\t", header=None, names = ["user_id", "movie_id","rating", "timestamp"])
data = data.drop(["timestamp"], axis = 1)

data["reward"] = np.where(data["rating"] < 5,0,1)
data.pop("rating")
data = data.reset_index(drop = True)

data_array = data.to_numpy()

Y = data_array[:,2]

X = []
for i in range(data_array.shape[0]):
    user_id = data_array[i,0]
    movie_id= data_array[i,1]
    
    X.append(np.concatenate((user_id_to_feature[user_id], movie_id_to_feature[movie_id])).copy())
X = np.vstack(X)

reward0_idx = np.where(Y == 0)[0]
reward1_idx = np.where(Y == 1)[0]

np.save(dataset_path+'/preprocess/X0_ml100k',X[reward0_idx,:])
np.save(dataset_path+'/preprocess/X1_ml100k',X[reward1_idx,:])