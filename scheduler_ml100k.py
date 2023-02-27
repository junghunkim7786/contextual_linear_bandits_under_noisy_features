import subprocess, os, datetime
import numpy as np
import real_plotting

os.environ["MKL_THREADING_LAYER"] = 'GNU'
plotting_dict = {'cut':0, 'title':'MovieLens 100K Dataset', 'y_lim':[0,0.13], 'y_ticks':[0, 0.05, 0.10], 'y_ticklabels':['0', '0.05' ,'0.10']}
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
seed_list = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234]
base_arg = ' --T 100000 --env ml100k --resultfoldertail _ctr_ml100k_{} '.format(now)

if os.path.exists('./real_datasets/ml100k/preprocess/X0_ml100k.npy'):
    print('Skip Preprocessing')
else:
    print('Preprocess')
    os.system("python3 ./real_preprocess/ml100k_preprocess.py")

for seed in seed_list:
    if os.path.exists('./real_models/AE_ml100k_s{}.pt'.format(seed)):
        print('Skip Training Autoencoder')
    else:
        print('Train Autoencoder')
        os.system(f"python3 ./real_preprocess/ml100k_aetrain.py --seed {seed}") 

    print('Run Contextual Linear Bandit Algorithms - Seed: {}'.format(seed))
    condition_arg = f'--seed {seed} --model_tail s{str(seed)}'
    os.system('python3 ./real_main.py'+ base_arg + condition_arg)

real_plotting.plotting(resultfoldertail='_ctr_ml100k_{}'.format(now), env='ml100k', plotting_dict=plotting_dict, fnametail='_{}_all'.format(now))