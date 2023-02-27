import subprocess, os, datetime
import numpy as np
import real_plotting

os.environ["MKL_THREADING_LAYER"] = 'GNU'
plotting_dict = {'cut':0, 'title':'Taobao Dataset', 'y_lim':[0,0.11], 'y_ticks':[0, 0.05, 0.10], 'y_ticklabels':['0', '0.05' ,'0.10']}
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
seed_list = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234]
base_arg = ' --T 1025 --env taobao --resultfoldertail _ctr_{} '.format(now)

PREPROCESS = True

if PREPROCESS:
    print('Preprocess')
    os.system("python3 ./real_preprocess/taobao_preprocess.py")

for seed in seed_list:
    print('Train Autoencoder')
    os.system(f"python3 ./real_preprocess/taobao_aetrain.py --seed {seed}")

    print('Run Contextual Linear Bandit Algorithms')
    condition_arg = f'--seed {seed} --model_tail s{str(seed)}'
    os.system('python3 ./real_main.py'+ base_arg + condition_arg)

real_plotting.plotting(resultfoldertail='_ctr_{}'.format(now), env='taobao', plotting_dict=plotting_dict, fnametail='_{}_all'.format(preseed))