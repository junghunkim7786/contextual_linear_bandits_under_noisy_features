import subprocess, os, argparse, datetime
import numpy as np
import plotting

print('preprocess')

exec(open("./preprocess/taobao_preprocess.py").read())

seed_list = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234]

base_arg = ' --env taobao --resultfoldertail _ctr '

#print(args.mask_ratio)
for seed in seed_list:
    condition_arg = '--seed {}'.format(seed)
    os.system('python3 ./real_main.py'+ base_arg + condition_arg)

plotting_dict = {'cut':2000, 'title':'Taobao Dataset', 'y_lim':[0,0.11], 'y_ticks':[0, 0.05, 0.10], 'y_ticklabels':['0', '0.05' ,'0.10']}
plotting.plotting(resultfoldertail='_ctr', env='taobao', plotting_dict=plotting_dict)