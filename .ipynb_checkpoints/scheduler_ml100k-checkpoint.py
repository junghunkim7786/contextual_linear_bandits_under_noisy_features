import subprocess, os, argparse, datetime
import numpy as np
import plotting

seed_list = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234]
base_arg = ' --env ml100k --resultfoldertail _ctr '

#print(args.mask_ratio)
for seed in seed_list:
    condition_arg = '--seed {}'.format(seed)
    os.system('python3 ./real_main.py'+ base_arg + condition_arg)


plotting_dict = {'cut':2000, 'title':'MovieLens 100K Dataset', 'y_lim':[0,0.13], 'y_ticks':[0, 0.05, 0.10], 'y_ticklabels':['0', '0.05' ,'0.10']}
plotting.plotting(resultfoldertail='_ctr', env='ml100k', plotting_dict=plotting_dict)