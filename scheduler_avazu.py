import subprocess, os, argparse, datetime
import numpy as np
import real_plotting

plotting_dict = {'cut':0, 'title':'Avazu CTR Dataset', 'y_lim':[0,0.11], 'y_ticks':[0, 0.05, 0.10], 'y_ticklabels':['0', '0.05' ,'0.10']}


parser = argparse.ArgumentParser()
parser.add_argument("-preseed", nargs='?', type=int, default = 34567)
parser.add_argument("-preprocess", action='store_true')
args = parser.parse_args()
preseed = args.preseed

if args.preprocess:
    print('Preseed: ',preseed)
    print('preprocess')
    os.system("python3 ./preprocess/avazu_prepreprocess.py")
    
# seed_list = [12345, 23456]
seed_list = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234]
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
for seed in seed_list:
    preseed=seed
    os.system("python3 ./preprocess/avazu_preprocess.py -seed {}".format(preseed))

    base_arg = ' --T 100000 --env avazu --resultfoldertail _ctr_{} --data_tail _mili_{} --model_tail _mili32_{} '.format(now, preseed, preseed)
    condition_arg = '--seed {}'.format(seed)
    os.system('python3 ./real_main.py'+ base_arg + condition_arg)

real_plotting.plotting(resultfoldertail='_ctr_{}'.format(now), env='avazu', plotting_dict=plotting_dict, fnametail='_{}_all'.format(preseed))
