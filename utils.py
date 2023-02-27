import argparse
import json

__all__ = ['parse_args']

ENV_LIST = ['ml100k', 'avazu', 'taobao']

def parse_args(parser):
    parser.add_argument("--env", choices=ENV_LIST, type=str)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--mask_ratio", "--np", type=float, help='1-p ; mask probabilty')
    parser.add_argument("--K", type=int)
    parser.add_argument("--reward1_ratio", type=float, help='The ratio of reward 1 arms from the sampling')
    parser.add_argument("--num_partial", type=int, help='Whole dataset is sampled from the loaded data, with given num')
    parser.add_argument("--model_tail", type=str, help="Indicating the model tail, e.g. '_avazu_12345")

    parser.add_argument("--resultfoldertail",type=str, default='')

    parser.add_argument("--T", type=int)
    args_tmp = parser.parse_args()

    with open('./jsons/{}.json'.format(args_tmp.env)) as json_file:
        args = argparse.Namespace(**json.load(json_file)) 
        args.__dict__.update(args_tmp.__dict__)

    return args