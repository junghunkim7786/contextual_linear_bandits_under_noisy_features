-For synthetic dataset:

1. Run `python3 syn_main.py 1 run` for (figure 1 (a,b)) or `python3 syn_main.py 2 run` for (figure 1 (c,d)).
(For loading results after saving, please run `python3 syn_main.py 1 load` or `python3 syn_main.py 2 load`)
2. Check the 'syn_result' folder

-For real CTR dataset:

1. Download dataset and unzip in ./datasets/{env}/raw/

*URls
Avazu: https://www.kaggle.com/competitions/avazu-ctr-prediction/data 
Taobao: https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom
MovieLens: https://grouplens.org/datasets/movielens/100k/

2. Run 'python3 scheduler_{env}.py' (env: avazu, taobao, ml100k) 
3. Check the result_ctr folder

Notes: For training autoencoder models, please run ipynbs in the preprocess folder from start to end.
