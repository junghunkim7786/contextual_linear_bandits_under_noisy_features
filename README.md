

# Contextual Linear Bandits under Noisy Features: Towards Bayesian Oracles


## Synthetic datasets

For figure 1 (a,b), run:

```
python3 syn_main.py 1 run
```

For figure 1 (c,d), run:

```
python3 syn_main.py 2 run
```

For results, check the 'syn_result' folder


## Real CTR datasets

You can download dataset here:

URls:

* Taobao: https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom

* MovieLens: https://grouplens.org/datasets/movielens/100k/

* Avazu: https://www.kaggle.com/competitions/avazu-ctr-prediction/data 

After download, place three zip files in ./real_datasets/ .

Then run:
```
python3 real_datasetter.py
```
at the home directory.

For figure 2, run: 

```
python3 scheduler_taobao.py
```
```
python3 scheduler_ml100k.py
```
```
python3 scheduler_avazu.py
```
For results, check the 'result_ctr' folder.
