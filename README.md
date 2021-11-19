# POLIMI Recommender Systems Challenge 2020/2021
<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/recommender-system-2020-challenge-polimi)

Part of the Recommender Systems exam at Politecnico di Milano consists in a kaggle challenge. In this repository you can find all the files that have been used for the competition. 

Note that the base (non hybrid) recommenders come from [this repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi).

## Overview
<p align="justify">
The complete description of the problem to be solved can be found in the kaggle competition link (check the top of the read.me). Shortly, given the User Rating Matrix and the Item Content Matrix, the objective of the competition was to create the best recommender system for a book recommendation service by providing <b>10 recommended books</b> to each user. In particular the URM was composed by around <b>135k interactions</b>, <b>7947 users</b> and <b>25975 item</b>; the ICM instead contained, for each book, a subset of <b>20000 possible tokens</b> (some books had less than 10 tokens, other more than 40).

Note that the evaluation metric for this competition was the mean average precision at position 10 (**MAP@10**).
</p>
  
## Best Model
<p align="justify">
The final model used for the best submission is an hybrid recommeder created by averaging predictions of different models. The idea is that, if the composing models have all good performances and are different enough, the combined predictions will improve since different models are able to capture different aspects of the problem. The final hybrid is the results of several steps:
  </p>
<ol>
  <li><p align="justify"><b>Simple hybrids</b>: item scores of two basic recommenders are normalized and combined, hyperparameters jointly optimized. P3Alpha + ItemKNNCBF gave the best results (MAP@10 = 0.08856 on public leaderboard)</p></li>
  <li><p align="justify"><b>Multilevel hybrids</b>: Instead of a simple recommender, pass to an hybrid other two hybrids as components (Basic block:  P3Alpha + ItemKNNCBF hybrid). Just normalize and mix scores. We may use the same hybrids with different hyperparameters; also some are trained with just URM, others with URM concatenated with ICM (MAP@10 = 0.09159 on public).</p></li>
  <li><p align="justify"><b>Specialized hybrids</b>: the basic idea is to tune hyperparameters of some hybrids to make better predictions only for cold or only for warm users. In practice: set a threshold, force an hybrid to make random predictions if the user profile lenght is below/above it, and do hyperparameter tuning. Then combine different specialized hybrids in multilevel way: the final recommender contains specialized hybrids for 4 user groups created by counting the number of user interactions (MAP@10 = 0.09509 on public).</p></li>
  <li><p align="justify"><b>IALS</b>: add IALS recommeder to the final hybrid (very different model from the previous ones). Using URM concatenated with ICM improved performance in CF and CBF algorithms, and improved also this ML model. Since this algorithm is very slow, tune with max 300 factors, and assume will work for more; also tune carefully the hyperparameter alpha.</p></li>
</ol>
<p align="justify">
Best model overall: hybrid of previous best multilevel specialized hybrid and IALS with n_factors = 1200 and alpha = 25, MAP@10 = 0.09877 (public), 0.10803 (private).
</p>


## Recommenders
<p align="justify">
In this repo you can find the implementation of different recommender systems; in particular the following models can be found in the <i>Recommenders</i> folder:
</p>

- Item and User based Collaborative Filtering
- Item Content Based Filtering
- P3Alpha and RP3Beta Graph Based models
- Pure SVD and Implicit Alternating Least Squares models
- Slim BPR and Slim ElasticNet
- Hybrids and multi-level hubrids used for the final ensamble

## Requirements
The `requirements.txt` file lists all the libraries that are necessary to run the scripts. <b>Install them</b> using:

```
pip install -r requirements.txt
```

## Cython
<p align="justify">
Some of the models use Cython implementations. As written in the original repository you have to <b>compile all Cython algorithms</b>. 
In order to compile you must first have installed: gcc and python3 dev. Under Linux those can be installed with the following commands:
</p>

```
sudo apt install gcc 
sudo apt-get install python3-dev
```
  
<p align="justify">
If you are using Windows as operating system, the installation procedure is a bit more complex. 
You may refer to <a href="https://github.com/cython/cython/wiki/InstallingOnWindows">the official guide</a>.
</p>

<p align="justify">
Now you can compile all Cython algorithms by running the following command. 
The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. 
During the compilation <b>you may see some warnings</b>. 
</p>
  
```
python run_compile_all_cython.py
```

## Visualization
To see a plot of MAP@10 for the best model and the hybrids composing it on various user groups, you can run the following command:
```
python HybridFinalParall.py
```
<p align="justify">
Note that the script tries to train in parallel as many recommenders as possible, and this may cause problems on machines with less than 16GB of RAM.
</p>
  
## Results
* Ranked 2nd among 70 teams 
* MAP@10 = 0.10803 on Kaggle's private leaderboard

