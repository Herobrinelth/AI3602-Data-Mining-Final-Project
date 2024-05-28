# Movie Recommendation System
## Final Project of AI3602 Data Mining
## Tianhua Li, Yuchen Mao, Nange Wang

* Environment
  * python 3.10 + pytorch 1.13.0

* Data
  
  * [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/) Highly recommanded to utilize.
  * [Recipe Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_interactions.csv) Too large to utilize.
  * [Yelp Dataset](https://github.com/Yelp/dataset-examples?tab=readme-ov-file) Too large to utilize.

* Algorithms 

  * User-based Collaborative Filtering
  * Similar Item Recommendation (Min-Hash)
  * Cluster (PCA dimension reduction & EM algorithm)
  * Neutral Collaborative Filtering
  * LSTM
  * Movie clustering & Neutral Collaborative Filtering

* Usage

  * Activate all: `bash activate_all.sh` on bash.
  * If you want to activate algorithms separately, run `bash Scripts/{method.sh}` on bash.
  * To change training configurations, please refer to `Tools/Parameter.py`.
