import pickle
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import os
import sys
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import contextlib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import Tools.Parameter as Parameter
import Tools.Dataset as Dataset

class BagUserCluster:
    def __init__(self, train_rating):
        self.usernum = 6040
        self.movienum = 3952
        self.k = 5
        self.user_film_mat = np.zeros((self.usernum,self.movienum))
        self.reduced_dim = 1000
        print("Initialize...")
        for user in tqdm(train_rating.keys()):
            for film, score in train_rating[user].items():
                self.user_film_mat[int(user)-1][int(film)-1] = score
        print("Utilizing PCA...")
        self.pca = PCA(n_components=self.reduced_dim)
        self.pca.fit(self.user_film_mat)
        self.user_film_mat = self.pca.transform(self.user_film_mat)


    def fit(self):
        print("Fitting...")
        self.gm = GaussianMixture(n_components=self.k)
        cluster_labels = self.gm.fit_predict(self.user_film_mat)
        tmp = []
        for i in tqdm(cluster_labels):
            address_index = [x + 1 for x in range(len(cluster_labels)) if cluster_labels[x] == i]
            tmp.append([i, address_index])
        self.dict_address = dict(tmp)
        cluster_size = []
        for i in range(self.k):
            cluster_size.append(len(self.dict_address[i]))
        print("cluster size:\n",cluster_size)

    def recommand(self, test_rating):
        print("Recommanding...")
        with open(os.path.join(Parameter.output_root, 'ml-1m', "output_cluster.txt"), "w") as file:
            with contextlib.redirect_stdout(file):
                user_film = np.zeros((self.usernum, self.movienum))
                for user in test_rating.keys():
                    for film, score in test_rating[user].items():
                        user_film[int(user)-1][int(film)-1] = score
                reduced_mat = self.pca.transform(user_film)
                cluster_labels = self.gm.predict(reduced_mat)
                print("test ", cluster_labels[:100])
                for idx, category in tqdm(enumerate(cluster_labels)):
                    if str(idx+1) not in test_rating.keys():
                        continue
                    cluster = self.dict_address[category]
                    watched_movies = test_rating[str(idx+1)]
                    candidate_movies = dict()
                    for similar_user in cluster:
                        if str(similar_user) in test_rating:
                            for movies, score in test_rating[str(similar_user)].items():
                                if movies in watched_movies:
                                    continue
                                candidate_movies.setdefault(movies, 0)
                                candidate_movies[movies] += score
                    candidate_movies_order = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
                    print("recom movie",candidate_movies_order[:10])
        return 0

if __name__ == '__main__':
    train_rating = Dataset.LoadRatingDataset(Parameter.train_path)
    test_rating = Dataset.LoadRatingDataset(Parameter.test_path)
    bag_em = BagUserCluster(train_rating)
    bag_em.fit()
    bag_em.recommand(test_rating)