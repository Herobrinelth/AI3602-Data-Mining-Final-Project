import itertools
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import Tools.Parameter as Parameter

class ML1m(Dataset):
    def __init__(self, data_path, sep='::', header=None):
        data = pd.read_csv(data_path, sep=sep, header=header, engine='python').to_numpy()[:, :3]
        print("User ID ranging from {} to {}, object ID ranging from {} to {}.".format(np.min(data[:, 0]), np.max(data[:, 0]), 
                                                                                       np.min(data[:, 1]), np.max(data[:, 1])))
        self.features = data[:, :2].astype(np.compat.long)-1
        self.targets = self.__process_score(data[:, -1]).astype(np.float32)
        self.feature_dims = np.max(self.features, axis=0)+1
        self.user_field_idx = np.array((0,), dtype=np.compat.long)
        self.item_field_idx = np.array((1,), dtype=np.compat.long)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return self.features.shape[0]

    def __process_score(self, score):
        score[score<=3] = 0
        score[score>3] = 1
        return score

def LoadRatingDataset(datafile):
    """
    :param name: the name of dataset like ml-1m, ml-100k
    :return: nested dictionary in forms of {user_id:{movie_id:score}}
    """
    ratings = {}
    with open(datafile) as f:
        for line in itertools.islice(f, 0, None):
            user, movie, rate = line.strip('\r\n').split(Parameter.seperator)[:3]
            ratings.setdefault(user, {})
            ratings[user][movie] = int(rate)
    return ratings

def LoadRatingDataset_mat(datafile):
    """
    :param name: the name of dataset like ml-1m, ml-100k
    :return: nested dictionary in forms of {user_id:{movie_id:score}}
    """
    ratings = {}
    with open(datafile) as f:
        for line in itertools.islice(f, 0, None):
            user, movie, rate = line.strip('\r\n').split(Parameter.seperator)[:3]
            ratings.setdefault(user, {})
            ratings[user][movie] = int(rate)

    like_matrix = [[] for _ in range(6040)]
    for user in ratings.keys():
        for film, score in ratings[user].items():
            like_matrix[int(user) - 1].append(int(film) - 1)

    tmp = []
    for user in ratings.keys():
        tmp.append(len(like_matrix[int(user) - 1]))

    return tmp

def LoadMovieDataset(name='ml-1m'):
    ratings = {}
    with open(Parameter.movies_path, encoding='gbk', errors='ignore') as f:
        for line in itertools.islice(f, 0, None):
            movie_id, movie_name = line.strip('\r\n').split(Parameter.seperator)[:2]
            ratings[movie_id] = movie_name
    return ratings

class DivideData(object):

    def __init__(self):
        self.trainset = []
        self.testset = []

    def generate_data_set(self, filename, train_path, test_path, pivot=0.8):
        a, b = 0, 0
        for line in self.loadfile(filename):
            if random.random() < pivot:
                self.trainset.append(line)
                a += 1
            else:
                self.testset.append(line)
                b += 1

        self.export_file(train_path, self.trainset)
        self.export_file(test_path, self.testset)

    @staticmethod
    def loadfile(filename):
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
        fp.close()

    @staticmethod
    def export_file(filename, data):
        with open(filename, "w") as f:
            for line in data:
                f.write(line)
                f.write("\n")

def get_cluster_movie():
    genres = []
    movieids=[]
    with open(Parameter.movies_path, encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            movieid, _, genre = line.strip('\r\n').split(Parameter.seperator)
            genres.append(genre)
            movieids.append(movieid)
    label_encoder = LabelEncoder()
    encoded_list = label_encoder.fit_transform(genres)
    print("New classes ranging from {} to {}.".format(np.min(encoded_list), np.max(encoded_list)))
    dic={}
    for i in range(len(encoded_list)):
        dic.update({movieids[i]:encoded_list[i]})
    with open(Parameter.rating_path, 'r') as infile:
        lines = infile.readlines()

    with open(Parameter.cluster_path, 'w') as outfile:
        for i, line in enumerate(lines):
            line = line.strip()
            user, movie, score, timestamp = line.split('::')
            movie = str(dic[movie])
            new_line = f'{user}::{movie}::{score}::{timestamp}\n'
            outfile.write(new_line)
    func = DivideData()
    func.generate_data_set(Parameter.cluster_path, Parameter.train_cluster, Parameter.test_cluster)

if __name__ == '__main__':
    test = LoadRatingDataset_mat(Parameter.rating_path)
    plt.plot(test)
    plt.savefig(os.path.join(Parameter.output_root, "test.jpg"), format='jpg')
    plt.violinplot(test, showmedians=True)
    plt.savefig(os.path.join(Parameter.output_root, "test_violin.jpg"), format='jpg')
    get_cluster_movie()
    # func = DivideData()
    # func.generate_data_set('dataset/yelp/yelp_reviews.dat', 'dataset/yelp/train_set.dat', 'dataset/yelp/test_set.dat')