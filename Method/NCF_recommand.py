import torch
from tqdm import tqdm
import numpy as np
import contextlib
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import Tools.Dataset as Dataset
import Tools.Parameter as Parameter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NCFrecommand:
    def __init__(self, model_path):
        print("Reading checkpoint from {}".format(model_path))
        self.model = torch.load(model_path).to(device)
        self.movienum = 3952
        self.usernum = 6040

    def recommand(self, user:int):
        """
        :param user: integer, starting from 1 to 6040
        :return: top-10 recommand movie [int,int,...,int] from 1 to 3592
        """
        user_movie = np.zeros((6040,2), dtype=np.compat.long)
        for i in range(self.movienum):
            user_movie[i][0] = user-1
            user_movie[i][1] = i
        user_movie = torch.from_numpy(user_movie).to(device)
        test = self.model(user_movie)
        all_movies = test.detach().cpu().numpy()
        recommand = sorted(range(1, self.movienum+1), key = lambda k:all_movies[k-1], reverse=True)
        return recommand

if __name__ == "__main__":
    train_rating = Dataset.LoadRatingDataset(Parameter.train_path)
    test_rating = Dataset.LoadRatingDataset(Parameter.test_path)
    modelpath = Parameter.model_path
    ncf = NCFrecommand(modelpath)
    accurate = 0
    recall = 0
    for user in tqdm(test_rating.keys()):
        watched_movies = train_rating[user]
        list_movie = ncf.recommand(int(user))
        cnt = 0
        rec_movie = []
        while len(rec_movie)<100:
            if str(list_movie[cnt]) in watched_movies:
                cnt += 1
                continue
            else:
                rec_movie.append(list_movie[cnt])
                cnt += 1
        test_movies = test_rating[user]
        hit = 0
        for i in rec_movie:
            if str(i) in test_movies:
                hit += 1
        total = len(test_movies)
        if total != 0:
            accurate += (hit/total)
        recall += (hit/100)
    with open(os.path.join(Parameter.output_root, "output_NCF.txt"), "w") as file:
        with contextlib.redirect_stdout(file):
            print(Parameter.model_path)
            print("acc cnt", accurate)
            print("accurate",accurate/len(test_rating.keys()))
            print("recall cnt", recall)
            print("recall",recall/len(test_rating.keys()))