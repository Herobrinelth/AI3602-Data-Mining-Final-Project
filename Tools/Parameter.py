import os

# root
output_root = 'Output'
dataset_root = 'dataset'
dataset = 'ml-1m' # ['ml-1m', 'yelp', 'recipe']
seperator = '::'

# input file path
rating_path = os.path.join(dataset_root, dataset, 'ratings.dat')
cluster_path = os.path.join(dataset_root, dataset, 'ratings_cluster.dat')
movies_path = os.path.join(dataset_root, dataset, 'movies.dat')
user_path = os.path.join(dataset_root, dataset, 'users.dat')


# UserCF parameter
num_sim_user = 25
ucf_recom_movies = 10

# split file path
train_path = os.path.join(dataset_root, dataset, 'train_set.dat')
test_path = os.path.join(dataset_root, dataset, 'test_set.dat')
train_cluster = os.path.join(dataset_root, dataset, 'train_set_cluster.dat')
test_cluster = os.path.join(dataset_root, dataset, 'test_set_cluster.dat')

# cluster path
reduced_mat_path = os.path.join(output_root, dataset, 'reduced_mat.pkl')
EM_mu_sigma_alpha_path = os.path.join(output_root, dataset, 'EM_mu_sigma_alpha.pkl')

# NCF path
model_path = os.path.join(output_root, dataset, "NCFmodel.pkl")
model_cluster = os.path.join(output_root, dataset, "NCFmodel_cluster.pkl")
early_stop = 5

# LSTM path
lstm_path = os.path.join(output_root, dataset, "LSTM.pkl")

# movie similaritem path
movie_similarity_matrices_path = os.path.join(output_root, dataset, 'MovieSimMat.pkl')
movie_signature_path = os.path.join(output_root, dataset, 'MovieSignature.pkl')

# user similaritem path
user_similarity_matrices_path = os.path.join(output_root, dataset, "UserSimMat.pkl")

# NCF configuration
ncf_config = {
    "epoches": 50,
    "lr": 1e-3,
    "batch_size": 256,
    "kwargs": {
        'embed_size':128,
        'hidden_nbs':[512, 1024, 512],
        'dropout':0.1
    }
}

# LSTM configuration
lstm_config = {
    "epoches": 50,
    "lr": 1e-3,
    "batch_size": 1024,
    "kwargs": {
        'embed_size':256,
        'dropout':0.1,
        'gru_layers':2
    }
}

# NCF cluster configuration
ncfclu_config = {
    "epoches": 50,
    "lr": 1e-3,
    "batch_size": 256,
    "kwargs": {
        'embed_size':128,
        'hidden_nbs':[512, 1024, 512],
        'dropout':0.1
    }
}