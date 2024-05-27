# input file path
rating_path = 'dataset/ml-1m/ratings.dat'
movies_path = 'dataset/ml-1m/movies.dat'
user_path = 'dataset/ml-1m/users.dat'
seperator = '::'

# output root
output_root = 'Output'

# UserCF parameter
num_sim_user = 25
ucf_recom_movies = 10

# split file path
train_path = 'dataset/train_set.dat'
test_path = "dataset/test_set.dat"

# cluster path
reduced_mat_path = 'Output/reduced_mat.pkl'
EM_mu_sigma_alpha_path = 'Output/EM_mu_sigma_alpha.pkl'

# NCF path
model_path = "Output/NCFmodel.pkl"
early_stop = 10

# LSTM path
lstm_path = "Output/LSTM.pkl"

# movie similaritem path
movie_similarity_matrices_path = 'Output/MovieSimMat.pkl'
movie_signature_path = 'Output/MovieSignature.pkl'

# user similaritem path
user_similarity_matrices_path = "Output/UserSimMat.pkl"

# NCF configuration
ncf_config = {
    "epoches": 50,
    "lr": 1e-3,
    "batch_size": 1024,
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