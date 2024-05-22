# input file path
rating_path = 'dataset/ml-1m/ratings.dat'
movies_path = 'dataset/ml-1m/movies.dat'
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

# recommend path
model_path = "Output/NCFmodel.pkl"

# train path
train_eer_path = "Output/train_error.pkl"
test_eer_path = "Output/test_error.pkl"

# movie similaritem path
movie_similarity_matrices_path = 'Output/MovieSimMat.pkl'
movie_signature_path = 'Output/MovieSignature.pkl'

# user similaritem path
user_similarity_matrices_path = "Output/UserSimMat.pkl"

# NCF configuration
ncf_config = {
    "epoches": 50,
    "lr": 5e-4,
    "batch_size": 4096,
    "kwargs": {
        'embed_size':128,
        'hidden_nbs':[1024, 4096, 1024],
        'dropout':0.1
    }
}