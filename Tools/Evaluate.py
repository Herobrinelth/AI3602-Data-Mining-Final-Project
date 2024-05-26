import numpy as np
import matplotlib.pyplot as plt



def CountEval(recommandation, truth):
    hit = 0
    total = len(truth)
    for movie,_ in recommandation:
        if movie in truth:
            hit += 1

    return hit,total

def Plot_errors(train_error, test_error, output_dir=".", train_filename="train_error.jpg", test_filename="test_error.jpg"):
    epochs = range(1, len(train_error) + 1)
    
    train_rmse = [error[0] for error in train_error]
    train_mae = [error[1] for error in train_error]
    train_auc = [error[2] for error in train_error]
    
    test_rmse = [error[0] for error in test_error]
    test_mae = [error[1] for error in test_error]
    test_auc = [error[2] for error in test_error]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, '-.p', label='Train RMSE')
    plt.plot(epochs, train_mae, '-.p', label='Train MAE')
    plt.plot(epochs, train_auc, '-.p', label='Train AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Error over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/{train_filename}", format='jpg')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, test_rmse, '-.p', label='Test RMSE')
    plt.plot(epochs, test_mae, '-.p', label='Test MAE')
    plt.plot(epochs, test_auc, '-.p', label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Testing Error over Epochs')
    plt.legend()
    plt.savefig(f"{output_dir}/{test_filename}", format='jpg')
    plt.close()





