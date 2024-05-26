import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, roc_auc_score
import pickle
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from DNN.model import NCF
import Tools.Parameter as Parameter
 
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def get_metrics(predict_vec, target_vec):
    predict_vec, target_vec = np.asarray(predict_vec, dtype=float), np.asarray(target_vec, dtype=int)
    value_rmse = mse(predict_vec, target_vec)
    value_mae = mae(predict_vec, target_vec)
    value_auc = roc_auc_score(target_vec, predict_vec)
    return value_rmse, value_mae, value_auc

class ML1m(Dataset):
    '''
    加载movielens 1m的数据
    '''
    def __init__(self, data_path, sep='::', header=None):
        '''
        :param data_path: 评分文件
        :param sep: 切割符
        :param header: 是否有标题
        '''
        data = pd.read_csv(data_path, sep=sep, header=header, engine='python').to_numpy()[:, :3]
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
        '''
        分数小于等于3的就是0, 大于3的就是1
        :param score:
        :return:
        '''
        score[score<=3] = 0
        score[score>3] = 1
        return score

def test(model, dataloader, device):
    '''
    简单统计
    :param model:
    :param dataloader:
    :param device:
    :return:
    '''
    model.eval()
    target_vec, predict_vec = [], []
    with torch.no_grad():
        for feature, target in tqdm(dataloader):
            feature, target = feature.to(device), target.to(device)
            predict = model(feature)
            predict_vec.extend(predict.tolist())
            target_vec.extend(target.tolist())
    return get_metrics(predict_vec, target_vec)

def train(model, dataloader, opt, criterion, device, log_interval=100):
    '''
    训练一次模型
    :param model: 模型
    :param dataloader: 数据加载器
    :param opt: 优化器
    :param criterion: 损失函数
    :param device: 运行设备
    :param log_interval: 日志打印间隔
    :return:
    '''
    model.train()
    total_loss = 0
    tk = tqdm(dataloader, smoothing=0, mininterval=1.0)
    for i, (features, target) in enumerate(tk):
        features, target = features.to(device), target.to(device)
        out = model(features)
        loss = criterion(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.cpu().item()
        if (i+1) % log_interval == 0:
            tk.set_postfix(loss=total_loss/log_interval)
            total_loss = 0

if __name__ == "__main__":
    epoches = Parameter.ncf_config["epoches"]
    batch_size = Parameter.ncf_config["batch_size"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on {}".format(device))
    kwargs = Parameter.ncf_config["kwargs"]
    lr = Parameter.ncf_config["lr"]
    train_dataset = ML1m(Parameter.train_path)
    test_dataset = ML1m(Parameter.test_path)
    dataloader_train = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)
    model = NCF(device=device, feature_dims=train_dataset.feature_dims,
                embed_size=kwargs['embed_size'],
                hidden_nbs=kwargs['hidden_nbs'],
                user_field_idx=train_dataset.user_field_idx,
                item_field_idx=train_dataset.item_field_idx,
                dropout=kwargs['dropout'])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f = torch.nn.BCELoss()
    train_error = []
    test_error = []
    for e in range(epoches):
        print("Training {} epoch of {}".format(e+1, epoches))
        train(model, dataloader_train, opt, criterion=loss_f, device=device, log_interval=100)
        print("Validating on train set")
        value_rmse, value_mae, value_auc = test(model, dataloader_train, device)
        train_error.append((value_rmse, value_mae, value_auc))
        print("Validating on test set")
        value_rmse, value_mae, value_auc = test(model, dataloader_test, device)
        test_error.append((value_rmse, value_mae, value_auc))
    torch.save(model, Parameter.model_path)
    with open(Parameter.train_eer_path, 'wb') as f:
        pickle.dump(train_error, f, pickle.HIGHEST_PROTOCOL)
    with open(Parameter.test_eer_path, 'wb') as f:
        pickle.dump(test_error, f, pickle.HIGHEST_PROTOCOL)