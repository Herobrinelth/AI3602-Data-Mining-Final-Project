import torch
import torch.nn as nn
from .layers import Feature_Embedding, My_MLP

class NCF(nn.Module):
    def __init__(self, device, feature_dims, embed_size, hidden_nbs, user_field_idx, item_field_idx, dropout=0):
        super(NCF, self).__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = Feature_Embedding(feature_dims=feature_dims, embed_size=embed_size, device=device)
        self.embed_out_dim = len(feature_dims) * embed_size
        self.mlp = My_MLP(device=device, input_dim=self.embed_out_dim,
                          hidden_nbs=hidden_nbs,
                          out_dim=hidden_nbs[-1], last_act=None, drop_rate=dropout)
        self.fc = nn.Linear(hidden_nbs[-1]+embed_size, 1).to(device)

    def forward(self, data):
        data = self.embedding(data)
        user_x = data[:, self.user_field_idx].squeeze(1)
        item_x = data[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        data = data.view(-1, self.embed_out_dim)
        out_mlp = self.mlp(data)
        data = torch.cat([gmf, out_mlp], dim=-1)
        out = torch.sigmoid(self.fc(data)).squeeze(1)
        return out
    
class LSTM(nn.Module):
    def __init__(self, device, feature_dims, embed_size, dropout=0, gru_layers=1):
        super(LSTM, self).__init__()
        self.embedding = Feature_Embedding(feature_dims=feature_dims, embed_size=embed_size, device=device)
        self.embed_out_dim = len(feature_dims) * embed_size
        self.fc = nn.Sequential(
            nn.Linear(self.embed_out_dim, self.embed_out_dim//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_out_dim//2, 1)
            ).to(device)
        self.gru = nn.GRU(embed_size, embed_size, gru_layers, batch_first = True).to(device)

    def forward(self, data):
        data = self.embedding(data)
        item_x = data
        item_x, _ = self.gru(item_x)
        item_x = item_x.reshape(item_x.shape[0], -1)
        out = torch.sigmoid(self.fc(item_x)).squeeze(1)
        return out
