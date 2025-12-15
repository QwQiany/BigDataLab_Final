from torch import nn
from models import HGNN_conv
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, n_hid2, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)   #定义第一层图卷积
        self.hgc2 = HGNN_conv(n_hid, n_class)   #定义第二层图卷积
        self.attention = Attention(n_hid2)
        
        self.MLP = nn.Sequential(
            nn.Linear(n_hid2, n_class),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, G):

        x = F.relu(self.hgc1(x, G))        #执行图卷积和激活函数
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        x = torch.stack([x,x],dim=1)
        x, att = self.attention(x)
        x = self.MLP(x)
        return x
