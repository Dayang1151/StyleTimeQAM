import numpy as np
import torch
import torch.nn as nn
from models.RE2.modules.prediction import registry as prediction

class CoAttention(nn.Module):
    def __init__(self, hidden_size, init_U='randn'):
        super().__init__()
        if init_U == 'zeros':
            self.U = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        else:
            self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, Q, A):
        # (batch_size,out_channels,seq_len)
        G = Q.transpose(-1, -2) @ self.U.expand(Q.shape[0], -1, -1) @ A
        G = torch.tanh(G)
        Q_pooling = G.max(dim=-1)[0]
        A_pooling = G.max(dim=-2)[0]
        Q_pooling = Q_pooling.softmax(dim=-1)
        A_pooling = A_pooling.softmax(dim=-1)
        rq = Q @ Q_pooling.unsqueeze(-1)
        ra = A @ A_pooling.unsqueeze(-1)
        rq = rq.squeeze(-1)
        ra = ra.squeeze(-1)
        return rq, ra

class AP_CNN(nn.Module):
    def __init__(self, arg,device="gpu"):
        super(AP_CNN, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(arg.vocabs_size + 1, embedding_dim=arg.embedding_dim)
        self.embedding.to(device)
        self.conv1 = nn.Conv1d(in_channels=arg.embedding_dim, out_channels=arg.out_channels,kernel_size=arg.kernel_sizes,padding=arg.padding)
        self.fc = nn.Linear(int(arg.out_channels * arg.max_length / 2), arg.hidden_size)
        self.prediction = prediction[arg.prediction](arg)
        self.coAttention = CoAttention(arg.hidden_size,arg.init_U)


    def forward(self, q, a):
        # (batch_size,seq_len)
        q = self.embedding(q)
        a = self.embedding(a)
        # (batch_size,seq_len,embedding)
        q = q.permute(0, 2, 1)
        a = a.permute(0, 2, 1)
        # (batch_size,embedding,seq_len)
        q = self.conv1(q)
        a = self.conv1(a)
        # (batch_size,out_channels,seq_len)
        rq, ra = self.coAttention(q, a)
        logits = self.prediction(rq, ra)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities