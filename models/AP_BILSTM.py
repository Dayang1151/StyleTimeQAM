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
        # (batch_size,seq_len,hidden_size*2)
        G = Q.transpose(-1, -2) @ self.U.expand(Q.shape[0], -1, -1) @ A
        # (batch_size,hidden_size*2,hidden_size*2)
        G = torch.tanh(G)
        # (batch_size,hidden_size*2,hidden_size*2)
        Q_pooling = G.max(dim=-1)[0]
        A_pooling = G.max(dim=-2)[0]
        # (batch_size,hidden_size*2)
        Q_pooling = Q_pooling.softmax(dim=-1)
        A_pooling = A_pooling.softmax(dim=-1)
        # (batch_size,hidden_size*2)
        rq = Q @ Q_pooling.unsqueeze(-1)
        ra = A @ A_pooling.unsqueeze(-1)
        # (batch_size,seq_len,1)
        rq = rq.squeeze(-1)
        ra = ra.squeeze(-1)
        # (batch_Size,sqlen)
        return rq, ra

class AP_BILSTM(nn.Module):
    def __init__(self, arg,device="gpu"):
        super(AP_BILSTM, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(arg.vocabs_size + 1, embedding_dim=arg.embedding_dim)
        self.embedding.to(device)
        self.LSTM = nn.LSTM(input_size=arg.embedding_dim, hidden_size=arg.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.prediction = prediction[arg.prediction](arg)
        self.coAttention = CoAttention(arg.lstm_hidden_size*2,arg.init_U)


    def forward(self, q, a):
        # (batch_size,seq_len)
        q = self.embedding(q)
        a = self.embedding(a)
        # (batch_size,seq_len,embedding)
        q,(h,c) = self.LSTM(q)
        a,(h,c) = self.LSTM(a)
        # (batch_size,seq_len,hidden_size*2)
        q = q.permute(0, 2, 1)
        a = a.permute(0, 2, 1)
        rq, ra = self.coAttention(q, a)
        # (batch_size,seq_len)
        logits = self.prediction(rq, ra)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities