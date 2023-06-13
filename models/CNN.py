import numpy as np
import torch
import torch.nn as nn
from models.RE2.modules.prediction import registry as prediction

class CNN(nn.Module):
    def __init__(self, arg,device="gpu"):
        super(CNN, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(arg.vocabs_size + 1, embedding_dim=arg.embedding_dim)
        self.embedding.to(device)
        self.conv1 = nn.Conv1d(in_channels=arg.embedding_dim, out_channels=arg.out_channels,kernel_size=arg.kernel_sizes,padding=arg.padding)
        self.pooling = nn.AvgPool1d(kernel_size=2)
        # self.dropout = nn.Dropout(model_param['dropout'])
        self.fc = nn.Linear(int(arg.out_channels * arg.max_length / 2), arg.hidden_size)
        self.prediction = prediction[arg.prediction](arg)


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
        q = self.pooling(q)
        a = self.pooling(a)
        # (batch_size,out_channels,seq_len/kernel_size)
        q = torch.flatten(q, 1)
        a = torch.flatten(a, 1)
        # (batch_size,out_channels*seq_len/kernel_size)
        q = self.fc(q)
        a = self.fc(a)
        # (batch_size,50)
        logits = self.prediction(q, a)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities
