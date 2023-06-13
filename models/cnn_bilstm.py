import torch
import torch.nn as nn
from models.RE2.modules.prediction import registry as prediction

class CNN_BILSTM(nn.Module):
    def __init__(self, args, embeddings, device="gpu"):
        super(CNN_BILSTM, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embedding.float()
        self.embedding.weight.requires_grad = True
        self.embedding.to(device)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=50,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )
        self.LSTM = nn.LSTM(input_size=50, hidden_size=50, bidirectional=True, batch_first=True)
        self.l3 = nn.Linear(args.max_length * 50 * 2,args.hidden_size)  # 特征输入
        # self.l3 = nn.Linear(model_param['context_num'] * model_param['seq_length'] * model_param['hidden_size'] * 2,
        #                     model_param['n_class'])  # 特征输入
        # self.l4 = nn.Dropout(args.dropout)
        # self.l5 = nn.BatchNorm1d(args.num_classes)
        self.prediction = prediction[args.prediction](args)


    def forward(self, a,b):  # (batch_size,seq_length)
        a = self.embedding(a)  # (batch_size,seq_length,embedding)
        b = self.embedding(b)

        a = a.permute(0, 2, 1)  # (batch_size,embedding,seq_length)
        b = b.permute(0, 2, 1)


        a = self.conv2(a)  # (batch_size,out_channels,seq_length)
        b = self.conv2(b)

        a = a.permute(0, 2, 1)   # (batch_size,seq_length,out_channels)
        b = b.permute(0, 2, 1)

        out_a, (final_hidden_state, final_cell_state) = self.LSTM(a)  # (batch_size,seq_length,hidden_size * 2)
        out_b, (final_hidden_state, final_cell_state) = self.LSTM(b)

        out_a = out_a.reshape(-1, out_a.shape[1] * out_a.shape[2])  # (batch_size,seq_length * hidden_size * 2)
        out_b = out_b.reshape(-1, out_b.shape[1] * out_b.shape[2])

        out_a = self.l3(out_a)  # (batch_size,hidden_size)
        out_b = self.l3(out_b)  # (batch_size,hidden_size)

        logits = self.prediction(out_a, out_b)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities
