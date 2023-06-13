import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RE2.modules.prediction import registry as prediction

class BI_LSTM(nn.Module):
  def __init__(self, arg,device="gpu"):
    super(BI_LSTM,self).__init__()
    self.device = device
    self.embedding = nn.Embedding(arg.vocabs_size + 1, embedding_dim=arg.embedding_dim)
    self.embedding.to(device)
    self.LSTM = nn.LSTM(input_size=arg.embedding_dim, hidden_size=arg.hidden_size, bidirectional=True, batch_first=True)
    self.prediction = prediction[arg.prediction](arg)
    self.l3 = nn.Linear(arg.max_length * arg.hidden_size * 2, arg.hidden_size)  # 特征输入

  def forward (self,question,anwser):  # (batch_size,seq_length）
    q_emb = self.embedding(question)  # (batch_size,seq_length,embedding)
    a_emb = self.embedding(anwser)


    Q_out, (final_hidden_state, final_cell_state) = self.LSTM(q_emb)
    A_out, (final_hidden_state, final_cell_state) = self.LSTM(a_emb)  #(batch_size,seq_length,hidden_size*2)

    Q_out = Q_out.reshape(-1, Q_out.shape[1] * Q_out.shape[2])  # (batch_size,seq_length * hidden_size * 2)
    A_out = A_out.reshape(-1, A_out.shape[1] * A_out.shape[2])

    out_a = self.l3(Q_out)  # (batch_size,hidden_size)
    out_b = self.l3(A_out)  # (batch_size,hidden_size)

    logits = self.prediction(out_a, out_b)
    probabilities = nn.functional.softmax(logits, dim=-1)
    return logits, probabilities

    # rq = Q_out.max(dim=-1)[0]
    # ra = A_out.max(dim=-1)[0]
    #
    # cos = torch.cosine_similarity(rq,ra,dim=1)
    # cos = cos.unsqueeze(1)
    #
    # y_cos = torch.sigmoid(cos)#(batch_size,1)
    # return y_cos
#    


