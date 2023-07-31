import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


# from d2l import torch as d2l


class Attention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)

        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out, scores


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)


class DotProductAttention(nn.Module):


    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def user_connect(user_embed, batch_size, context_num):  # user_embed (batch_Size + 100,1,embedding_size)
    a = user_embed[:context_num].unsqueeze(0)  # user_embed (1,context_num,1,embedding_size)
    for i in range(1, batch_size):
        b = user_embed[i:i + context_num].unsqueeze(0)
        a = torch.cat([a, b])

    return a  # a (batch_size,context_num,1,embedding_size)


def future_user(user_embed, batch_size, context_num):
    # user_embed(batch_size + 100,1,emb_dim)
    # a = user_embed[1:batch_size + 1] # user_embed (batch_size,1,emb_dim)

    res = user_embed[1:2]  # user_embed (1,1,emb_dim)
    for j in range(1, context_num):
        res = torch.cat([res, user_embed[1 + j:2 + j]], dim=1)  # user_embed (1,100,emb_dim)
    for i in range(2, batch_size + 1):
        a = user_embed[i:i + 1]  # user_embed (1,1,emb_dim)
        for j in range(1, context_num):
            a = torch.cat([a, user_embed[i + j:i + j + 1]], dim=1)
        res = torch.cat([res, a], dim=0)
    return res  # (batchsize,100,emb_dim)


class STQAM(nn.Module):
    def __init__(self, args, device='gpu'):
        super(STQAM, self).__init__()
        self.device = device
        self.batch_size = args.batch_size
        self.embedding_sen = nn.Embedding(args.vocabs_size + 1, embedding_dim=args.embedding_dim)
        self.embedding_user = nn.Embedding(args.user_num + 1, embedding_dim=args.embedding_dim)
        self.embedding_sen.to(device)
        self.embedding_user.to(device)

        self.embedding_size = args.embedding_dim

        self.LSTM = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.lstm_hidden_size, bidirectional=True,
                            batch_first=True)
        self.pooling = nn.AvgPool1d(kernel_size=2)  # emd_size / 2
        self.attention = DotProductAttention(0.5)
        self.self_attention = nn.MultiheadAttention(embed_dim=args.embedding_dim * 2, num_heads=1)
        self.linear = nn.Linear(in_features=args.embedding_dim * 2, out_features=args.num_classes)
        self.linear_test = nn.Linear(in_features=args.embedding_dim, out_features=args.num_classes)
        self.linear_match_user = nn.Linear(in_features=args.embedding_dim * 2, out_features=args.embedding_dim)
        self.linear_match_sen = nn.Linear(in_features=args.embedding_dim * 4, out_features=args.embedding_dim)
        self.time_linear = nn.Linear(in_features=args.embedding_dim, out_features=1)

    def forward(self, sentence, user, s_time,t_p):
        # print(s_time)
        # sentence:(batch_size + 200,sen_len)  user:(batch_size + 100,1) s_time:(batch_size,100)

        #   User Style-Aware Attention Mechanism for Question Extraction start

        sen_emd = self.embedding_sen(sentence)  # conversation embedding  (batch_size + 200,sen_len,embedding_dim)

        sen_lstmed, (final_hidden_state, final_cell_state) = self.LSTM(
            sen_emd)  # conversation BILSTM层  (batch_size + 200,sen_len,embedding_dim(hidden) * 2)

        sen_pooled = self.pooling(sen_lstmed)  # conversation pooling (batch_size + 200,seq_len,embedding_dim)
        sen_pooled = sen_pooled[100:200 + self.batch_size]  # (batch_size + 100,seq_len,embedding_dim)

        user = user.unsqueeze(1)  # (batch_size,1)
        user_embed = self.embedding_user(user)  # user embedding (batch_size + 100,1,embedding_dim)

        att_output = self.attention(user_embed, sen_pooled, sen_pooled)  # attention (batch_size + 100,1,embedding_dim)

        conn = torch.cat([user_embed, att_output], dim=-1)  #  (batch_size + 100,1,embedding_dim * 2)

        s_att_output, _ = self.self_attention(conn, conn, conn)  #  (batch_size + 100,1,embedding_dim * 2)

        s_output = s_att_output.reshape(s_att_output.shape[0], -1)  # (batch_size + 100,embedding_dim * 2)

        label_logits = self.linear(s_output)[:self.batch_size]  #  (batch_size,2)

        label_probs = nn.functional.softmax(label_logits, dim=-1)  # softmax(batch_size ,2)

        #  User Style-Aware Attention Mechanism for Question Extraction end

        #  Time-Aware Attention Mechanism for Answering Matching start
        #  matching degrees between user style start
        # user_embed (batch_Size,1,embedding_dim)
        user_conn = user_connect(user_embed=user_embed, batch_size=self.batch_size,
                                 context_num=100)  # (batch_size,100,1,embedding_dim)
        user_embed_match = user_embed.unsqueeze(1)  # user_embed (batch_Size + 100,1,1,embedding_size)
        user_embed_match = user_embed_match.expand(self.batch_size + 100, 100, 1,
                                                   self.embedding_size)  # user_embed (batch_Size,100,1,embedding_size)
        user_final = torch.cat([user_embed_match[:self.batch_size], user_conn],
                               dim=2)  # user_embed (batch_Size,100,2,embedding_size)
        user_final = user_final.reshape(user_final.shape[0], user_final.shape[1],
                                        -1)  # user_embed (batch_Size,100,2 * embedding_size)
        #  matching degrees between user style end

        #  matching degrees between text information start
        # att_output (batch_size + 200,1,embedding_dim)
        sen_conn = user_connect(user_embed=s_att_output, batch_size=self.batch_size, context_num=100)
        sen_att_out = s_att_output.unsqueeze(1)  # sen_att_out (batch_Size + 100,1,1,embedding_size)
        sen_att_out = sen_att_out.expand(self.batch_size + 100, 100, 1, self.embedding_size * 2)
        sen_final = torch.cat([sen_att_out[:self.batch_size], sen_conn],
                              dim=2)  # sen_final (batch_Size,100,2,embedding_size)
        sen_final = sen_final.reshape(sen_final.shape[0], sen_final.shape[1],
                                      -1)  # sen_final (batch_Size,100,2 * embedding_size)
        #  matching degrees between text information end

        # user_embed (batch_Size + 200,100,2,embedding_size)

        user_output = self.linear_match_user(user_final)  # user_output (batch_Size,100,embedding_size)
        user_output = nn.functional.relu(user_output)  # user_embed (batch_Size,100,embedding_size)

        sen_output = self.linear_match_sen(sen_final)  # sen_output (batch_Size,100,embedding_size)
        sen_output = nn.functional.relu(sen_output)  # user_embed (batch_Size+200,100,embedding_size)

        #   s_time (batch_size,100)
        # print(user_embed)
        user_embed = future_user(user_embed, self.batch_size, 100)
        user_time = self.time_linear(user_embed)  # (batch_size,100,1)
        user_time = nn.functional.softmax(user_time).reshape(user_time.shape[0], -1)  # (batch_Size,100,1)
        s_time = nn.functional.normalize(s_time.float(), p=1, dim=1)
        # print(s_time)
        # for i in range(len(s_time)):
        #     if 0 in s_time[i]:
        #         for j in range(len(s_time[i])):
        #             if s_time[i][j] == 0:
        #                 s_time[i][j] = 1e-9
        # for k in range(len(s_time)):
        fun_time = torch.exp(torch.div(s_time, user_time)).unsqueeze(-1)  # s_time (batch_size,100,1)

        # (batch_size,100,1)
        label_final = torch.argmax(label_probs, dim=-1).unsqueeze(-1)  # (batch_size,1)
        label_final = label_final.expand(label_final.shape[0], 100).unsqueeze(-1)  # (batch_size,100,1)


        if t_p == 0:
            #  full model
            final_match = torch.cat([user_output, sen_output], dim=-1)  # final_match (batch_Size,100,embedding_size * 2)
            final_match = torch.mul(torch.mul(final_match, label_final),
                                    fun_time)  # final_match (batch_Size,100,embedding_size * 2)
            match_logits = self.linear(final_match)  # (batch_size,100,2)

        elif t_p == 1:
            #  without part of label
            final_match = torch.cat([user_output, sen_output], dim=-1)  # final_match (batch_Size,100,embedding_size * 2)
            final_match = torch.mul(final_match, fun_time)  # final_match (batch_Size,100,embedding_size * 2)
            match_logits = self.linear(final_match)  # (batch_size,100,2)

        elif t_p == 2:
            # without timestamp
            final_match = torch.cat([user_output, sen_output], dim=-1)  # final_match (batch_Size,100,embedding_size * 2)
            final_match = torch.mul(final_match, label_final)  # final_match (batch_Size,100,embedding_size * 2)
            match_logits = self.linear(final_match)  # (batch_size,100,2)

        elif t_p == 3:
            # without sentences
            final_match = user_output
            final_match = torch.mul(torch.mul(final_match, label_final),
                                    fun_time)  # final_match (batch_Size,100,embedding_size)
            match_logits = self.linear_test(final_match)  # (batch_size,100,2)

        else:
            # without users
            final_match = sen_output
            final_match = torch.mul(torch.mul(final_match, label_final),
                                    fun_time)  # final_match (batch_Size,100,embedding_size)
            match_logits = self.linear_test(final_match)  # (batch_size,100,2)



        match_logits = match_logits.reshape(match_logits.shape[0] * match_logits.shape[1], -1)  # (batch_size * 100,2)
        match_probs = nn.functional.softmax(match_logits)  # (batch_size * 100,2)
        return label_logits, label_probs, match_logits, match_probs
        # return label_logits,label_probs
