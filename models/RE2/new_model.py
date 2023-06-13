import torch
import torch.nn as nn
import numpy as np
from models.RE2.modules import Module, ModuleList, ModuleDict
from models.RE2.modules.encoder import Encoder
from models.RE2.modules.alignment import registry as alignment
from models.RE2.modules.fusion import registry as fusion
from models.RE2.modules.connection import registry as connection
from models.RE2.modules.pooling import Pooling
from models.RE2.modules.prediction import registry as prediction

class RE2(Module):
    def __init__(self, args, device="gpu"):
        super().__init__()
        self.dropout = args.dropout
        self.device = device
        # self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        # self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings))
        # self.embedding.float()
        # self.embedding.weight.requires_grad = True
        self.embedding = nn.Embedding(args.vocabs_size + 1, embedding_dim=args.embedding_dim)
        self.embedding.to(device)
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])
        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, a, b):
        # (batch_size,max_length)
        torch.set_printoptions(threshold=np.inf)
        print(a, b)
        mask_a = torch.ne(a, 0).unsqueeze(2).to(self.device)
        mask_b = torch.ne(b, 0).unsqueeze(2).to(self.device)
        a = self.embedding(a)  # (batch_size,max_length,embedding)
        b = self.embedding(b)
        res_a, res_b = a, b
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a) # (batch_size,max_length,embedding)
        b = self.pooling(b, mask_b)
        # print(a,b)
        logits = self.prediction(a, b)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities
