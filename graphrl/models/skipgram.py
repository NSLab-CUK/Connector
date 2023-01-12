import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from graphrl.models.base import BaseModel

class SkipGram(BaseModel):
    def __init__(self, embedding_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim

        self.u_embedding = nn.Embedding(self.embedding_size, self.embedding_dim, sparse=True)
        self.v_embedding = nn.Embedding(self.embedding_size, self.embedding_dim, sparse=True)

        init_range = 1.0 / self.embedding_dim
        init.uniform_(self.u_embedding.weight.data, -init_range, init_range)
        init.constant_(self.v_embedding.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        u_pos = self.u_embedding(pos_u)
        v_pos = self.v_embedding(pos_v)
        v_neg = self.u_embedding(neg_v)

        score = torch.sum(torch.mul(u_pos, v_pos), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(v_neg, u_pos.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embedding.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.embedding_dim))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))