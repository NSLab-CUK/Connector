import torch
import torch.nn as nn
import torch.nn.functional as F
from graphrl.utils.model_utils import _L2_loss_mean
from graphrl.models.base import BaseModel


class TransR(BaseModel):

    def __init__(self, ent_dim, rel_dim, n_entities, n_relations, kg_l2loss_lambda=1e-5, training_neg_rate = 1, device="cpu"):
        
        super(TransR, self).__init__()
        self.device = device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = ent_dim
        self.relation_dim = rel_dim

        self.kg_l2loss_lambda = kg_l2loss_lambda

        self.training_neg_rate = training_neg_rate

        self.entity_embed = nn.Embedding(
            self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.projection_matrix = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.projection_matrix)

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.projection_matrix[r]

        all_embed = self.entity_embed.weight

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)

        wr_mul_h = torch.bmm(head_embed.unsqueeze(1), W_r).squeeze(1)
        wr_mul_pos_t = torch.bmm(tail_pos_embed.unsqueeze(1), W_r).squeeze(1)
        wr_mul_neg_t = torch.bmm(tail_neg_embed.unsqueeze(1), W_r).squeeze(1)

        # Trans R
        pos_score = torch.sum(
            torch.pow(wr_mul_h + r_embed - wr_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(wr_mul_h + r_embed - wr_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(wr_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(wr_mul_pos_t) + _L2_loss_mean(
            wr_mul_neg_t)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def forward(self, *input):
        return self.calc_triplet_loss(*input)